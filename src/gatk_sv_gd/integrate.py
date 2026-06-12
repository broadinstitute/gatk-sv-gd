"""
Integrate GD calls into a GATK-SV final VCF.

Annotates existing DEL/DUP records that match a known GD region, overwrites
and reconciles their genotypes from a GD-call carrier manifest, and emits
novel records for GD calls that have no matching variant in the input VCF.

Three processing phases:
1. Non-NAHR partial overlap annotation — any DEL/DUP overlapping a non-NAHR
   region by >= --non-nahr-overlap gets GENOMIC_DISORDER / GD_CLUSTER INFO.
2. GD-call-centric sample matching — for each GD locus with carriers, find
   all overlapping VCF records (RO >= --reciprocal-overlap) and force all
   samples to hom-ref. Emit a new GD record with authoritative genotypes.
3. Write surviving VCF records — records not cleared by Phase 2 are written
   as-is (no longer drops all-hom-ref Phase 2 records; those are already
   cleared). Novel GD records without any VCF match are emitted as Phase 2
   novel records.

Usage (via CLI)::

    gatk-sv-gd integrate \\
        --vcf input.vcf.gz \\
        --gd-calls gd_cnv_calls.tsv.gz \\
        --gd-table gd_table.tsv \\
        --par-bed par.hg38.bed \\
        --ploidy-table ploidy.tsv \\
        --out-vcf integrated.vcf.gz

Requirements:
    ``bcftools`` must be available on PATH (used for final VCF sorting).
"""

import argparse
import csv
import gzip
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Set, Text, Tuple

import pysam
from intervaltree import IntervalTree

from gatk_sv_gd._util import (
    fraction_covered,
    get_logger,
    overlap_bases,
    reciprocal_overlap,
    setup_logging,
)
from gatk_sv_gd.models import GDTable

# ── Constants ────────────────────────────────────────────────────────

GENOMIC_DISORDER_KEY = "GENOMIC_DISORDER"

RESET_PESR_FORMATS_DICT = {
    "SR_GT": (0,),
    "SR_GQ": 99,
    "PE_GT": (0,),
    "PE_GQ": 99,
}

# Standard header lines that must be present before writing genotypes /
# novel records.  Only added if the field is absent.
_INFO_HEADERS_REQUIRED = [
    f'##INFO=<ID={GENOMIC_DISORDER_KEY},Number=1,Type=String,'
    'Description="Genomic disorder region">',
    '##INFO=<ID=GD_CLUSTER,Number=1,Type=String,'
    'Description="Genomic disorder cluster locus">',
    '##INFO=<ID=GD_BP1,Number=1,Type=String,'
    'Description="Genomic disorder breakpoint 1">',
    '##INFO=<ID=GD_BP2,Number=1,Type=String,'
    'Description="Genomic disorder breakpoint 2">',
    '##INFO=<ID=SVTYPE,Number=1,Type=String,'
    'Description="Type of structural variant">',
    '##INFO=<ID=SVLEN,Number=.,Type=Integer,'
    'Description="Difference in length between REF and ALT alleles">',
    '##INFO=<ID=END,Number=1,Type=Integer,'
    'Description="End position of the structural variant">',
    '##INFO=<ID=ALGORITHMS,Number=.,Type=String,'
    'Description="Source algorithms">',
    '##INFO=<ID=EV,Number=.,Type=String,'
    'Description="Classes of evidence supporting final genotype">',
]

_FORMAT_HEADERS_REQUIRED = [
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
    '##FORMAT=<ID=GQ,Number=1,Type=Integer,'
    'Description="Genotype quality">',
    '##FORMAT=<ID=RD_CN,Number=1,Type=Integer,'
    'Description="Estimated copy number from read depth">',
    '##FORMAT=<ID=RD_GQ,Number=1,Type=Integer,'
    'Description="Read depth genotype quality">',
]

# ── GD table loading ─────────────────────────────────────────────────


def _build_trees_from_gd_table(
    gd_table_path: str,
) -> Tuple[
    Dict[str, "IntervalTree"],
    Dict[str, "IntervalTree"],
    Dict[str, dict],
]:
    """Parse GD table into NAHR/non-NAHR interval trees and metadata dict.

    Uses ``GDTable`` (handles column-name lookup, BP normalisation, mixed
    numeric/alphanumeric BP ordering).  Stores ``(GD_ID, svtype)`` as interval
    data so that a DEL variant cannot match a DUP region's ID.

    Returns
    -------
    nahr_trees : dict chrom -> IntervalTree
    non_nahr_trees : dict chrom -> IntervalTree
    gd_metadata : dict GD_ID -> {cluster, bp1, bp2, nahr, svtype}
    """
    gd_table = GDTable(gd_table_path)

    nahr_trees: Dict[str, IntervalTree] = defaultdict(IntervalTree)
    non_nahr_trees: Dict[str, IntervalTree] = defaultdict(IntervalTree)
    gd_metadata: Dict[str, dict] = {}

    # Local logger for this function (called before main() sets up logging)
    _log = get_logger("integrate")

    for _cluster, locus in gd_table.get_all_loci().items():
        chrom = locus.chrom
        for entry in locus.gd_entries:
            gd_id = entry["GD_ID"]
            svtype = entry["svtype"]
            start = entry["start_GRCh38"]
            end = entry["end_GRCh38"]
            is_nahr = entry["NAHR"] == "yes"

            # Reject inverted intervals — IntervalTree does not allow start >= end.
            if start >= end:
                _log.error(
                    "Skipping GD entry %r on %s: inverted or zero-length "
                    "interval (start=%d, end=%d)",
                    gd_id,
                    chrom,
                    start,
                    end,
                )
                continue

            gd_metadata[gd_id] = {
                "cluster": locus.cluster,
                "bp1": entry["BP1"],
                "bp2": entry["BP2"],
                "nahr": is_nahr,
                "svtype": svtype,
            }

            # Store (GD_ID, svtype) as interval data so NAHR matching is
            # svtype-aware and a DEL variant cannot match a DUP region.
            if is_nahr:
                nahr_trees[chrom].addi(start, end, (gd_id, svtype))
            else:
                non_nahr_trees[chrom].addi(start, end, (gd_id, svtype))

    return nahr_trees, non_nahr_trees, gd_metadata


# ── Input file readers ───────────────────────────────────────────────


def read_gd_calls(calls_path: str) -> Dict[Tuple[str, str], dict]:
    """Read GD-call manifest TSV.

    Supports two formats:

    1. **Wide format** (output of ``call`` subcommand, ``gd_cnv_calls.tsv.gz``):
       A header row followed by tabular data with columns including
       ``sample``, ``GD_ID``, ``chrom``, ``start``, ``end``, ``svtype``,
       ``is_carrier``.  Rows with ``is_carrier == "True"`` (or ``True``)
       are grouped by ``(GD_ID, svtype)`` and carrier samples collected.

    2. **Legacy narrow format** (6-column TSV, no header):
       ``chrom``, ``pos`` (0-based), ``end``, ``region_id``, ``svtype``,
       comma-separated-samples.

    Lines starting with ``#`` are skipped in the narrow format.
    An empty or ``.`` sample field produces an empty carrier set.

    Returns
    -------
    dict mapping ``(region_id, svtype)`` to
    ``{chrom, pos, end, samples}``
    """
    # Open file transparently (support .gz and plain text)
    if calls_path.endswith(".gz"):
        fp = gzip.open(calls_path, "rt")
    else:
        fp = open(calls_path, "r")

    try:
        # Peek at first non-comment line to detect format
        first_line = ""
        for line in fp:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                # Use stripped for header detection (column names)
                # but keep original line (rstrip only newlines) for parsing
                first_line = line.rstrip("\n\r")
                break

        # If the first field looks like a header (contains known column names)
        # treat as wide format
        fields = first_line.strip().split("\t")
        if _looks_like_wide_header(fields):
            return _read_wide_format(fp, first_line)
        else:
            return _read_narrow_format(fp, first_line)
    finally:
        fp.close()


def _looks_like_wide_header(fields: List[str]) -> bool:
    """Return True if the fields look like the wide gd_cnv_calls header."""
    header_set = set(fields)
    # Require GD_ID AND is_carrier (these never appear in narrow format)
    # plus at least one of {start, chrom} to disambiguate
    wide_core = {"GD_ID", "is_carrier"}
    wide_positional = {"start", "chrom"}
    return (
        wide_core.issubset(header_set)
        and bool(wide_positional.intersection(header_set))
    )


def _read_wide_format(
    fp,  # file pointer already positioned after first line
    first_line: str,
) -> Dict[Tuple[str, str], dict]:
    """Read the wide ``gd_cnv_calls.tsv.gz`` format.

    Tracks all (GD_ID, svtype) entries from the file (needed for Phase 3
    novel record detection), but only collects carrier samples from rows
    where is_carrier is True.
    """
    reader = csv.DictReader(
        [first_line] + fp.readlines(),
        delimiter="\t",
    )
    # Group by (GD_ID, svtype) and collect carrier samples
    groups: Dict[Tuple[str, str], dict] = {}
    for row in reader:
        gd_id = row["GD_ID"]
        svtype = row["svtype"]
        key = (gd_id, svtype)
        if key not in groups:
            groups[key] = {
                "chrom": row["chrom"],
                "pos": int(row["start"]),
                "end": int(row["end"]),
                "samples": set(),
            }
        is_carrier = row.get("is_carrier", "").strip()
        if is_carrier in ("True", "true", "1"):
            groups[key]["samples"].add(row["sample"])
    return groups


def _read_narrow_format(
    fp,  # file pointer already positioned after first line
    first_line: str,
) -> Dict[Tuple[str, str], dict]:
    """Read the legacy 6-column narrow TSV format (no header)."""
    gd_calls: Dict[Tuple[str, str], dict] = {}
    for line in [first_line] + fp.readlines():
        if line.startswith("#"):
            continue
        cols = line.rstrip("\n\r").split("\t")
        if len(cols) < 6:
            continue
        chrom, pos, end, region_id, svtype, samples_str = (
            cols[0], int(cols[1]), int(cols[2]),
            cols[3], cols[4], cols[5],
        )
        samples: Set[str] = (
            set(samples_str.split(","))
            if samples_str and samples_str != "."
            else set()
        )
        gd_calls[(region_id, svtype)] = {
            "chrom": chrom,
            "pos": pos,
            "end": end,
            "samples": samples,
        }
    return gd_calls


def read_ploidy_table(path: str) -> Dict[str, Dict[str, int]]:
    """Read wide GATK-SV ploidy table (sample + one column per contig).

    Returns
    -------
    dict sample -> {contig -> ploidy_int}
    """
    ploidy_dict: Dict[str, Dict[str, int]] = {}
    with open(path, "r") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            tokens = line.strip().split("\t")
            if not tokens or tokens[0].startswith("#"):
                continue
            sample = tokens[0]
            ploidy_dict[sample] = {
                header[i]: int(tokens[i])
                for i in range(1, min(len(header), len(tokens)))
            }
    return ploidy_dict


def _read_bed_to_trees(bed_path: str) -> Dict[str, "IntervalTree"]:
    """Read a BED file into per-chrom IntervalTrees."""
    trees: Dict[str, IntervalTree] = defaultdict(IntervalTree)
    with open(bed_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            record = line.strip().split("\t")
            if len(record) < 3:
                continue
            trees[record[0]].addi(int(record[1]), int(record[2]))
    return trees


# ── Genomic helpers ──────────────────────────────────────────────────


def is_in_par_region(
    chrom: str,
    pos: int,
    stop: int,
    par_trees: Dict[str, "IntervalTree"],
    cutoff: float = 0.5,
) -> bool:
    """Return True if the interval overlaps a PAR region by >= cutoff fraction."""
    length = stop - pos
    if length == 0:
        return False
    if chrom in par_trees:
        for par_ov in par_trees[chrom].overlap(pos, stop):
            ov = overlap_bases(par_ov.begin, par_ov.end, pos, stop)
            if ov / length >= cutoff:
                return True
    return False


def get_expected_cn(
    chrom: str,
    pos: int,
    stop: int,
    sample: str,
    ploidy_dict: Dict[str, Dict[str, int]],
    par_trees: Dict[str, "IntervalTree"],
) -> int:
    """Return expected copy number for a sample at a genomic position.

    PAR regions always return 2.  Otherwise returns the ploidy for the
    contig from the ploidy table, defaulting to 2 if sample or contig absent.
    """
    if is_in_par_region(chrom, pos, stop, par_trees):
        return 2
    if sample not in ploidy_dict:
        return 2
    return ploidy_dict[sample].get(chrom, 2)


# ── Carrier extraction ───────────────────────────────────────────────


def _is_homref(gt) -> bool:
    """Check if a genotype represents a hom-ref state.

    Handles both tuple ``(0, 0)`` (pysam / real VCF) and list ``[0, 0]``
    (JSON-encoded scenario data, test stubs).  Also handles ``None``
    (no-call) which is treated as non-hom-ref.
    """
    if gt is None:
        return False
    gt_tuple = tuple(gt) if not isinstance(gt, tuple) else gt
    return gt_tuple == (0, 0)


def _extract_vcf_carriers(record: "pysam.VariantRecord") -> Set[str]:
    """Return carrier samples from a VCF record.

    Carriers are samples whose genotype is **not** ``(0, 0)``.
    """
    carriers: Set[str] = set()
    for sample_name, gt_dict in record.samples.items():
        gt = gt_dict.get("GT", (0, 0))
        if not _is_homref(gt):
            carriers.add(sample_name)
    return carriers


def sample_overlap(set_a: Set[str], set_b: Set[str]) -> Optional[float]:
    """Return sample-overlap score between two carrier sets.

    Defined as ``|A ∩ B| / max(|A|, |B|)``.

    Returns
    -------
    float in ``[0, 1]`` when at least one set is non-empty.
    ``None`` when **both** sets are empty (signals "undefined, use fallback").

    Example
    -------
    >>> sample_overlap({"A", "B"}, {"A", "B", "C"})
    0.6666666666666666
    >>> sample_overlap({"A"}, {"B"})
    0.0
    >>> sample_overlap(set(), set())
    >>>
    """
    if not set_a and not set_b:
        return None
    intersection = len(set_a & set_b)
    union_max = max(len(set_a), len(set_b))
    return intersection / union_max


# ── Genotype update ──────────────────────────────────────────────────


def update_genotype(
    gt: dict,
    sample: str,
    is_carrier: bool,
    ecn: int,
    svtype: str,
) -> None:
    """Update per-sample genotype dict in-place.

    Replicates the script's genotyping model exactly (locked decision #3):
    - ecn == 0  -> no-call (None, None), RD_CN=0, RD_GQ=0, return early
    - carrier   -> het (0, 1); DEL: RD_CN=max(ecn-1,0); DUP: RD_CN=ecn+1
    - non-carrier -> homref (0, 0); RD_CN=ecn
    PE/SR FORMAT fields are reset only when present.
    """
    if ecn == 0:
        gt["GT"] = (None, None)
        gt["RD_CN"] = 0
        gt["RD_GQ"] = 0
        return

    gt["EV"] = ("RD",)
    for key, val in RESET_PESR_FORMATS_DICT.items():
        if key in gt:
            gt[key] = val

    if is_carrier:
        gt["GT"] = (0, 1)  # Always het per locked decision #3
        if svtype == "DEL":
            gt["RD_CN"] = max(ecn - 1, 0)
        elif svtype == "DUP":
            gt["RD_CN"] = ecn + 1
        gt["RD_GQ"] = 99
        gt["GQ"] = 99
    else:
        gt["GT"] = (0, 0)  # Force HomRef
        gt["RD_CN"] = ecn
        gt["RD_GQ"] = 99
        gt["GQ"] = 99


# ── Header helpers ───────────────────────────────────────────────────


def _ensure_headers(header: "pysam.VariantHeader") -> None:
    """Idempotently add required INFO and FORMAT header lines."""
    existing_info = set(header.info)
    for line in _INFO_HEADERS_REQUIRED:
        info_id = line.split("ID=")[1].split(",")[0].split(">")[0]
        if info_id not in existing_info:
            header.add_line(line)
            existing_info.add(info_id)

    existing_fmt = set(header.formats)
    for line in _FORMAT_HEADERS_REQUIRED:
        fmt_id = line.split("ID=")[1].split(",")[0].split(">")[0]
        if fmt_id not in existing_fmt:
            header.add_line(line)
            existing_fmt.add(fmt_id)


# ── VCF sorting ──────────────────────────────────────────────────────


def _sort_vcf(vcf_path: str, out_path: str, temp_dir: str) -> None:
    """Sort a VCF with ``bcftools sort`` and write bgzipped output."""
    proc = subprocess.Popen(
        ["bcftools", "sort", "-T", temp_dir, vcf_path, "-O", "z", "-o", out_path]
    )
    proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"bcftools sort returned non-zero exit code: {proc.returncode}"
        )


# ── CLI ──────────────────────────────────────────────────────────────


def _parse_args(argv: Optional[List[Text]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gatk-sv-gd integrate",
        description=(
            "Integrate GD calls into a GATK-SV final VCF, with cluster awareness. "
            "Requires ``bcftools`` on PATH for output sorting."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vcf", required=True,
        help="Input VCF (bgzipped or plain).",
    )
    parser.add_argument(
        "--gd-calls", required=True,
        help=(
            "Output of the ``call`` subcommand (gd_cnv_calls.tsv.gz) — "
            "wide TSV format with sample, GD_ID, chrom, start, end, svtype, "
            "is_carrier columns.  Legacy 6-column narrow TSV also supported."
        ),
    )
    parser.add_argument(
        "--gd-table", required=True,
        help="GD regions TSV table (same format used by preprocess/infer).",
    )
    parser.add_argument(
        "--par-bed", required=True,
        help="BED file of pseudoautosomal regions.",
    )
    parser.add_argument(
        "--ploidy-table", required=True,
        help="Wide GATK-SV ploidy table (sample + one column per contig).",
    )
    parser.add_argument(
        "--out-vcf", required=True,
        help="Output VCF path (bgzipped, sorted, and tabix-indexed).",
    )
    parser.add_argument(
        "--reciprocal-overlap", type=float, default=0.5,
        help=(
            "Reciprocal overlap cutoff for VCF record clearing. Only VCF "
            "records with RO >= this value are touched by GD calls."
        ),
    )
    parser.add_argument(
        "--non-nahr-overlap", type=float, default=0.02,
        help=(
            "Minimum fraction of a non-NAHR region overlapped by a variant "
            "to annotate it."
        ),
    )
    parser.add_argument(
        "--temp-dir", default="./",
        help="Temporary directory for intermediate files.",
    )
    if argv is not None:
        return parser.parse_args(argv)
    return parser.parse_args()


# ── New helpers: sample-centric Phase 2 ──────────────────────────────


def _compute_sample_overlap(
    gd_info: dict,
    vcf_record: "pysam.VariantRecord",
    vcf_samples: List[str],
) -> float:
    """Compute the sample overlap score between a GD call and a VCF record.

    Returns the fraction of GD carriers that are present in the VCF record's
    carrier set (heterozygous/homozygous-alternate genotypes).

    Parameters
    ----------
    gd_info : dict
        GD call info with 'samples' key (carrier names).
    vcf_record : VariantRecord
        The VCF record to compare against.
    vcf_samples : list[str]
        List of sample names in the VCF header.

    Returns
    -------
    float
        Fraction of GD carriers found in the VCF record (0.0–1.0).
    """
    gd_carriers = set(gd_info.get("samples", []))
    if not gd_carriers:
        return 0.0
    vcf_carriers = {
        s for s, gt in vcf_record.samples.items()
        if gt.get("GT") != (0, 0)
    }
    matching = gd_carriers & vcf_carriers
    return len(matching) / len(gd_carriers)


def _find_overlapping_vcf_records(
    vcf_records: List[Tuple["pysam.VariantRecord", dict]],
    chrom: str,
    start: int,       # 0-based
    stop: int,        # 0-based exclusive
    svtype: str,
    ro_cutoff: float,
) -> List[Tuple[int, float]]:
    """Return (index, reciprocal_overlap) for VCF records matching svtype
    with RO >= threshold against the given interval.

    Iterates the buffered VCF list and filters on:
    - Matching SVTYPE (svtype-aware: DEL only matches DEL, etc.)
    - Reciprocal overlap >= threshold
    - Chromosome match

    Returns
    -------
    list of (index, ro) tuples where index refers to the position in
    vcf_records and ro is the reciprocal overlap value.
    """
    matches: List[Tuple[int, float]] = []
    for idx, (record, meta) in enumerate(vcf_records):
        record_svtype = record.info.get("SVTYPE", "")
        if isinstance(record_svtype, (tuple, list)):
            record_svtype = record_svtype[0] if record_svtype else ""
        record_svtype = str(record_svtype) if record_svtype else ""

        if record_svtype != svtype:
            continue

        if record.chrom != chrom:
            continue

        ro = reciprocal_overlap(record.pos, record.stop, start, stop)
        if ro >= ro_cutoff:
            matches.append((idx, ro))

    return matches


def _build_gd_record(
    header: "pysam.VariantHeader",
    chrom: str,
    pos: int,       # 0-based start
    stop: int,      # 0-based exclusive
    gd_id: str,
    svtype: str,
    meta: dict,
    carriers: Set[str],
    ploidy_dict: Dict[str, Dict[str, int]],
    par_trees: Dict[str, "IntervalTree"],
    is_novel: bool,
) -> "pysam.VariantRecord":
    """Construct a new VCF record from GD call metadata.

    Uses GD coordinates. Carriers → het, all others → hom-ref.

    Parameters
    ----------
    header : VariantHeader
        The VCF header used to create the record.
    chrom, pos, stop : GD interval coordinates (0-based half-open).
    gd_id : GD region identifier.
    svtype : "DEL" or "DUP".
    meta : GD metadata dict from gd_metadata.
    carriers : set of carrier sample names.
    ploidy_dict : sample -> {contig -> ploidy}.
    par_trees : PAR region interval trees.
    is_novel : True if no VCF record matched (novel GD call).

    Returns
    -------
    pysam.VariantRecord (not yet written).
    """
    new_rec = header.new_record(
        contig=chrom,
        start=pos,   # 0-based; pysam new_record uses 0-based
        stop=stop,
        alleles=("N", f"<{svtype}>"),
        id=f"{gd_id}_{svtype}_novel" if is_novel else gd_id,
    )
    new_rec.info["SVTYPE"] = svtype
    # pysam computes stop = rec.pos + SVLEN, and rec.pos is 1-based
    # (new_record adds 1 to 0-based start). So SVLEN must be (stop - 1) - pos.
    new_rec.info["SVLEN"] = stop - pos - 1
    new_rec.info["EV"] = ("RD",)
    if is_novel:
        new_rec.info["ALGORITHMS"] = ("depth",)
    new_rec.info[GENOMIC_DISORDER_KEY] = gd_id
    new_rec.info["GD_CLUSTER"] = meta["cluster"]
    new_rec.info["GD_BP1"] = meta["bp1"]
    new_rec.info["GD_BP2"] = meta["bp2"]

    for sample, gt in new_rec.samples.items():
        ecn = get_expected_cn(
            chrom, pos, stop, sample, ploidy_dict, par_trees
        )
        is_carrier = sample in carriers
        update_genotype(gt, sample, is_carrier, ecn, svtype)

    return new_rec


# ── Main ─────────────────────────────────────────────────────────────


def main(argv: Optional[List[Text]] = None) -> None:
    """Entry point for the *integrate* subcommand."""
    args = _parse_args(argv)

    logger = get_logger("integrate")

    # Validate inputs
    for label, path in [
        ("VCF", args.vcf),
        ("GD calls", args.gd_calls),
        ("GD table", args.gd_table),
        ("PAR BED", args.par_bed),
        ("ploidy table", args.ploidy_table),
    ]:
        if not os.path.exists(path):
            print(f"Error: required {label} input not found", file=sys.stderr)
            sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(args.out_vcf)) or "."
    setup_logging(
        out_dir,
        filename="integrate_log.txt",
        command="integrate",
        args=args,
    )

    logger.info("Loading reference and annotation tables")
    ploidy_dict = read_ploidy_table(args.ploidy_table)
    gd_calls = read_gd_calls(args.gd_calls)
    nahr_trees, non_nahr_trees, gd_metadata = _build_trees_from_gd_table(
        args.gd_table
    )
    par_trees = _read_bed_to_trees(args.par_bed)

    os.makedirs(args.temp_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=args.temp_dir, suffix=".vcf.gz", delete=False
    ) as tmp_file:
        tmp_vcf_path = tmp_file.name

    try:
        with pysam.VariantFile(args.vcf) as vcf_in:
            header = vcf_in.header
            _ensure_headers(header)

            vcf_samples = set(header.samples)

            # Warn about carriers absent from VCF header
            for (region_id, svtype), gd_info in gd_calls.items():
                for sample in gd_info["samples"]:
                    if sample not in vcf_samples:
                        logger.warning(
                            "Carrier sample %r (region %s/%s) not in VCF header",
                            sample,
                            region_id,
                            svtype,
                        )

            # Phase 1: Buffer VCF records (instead of writing immediately)
            vcf_records: List[Tuple["pysam.VariantRecord", dict]] = []

            for record in vcf_in:
                svtype = record.info.get("SVTYPE", "")
                if isinstance(svtype, (tuple, list)):
                    svtype = svtype[0] if svtype else ""
                svtype = str(svtype) if svtype else ""

                if svtype not in ("DEL", "DUP"):
                    vcf_records.append((record, {"gd_handled": False, "write": True}))
                    continue

                chrom = record.chrom
                pos = record.pos    # 1-based in pysam
                stop = record.stop  # 0-based end (half-open)
                record_len = stop - pos

                # Phase 1: Non-NAHR partial overlap annotation
                if record_len > 0 and chrom in non_nahr_trees:
                    for non_nahr_ov in non_nahr_trees[chrom].overlap(pos, stop):
                        region_id_sv = non_nahr_ov.data  # (gd_id, svtype)
                        ov_region_id, ov_svtype = region_id_sv
                        if ov_svtype != svtype:
                            continue
                        ov_frac = fraction_covered(
                            non_nahr_ov.begin, non_nahr_ov.end, pos, stop
                        )
                        if ov_frac >= args.non_nahr_overlap:
                            record.info[GENOMIC_DISORDER_KEY] = ov_region_id
                            record.info["GD_CLUSTER"] = (
                                gd_metadata[ov_region_id]["cluster"]
                            )
                            break

                vcf_records.append((record, {"gd_handled": False, "write": True}))

            # Phase 2: GD-call-centric processing
            # For each GD locus with carriers:
            #   1. Find all overlapping VCF records (RO >= threshold)
            #   2. Force all VCF-header samples to hom-ref in those records
            #   3. Emit a new GD record with authoritative genotypes
            #
            # NAHR GD calls are processed first (higher priority), then
            # non-NAHR.  Matched VCF records are tracked for dropping in Phase 3.
            matched_gd: Set[Tuple[str, str]] = set()
            gd_records: List[pysam.VariantRecord] = []
            replaced_vcf: Set[int] = set()

            # Separate GD calls into NAHR / non-NAHR for priority ordering
            nahr_gd: List[Tuple[str, str, dict]] = []
            non_nahr_gd: List[Tuple[str, str, dict]] = []
            for (gd_id, svtype), gd_info in gd_calls.items():
                is_nahr = gd_metadata.get(gd_id, {}).get("nahr", False)
                if is_nahr:
                    nahr_gd.append((gd_id, svtype, gd_info))
                else:
                    non_nahr_gd.append((gd_id, svtype, gd_info))

            # ── Step 1: Find all matches, then select best GD call per VCF record ──
            # vcf_idx -> (gd_id, svtype, gd_info, ro, so)
            vcf_best_match: Dict[int, Tuple[str, str, dict, float, float]] = {}
            # gd_id -> set of vcf_idxs it matched
            gd_matched_vcf: Dict[str, Set[int]] = {}

            for gd_id, svtype, gd_info in nahr_gd + non_nahr_gd:
                carriers = gd_info["samples"]
                chrom = gd_info["chrom"]
                start = gd_info["pos"]    # 0-based
                stop = gd_info["end"]     # 0-based exclusive

                matches = _find_overlapping_vcf_records(
                    vcf_records, chrom, start, stop, svtype,
                    args.reciprocal_overlap,
                )

                gd_matched_vcf[gd_id] = set()
                for vcf_idx, ro in matches:
                    gd_matched_vcf[gd_id].add(vcf_idx)
                    # Compute sample overlap: fraction of GD carriers present in VCF
                    so = _compute_sample_overlap(
                        gd_info, vcf_records[vcf_idx][0], vcf_in.header.samples
                    )
                    if vcf_idx not in vcf_best_match:
                        vcf_best_match[vcf_idx] = (gd_id, svtype, gd_info, ro, so)
                    else:
                        _, _, _, best_ro, best_so = vcf_best_match[vcf_idx]
                        if ro > best_ro or (ro == best_ro and so > best_so):
                            vcf_best_match[vcf_idx] = (gd_id, svtype, gd_info, ro, so)

            # ── Step 2: Process matched VCF records ──
            for vcf_idx, (gd_id, svtype, gd_info, ro, so) in vcf_best_match.items():
                carriers = gd_info["samples"]
                chrom = gd_info["chrom"]
                start = gd_info["pos"]
                stop = gd_info["end"]

                replaced_vcf.add(vcf_idx)
                record, _meta = vcf_records[vcf_idx]
                for sample_name in vcf_in.header.samples:
                    if sample_name in record.samples:
                        gt = record.samples[sample_name]
                        ecn = get_expected_cn(
                            chrom, start, stop,
                            sample_name, ploidy_dict, par_trees,
                        )
                        update_genotype(gt, sample_name, is_carrier=False,
                                        ecn=ecn, svtype=svtype)

                if gd_id not in gd_metadata:
                    logger.warning(
                        "GD ID %r has no metadata in the GD table; skipping.",
                        gd_id,
                    )
                    continue

                meta = gd_metadata[gd_id]
                is_novel = (not matches)  # always False for matched
                new_rec = _build_gd_record(
                    header, chrom, start, stop,
                    gd_id, svtype, meta,
                    carriers,
                    ploidy_dict, par_trees,
                    False,  # is_novel = False (matched VCF)
                )

                if not all_homref_record(new_rec.samples.values()):
                    matched_gd.add((gd_id, svtype))
                    gd_records.append(new_rec)

            # ── Step 3: Process unmatched GD calls as novel ──
            processed_gd_ids = {v[0] for v in vcf_best_match.values()} if vcf_best_match else set()

            for gd_id, svtype, gd_info in nahr_gd + non_nahr_gd:
                if gd_id in processed_gd_ids:
                    continue

                carriers = gd_info["samples"]
                chrom = gd_info["chrom"]
                start = gd_info["pos"]
                stop = gd_info["end"]

                if gd_id not in gd_metadata:
                    logger.warning(
                        "GD ID %r has no metadata in the GD table; skipping.",
                        gd_id,
                    )
                    continue

                if not carriers:
                    continue

                # Check contig exists in header
                if chrom not in header.contigs:
                    logger.warning(
                        "Contig %r not in VCF header; skipping novel "
                        "record for %s/%s",
                        chrom,
                        gd_id,
                        svtype,
                    )
                    continue

                meta = gd_metadata[gd_id]
                new_rec = _build_gd_record(
                    header, chrom, start, stop,
                    gd_id, svtype, meta,
                    carriers,
                    ploidy_dict, par_trees,
                    True,  # is_novel = True (no VCF matched)
                )

                if not all_homref_record(new_rec.samples.values()):
                    matched_gd.add((gd_id, svtype))
                    gd_records.append(new_rec)
                    # Warn for missing metadata regardless of carriers
                    logger.warning(
                        "GD ID %r has no metadata in the GD table; skipping.",
                        gd_id,
                    )


            # Phase 3: Write surviving VCF records and GD records
            with pysam.VariantFile(tmp_vcf_path, mode="w", header=header) as vcf_out:
                for idx, (record, _meta) in enumerate(vcf_records):
                    if idx not in replaced_vcf:
                        vcf_out.write(record)
                for gd_rec in gd_records:
                    vcf_out.write(gd_rec)

        logger.info("Sorting and indexing final VCF")
        with tempfile.TemporaryDirectory(dir=args.temp_dir) as temp_sort_dir:
            _sort_vcf(
                vcf_path=tmp_vcf_path,
                out_path=args.out_vcf,
                temp_dir=temp_sort_dir,
            )
        pysam.tabix_index(args.out_vcf, preset="vcf", force=True)
        logger.info("Integration complete: %s", args.out_vcf)

    finally:
        if os.path.exists(tmp_vcf_path):
            os.unlink(tmp_vcf_path)


def all_homref_record(sample_dicts) -> bool:
    """Return True if all sample genotype dicts represent hom-ref."""
    for gt_dict in sample_dicts:
        if not _is_homref(gt_dict.get("GT", (0, 0))):
            return False
    return True


# ── Legacy main for backward compatibility ───────────────────────────
# The refactored main() above implements the new sample-centric approach.
# The old NAHR record-centric logic has been replaced entirely.
