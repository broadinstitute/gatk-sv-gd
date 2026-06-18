"""
Integrate GD calls into a GATK-SV final VCF.

Single-pass streaming design — O(GD entries) memory, never O(records):

1. Build a ``gd_call_index`` (chrom → IntervalTree of NAHR gd_calls with
   svtype, start, stop) from the GD-calls TSV.  Identify NAHR / non-NAHR by
   the GD table (``nahr_trees`` / ``non_nahr_trees``), not by gd_calls
   membership.

2. Stream the input VCF once.  For each record:

   a. **NAHR drop** — DEL/DUP matching a NAHR ``gd_call_index`` entry by
      SVTYPE + reciprocal overlap >= ``--reciprocal-overlap``:
      *do not write*; mark the gd_call as matched.
   b. **Non-NAHR annotate** — else, if the record overlaps a non-NAHR region
      by ``fraction_covered >= --non-nahr-overlap``:
      set ``GENOMIC_DISORDER`` / ``GD_CLUSTER`` INFO in place and *write*.
      Every overlapping record is annotated; none are dropped.  The existing
      per-sample genotypes ARE the non-NAHR calls.
   c. Otherwise — *write* unchanged.

   Records are written to a coordinate-sorted passthrough temp file (order
   preserved from sorted input).

3. After the stream: for each NAHR gd_call emit one ``_build_gd_record``
   (``is_novel`` = gd_call not matched) with carriers from the TSV into a
   small separate GD-records file.

4. Sort only the small GD file (``bcftools sort``); merge with the
   already-sorted passthrough via ``bcftools concat -a`` (allow-overlaps
   positional merge) → final output, bgzipped + tabix-indexed.

Non-NAHR gd_calls rows are not expected in the calls TSV (``infer``/``call``
emit nothing for non-NAHR sites); they are handled exclusively via
``non_nahr_trees`` derived from the GD table.

Usage (via CLI)::

    gatk-sv-gd integrate \\
        --vcf input.vcf.gz \\
        --gd-calls gd_cnv_calls.tsv.gz \\
        --gd-table gd_table.tsv \\
        --par-bed par.hg38.bed \\
        --ploidy-table ploidy.tsv \\
        --out-vcf integrated.vcf.gz

Requirements:
    ``bcftools`` must be available on PATH (used for final VCF sorting and
    merging).
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


# ── Genotype update ──────────────────────────────────────────────────


def update_genotype(
    gt: dict,
    sample: str,
    is_carrier: bool,
    ecn: int,
    svtype: str,
) -> None:
    """Update per-sample genotype dict in-place.

    Ploidy-aware genotype arity (supersedes the old "always het (0,1)"
    decision): the number of alleles in ``GT`` matches ``ecn`` (the
    expected copy number / contig ploidy, with PAR forced to 2), so a
    haploid contig (e.g. male chrX non-PAR, ecn==1) gets a haploid
    genotype rather than a diploid het.

    - ecn == 0    -> no-call (None, None), RD_CN=0, RD_GQ=0, return early
    - carrier     -> GT = (0,) * (ecn - 1) + (1,)  (one alt allele, rest ref)
                     DEL: RD_CN=max(ecn-1,0); DUP: RD_CN=ecn+1
    - non-carrier -> GT = (0,) * ecn  (all-ref); RD_CN=ecn
    PE/SR FORMAT fields are reset only when present.

    For ecn == 2 (the diploid default) this is byte-identical to the old
    behaviour: carrier -> (0, 1), non-carrier -> (0, 0).
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
        gt["GT"] = (0,) * (ecn - 1) + (1,)
        if svtype == "DEL":
            gt["RD_CN"] = max(ecn - 1, 0)
        elif svtype == "DUP":
            gt["RD_CN"] = ecn + 1
        gt["RD_GQ"] = 99
        gt["GQ"] = 99
    else:
        gt["GT"] = (0,) * ecn  # Force HomRef
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


# ── VCF sorting and merging ──────────────────────────────────────────


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


def _concat_vcf(passthrough_path: str, gd_sorted_path: str, out_path: str) -> None:
    """Merge a coordinate-sorted passthrough VCF with sorted GD records.

    Uses ``bcftools concat --allow-overlaps`` (positional merge) so that
    NAHR GD records are interleaved with passthrough records by coordinate
    without re-sorting the (potentially large) passthrough file.
    """
    proc = subprocess.Popen(
        [
            "bcftools", "concat", "--allow-overlaps",
            passthrough_path, gd_sorted_path,
            "-O", "z", "-o", out_path,
        ]
    )
    proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"bcftools concat returned non-zero exit code: {proc.returncode}"
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
        "--non-nahr-max-size-ratio", type=float, default=2.0,
        help=(
            "Maximum allowed ratio of variant length to matched non-NAHR region "
            "length. Variants larger than RATIO * region_length are not annotated "
            "(default: 2.0)."
        ),
    )
    parser.add_argument(
        "--temp-dir", default="./",
        help="Temporary directory for intermediate files.",
    )
    if argv is not None:
        return parser.parse_args(argv)
    return parser.parse_args()


# ── GD-call index builder ─────────────────────────────────────────────


def _build_gd_call_index(
    gd_calls: Dict[Tuple[str, str], dict],
    gd_metadata: Dict[str, dict],
) -> Dict[str, "IntervalTree"]:
    """Build a per-chrom IntervalTree index of NAHR gd_calls for fast lookup.

    Only NAHR gd_calls entries are indexed (non-NAHR is handled exclusively
    via ``non_nahr_trees`` from the GD table).  Each interval stores
    ``(gd_id, svtype, start, stop)`` so the drop test can check SVTYPE and
    reciprocal overlap without re-reading the gd_calls dict.

    Non-NAHR entries (``gd_metadata[gd_id]["nahr"] is False``) are silently
    skipped — they are annotate-only and never drop VCF records.
    """
    index: Dict[str, IntervalTree] = defaultdict(IntervalTree)
    for (gd_id, svtype), gd_info in gd_calls.items():
        # Fallback: treat as NAHR when not in metadata (caller is NAHR by default)
        is_nahr = gd_metadata.get(gd_id, {}).get("nahr", True)
        if not is_nahr:
            # Non-NAHR entries: annotate-only path, not drop+replace.
            continue
        chrom = gd_info["chrom"]
        start = gd_info["pos"]
        stop = gd_info["end"]
        if start < stop:
            index[chrom].addi(start, stop, (gd_id, svtype, start, stop))
    return index


# ── Record builder ───────────────────────────────────────────────────


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
    # Only set BP fields when non-empty (fallback metadata may use "").
    if meta.get("bp1"):
        new_rec.info["GD_BP1"] = meta["bp1"]
    if meta.get("bp2"):
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
    """Entry point for the *integrate* subcommand.

    Single-pass streaming over the input VCF — memory is O(GD entries),
    never O(records).  NAHR gd_calls drop+replace overlapping records; non-NAHR
    regions annotate overlapping records in place (no drop, no synthesized
    record).  See module docstring for the full algorithm.
    """
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

    # Build the per-chrom index of NAHR gd_calls for O(log N) drop decisions.
    gd_call_index = _build_gd_call_index(gd_calls, gd_metadata)

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

        # matched_calls: set of (gd_id, svtype) whose VCF records were dropped.
        # is_novel = (gd_id, svtype) not in matched_calls after the stream.
        matched_calls: Set[Tuple[str, str]] = set()

        with tempfile.NamedTemporaryFile(
            dir=args.temp_dir, suffix=".passthrough.vcf.gz", delete=False
        ) as _pt:
            passthrough_path = _pt.name
        with tempfile.NamedTemporaryFile(
            dir=args.temp_dir, suffix=".gd_records.vcf.gz", delete=False
        ) as _gd:
            gd_records_path = _gd.name
        with tempfile.NamedTemporaryFile(
            dir=args.temp_dir, suffix=".gd_sorted.vcf.gz", delete=False
        ) as _gs:
            gd_sorted_path = _gs.name

        try:
            # ── Single-pass streaming ────────────────────────────────
            with pysam.VariantFile(
                passthrough_path, mode="w", header=header
            ) as vcf_pass:
                for record in vcf_in:
                    svtype = record.info.get("SVTYPE", "")
                    if isinstance(svtype, (tuple, list)):
                        svtype = svtype[0] if svtype else ""
                    svtype = str(svtype) if svtype else ""

                    chrom = record.chrom
                    start = record.start  # 0-based start
                    stop = record.stop    # 0-based end (half-open)
                    record_len = stop - start

                    # ── NAHR drop check (constraint 4 / 5) ──────────
                    # Check if this record is matched by any NAHR gd_call
                    # (any svtype).  If matched → drop the record (write
                    # nothing) and mark the gd_call as matched.
                    # NAHR drop takes precedence over non-NAHR annotation
                    # (constraint 5).
                    dropped = False
                    if chrom in gd_call_index and record_len > 0:
                        for iv in gd_call_index[chrom].overlap(start, stop):
                            iv_gd_id, iv_svtype, iv_start, iv_stop = iv.data
                            if iv_svtype != svtype:
                                continue
                            ro = reciprocal_overlap(start, stop, iv_start, iv_stop)
                            if ro >= args.reciprocal_overlap:
                                matched_calls.add((iv_gd_id, iv_svtype))
                                dropped = True
                                # Continue checking other overlapping gd_calls
                                # so all matching entries are marked.

                    if dropped:
                        continue  # Record dropped — NAHR wins.

                    # Non-DEL/DUP records pass through (after NAHR drop check).
                    # Non-NAHR annotation is DEL/DUP-specific.
                    if svtype not in ("DEL", "DUP"):
                        vcf_pass.write(record)
                        continue

                    # ── Non-NAHR annotation (constraint 3) ──────────
                    # Any DEL/DUP record overlapping a non-NAHR region
                    # by fraction_covered >= threshold gets GENOMIC_DISORDER /
                    # GD_CLUSTER in place.  All such records are kept.
                    if record_len > 0 and chrom in non_nahr_trees:
                        for non_nahr_ov in non_nahr_trees[chrom].overlap(start, stop):
                            region_id_sv = non_nahr_ov.data  # (gd_id, svtype)
                            ov_region_id, ov_svtype = region_id_sv
                            if ov_svtype != svtype:
                                continue
                            region_len = non_nahr_ov.end - non_nahr_ov.begin
                            if region_len > 0 and record_len > args.non_nahr_max_size_ratio * region_len:
                                continue
                            ov_frac = fraction_covered(
                                non_nahr_ov.begin, non_nahr_ov.end, start, stop
                            )
                            if ov_frac >= args.non_nahr_overlap:
                                record.info[GENOMIC_DISORDER_KEY] = ov_region_id
                                record.info["GD_CLUSTER"] = (
                                    gd_metadata[ov_region_id]["cluster"]
                                )
                                # Annotate with the first matching non-NAHR
                                # region (deterministic: first overlap wins).
                                break

                    vcf_pass.write(record)

            # ── Emit one GD record per NAHR gd_call ─────────────────
            # Writes GD records to a small separate file (not the large
            # passthrough). Only NAHR gd_calls go through _build_gd_record.
            gd_records: List["pysam.VariantRecord"] = []

            for (gd_id, svtype), gd_info in gd_calls.items():
                # Only process NAHR entries through the drop+replace path.
                is_nahr = gd_metadata.get(gd_id, {}).get("nahr", True)
                if not is_nahr:
                    # Non-NAHR: annotate-only; no synthesized record.
                    continue

                carriers = gd_info["samples"]
                chrom = gd_info["chrom"]
                start = gd_info["pos"]    # 0-based
                stop = gd_info["end"]     # 0-based exclusive

                # Skip inverted or zero-length intervals (coordinates from
                # gd_calls may not have been validated by the GD table loader).
                if start >= stop:
                    logger.warning(
                        "GD call %r/%s has inverted/zero interval "
                        "(start=%d >= stop=%d); skipping.",
                        gd_id, svtype, start, stop,
                    )
                    continue

                # T2: GD-ID fallback — use the gd_calls coordinates and a
                # synthetic meta when gd_id is absent from the GD table.
                # This handles calls produced against an older GD-table version.
                if gd_id not in gd_metadata:
                    logger.warning(
                        "GD ID %r has no metadata in the GD table; "
                        "using fallback metadata.",
                        gd_id,
                    )
                    meta = {
                        "cluster": gd_id,
                        "bp1": "",
                        "bp2": "",
                        "nahr": True,
                        "svtype": svtype,
                    }
                else:
                    meta = gd_metadata[gd_id]

                is_novel = (gd_id, svtype) not in matched_calls

                # Novel call with no carriers: nothing to assert and nothing
                # to clear → skip entirely.
                if is_novel and not carriers:
                    continue

                # Novel call on a contig absent from the header: cannot
                # place a record.
                if is_novel and chrom not in header.contigs:
                    logger.warning(
                        "Contig %r not in VCF header; skipping novel "
                        "record for %s/%s",
                        chrom,
                        gd_id,
                        svtype,
                    )
                    continue

                new_rec = _build_gd_record(
                    header, chrom, start, stop,
                    gd_id, svtype, meta,
                    carriers,
                    ploidy_dict, par_trees,
                    is_novel,
                )

                if not all_homref_record(new_rec.samples.values()):
                    gd_records.append(new_rec)

            # Write GD records to the small separate file.
            with pysam.VariantFile(
                gd_records_path, mode="w", header=header
            ) as vcf_gd:
                for gd_rec in gd_records:
                    vcf_gd.write(gd_rec)

            # ── Index passthrough before concat ──────────────────────
            # pysam.tabix_index bgzips the file in place (if not already
            # bgzipped) and writes the .tbi index.  bcftools concat
            # --allow-overlaps requires every input to be bgzipped + indexed.
            pysam.tabix_index(passthrough_path, preset="vcf", force=True)

            # ── Sort GD file + merge with passthrough ────────────────
            logger.info("Sorting GD records and merging with passthrough VCF")
            with tempfile.TemporaryDirectory(dir=args.temp_dir) as temp_sort_dir:
                _sort_vcf(
                    vcf_path=gd_records_path,
                    out_path=gd_sorted_path,
                    temp_dir=temp_sort_dir,
                )
            # Index the sorted GD file — also required by --allow-overlaps.
            pysam.tabix_index(gd_sorted_path, preset="vcf", force=True)
            _concat_vcf(passthrough_path, gd_sorted_path, args.out_vcf)
            pysam.tabix_index(args.out_vcf, preset="vcf", force=True)
            logger.info("Integration complete: %s", args.out_vcf)

        finally:
            for path in (passthrough_path, gd_records_path, gd_sorted_path):
                if os.path.exists(path):
                    os.unlink(path)


def all_homref_record(sample_dicts) -> bool:
    """Return True if all sample genotype dicts represent hom-ref."""
    for gt_dict in sample_dicts:
        if not _is_homref(gt_dict.get("GT", (0, 0))):
            return False
    return True


# ── Legacy main for backward compatibility ───────────────────────────
# The refactored main() above implements the new sample-centric approach.
# The old NAHR record-centric logic has been replaced entirely.
