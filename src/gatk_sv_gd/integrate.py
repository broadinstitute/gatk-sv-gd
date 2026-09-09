"""
Integrate GD calls into a GATK-SV final VCF.

Positive NAHR GD calls replace sufficiently overlapping original DEL/DUP calls
in each sample, regardless of copy-state or event-type disagreement. Matched
VCF calls are removed entirely; GD coordinates, genotypes, and qualities win.
Canonical events are reevaluated using complete-cohort GD calls, including
negative evaluations. Other original calls retain their coordinates and fields.

Only direct GD overlaps are considered; VCF overlap chains do not extend the
replacement region. The sorted input streams once, retaining direct overlaps
for sample-level replacement. Reconciled records are sorted with bcftools and
merged with passthrough records into a bgzipped, tabix-indexed VCF.

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
import json
import math
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
    "SR_GT": None,
    "SR_GQ": None,
    "PE_GT": None,
    "PE_GQ": None,
}

# Standard header lines that must be present before writing genotypes /
# novel records.  Only added if the field is absent.
_INFO_HEADERS_REQUIRED = [
    '##INFO=<ID=GD_ATYPICAL,Number=0,Type=Flag,Description="Noncanonical GD event breakpoints">',
    '##INFO=<ID=GD_CALL_IDS,Number=.,Type=String,Description="GD calls represented by this event">',
    '##INFO=<ID=GD_SOURCE_IDS,Number=.,Type=String,Description="Original VCF events replaced by this GD event">',
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
    '##INFO=<ID=SVLEN,Number=1,Type=Integer,'
    'Description="Length of affected segment on the reference">',
    '##INFO=<ID=END,Number=1,Type=Integer,'
    'Description="End position of the structural variant">',
    '##INFO=<ID=ALGORITHMS,Number=.,Type=String,'
    'Description="Source algorithms">',
    '##INFO=<ID=EV,Number=.,Type=String,'
    'Description="Classes of evidence supporting final genotype">',
]

_FORMAT_HEADERS_REQUIRED = [
    '##FORMAT=<ID=ECN,Number=1,Type=Integer,Description="Expected copy number for ref genotype">',
    '##FORMAT=<ID=EV,Number=.,Type=String,Description="Classes of evidence supporting final genotype">',
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
                "start": start,
                "end": end,
                "chrom": chrom,
                "start_range": locus.breakpoints[locus.breakpoint_names.index(entry["BP1"])],
                "end_range": locus.breakpoints[locus.breakpoint_names.index(entry["BP2"])],
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
       are grouped by ``(GD_ID, svtype)`` and carrier samples collected,
       along with optional per-sample total copy numbers from ``cn_state``
       and posterior summaries from ``cn_probabilities``.

    2. **Legacy narrow format** (6-column TSV, no header):
       ``chrom``, ``pos`` (0-based), ``end``, ``region_id``, ``svtype``,
       comma-separated-samples.

    Lines starting with ``#`` are skipped in the narrow format.
    An empty or ``.`` sample field produces an empty carrier set.

    Returns
    -------
    dict mapping ``(region_id, svtype)`` to
    ``{chrom, pos, end, samples}``, plus ``copy_states`` and
    ``copy_probabilities`` and ``evaluated_samples`` in wide format.
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
                "copy_states": {},
                "copy_probabilities": {},
                "evaluated_samples": set(),
                "carrier_calls": {},
                "intervals": set(),
            }
        groups[key]["evaluated_samples"].add(row["sample"])
        coordinates = (row["chrom"], int(row["start"]), int(row["end"]))
        groups[key]["intervals"].add(coordinates)
        probability_text = (row.get("cn_probabilities") or "").strip()
        probabilities = None
        if probability_text.lower() not in ("", ".", "nan", "na"):
            probabilities = _parse_copy_probabilities(probability_text)
            groups[key]["copy_probabilities"][row["sample"]] = probabilities
        is_carrier = row.get("is_carrier", "").strip()
        if is_carrier in ("True", "true", "1"):
            groups[key]["samples"].add(row["sample"])
            cn = None
            cn_text = (row.get("cn_state") or "").strip()
            if cn_text.lower() not in ("", ".", "nan", "na"):
                cn = float(cn_text)
                if not cn.is_integer() or cn < 0:
                    raise ValueError("cn_state must be a non-negative integer or missing")
                groups[key]["copy_states"][row["sample"]] = int(cn)
            groups[key]["carrier_calls"].setdefault(row["sample"], []).append({
                "chrom": coordinates[0], "pos": coordinates[1], "end": coordinates[2],
                "cn_state": int(cn) if cn is not None else None,
                "cn_probabilities": probabilities,
            })
    return groups


def _parse_copy_probabilities(value: str) -> dict:
    """Read a total-CN posterior summary; missing mass is null uncertainty."""
    try:
        raw = json.loads(value)
        probabilities = {int(cn): float(prob) for cn, prob in raw.items()}
        if any(str(int(cn)) != cn for cn in raw):
            raise ValueError("Copy numbers must be non-negative integers")
        if any(cn < 0 or not math.isfinite(p) or p < 0 or p > 1 + 1e-6 for cn, p in probabilities.items()):
            raise ValueError("Invalid copy-state probability")
        if sum(probabilities.values()) > 1 + 1e-6:
            raise ValueError("Copy-state probability mass exceeds one")
        return {cn: min(p, 1.0) for cn, p in probabilities.items()}
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("cn_probabilities must be a JSON object of total CN to probability (mass <= 1)") from exc


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


# ── Genotype update ──────────────────────────────────────────────────


def update_genotype(
    gt: dict,
    sample: str,
    is_carrier: bool,
    ecn: int,
    svtype: str,
    cn_state: Optional[int] = None,
    baseline_cn: Optional[int] = None,
    cn_probabilities: Optional[Dict[int, float]] = None,
) -> None:
    """Write GATK-SV diploid-encoded GT and biological contig ploidy (ECN).

    baseline_cn may differ from ECN in PAR, where the reference copy number
    is two. Each lost/gained copy contributes an alternate allele, capped at
    two: haploid DEL CN=0 is 0/1; haploid DUP CN>=3 is 1/1. Zero-ploidy
    samples have missing genotype/evidence fields. Missing copy states use
    the legacy single-alt assignment. Qualities use body-averaged model
    probabilities when available and remain missing for older calls TSVs.
    """
    gt["ECN"] = ecn
    if ecn == 0:
        for key in list(gt):
            if key != "ECN":
                gt[key] = None
        gt["GT"] = (None, None)
        gt["RD_CN"] = None
        gt["RD_GQ"] = None
        gt["GQ"] = None
        gt["EV"] = (".",)
        return

    ref_cn = ecn if baseline_cn is None else baseline_cn
    gt["EV"] = ("RD",)
    for key, val in RESET_PESR_FORMATS_DICT.items():
        if key in gt:
            gt[key] = val

    if is_carrier:
        alt_count = 1
        if cn_state is not None:
            if cn_state < 0 or int(cn_state) != cn_state:
                raise ValueError("cn_state must be a non-negative integer")
            cn_state = int(cn_state)
            delta = ref_cn - cn_state if svtype == "DEL" else cn_state - ref_cn
            if delta <= 0:
                raise ValueError("Carrier cn_state must agree with svtype and expected copy number")
            alt_count = min(delta, 2)
        gt["GT"] = (0, 1) if alt_count == 1 else (1, 1)
        if cn_state is not None:
            gt["RD_CN"] = cn_state
        elif svtype == "DEL":
            gt["RD_CN"] = max(ref_cn - 1, 0)
        elif svtype == "DUP":
            gt["RD_CN"] = ref_cn + 1
    else:
        gt["GT"] = (0, 0)
        gt["RD_CN"] = ref_cn
    gt["RD_GQ"] = None
    gt["GQ"] = None
    if cn_probabilities and gt.get("RD_CN") is not None:
        selected_alt_count = sum(gt["GT"])
        genotype_probability = 0.0
        for cn, probability in cn_probabilities.items():
            delta = ref_cn - cn if svtype == "DEL" else cn - ref_cn
            if min(max(delta, 0), 2) == selected_alt_count:
                genotype_probability += probability
        gt["GQ"] = _probability_to_gq(genotype_probability)
        gt["RD_GQ"] = _probability_to_gq(cn_probabilities.get(gt["RD_CN"], 0.0))


def _probability_to_gq(probability: float) -> int:
    """Rounded Phred error probability, capped at 99."""
    return int(round(-10 * math.log10(max(1 - min(probability, 1.0), 10 ** -9.9))))


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
            "Minimum reciprocal overlap for a positive GD call to replace an entire VCF call "
            "in the same sample, regardless of copy state or DEL/DUP type (default: 0.5)."
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
    ``(gd_id, svtype, start, stop)`` for locating directly overlapping VCF records.
    Every sample-specific interval is indexed.

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
        intervals = gd_info.get("intervals", {(gd_info["chrom"], gd_info["pos"], gd_info["end"])})
        for chrom, start, stop in sorted(intervals):
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
    copy_states: Optional[Dict[str, int]] = None,
    copy_probabilities: Optional[Dict[str, dict]] = None,
) -> "pysam.VariantRecord":
    """Construct a new VCF record from GD call metadata.

    Uses GD coordinates and copy-state genotypes; non-carriers are hom-ref.

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
    copy_states : optional sample -> total copy number for carriers.
    copy_probabilities : optional sample -> total CN posterior summary.

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
        baseline_cn = get_expected_cn(
            chrom, pos, stop, sample, ploidy_dict, par_trees
        )
        ecn = ploidy_dict.get(sample, {}).get(chrom, 2)
        is_carrier = sample in carriers
        update_genotype(
            gt, sample, is_carrier, ecn, svtype,
            cn_state=(copy_states or {}).get(sample),
            baseline_cn=baseline_cn,
            cn_probabilities=(copy_probabilities or {}).get(sample),
        )

    return new_rec


# ── Main ─────────────────────────────────────────────────────────────


def main(argv: Optional[List[Text]] = None) -> None:
    """Entry point for the *integrate* subcommand.

    Replace overlapping VCF calls per sample; annotate non-NAHR regions.
    See the module docstring for replacement and breakpoint rules.
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

    for region_id, svtype in gd_calls:
        if svtype not in ("DEL", "DUP") and gd_metadata.get(region_id, {}).get("nahr", True):
            raise ValueError(f"Unsupported GD copy-number SVTYPE {svtype}; expected DEL or DUP")
        if region_id not in gd_metadata:
            logger.warning("GD entry %s/%s missing from GD table; using call coordinates", region_id, svtype)

    os.makedirs(args.temp_dir, exist_ok=True)

    # Index all evaluated NAHR intervals, including sample-specific coordinates.
    gd_call_index = _build_gd_call_index(gd_calls, gd_metadata)

    with pysam.VariantFile(args.vcf) as vcf_in:
        header = vcf_in.header
        _ensure_headers(header)

        vcf_samples = set(header.samples)

        # A wide table distinguishes evaluated non-carriers from missing samples.
        # Reject incomplete NAHR entries before opening any output or dropping records.
        for (region_id, svtype), gd_info in gd_calls.items():
            if not gd_metadata.get(region_id, {}).get("nahr", True):
                continue
            evaluated = gd_info.get("evaluated_samples")
            if evaluated is not None:
                missing = vcf_samples - evaluated
                if missing:
                    raise ValueError(
                        f"GD calls for {region_id}/{svtype} are missing evaluations for "
                        f"{len(missing)} VCF sample(s); provide a row for every sample, "
                        "including non-carriers, before integration"
                    )

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

        # Register contigs before any output header is written. Positive GD
        # calls on a contig absent from the original VCF must still be emitted.
        for (gd_id, svtype), info in gd_calls.items():
            if not gd_metadata.get(gd_id, {}).get("nahr", True) or not info["samples"]:
                continue
            for chrom, start, end in sorted(info.get("intervals", {(info["chrom"], info["pos"], info["end"])})):
                if start < 0 or start >= end:
                    raise ValueError(f"Invalid interval for detected GD call {gd_id}/{svtype}")
                if chrom not in header.contigs:
                    header.contigs.add(chrom)
        affected_records = []

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
                from gatk_sv_gd.reconcile import affected_records as select_affected_records

                for record, loci in select_affected_records(vcf_in, gd_call_index):
                    svtype = record.info.get("SVTYPE", "")
                    if isinstance(svtype, (tuple, list)):
                        svtype = svtype[0] if svtype else ""
                    svtype = str(svtype) if svtype else ""

                    chrom = record.chrom
                    start = record.start  # 0-based start
                    stop = record.stop    # 0-based end (half-open)
                    record_len = stop - start

                    # Other variant classes pass through unchanged.
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

                    if loci and svtype in ("DEL", "DUP") and record_len > 0:
                        affected_records.append((record.copy(), set(loci)))
                        continue

                    vcf_pass.write(record)

            from gatk_sv_gd.reconcile import reconcile_records

            gd_records = reconcile_records(
                header, gd_calls, gd_metadata, affected_records,
                ploidy_dict, par_trees, args.reciprocal_overlap,
            )

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
    """Return True when no sample has a called alternate allele.

    Reference and missing genotypes (including zero-ploidy allosomes) cannot
    justify emitting a GD variant. Partial calls with an alternate allele can.
    """
    return not any(
        allele is not None and allele > 0
        for gt_dict in sample_dicts
        for allele in (gt_dict.get("GT") or ())
    )


# ── Legacy main for backward compatibility ───────────────────────────
# The refactored main() above implements the new sample-centric approach.
# The old NAHR record-centric logic has been replaced entirely.
