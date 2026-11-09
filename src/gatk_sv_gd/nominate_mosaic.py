"""
Nominate potential mosaic GD carriers from read depth.

Backs the ``NominateMosaicGDCarriers`` WDL.  Two subcommands:

``nominate``
    Per batch.  For every autosomal GD interval in the GD table, builds the
    normalised depth (the ``plot.py`` raw-depth procedure: each sample scaled
    so its autosomal median is CN 2; high-res bins substituted for intervals
    shorter than 10x the low-res median bin size), drops samples whose depth is
    abnormal in the flanks or in cluster intervals outside the interval (the
    ``plot_mosaic`` procedure), and reports samples whose median depth over the
    interval, excluding bins that touch a segmental duplication, lies in a
    mosaic window.
``merge``
    Once.  Concatenates the per-batch tables (listed in a file, so hundreds of
    batches do not hit command-line limits).
"""

import argparse
import gzip
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gatk_sv_gd.models import GDLocus, GDTable
from gatk_sv_gd.plot import (
    _build_raw_region_df,
    _compute_raw_sample_medians,
    estimate_lowres_bin_size,
)
from gatk_sv_gd.plot_mode_depth import read_count_matrix
from gatk_sv_gd.plot_mosaic import (
    flag_abnormal_samples,
    load_segdup_intervals,
    overlaps_segdup,
)

DEL_RANGE = (1.2, 1.8)
DUP_RANGE = (2.2, 2.8)
SEX_CHROMS = {"chrX", "chrY", "X", "Y"}
MIN_FLANK, MAX_FLANK = 50_000, 1_000_000  # as in bins.py

CANDIDATE_COLUMNS = ["gd_id", "sample", "batch", "median_depth", "mosaic_type",
                     "chr", "start", "end", "svtype", "n_bins"]
SUMMARY_COLUMNS = ["batch", "gd_id", "chr", "start", "end", "n_samples",
                   "n_bins", "n_abnormal", "n_candidates"]


def gd_intervals(table: GDTable) -> List[dict]:
    """Distinct autosomal (chr, start, end) intervals with their cluster locus.

    Each item has the cluster ``locus`` (used for flank and cluster-interval
    checks), the interval coordinates, and ``entries``: the table rows
    (GD_ID, svtype) that share the interval.
    """
    out: Dict[Tuple[str, int, int], dict] = {}
    for locus in table.loci.values():
        if locus.chrom in SEX_CHROMS:
            continue
        for entry in locus.gd_entries:
            key = (locus.chrom, int(entry["start_GRCh38"]), int(entry["end_GRCh38"]))
            item = out.setdefault(key, {"locus": locus, "chrom": key[0], "start": key[1],
                                        "end": key[2], "entries": []})
            item["entries"].append((str(entry["GD_ID"]), str(entry["svtype"])))
    return list(out.values())


def pick_gd_id(entries: Sequence[Tuple[str, str]], mosaic_type: str) -> Tuple[str, str]:
    """GD_ID (and svtype) of the interval's row matching *mosaic_type*, else the first."""
    for gd_id, svtype in entries:
        if svtype == mosaic_type:
            return gd_id, svtype
    return entries[0]


def flank_size(locus: GDLocus) -> int:
    """Flank on each side of the cluster: its body size clipped to 50 kb - 1 Mb."""
    return int(min(max(locus.end - locus.start, MIN_FLANK), MAX_FLANK))


def interval_medians(depth: pd.DataFrame, chrom: str, start: int, end: int,
                     segdups) -> Tuple[pd.Series, int]:
    """Per-sample median depth over [start, end) without segdup-touching bins."""
    starts = depth["Start"].to_numpy()
    ends = depth["End"].to_numpy()
    mid = (starts + ends) / 2
    keep = (mid >= start) & (mid < end) & ~overlaps_segdup(segdups, chrom, starts, ends)
    sample_cols = [c for c in depth.columns if c not in ("Chr", "Start", "End")]
    if not keep.any():
        return pd.Series(np.nan, index=sample_cols), 0
    med = np.nanmedian(depth.loc[keep, sample_cols].to_numpy(dtype=float), axis=0)
    return pd.Series(med, index=sample_cols), int(keep.sum())


def nominate_batch(
    counts: pd.DataFrame,
    gd_table_path: str,
    batch: str,
    segdups,
    highres_path: Optional[str] = None,
    del_range: Tuple[float, float] = DEL_RANGE,
    dup_range: Tuple[float, float] = DUP_RANGE,
    min_bins: int = 1,
    max_deviation: float = 0.4,
    window_bins: int = 5,
    min_deviant_bins: int = 3,
    max_population_fraction: float = 0.05,
    nahr_only: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Candidate mosaic events and a per-interval summary for one batch."""
    table = GDTable(gd_table_path)
    samples = [c for c in counts.columns if c not in ("Chr", "Start", "End")]
    medians = _compute_raw_sample_medians(counts)
    usable = [s for s in samples if s in medians]
    dropped = sorted(set(samples) - set(usable))
    if dropped:
        print(f"WARNING: {batch}: no usable autosomal median for {len(dropped)} "
              f"sample(s): {', '.join(dropped[:10])}{' ...' if len(dropped) > 10 else ''}",
              file=sys.stderr)
    cand_parts, summary_rows = [], []
    if not usable:
        return (pd.DataFrame(columns=CANDIDATE_COLUMNS),
                pd.DataFrame(columns=SUMMARY_COLUMNS))
    bin_size = estimate_lowres_bin_size(counts)

    for item in gd_intervals(table):
        locus, chrom = item["locus"], item["chrom"]
        start, end = item["start"], item["end"]
        if nahr_only and not locus.is_nahr:
            continue
        first_id = item["entries"][0][0]
        flank = flank_size(locus)
        depth = _build_raw_region_df(
            locus, max(0, min(locus.start, start) - flank), max(locus.end, end) + flank,
            counts, usable, medians, bin_size, highres_path=highres_path)
        if depth is None or len(depth) == 0:
            print(f"WARNING: {batch}: no depth bins for {first_id}", file=sys.stderr)
            continue

        med, n_bins = interval_medians(depth, chrom, start, end, segdups)
        flags = flag_abnormal_samples(
            depth, locus, first_id, {}, usable, max_deviation, window_bins,
            min_deviant_bins, max_population_fraction, segdups)
        abnormal = set(flags.loc[flags["abnormal"], "sample"])

        in_window = med.between(*del_range) | med.between(*dup_range)
        hit = med[in_window & ~med.index.isin(abnormal)] if n_bins >= min_bins else med.iloc[:0]
        n_cand = len(hit)
        if n_cand:
            types = np.where(hit.to_numpy() < 2.0, "DEL", "DUP")
            rows = []
            for sample, value, mtype in zip(hit.index, hit.to_numpy(), types):
                gd_id, svtype = pick_gd_id(item["entries"], str(mtype))
                rows.append((gd_id, sample, batch, round(float(value), 4), str(mtype),
                             chrom, start, end, svtype, n_bins))
            cand_parts.append(pd.DataFrame(rows, columns=CANDIDATE_COLUMNS))
        summary_rows.append((batch, first_id, chrom, start, end, len(usable), n_bins,
                             len(abnormal), n_cand))

    candidates = (pd.concat(cand_parts, ignore_index=True)
                  if cand_parts else pd.DataFrame(columns=CANDIDATE_COLUMNS))
    return candidates, pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)


def cmd_nominate(args) -> None:
    counts = read_count_matrix(args.low_res_counts)
    segdups = load_segdup_intervals(args.segdup_bed)
    candidates, summary = nominate_batch(
        counts, args.gd_table, args.batch, segdups, args.high_res_counts,
        (args.del_min, args.del_max), (args.dup_min, args.dup_max), args.min_bins,
        args.max_deviation, args.window_bins, args.min_deviant_bins,
        args.max_population_fraction, args.nahr_only)
    candidates.to_csv(f"{args.out_prefix}.candidates.tsv.gz", sep="\t", index=False)
    summary.to_csv(f"{args.out_prefix}.interval_summary.tsv.gz", sep="\t", index=False)
    print(f"{args.batch}: {len(candidates)} candidate events in "
          f"{candidates['sample'].nunique()} samples across {len(summary)} intervals")


def cmd_merge(args) -> None:
    with open(args.input_list) as handle:
        paths = [line.strip() for line in handle if line.strip()]
    n_rows = 0
    wrote_header = False
    opener = gzip.open if args.out.endswith(".gz") else open
    # Stream one table at a time so hundreds of batches stay in constant memory.
    with opener(args.out, "wt") as sink:
        for path in paths:
            part = pd.read_csv(path, sep="\t", compression="infer")
            part.to_csv(sink, sep="\t", index=False, header=not wrote_header)
            wrote_header = True
            n_rows += len(part)
        if not wrote_header:
            sink.write("\t".join(args.columns.split(",")) + "\n")
    print(f"merged {len(paths)} tables, {n_rows} rows")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gatk-sv-gd nominate-mosaic", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    n = sub.add_parser("nominate", help="Nominate candidate mosaic events for one batch")
    n.add_argument("--batch", required=True)
    n.add_argument("--gd-table", required=True)
    n.add_argument("--low-res-counts", required=True)
    n.add_argument("--high-res-counts", help="bgzipped, tabix-indexed high-res counts")
    n.add_argument("--segdup-bed", required=True)
    n.add_argument("--out-prefix", required=True)
    n.add_argument("--del-min", type=float, default=DEL_RANGE[0])
    n.add_argument("--del-max", type=float, default=DEL_RANGE[1])
    n.add_argument("--dup-min", type=float, default=DUP_RANGE[0])
    n.add_argument("--dup-max", type=float, default=DUP_RANGE[1])
    n.add_argument("--min-bins", type=int, default=1,
                   help="minimum non-segdup bins in the interval to report")
    n.add_argument("--max-deviation", type=float, default=0.4)
    n.add_argument("--window-bins", type=int, default=5)
    n.add_argument("--min-deviant-bins", type=int, default=3)
    n.add_argument("--max-population-fraction", type=float, default=0.05)
    n.add_argument("--nahr-only", action=argparse.BooleanOptionalAction, default=True)

    m = sub.add_parser("merge", help="Concatenate per-batch tables")
    m.add_argument("--input-list", required=True, help="file with one table path per line")
    m.add_argument("--out", required=True)
    m.add_argument("--columns", default=",".join(CANDIDATE_COLUMNS),
                   help="header to write when there are no inputs")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    {"nominate": cmd_nominate, "merge": cmd_merge}[args.command](args)


if __name__ == "__main__":
    main(sys.argv[1:])
