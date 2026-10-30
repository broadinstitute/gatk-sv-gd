"""
Most-common normalized read depth per sample at every NAHR-mediated GD locus.

Backs the ``PlotGDModeDepth`` WDL.  Three subcommands, run in this order:

``summarize``
    Per batch.  Reads ``cn_posteriors.tsv.gz`` from ``CallGenomicDisorderCNVs``.
    For every sample at every NAHR locus, rounds the normalized depth of each
    locus body bin to the nearest 0.1, finds the most commonly observed
    rounded value, and reports the median of the actual (unrounded) depths
    that rounded to it.  Flank bins are ignored.
``merge``
    Once.  Concatenates the per-batch tables into one dataset.
``plot``
    Once.  Scatter of per-sample median depth, one x position per locus in
    ``gd_table`` order (DEL and DUP at the same locus share a position).  Only
    loci with probable mosaic carriers are labeled.  Several datasets can be
    stacked, sharing the x-axis.

Expected behaviour of the mode depth (CN 2 baseline = 2.0): non-carriers sit
near 2, germline het DEL carriers near 1 (het DUP near 3), mosaic carriers
between (1-2 for DEL, 2-3 for DUP), and carriers of an event that covers only
part of the locus sit near 2 or near the carrier value, never between.

Loci on chrX and chrY are included; depth is not rescaled for ploidy.
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gatk_sv_gd.models import GDTable

ROUND_DECIMALS = 1

# Median depth windows (inclusive) that flag a probable mosaic carrier.
MOSAIC_DEL_RANGE = (1.2, 1.8)
MOSAIC_DUP_RANGE = (2.2, 2.8)

COLUMNS = ["batch", "cluster", "sample", "mode_depth", "median_depth",
           "n_bins", "mode_bins"]

COLOR_POINT_EVEN = "#2b2b2b"
COLOR_POINT_ODD = "#4a86b8"
COLOR_MOSAIC_BAND = "#e8dcf3"
COLOR_BASELINE = "#b0b0b0"


# =============================================================================
# Locus order
# =============================================================================


def locus_label(gd_id: str) -> str:
    """GD_ID text preceding the first "_DEL" or "_DUP"."""
    return re.split(r"_(?:DEL|DUP)", str(gd_id), maxsplit=1)[0]


def locus_order(gd_table_path: str) -> pd.DataFrame:
    """NAHR loci in gd_table order, one row per locus (cluster).

    Columns: cluster, label, chrom, x.  The label is the first GD_ID of the
    locus up to its "_DEL"/"_DUP" suffix, so a DEL and DUP at the same locus
    share one label and one x position.
    """
    table = GDTable(gd_table_path)
    raw = pd.read_csv(gd_table_path, sep="\t")
    rank: Dict[str, int] = {}
    for i, row in raw.iterrows():
        cluster = row["cluster"]
        if pd.isna(cluster) or cluster == "":
            cluster = GDTable._standalone_locus_key(row)
        rank.setdefault(str(cluster), i)

    rows = []
    for cluster, locus in table.loci.items():
        if not locus.is_nahr:
            continue
        rows.append((str(cluster), locus_label(locus.display_label), locus.chrom,
                     rank.get(str(cluster), len(rank))))
    out = pd.DataFrame(rows, columns=["cluster", "label", "chrom", "rank"])
    out = out.sort_values("rank", kind="stable").reset_index(drop=True)
    out["x"] = np.arange(len(out))
    return out[["cluster", "label", "chrom", "x"]]


# =============================================================================
# summarize
# =============================================================================


def mode_of_rounded(counts: pd.Series) -> float:
    """Most common rounded depth; ties go to the value nearest 2."""
    top = counts[counts == counts.max()]
    return float(min(top.index, key=lambda v: (abs(v - 2.0), v)))


def summarize_depth(chunks, clusters: Sequence[str], batch: str) -> pd.DataFrame:
    """Per (cluster, sample): mode of 0.1-rounded body-bin depth, and the
    median of the actual depths that rounded to that mode.

    *chunks* yields DataFrames with cluster, interval, sample and depth.
    """
    wanted = set(clusters)
    parts = []
    for chunk in chunks:
        chunk = chunk[chunk["cluster"].astype(str).isin(wanted)]
        chunk = chunk[~chunk["interval"].astype(str).str.contains("flank", case=False)]
        depth = chunk["depth"].astype(float)
        keep = np.isfinite(depth)
        if not keep.any():
            continue
        parts.append(pd.DataFrame({
            "cluster": chunk["cluster"].astype(str)[keep].to_numpy(),
            "sample": chunk["sample"].astype(str)[keep].to_numpy(),
            "depth": depth[keep].to_numpy(),
            "rounded": np.round(depth[keep].to_numpy(), ROUND_DECIMALS),
        }))
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    bins = pd.concat(parts, ignore_index=True)

    counts = bins.groupby(["cluster", "sample", "rounded"]).size()
    rows = []
    for (cluster, sample), series in counts.groupby(level=[0, 1]):
        series = series.droplevel([0, 1])
        mode = mode_of_rounded(series)
        rows.append((cluster, sample, mode, int(series.sum()), int(series.loc[mode])))
    modes = pd.DataFrame(rows, columns=["cluster", "sample", "mode_depth",
                                        "n_bins", "mode_bins"])
    hit = bins.merge(modes[["cluster", "sample", "mode_depth"]], how="inner",
                     left_on=["cluster", "sample", "rounded"],
                     right_on=["cluster", "sample", "mode_depth"])
    med = (hit.groupby(["cluster", "sample"])["depth"].median()
           .rename("median_depth").reset_index())
    out = modes.merge(med, on=["cluster", "sample"])
    out["batch"] = batch
    return out[COLUMNS]


def cmd_summarize(args) -> None:
    order = locus_order(args.gd_table)
    chunks = pd.read_csv(
        args.cn_posteriors, sep="\t", compression="infer",
        usecols=["cluster", "interval", "sample", "depth"], chunksize=500_000)
    out = summarize_depth(chunks, list(order["cluster"]), args.batch)
    out.to_csv(args.out, sep="\t", index=False, compression="infer")
    print(f"{args.batch}: {out['sample'].nunique()} samples, "
          f"{out['cluster'].nunique()} loci, {len(out)} values")


# =============================================================================
# merge
# =============================================================================


def cmd_merge(args) -> None:
    parts = [pd.read_csv(p, sep="\t", compression="infer") for p in args.inputs]
    merged = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
    merged.to_csv(args.out, sep="\t", index=False, compression="infer")
    print(f"{len(parts)} batches, {merged['sample'].nunique()} samples, "
          f"{len(merged)} values")


# =============================================================================
# plot
# =============================================================================


def mosaic_flags(data: pd.DataFrame,
                 del_range: Tuple[float, float] = MOSAIC_DEL_RANGE,
                 dup_range: Tuple[float, float] = MOSAIC_DUP_RANGE) -> pd.Series:
    """Boolean per row: median depth in a mosaic window (between CN 1-2 or 2-3)."""
    d = data["median_depth"]
    return d.between(*del_range) | d.between(*dup_range)


def probable_mosaic_loci(datasets: Sequence[pd.DataFrame], min_samples: int,
                         del_range=MOSAIC_DEL_RANGE,
                         dup_range=MOSAIC_DUP_RANGE) -> List[str]:
    """Clusters with at least *min_samples* mosaic-range samples in any dataset."""
    hits = set()
    for data in datasets:
        flagged = data[mosaic_flags(data, del_range, dup_range)]
        n = flagged.groupby("cluster")["sample"].nunique()
        hits.update(n[n >= min_samples].index.astype(str))
    return sorted(hits)


def draw_mode_depth_plot(
    datasets: Sequence[pd.DataFrame],
    dataset_labels: Sequence[str],
    order: pd.DataFrame,
    out_pdf: str,
    width: float = 3.5,
    height: float = 1.8,
    min_mosaic_samples: int = 1,
    jitter: float = 0.15,
    ymin: float = 0.0,
    ymax: float = 4.0,
    seed: int = 0,
    del_range=MOSAIC_DEL_RANGE,
    dup_range=MOSAIC_DUP_RANGE,
) -> List[str]:
    """Draw stacked scatter panels; return labels of the loci tick-labeled.

    *height* is the whole figure; stacked panels share it.  The y-axis tops
    out at *ymax* or the largest plotted value, whichever is smaller.
    """
    plt.rcParams.update({
        "pdf.fonttype": 42, "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 1.5, "ytick.major.size": 1.5,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
    })
    n = len(datasets)
    fig, axes = plt.subplots(n, 1, figsize=(width, height), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    rng = np.random.RandomState(seed)
    xmap = dict(zip(order["cluster"], order["x"]))

    mosaic_clusters = probable_mosaic_loci(
        datasets, min_mosaic_samples, del_range, dup_range)
    for ax, data, label in zip(axes, datasets, dataset_labels):
        data = data[data["cluster"].isin(xmap)]
        x = data["cluster"].map(xmap).to_numpy(float)
        x = x + rng.uniform(-jitter, jitter, len(x))
        y = data["median_depth"].to_numpy(float)
        for cluster in mosaic_clusters:
            ax.axvspan(xmap[cluster] - 0.4, xmap[cluster] + 0.4,
                       color=COLOR_MOSAIC_BAND, linewidth=0, zorder=0)
        # Vector filled dots; colors alternate between adjacent loci.
        colors = np.where(data["cluster"].map(xmap).to_numpy(int) % 2 == 0,
                          COLOR_POINT_EVEN, COLOR_POINT_ODD)
        ax.scatter(x, y, s=1.6, c=colors, linewidths=0, clip_on=False, zorder=3)
        for level in (1, 2, 3):
            ax.axhline(level, color=COLOR_BASELINE, lw=0.3, zorder=1)
        top = min(ymax, float(y.max())) if len(y) else ymax
        ax.set_ylim(ymin, top)
        ax.set_yticks(range(int(np.ceil(ymin)), int(np.floor(top)) + 1))
        ax.set_ylabel("Normalized read depth", fontsize=6)
        ax.tick_params(labelsize=5.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if n > 1:
            ax.set_title(label, fontsize=6, loc="left", pad=1, color="black")

    ax = axes[-1]
    labeled = order[order["cluster"].isin(mosaic_clusters)]
    ax.set_xticks(labeled["x"])
    ax.set_xticklabels(labeled["label"], rotation=60, ha="right", fontsize=5.5,
                       rotation_mode="anchor")
    ax.set_xlim(-0.7, len(order) - 0.3)
    fig.tight_layout(pad=0.3, h_pad=0.4)
    fig.savefig(out_pdf)
    plt.close(fig)
    return list(labeled["label"])


def cmd_plot(args) -> None:
    order = locus_order(args.gd_table)
    datasets = [pd.read_csv(p, sep="\t", compression="infer") for p in args.data]
    if args.labels:
        if len(args.labels) != len(datasets):
            raise SystemExit("ERROR: --labels must match the number of --data files")
        labels = args.labels
    else:
        labels = [os.path.basename(p).split(".")[0] for p in args.data]
    shown = draw_mode_depth_plot(
        datasets, labels, order, args.out_pdf,
        width=args.width, height=args.height,
        min_mosaic_samples=args.min_mosaic_samples, jitter=args.jitter,
        ymin=args.ymin, ymax=args.ymax, seed=args.seed,
        del_range=(args.mosaic_del_min, args.mosaic_del_max),
        dup_range=(args.mosaic_dup_min, args.mosaic_dup_max))
    print(f"Wrote {args.out_pdf}; labeled loci: {', '.join(shown) or 'none'}")


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gatk-sv-gd plot-mode-depth", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("summarize", help="Per-batch most-common depth per sample and locus")
    p.add_argument("--batch", required=True)
    p.add_argument("--gd-table", required=True)
    p.add_argument("--cn-posteriors", required=True)
    p.add_argument("--out", required=True, help="Output TSV(.gz)")

    p = sub.add_parser("merge", help="Concatenate per-batch tables")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("plot", help="Draw the stacked scatter PDF")
    p.add_argument("--gd-table", required=True)
    p.add_argument("--data", nargs="+", required=True,
                   help="Merged dataset(s); several are stacked, top to bottom")
    p.add_argument("--labels", nargs="+", help="One panel title per dataset")
    p.add_argument("--out-pdf", required=True)
    p.add_argument("--width", type=float, default=3.5)
    p.add_argument("--height", type=float, default=1.8,
                   help="Whole-figure height (inches); stacked panels share it")
    p.add_argument("--min-mosaic-samples", type=int, default=1)
    p.add_argument("--mosaic-del-min", type=float, default=MOSAIC_DEL_RANGE[0])
    p.add_argument("--mosaic-del-max", type=float, default=MOSAIC_DEL_RANGE[1])
    p.add_argument("--mosaic-dup-min", type=float, default=MOSAIC_DUP_RANGE[0])
    p.add_argument("--mosaic-dup-max", type=float, default=MOSAIC_DUP_RANGE[1])
    p.add_argument("--jitter", type=float, default=0.15)
    p.add_argument("--ymin", type=float, default=0.0)
    p.add_argument("--ymax", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    {"summarize": cmd_summarize, "merge": cmd_merge, "plot": cmd_plot}[args.command](args)


if __name__ == "__main__":
    main(sys.argv[1:])
