"""
Most-common normalized read depth per sample at every autosomal NAHR GD locus.

Backs the ``PlotGDModeDepth`` WDL.  Three subcommands, run in this order:

``summarize``
    Per batch.  Reads a low-resolution read-count matrix and normalizes it as
    ``plot.py`` does for its raw-depth overlay (counts scaled so each sample's
    autosomal median is CN 2, via ``_compute_raw_sample_medians`` and
    ``_build_raw_region_df``).  For every sample at every locus, bins between
    the locus breakpoints (the GD body intervals) are rounded to the nearest
    0.1, the most commonly observed rounded value is found, and the median of
    the actual (unrounded) depths that rounded to it is reported.
``merge``
    Once.  Concatenates the per-batch tables into one dataset.
``plot``
    Once.  Scatter of per-sample median depth, one x position per locus in
    ``gd_table`` order (DEL and DUP at the same locus share a position).  Only
    loci with probable mosaic carriers are labeled.  Several datasets can be
    stacked, sharing the x-axis.

Expected behaviour of the reported depth (CN 2 baseline = 2.0): non-carriers
sit near 2, germline het DEL carriers near 1 (het DUP near 3), mosaic carriers
between (1-2 for DEL, 2-3 for DUP), and carriers of an event that covers only
part of the locus sit near 2 or near the carrier value, never between.

Loci on chrX and chrY are excluded: normalization is autosomal and depth is
not rescaled for ploidy.
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

from gatk_sv_gd.models import GDLocus, GDTable
from gatk_sv_gd.plot import (
    _build_raw_region_df,
    _compute_raw_sample_medians,
    estimate_lowres_bin_size,
)

ROUND_DECIMALS = 1
SKIP_CHROMS = {"chrX", "chrY", "X", "Y"}
RASTER_DPI = 1000

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
        if not locus.is_nahr or str(locus.chrom) in SKIP_CHROMS:
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


def mode_and_median(values: np.ndarray) -> Optional[Tuple[float, float, int, int]]:
    """Most common 0.1-rounded value of *values*, and the median of the actual
    values that rounded to it.

    Ties go to the rounded value nearest 2, then the lower one.  Returns
    ``(mode, median, n_bins, mode_bins)`` or ``None`` when no finite values.
    """
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    scale = 10 ** ROUND_DECIMALS
    codes = np.round(values * scale).astype(np.int64)
    uniq, counts = np.unique(codes, return_counts=True)
    top = uniq[counts == counts.max()]
    mode_code = min(top, key=lambda c: (abs(c - 2 * scale), c))
    in_mode = values[codes == mode_code]
    return (round(mode_code / scale, ROUND_DECIMALS), float(np.median(in_mode)),
            int(values.size), int(in_mode.size))


def body_intervals(locus: GDLocus) -> List[Tuple[int, int]]:
    """Intervals between adjacent breakpoints (breakpoint ranges excluded)."""
    return [(s, e) for s, e, _ in locus.get_intervals() if e > s]


def summarize_locus(depth: pd.DataFrame, locus: GDLocus, batch: str) -> pd.DataFrame:
    """Per-sample most-common depth over the body bins of one locus.

    *depth* is a normalized region frame (Chr, Start, End, one column per
    sample), as returned by ``_build_raw_region_df``.
    """
    if depth is None or len(depth) == 0:
        return pd.DataFrame(columns=COLUMNS)
    mids = (depth["Start"].to_numpy() + depth["End"].to_numpy()) / 2
    in_body = np.zeros(len(depth), dtype=bool)
    for start, end in body_intervals(locus):
        in_body |= (mids >= start) & (mids < end)
    samples = [c for c in depth.columns if c not in ("Chr", "Start", "End")]
    matrix = depth.loc[in_body, samples].to_numpy(dtype=float)
    rows = []
    for j, sample in enumerate(samples):
        res = mode_and_median(matrix[:, j])
        if res is not None:
            rows.append((batch, locus.cluster, sample, res[0], res[1], res[2], res[3]))
    return pd.DataFrame(rows, columns=COLUMNS)


def read_count_matrix(path: str) -> pd.DataFrame:
    """Read a low-res count matrix (Chr, Start, End, samples) compactly."""
    header = pd.read_csv(path, sep="\t", compression="infer", nrows=0).columns
    coord = {"#Chr", "Chr", "Start", "End"}
    dtypes = {c: np.float32 for c in header if c not in coord}
    df = pd.read_csv(path, sep="\t", compression="infer", dtype=dtypes)
    if "#Chr" in df.columns:
        df = df.rename(columns={"#Chr": "Chr"})
    df["Start"] = df["Start"].astype(int)
    df["End"] = df["End"].astype(int)
    return df


def summarize_matrix(counts: pd.DataFrame, gd_table_path: str,
                     batch: str) -> pd.DataFrame:
    """Normalize *counts* as plot.py does and summarize every autosomal NAHR locus."""
    order = locus_order(gd_table_path)
    loci_by_key = {str(k): v for k, v in GDTable(gd_table_path).loci.items()}
    samples = [c for c in counts.columns if c not in ("Chr", "Start", "End")]
    medians = _compute_raw_sample_medians(counts)
    usable = [s for s in samples if s in medians]
    dropped = sorted(set(samples) - set(usable))
    if dropped:
        print(f"WARNING: no usable autosomal median for {len(dropped)} sample(s): "
              f"{', '.join(dropped[:10])}{' ...' if len(dropped) > 10 else ''}",
              file=sys.stderr)
    if not usable:
        return pd.DataFrame(columns=COLUMNS)
    bin_size = estimate_lowres_bin_size(counts)

    parts = []
    for cluster in order["cluster"]:
        locus = loci_by_key[cluster]
        depth = _build_raw_region_df(
            locus, locus.start, locus.end, counts, usable, medians, bin_size,
            highres_path=None)
        res = summarize_locus(depth, locus, batch)
        if len(res) == 0:
            print(f"WARNING: no body bins for locus {locus.display_label} in {batch}",
                  file=sys.stderr)
        res["cluster"] = str(cluster)
        parts.append(res)
    return pd.concat(parts, ignore_index=True)[COLUMNS] if parts else \
        pd.DataFrame(columns=COLUMNS)


def cmd_summarize(args) -> None:
    counts = read_count_matrix(args.low_res_counts)
    out = summarize_matrix(counts, args.gd_table, args.batch)
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
        # Dots are one high-resolution raster layer (a vector dot per sample
        # makes multi-MB PDFs that many viewers fail to draw); text, axes and
        # bands stay vector.  Colors alternate between adjacent loci.
        colors = np.where(data["cluster"].map(xmap).to_numpy(int) % 2 == 0,
                          COLOR_POINT_EVEN, COLOR_POINT_ODD)
        ax.scatter(x, y, s=1.6, c=colors, linewidths=0, clip_on=False, zorder=3,
                   rasterized=True)
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
    fig.savefig(out_pdf, dpi=RASTER_DPI)
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
    p.add_argument("--low-res-counts", required=True,
                   help="Low-resolution read-count matrix (.rd.txt.gz style)")
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
