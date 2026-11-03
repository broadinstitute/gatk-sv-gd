"""
Mosaic-carrier depth comparison plot for one GD locus.

Backs the ``PlotGDMosaicDepth`` WDL.  Four subcommands, run in this order:

``classify``
    Per batch.  Reads the ``CallGenomicDisorderCNVs`` outputs and labels every
    sample modelled at the locus's cluster as a germline carrier of the target
    GD, a non-carrier of every GD in the cluster, or neither.  With the
    batch's low-res counts it also flags samples whose normalised depth is
    abnormal in the flanks or in cluster intervals outside the target GD (a
    larger or different event), so ``select`` leaves them out of the
    germline and non-carrier comparators.
``select``
    Once.  Finds the batches holding the mosaic carriers, draws random
    non-carrier comparators from those batches, and draws min(requested,
    available) germline carriers from them too, adding the fewest extra
    batches needed when they hold too few.
``extract``
    Per batch holding a selected sample.  Builds normalised depth for the
    selected samples with the low-res procedure ``plot.py`` uses for its "raw
    normalised" column (autosomal-median scaling to CN 2, per-region
    rebinning).  High-res bins are not used, for a smoother trace.
``plot``
    Once.  Draws the compact publication panel as a PDF.
"""

import argparse
import itertools
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from gatk_sv_gd.annotations import (
    FlankCompressor,
    SegDupAnnotation,
    _build_gap_positions,
    _depths_with_gaps,
)
from gatk_sv_gd.models import GDLocus, GDTable
from gatk_sv_gd.plot import (
    _build_raw_region_df,
    _compute_raw_sample_medians,
    _rebin_region_df,
    estimate_lowres_bin_size,
)

GROUP_MOSAIC = "mosaic"
GROUP_GERMLINE = "germline"
GROUP_NON_CARRIER = "non_carrier"

STATUS_GERMLINE = "germline_carrier"
STATUS_NON_CARRIER = "non_carrier"
STATUS_OTHER = "other"

COLOR_NON_CARRIER = "#d3d3d3"
COLOR_GERMLINE = "#4d4d4d"
COLOR_MOSAIC = "#a45cc2"
COLOR_MOSAIC_OUTLINE = "#34104a"
COLOR_SEGDUP = "#e08a1e"
COLOR_BREAKPOINT = "#c0392b"


# =============================================================================
# Shared helpers
# =============================================================================


def _resolve_locus(gd_table_path: str, gd_id: str) -> GDLocus:
    """Return the locus (whole cluster, when clustered) containing *gd_id*."""
    table = GDTable(gd_table_path)
    for locus in table.loci.values():
        if any(str(e["GD_ID"]) == str(gd_id) for e in locus.gd_entries):
            return locus
    raise SystemExit(f"ERROR: GD_ID {gd_id!r} not found in GD table")


def _read_lines(path: str) -> List[str]:
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def _read_depth_columns(path: str, samples: List[str]) -> pd.DataFrame:
    """Read coordinate columns plus *samples* from a read-count matrix."""
    wanted = set(samples) | {"#Chr", "Chr", "Start", "End"}
    df = pd.read_csv(path, sep="\t", compression="infer",
                     usecols=lambda c: c in wanted)
    if "#Chr" in df.columns:
        df = df.rename(columns={"#Chr": "Chr"})
    df["Start"] = df["Start"].astype(int)
    df["End"] = df["End"].astype(int)
    return df


# =============================================================================
# classify
# =============================================================================


def classify_samples(
    calls_df: pd.DataFrame,
    cluster_samples: List[str],
    cluster: str,
    gd_id: str,
) -> pd.DataFrame:
    """Label each sample at *cluster* as germline carrier / non-carrier / other.

    A germline carrier is called for the target GD and for no other GD in the
    cluster.  A non-carrier is called for none.  Samples called for any other
    GD in the cluster (alone or together with the target) are ``other``.
    """
    carrier_gds: Dict[str, set] = {}
    if len(calls_df) > 0:
        hits = calls_df[
            (calls_df["cluster"].astype(str) == str(cluster))
            & (calls_df["is_carrier"].astype(str).str.lower() == "true")
        ]
        for sample, gd in zip(hits["sample"].astype(str), hits["GD_ID"].astype(str)):
            carrier_gds.setdefault(sample, set()).add(gd)

    status = []
    for sample in cluster_samples:
        gds = carrier_gds.get(sample, set())
        if not gds:
            status.append(STATUS_NON_CARRIER)
        elif gds == {str(gd_id)}:
            status.append(STATUS_GERMLINE)
        else:
            status.append(STATUS_OTHER)
    return pd.DataFrame({"sample": cluster_samples, "status": status})


def _load_cluster_bins(cn_posteriors: str, cluster: str) -> pd.DataFrame:
    """Unique (sample, Start, End) rows of one cluster from cn_posteriors."""
    parts = []
    usecols = ["cluster", "chr", "start", "end", "sample"]
    for chunk in pd.read_csv(cn_posteriors, sep="\t", compression="infer",
                             usecols=usecols, chunksize=500_000):
        sub = chunk[chunk["cluster"].astype(str) == str(cluster)]
        if len(sub):
            parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=usecols)
    return pd.concat(parts, ignore_index=True)


def _max_deviant_in_window(deviant: np.ndarray, window: int) -> int:
    """Largest count of True values in any run of *window* consecutive entries."""
    window = min(window, len(deviant))
    if window <= 0:
        return 0
    csum = np.concatenate([[0], np.cumsum(deviant, dtype=int)])
    return int((csum[window:] - csum[:-window]).max())


def load_segdup_intervals(path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Merged segdup intervals per chromosome (sorted starts, ends) from a BED."""
    bed = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2],
                      names=["chr", "start", "end"], comment="#",
                      compression="gzip" if path.endswith(".gz") else None)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for chrom, grp in bed.groupby("chr"):
        starts: List[int] = []
        ends: List[int] = []
        for a, b in grp.sort_values("start")[["start", "end"]].itertuples(index=False):
            if starts and a <= ends[-1]:
                ends[-1] = max(ends[-1], b)
            else:
                starts.append(a)
                ends.append(b)
        out[str(chrom)] = (np.array(starts), np.array(ends))
    return out


def overlaps_segdup(segdups: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]],
                    chrom: str, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """True for bins overlapping any segdup by at least 1 bp."""
    hit = np.zeros(len(starts), dtype=bool)
    if not segdups or str(chrom) not in segdups:
        return hit
    sd_starts, sd_ends = segdups[str(chrom)]
    idx = np.searchsorted(sd_starts, ends, side="left") - 1  # last SD starting before bin end
    valid = idx >= 0
    hit[valid] = sd_ends[idx[valid]] > starts[valid]
    return hit


def flag_abnormal_samples(
    depth: Optional[pd.DataFrame],
    locus: GDLocus,
    gd_id: str,
    ploidy: Dict[str, int],
    samples: List[str],
    max_deviation: float,
    window_bins: int = 5,
    min_deviant_bins: int = 3,
    max_population_fraction: float = 0.05,
    segdups: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> pd.DataFrame:
    """Flag samples with abnormal depth outside the target GD.

    *depth* is the normalised region frame (Chr, Start, End, one column per
    sample; CN 2 = 2.0).  Bins are grouped into the left flank, the right
    flank, and each cluster interval whose midpoint lies outside the target
    GD's span.  Bins overlapping a segdup (*segdups*, see
    :func:`load_segdup_intervals`) are ignored, as are bins where more than
    max(2, *max_population_fraction* x samples) samples deviate (common
    polymorphism, not an abnormal sample).

    A sample's bin deviates when it differs from the batch median of the
    samples sharing its ploidy (the ploidy itself when fewer than 5 share it)
    by more than *max_deviation*.  A sample is abnormal when, in any window of
    *window_bins* consecutive usable bins of one group, at least
    *min_deviant_bins* deviate in the same direction (all low or all high).
    Samples missing from *depth* are not flagged.

    Returns columns sample, abnormal, abnormal_region, max_deviation (the
    largest windowed count of deviating bins).
    """
    gd_start, gd_end = locus.get_gd_intervals()[str(gd_id)]
    present = [s for s in samples if depth is not None and s in depth.columns]
    worst = {s: (0, "") for s in samples}
    if depth is not None and len(depth) and present:
        starts = depth["Start"].to_numpy()
        ends = depth["End"].to_numpy()
        mid = (starts + ends) / 2
        base_ok = ~overlaps_segdup(segdups, locus.chrom, starts, ends)
        regions: List[Tuple[str, np.ndarray]] = [
            ("left_flank", mid < locus.start), ("right_flank", mid >= locus.end)]
        for start, end, name in locus.get_intervals():
            if end <= start or gd_start <= (start + end) / 2 < gd_end:
                continue
            regions.append((name, (mid >= start) & (mid < end)))

        by_ploidy: Dict[int, List[str]] = {}
        for s in present:
            by_ploidy.setdefault(int(ploidy.get(s, 2)), []).append(s)
        for pl, group in by_ploidy.items():
            values = depth[group].to_numpy(dtype=float)
            if len(group) >= 5:
                ref = np.nanmedian(values, axis=1)
            else:
                ref = np.full(len(values), float(pl))
            res = values - ref[:, None]
            n_dev = (np.abs(res) > max_deviation).sum(axis=1)
            usable = base_ok & (n_dev <= max(2, max_population_fraction * len(group)))
            for name, mask in regions:
                m = mask & usable
                if m.sum() < min_deviant_bins:
                    continue
                r = res[m]
                for j, sample in enumerate(group):
                    col = r[:, j]
                    col = col[np.isfinite(col)]
                    n = max(_max_deviant_in_window(col < -max_deviation, window_bins),
                            _max_deviant_in_window(col > max_deviation, window_bins))
                    if n > worst[sample][0]:
                        worst[sample] = (n, name)

    rows = []
    for sample in samples:
        n, name = worst[sample]
        bad = n >= min_deviant_bins
        rows.append((sample, bool(bad), name if bad else "", n))
    return pd.DataFrame(rows, columns=["sample", "abnormal", "abnormal_region",
                                       "max_deviation"])


def _ploidy_for(ploidy_path: Optional[str], chrom: str,
                samples: List[str]) -> Dict[str, int]:
    ploidy = {s: 2 for s in samples}
    if not ploidy_path:
        return ploidy
    pdf = pd.read_csv(ploidy_path, sep="\t")
    if not {"sample", "contig", "ploidy"}.issubset(pdf.columns):
        return ploidy
    pdf = pdf[pdf["contig"].astype(str) == str(chrom)]
    for sample, p in zip(pdf["sample"].astype(str), pdf["ploidy"]):
        if sample in ploidy and pd.notna(p):
            ploidy[sample] = int(p)
    return ploidy


def cmd_classify(args) -> None:
    locus = _resolve_locus(args.gd_table, args.gd_id)
    try:
        calls_df = pd.read_csv(args.calls, sep="\t", compression="infer")
    except pd.errors.EmptyDataError:
        calls_df = pd.DataFrame()

    bins = _load_cluster_bins(args.cn_posteriors, locus.cluster)
    samples = sorted(bins["sample"].astype(str).unique())
    classes = classify_samples(calls_df, samples, locus.cluster, args.gd_id)
    ploidy = _ploidy_for(args.ploidy_table, locus.chrom, samples)
    classes["batch"] = args.batch
    classes["ploidy"] = classes["sample"].map(ploidy)

    unique_bins = (
        bins[["start", "end"]].drop_duplicates().rename(
            columns={"start": "Start", "end": "End"}).sort_values("Start")
    )

    depth = None
    if args.low_res_counts and len(unique_bins):
        raw = _read_depth_columns(args.low_res_counts, samples)
        medians = _compute_raw_sample_medians(raw)
        usable = [s for s in samples if s in medians]
        if usable:
            depth = _build_raw_region_df(
                locus, int(unique_bins["Start"].min()), int(unique_bins["End"].max()),
                raw, usable, medians, estimate_lowres_bin_size(raw),
                highres_path=None)
    segdups = load_segdup_intervals(args.segdup_bed) if args.segdup_bed else None
    flags = flag_abnormal_samples(depth, locus, args.gd_id, ploidy, samples,
                                  args.max_flank_deviation, args.flank_window_bins,
                                  args.flank_min_deviant_bins,
                                  args.flank_max_population_fraction, segdups)
    classes = classes.merge(flags[["sample", "abnormal", "abnormal_region"]],
                            on="sample", how="left")
    classes[["sample", "batch", "status", "ploidy", "abnormal", "abnormal_region"]].to_csv(
        f"{args.out_prefix}.classes.tsv", sep="\t", index=False)

    with open(f"{args.out_prefix}.region.json", "w") as handle:
        json.dump({
            "batch": args.batch,
            "cluster": locus.cluster,
            "chrom": locus.chrom,
            "start": int(unique_bins["Start"].min()) if len(unique_bins) else None,
            "end": int(unique_bins["End"].max()) if len(unique_bins) else None,
            "n_samples": len(samples),
        }, handle)
    print(f"{args.batch}: {len(samples)} samples at {locus.cluster}; "
          f"{(classes['status'] == STATUS_GERMLINE).sum()} germline carriers, "
          f"{(classes['status'] == STATUS_NON_CARRIER).sum()} non-carriers; "
          f"{int(classes['abnormal'].sum())} with abnormal flank/cluster depth")


# =============================================================================
# select
# =============================================================================


MAX_COMBOS = 50_000


def choose_extra_batches(counts: Dict[str, int], need: int,
                         rng: np.random.RandomState) -> List[str]:
    """Fewest batches whose germline counts sum to at least *need*.

    Ties between equally small batch sets are broken at random.  Returns []
    when *need* is not positive; raises if the batches cannot supply *need*.
    """
    if need <= 0:
        return []
    names = sorted(counts)
    ordered = sorted(counts.values(), reverse=True)
    if sum(ordered) < need:
        raise ValueError("not enough germline carriers in the given batches")
    k, running = 0, 0
    while running < need:
        running += ordered[k]
        k += 1

    if math.comb(len(names), k) <= MAX_COMBOS:
        subsets = list(itertools.combinations(names, k))
    else:
        top = tuple(sorted(names, key=lambda b: (-counts[b], b))[:k])
        subsets = [top] + [
            tuple(sorted(rng.choice(names, size=k, replace=False)))
            for _ in range(MAX_COMBOS)]
    valid = [c for c in subsets if sum(counts[b] for b in c) >= need]
    return sorted(valid[rng.randint(len(valid))])


def select_samples(
    classes: pd.DataFrame,
    mosaic_samples: List[str],
    n_germline: int,
    n_non_carrier: int,
    seed: int,
) -> pd.DataFrame:
    """Pick mosaic carriers plus random comparators.

    Comparators share the mosaic carriers' (most common) ploidy on the locus
    contig.  Non-carriers come only from batches containing a mosaic carrier.
    Germline carriers number ``min(n_germline, total available)``: they come
    from the mosaic carriers' batches first; if those hold too few, the
    smallest possible number of additional batches (random among ties)
    supplies the rest.
    """
    classes = classes.drop_duplicates("sample")
    known = set(classes["sample"])
    missing = [s for s in mosaic_samples if s not in known]
    if missing:
        raise SystemExit(
            "ERROR: mosaic sample(s) not modelled at this locus in any batch: "
            + ", ".join(missing))

    mosaic = classes[classes["sample"].isin(mosaic_samples)]
    mosaic_batches = set(mosaic["batch"])
    ploidy = int(mosaic["ploidy"].mode().iloc[0])

    if "abnormal" in classes.columns:
        bad = classes["abnormal"].fillna(False).astype(bool)
        flagged = classes[bad & classes["sample"].isin(mosaic_samples)]
        if len(flagged):
            print("WARNING: mosaic sample(s) with abnormal flank/cluster depth "
                  "(kept): " + ", ".join(
                      f"{r.sample} ({r.abnormal_region})" for r in flagged.itertuples()),
                  file=sys.stderr)
        n_bad = int((bad & ~classes["sample"].isin(mosaic_samples)).sum())
        print(f"Excluding {n_bad} comparator sample(s) with abnormal flank/cluster "
              "depth", file=sys.stderr)
        classes = classes[~bad | classes["sample"].isin(mosaic_samples)]

    candidates = classes[
        ~classes["sample"].isin(mosaic_samples) & (classes["ploidy"] == ploidy)
    ].sort_values("sample")
    pool = candidates[candidates["batch"].isin(mosaic_batches)]
    rng = np.random.RandomState(seed)

    def draw(cand: pd.DataFrame, n: int, status: str) -> pd.DataFrame:
        if len(cand) < n:
            print(f"WARNING: only {len(cand)} {status} samples available "
                  f"(requested {n}); using all", file=sys.stderr)
            n = len(cand)
        idx = rng.choice(len(cand), size=n, replace=False)
        return cand.iloc[np.sort(idx)]

    germ_all = candidates[candidates["status"] == STATUS_GERMLINE]
    germ_local = germ_all[germ_all["batch"].isin(mosaic_batches)]
    target = min(n_germline, len(germ_all))
    if len(germ_local) >= target:
        germline = draw(germ_local, target, STATUS_GERMLINE)
    else:
        remote = germ_all[~germ_all["batch"].isin(mosaic_batches)]
        need = target - len(germ_local)
        extra = choose_extra_batches(
            remote.groupby("batch").size().to_dict(), need, rng)
        print(f"Germline carriers: {len(germ_local)} in mosaic batches; adding "
              f"batch(es) {', '.join(extra)} for {need} more", file=sys.stderr)
        germline = pd.concat([
            germ_local,
            draw(remote[remote["batch"].isin(extra)], need, STATUS_GERMLINE),
        ])

    out = pd.concat([
        mosaic.assign(group=GROUP_MOSAIC),
        germline.assign(group=GROUP_GERMLINE),
        draw(pool[pool["status"] == STATUS_NON_CARRIER], n_non_carrier,
             STATUS_NON_CARRIER).assign(group=GROUP_NON_CARRIER),
    ], ignore_index=True)
    return out[["sample", "batch", "group"]]


def cmd_select(args) -> None:
    classes = pd.concat(
        [pd.read_csv(p, sep="\t") for p in args.classes], ignore_index=True)
    mosaic_samples = _read_lines(args.mosaic_samples)
    if not mosaic_samples:
        raise SystemExit("ERROR: no mosaic samples given")
    selected = select_samples(
        classes, mosaic_samples, args.n_germline, args.n_non_carrier, args.seed)
    selected.to_csv(f"{args.out_prefix}.selected.tsv", sep="\t", index=False)

    # Every batch that contributes a selected sample needs depth extracted.
    used_batches = sorted(selected["batch"].unique())
    with open(f"{args.out_prefix}.batches.txt", "w") as handle:
        handle.write("".join(f"{b}\n" for b in used_batches))

    regions = [json.load(open(p)) for p in args.regions]
    # One flag per input region file, in input order, so a workflow can map
    # the flags back onto its own batch array.
    with open(f"{args.out_prefix}.use_batch.txt", "w") as handle:
        handle.write("".join(
            f"{str(r['batch'] in used_batches).lower()}\n" for r in regions))
    regions = [r for r in regions if r["batch"] in used_batches and r["start"] is not None]
    if not regions:
        raise SystemExit("ERROR: no bins found at this locus in the selected batches")
    region = {
        "cluster": regions[0]["cluster"],
        "chrom": regions[0]["chrom"],
        "start": min(r["start"] for r in regions),
        "end": max(r["end"] for r in regions),
    }
    with open(f"{args.out_prefix}.region.json", "w") as handle:
        json.dump(region, handle)
    print(selected["group"].value_counts().to_string())


# =============================================================================
# extract
# =============================================================================


def cmd_extract(args) -> None:
    locus = _resolve_locus(args.gd_table, args.gd_id)
    region = json.load(open(args.region))
    selected = pd.read_csv(args.selected, sep="\t")
    samples = sorted(selected[selected["batch"] == args.batch]["sample"].astype(str))
    if not samples:
        raise SystemExit(f"ERROR: no selected samples in batch {args.batch}")

    raw_counts_df = _read_depth_columns(args.low_res_counts, samples)
    medians = _compute_raw_sample_medians(raw_counts_df)
    usable = [s for s in samples if s in medians]
    dropped = sorted(set(samples) - set(usable))
    if dropped:
        print(f"WARNING: no usable autosomal median for: {', '.join(dropped)}",
              file=sys.stderr)
    bin_size = estimate_lowres_bin_size(raw_counts_df)

    depth = _build_raw_region_df(
        locus, int(region["start"]), int(region["end"]),
        raw_counts_df, usable, medians, bin_size,
        highres_path=None,
    )
    if depth is None or len(depth) == 0:
        raise SystemExit(f"ERROR: no depth bins found for batch {args.batch}")
    depth = _rebin_region_df(depth, locus)
    depth.to_csv(f"{args.out_prefix}.depth.tsv.gz", sep="\t", index=False,
                 compression="gzip")
    print(f"{args.batch}: {len(depth)} bins x {len(usable)} samples")


# =============================================================================
# plot
# =============================================================================


def _body_ticks_mb(start: int, end: int, max_ticks: int = 3) -> List[float]:
    """Round genomic positions (bp) spanning [start, end] for axis labels."""
    span_mb = max(end - start, 1) / 1e6
    step = next((s for s in (0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20)
                 if span_mb / s <= max_ticks), 50)
    first = np.ceil(start / 1e6 / step) * step
    ticks = np.arange(first, end / 1e6 + 1e-9, step)
    return [float(round(t * 1e6)) for t in ticks]


def _draw_annotation_strip(ax, locus: GDLocus, xform: FlankCompressor,
                           segdup: Optional[SegDupAnnotation],
                           region_start: int, region_end: int) -> None:
    """Segmental duplications and breakpoints, as in plot.py's annotation panel."""
    ax.set_xlim(0.0, xform.d_end)
    ax.set_ylim(0, 1)
    if segdup is not None:
        for sd_start, sd_end in segdup.get_regions_in_range(
                locus.chrom, region_start, region_end):
            ax.axvspan(xform(max(sd_start, region_start)),
                       xform(min(sd_end, region_end)),
                       ymin=0.08, ymax=0.42, color=COLOR_SEGDUP, alpha=0.7,
                       lw=0, zorder=2)
    for i, (bp_start, bp_end) in enumerate(locus.breakpoints):
        ds, de = xform(bp_start), xform(bp_end)
        ax.axvspan(ds, max(de, ds + xform.d_end * 0.002), ymin=0.5, ymax=0.78,
                   color=COLOR_BREAKPOINT, alpha=0.85, lw=0, zorder=3)
        name = locus.breakpoint_names[i] if i < len(locus.breakpoint_names) else str(i + 1)
        ax.text((ds + de) / 2, 0.86, f"BP{name}", ha="center", va="bottom",
                fontsize=4, color=COLOR_BREAKPOINT, clip_on=False)
    for d0, d1 in ((0.0, xform.d_body_start), (xform.d_body_end, xform.d_end)):
        if d1 > d0:
            ax.axvspan(d0, d1, color="0.85", alpha=0.55, lw=0, zorder=1)
    # Track labels, right-aligned just left of the strip.
    for label, y in (("Breakpoints", 0.64), ("Segmental\nduplications", 0.25)):
        ax.text(-0.02, y, label, transform=ax.transAxes, ha="right", va="center",
                fontsize=4.5, color="0.2", clip_on=False, linespacing=0.9)
    ax.axis("off")


def _draw_group(ax, df: pd.DataFrame, samples: List[str], xform: FlankCompressor,
                color: str, lw: float, alpha: float, zorder: int,
                outline: Optional[str] = None) -> None:
    starts, ends = df["Start"].values, df["End"].values
    mids = (starts + ends) / 2
    # NaN sentinels break the line across masked stretches; gaps are judged in
    # genomic space, and xform passes NaN through.
    x_genomic = _build_gap_positions(mids, starts, ends)
    x = xform(x_genomic)
    for sample in samples:
        if sample not in df.columns:
            continue
        y = _depths_with_gaps(df[sample].values.astype(float), x_genomic)
        line, = ax.plot(x, y, color=color, lw=lw, alpha=alpha, zorder=zorder,
                        solid_capstyle="round", solid_joinstyle="round")
        if outline:
            # Outline keeps overlapping mosaic lines individually distinguishable.
            line.set_path_effects([
                pe.Stroke(linewidth=lw + 0.5, foreground=outline), pe.Normal()])


def draw_mosaic_plot(
    locus: GDLocus,
    region: dict,
    selected: pd.DataFrame,
    depth_tables: List[pd.DataFrame],
    segdup: Optional[SegDupAnnotation],
    out_pdf: str,
    width: float = 2.0,
    height: float = 2.1,
    flank_scale: float = 0.15,
    ymin: float = 0.0,
    ymax: float = 4.0,
    title: Optional[str] = None,
) -> None:
    region_start, region_end = int(region["start"]), int(region["end"])
    xform = FlankCompressor(region_start, region_end, locus.start, locus.end,
                            flank_scale=flank_scale)

    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 5,
        "axes.linewidth": 0.5,
    })

    fig, (ax_ann, ax) = plt.subplots(
        2, 1, figsize=(width, height), sharex=True,
        gridspec_kw={"height_ratios": [0.26, 1], "hspace": 0.04})

    _draw_annotation_strip(ax_ann, locus, xform, segdup, region_start, region_end)

    # Faint breakpoint bands link the annotation strip to the depth traces.
    for bp_start, bp_end in locus.breakpoints:
        ax.axvspan(xform(bp_start), xform(bp_end), color=COLOR_BREAKPOINT,
                   alpha=0.08, lw=0, zorder=0)
    ax.axhline(2.0, color="0.6", lw=0.5, ls=(0, (3, 3)), zorder=1)

    counts = {}
    style = [
        (GROUP_NON_CARRIER, COLOR_NON_CARRIER, 0.3, 1.0, 2, None),
        (GROUP_GERMLINE, COLOR_GERMLINE, 0.3, 0.9, 3, None),
        (GROUP_MOSAIC, COLOR_MOSAIC, 0.3, 1.0, 4, COLOR_MOSAIC_OUTLINE),
    ]
    for group, color, lw, alpha, zorder, outline in style:
        members = selected[selected["group"] == group]
        counts[group] = 0
        for table in depth_tables:
            names = [s for s in members["sample"].astype(str) if s in table.columns]
            counts[group] += len(names)
            _draw_group(ax, table.sort_values("Start"), names, xform,
                        color, lw, alpha, zorder, outline)

    # Flanks are grayed out above the traces so context does not compete with
    # the locus body.
    for d0, d1 in ((0.0, xform.d_body_start), (xform.d_body_end, xform.d_end)):
        if d1 > d0:
            ax.axvspan(d0, d1, color="0.85", alpha=0.55, lw=0, zorder=6)

    ax.set_xlim(0.0, xform.d_end)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.arange(np.ceil(ymin), np.floor(ymax) + 1, 1))
    ax.set_ylabel("Normalized depth", fontsize=5.5, labelpad=1.5)
    ax.tick_params(axis="both", length=2, width=0.5, labelsize=5, pad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ticks = _body_ticks_mb(locus.start, locus.end)
    ax.set_xticks([xform(t) for t in ticks])
    ax.set_xticklabels([f"{t / 1e6:g}" for t in ticks])
    ax.set_xlabel(f"{locus.chrom} position (Mb)", fontsize=5.5, labelpad=1.5)

    if title is None:
        title = f"{locus.display_label}"
    ax_ann.set_title(title, loc="left", fontsize=5.5, pad=7 if locus.breakpoints else 2)

    handles = [
        Line2D([0], [0], color=COLOR_MOSAIC, lw=0.6,
               path_effects=[pe.Stroke(linewidth=0.8, foreground=COLOR_MOSAIC_OUTLINE),
                             pe.Normal()],
               label=f"Mosaic (n={counts[GROUP_MOSAIC]})"),
        Line2D([0], [0], color=COLOR_GERMLINE, lw=0.6,
               label=f"Germline (n={counts[GROUP_GERMLINE]})"),
        Line2D([0], [0], color=COLOR_NON_CARRIER, lw=0.6,
               label=f"Non-carrier (n={counts[GROUP_NON_CARRIER]})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False,
               fontsize=5, handlelength=1.4, labelspacing=0.3,
               bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.27, right=0.97, top=0.89, bottom=0.30)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def cmd_plot(args) -> None:
    locus = _resolve_locus(args.gd_table, args.gd_id)
    region = json.load(open(args.region))
    selected = pd.read_csv(args.selected, sep="\t")
    tables = [pd.read_csv(p, sep="\t", compression="infer") for p in args.depth]
    segdup = None
    if args.segdup_bed:
        segdup = SegDupAnnotation(args.segdup_bed)
    draw_mosaic_plot(
        locus, region, selected, tables, segdup, args.out_pdf,
        width=args.width, height=args.height, flank_scale=args.flank_scale,
        ymin=args.ymin, ymax=args.ymax, title=args.title,
    )
    print(f"Wrote {args.out_pdf}")


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mosaic-carrier depth comparison plot at a GD locus")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--gd-table", required=True)
        p.add_argument("--gd-id", required=True,
                       help="GD_ID of the target locus; its whole cluster is plotted")

    p = sub.add_parser("classify", help="Label samples in one batch")
    common(p)
    p.add_argument("--batch", required=True)
    p.add_argument("--calls", required=True, help="call/gd_cnv_calls.tsv.gz")
    p.add_argument("--cn-posteriors", required=True, help="infer/cn_posteriors.tsv.gz")
    p.add_argument("--ploidy-table", help="preprocess/ploidy_estimates.tsv")
    p.add_argument("--low-res-counts",
                   help="Batch low-res count matrix; enables flagging of samples with "
                        "abnormal depth in the flanks / other cluster intervals")
    p.add_argument("--max-flank-deviation", type=float, default=0.4,
                   help="A flank / other-cluster-interval bin deviates when its depth "
                        "differs from the sample's ploidy by more than this")
    p.add_argument("--flank-window-bins", type=int, default=5,
                   help="Sliding window length (bins) for the abnormal-depth test")
    p.add_argument("--flank-min-deviant-bins", type=int, default=3,
                   help="Flag a sample when at least this many bins of a window "
                        "deviate in the same direction")
    p.add_argument("--flank-max-population-fraction", type=float, default=0.05,
                   help="Ignore bins where more than this fraction of the batch deviates "
                        "(common polymorphism)")
    p.add_argument("--segdup-bed",
                   help="Segdup BED(.gz); bins overlapping it are ignored when "
                        "flagging abnormal flank / cluster-interval depth")
    p.add_argument("--out-prefix", required=True)
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("select", help="Choose mosaic carriers and comparators")
    p.add_argument("--classes", nargs="+", required=True)
    p.add_argument("--regions", nargs="+", required=True)
    p.add_argument("--mosaic-samples", required=True, help="One sample ID per line")
    p.add_argument("--n-germline", type=int, default=10)
    p.add_argument("--n-non-carrier", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-prefix", required=True)
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("extract", help="Normalised depth for one batch")
    common(p)
    p.add_argument("--batch", required=True)
    p.add_argument("--selected", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--low-res-counts", required=True)
    p.add_argument("--out-prefix", required=True)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("plot", help="Draw the PDF panel")
    common(p)
    p.add_argument("--selected", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--depth", nargs="+", required=True)
    p.add_argument("--segdup-bed")
    p.add_argument("--out-pdf", required=True)
    p.add_argument("--width", type=float, default=2.0, help="inches")
    p.add_argument("--height", type=float, default=2.1, help="inches")
    p.add_argument("--flank-scale", type=float, default=0.15)
    p.add_argument("--ymin", type=float, default=0.0)
    p.add_argument("--ymax", type=float, default=4.0)
    p.add_argument("--title")
    p.set_defaults(func=cmd_plot)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
