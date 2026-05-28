"""
Genomic Disorder (GD) CNV Detection from Binned Read Counts

This script detects copy number variants at known genomic disorder loci using
a hierarchical Bayesian model. Each locus (cluster) is defined by one or more
breakpoints, and the script determines which breakpoint set best explains the
observed depth signal for each sample.

Input:
    - Binned read count file (TSV)
    - GD table with locus definitions (TSV)
    - Optional exclusion interval BED files for masking (e.g. segdups, centromeres)

Output:
    - Per-locus CNV calls with breakpoint assignments
    - Copy number posteriors for each sample at each locus
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import pyro
import torch

if not hasattr(pyro, "enable_validation") and hasattr(pyro.distributions, "enable_validation"):
    pyro.enable_validation = pyro.distributions.enable_validation

from gatk_sv_gd import _util
from gatk_sv_gd._util import get_sample_columns, setup_logging
from gatk_sv_gd.bins import filter_low_quality_bins, read_data
from gatk_sv_gd.depth import CNVModel, DepthData, ExclusionMask, _windowed_relative_elbo_change
from gatk_sv_gd.models import GDTable
from gatk_sv_gd.output import build_ploidy_map, estimate_ploidy, write_locus_metadata, write_posterior_tables
from gatk_sv_gd.preprocess import (
    build_normalization_metadata,
    collect_all_locus_bins,
    load_preprocessed_data,
    write_normalization_metadata,
)


def _flatten_multi_args(arg_groups: List[List[str]]) -> List[str]:
    """Flatten argparse lists produced by repeated multi-value options."""
    return [value for group in arg_groups for value in group]


def _align_normalization_metadata(
    normalization_metadata: Optional[pd.DataFrame],
    sample_ids: List[str],
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Align per-sample raw-count medians to the modeled sample order."""
    if normalization_metadata is None or normalization_metadata.empty:
        raise ValueError(
            "Normalization metadata is required for the count-anchored spatial variance model."
        )

    required_columns = {"sample", "raw_count_median", "reference_bin_size"}
    missing_columns = required_columns.difference(normalization_metadata.columns)
    if missing_columns:
        raise ValueError(
            "Normalization metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    metadata = normalization_metadata.copy()
    metadata["sample"] = metadata["sample"].astype(str)
    metadata = metadata.drop_duplicates(subset=["sample"], keep="last")
    raw_count_lookup = metadata.set_index("sample")["raw_count_median"]

    missing_samples = [str(sample_id) for sample_id in sample_ids if str(sample_id) not in raw_count_lookup.index]
    if missing_samples:
        raise ValueError(
            "Normalization metadata is missing raw-count medians for samples: "
            f"{missing_samples[:5]}"
            + ("..." if len(missing_samples) > 5 else "")
        )

    reference_bin_sizes = metadata["reference_bin_size"].dropna().astype(float).unique()
    if len(reference_bin_sizes) != 1:
        raise ValueError(
            "Normalization metadata must contain exactly one reference_bin_size value."
        )

    aligned_raw_count_medians = np.asarray(
        [raw_count_lookup[str(sample_id)] for sample_id in sample_ids],
        dtype=np.float64,
    )
    return aligned_raw_count_medians, float(reference_bin_sizes[0])


def _write_training_loss_history(model: CNVModel, args: argparse.Namespace) -> None:
    """Persist ELBO history and log a compact convergence summary."""
    logger = _util.get_logger("infer")
    loss_history = getattr(model, "loss_history", None) or {}
    if "epoch" not in loss_history or "elbo" not in loss_history:
        loss_df = pd.DataFrame(columns=["epoch", "elbo"])
    else:
        loss_df = pd.DataFrame(loss_history)

    loss_path = os.path.join(args.output_dir, "training_loss.tsv")
    loss_df.to_csv(loss_path, sep="\t", index=False)
    logger.info("Wrote training loss history: epochs=%d", len(loss_df))

    if loss_df.empty:
        logger.warning("Training produced no ELBO history.")
        return

    elbo_history = loss_df["elbo"].to_numpy(dtype=np.float64)
    logger.info(
        "ELBO history summary: initial=%.4f final=%.4f best=%.4f",
        float(elbo_history[0]),
        float(elbo_history[-1]),
        float(np.min(elbo_history)),
    )

    relative_change = _windowed_relative_elbo_change(
        elbo_history,
        args.elbo_window,
    )
    if relative_change is None:
        logger.info(
            "ELBO convergence summary: final_window_change=unavailable window=%d target_rtol=%s",
            args.elbo_window,
            args.elbo_rtol,
        )
        return

    logger.info(
        "ELBO convergence summary: final_window_change=%.2e window=%d target_rtol=%s within_tolerance=%s",
        relative_change,
        args.elbo_window,
        args.elbo_rtol,
        relative_change < args.elbo_rtol,
    )


def run_gd_analysis(
    df: pd.DataFrame,
    gd_table: GDTable,
    exclusion_mask: Optional[ExclusionMask],
    hard_inclusion_mask: Optional[ExclusionMask],
    args: argparse.Namespace,
    device: str = "cpu",
    column_medians: Optional[np.ndarray] = None,
    lowres_median_bin_size: Optional[float] = None,
    normalization_metadata: Optional[pd.DataFrame] = None,
    preprocessed_bins: Optional[pd.DataFrame] = None,
    preprocessed_mappings=None,
    preprocessed_baf_summary: Optional[pd.DataFrame] = None,
    ploidy_map: Optional[dict] = None,
):
    """
    Run GD CNV analysis on all loci using a single unified model.

    This function performs model training and inference only.
    CNV calling is handled by downstream scripts (plot_gd_cnv_output.py).

    When *preprocessed_bins* and *preprocessed_mappings* are supplied (from
    the ``preprocess`` subcommand), bin collection is skipped entirely and
    the provided data is used directly.

    Args:
        df: DataFrame with normalized read depth
        gd_table: GDTable with locus definitions
        exclusion_mask: Optional ExclusionMask for filtering
        hard_inclusion_mask: Optional mask of regions whose overlapping
            bins are always retained during preprocessing.
        args: Command line arguments
        device: Torch device
        column_medians: Per-sample autosomal median raw counts (before
            normalisation).  Needed when ``args.high_res_counts`` is set.
        lowres_median_bin_size: Median bin size (bp) of the low-res file.
            Needed when ``args.high_res_counts`` is set.
        normalization_metadata: Optional per-sample raw-count medians and
            reference bin size used to build a count-anchored variance
            baseline.
        preprocessed_bins: Optional combined DataFrame from the
            ``preprocess`` subcommand.  When set, *preprocessed_mappings*
            must also be provided.
        preprocessed_mappings: Optional list of LocusBinMapping objects
            from the ``preprocess`` subcommand.
        ploidy_map: Optional ``{(sample, chrom): ploidy}`` lookup used
            for ploidy-adjusted quality filtering during bin collection.
    """
    if preprocessed_bins is not None and preprocessed_mappings is not None:
        # Use preprocessed data directly — skip bin collection
        combined_df = preprocessed_bins
        mappings = preprocessed_mappings
        included_loci = None  # already written by preprocess
    else:
        # Build quality-filter params dict.
        filter_params: dict = {
            "median_min": args.median_min,
            "median_max": args.median_max,
            "mad_max": args.mad_max,
        }
        highres_path: Optional[str] = getattr(args, "high_res_counts", None)

        # Collect all bins across all loci
        combined_df, mappings, included_loci = collect_all_locus_bins(
            df, gd_table, exclusion_mask,
            exclusion_threshold=args.exclusion_threshold,
            locus_padding=args.locus_padding,
            min_bins_per_interval=args.min_bins_per_interval,
            max_bins_per_interval=args.max_bins_per_interval,
            highres_counts_path=highres_path,
            column_medians=column_medians,
            lowres_median_bin_size=lowres_median_bin_size,
            filter_params=filter_params,
            exclusion_bypass_threshold=args.exclusion_bypass_threshold,
            min_rebin_coverage=args.min_rebin_coverage,
            min_flank_bases=args.min_flank_bases,
            min_flank_bins=args.min_flank_bins,
            min_flank_coverage=args.min_flank_coverage,
            ploidy_map=ploidy_map,
            hard_inclusion_mask=hard_inclusion_mask,
        )

    if len(combined_df) == 0:
        print("No bins to analyze!")
        return pd.DataFrame()

    # Create data object for all loci combined
    print("\nCreating combined depth data...")
    combined_data = DepthData(
        combined_df,
        device=device,
        dtype=torch.float32,
        clamp_threshold=args.clamp_threshold,
    )
    if preprocessed_baf_summary is not None:
        combined_data.attach_baf_summary(preprocessed_baf_summary, mappings)

    sample_raw_count_medians, reference_bin_size = _align_normalization_metadata(
        normalization_metadata,
        combined_data.sample_ids,
    )

    # Initialize and train a single model on all bins
    print("\nInitializing unified CNV model...")
    model = CNVModel(
        n_states=6,
        alpha_ref=args.alpha_ref,
        alpha_non_ref=args.alpha_non_ref,
        null_state_prior=args.null_state_prior,
        baf_temperature=args.baf_temperature,
        learn_baf_temperature=not args.fixed_baf_temperature and args.baf_temperature > 0,
        baf_temperature_prior_scale=args.baf_temperature_prior_scale,
        baf_outlier_rate=args.baf_outlier_rate,
        var_bias_bin=args.var_bias_bin,
        var_sample=args.var_sample,
        var_bin=args.var_bin,
        freeze_bin_bias=args.freeze_bin_bias,
        freeze_bin_var=args.freeze_bin_var,
        freeze_pair_state_priors=args.freeze_pair_state_priors,
        bin_size_factor=args.bin_size_factor,
        sample_raw_count_medians=sample_raw_count_medians,
        reference_bin_size=reference_bin_size,
        var_length_scale=args.var_length_scale,
        device=device,
        dtype=torch.float32,
        guide_type=args.guide_type,
    )

    print("\nTraining unified model on all GD loci bins...")
    model.train(
        data=combined_data,
        max_iter=args.max_iter,
        guide_warmup_iter=args.guide_warmup_iter,
        lr_init=args.lr_init,
        lr_min=args.lr_min,
        lr_decay=args.lr_decay,
        log_freq=args.log_freq,
        jit=not args.disable_jit,
        early_stopping=args.early_stopping,
        patience=args.patience,
        convergence_window=args.elbo_window,
        convergence_rtol=args.elbo_rtol,
    )
    _write_training_loss_history(model, args)

    # Get MAP estimates and posterior for all bins
    print("\nComputing MAP estimates...")
    map_estimates = model.get_map_estimates(combined_data)

    print("\nRunning discrete inference...")
    cn_posterior = model.run_discrete_inference(
        combined_data,
        n_samples=args.n_discrete_samples,
        log_freq=args.log_freq,
    )

    # Write comprehensive posterior tables
    write_posterior_tables(
        combined_data,
        map_estimates,
        cn_posterior,
        mappings,
        args.output_dir,
    )

    # Write locus metadata for downstream calling/plotting (skip if
    # preprocessed data was loaded — preprocess already wrote these files).
    if included_loci is not None:
        write_locus_metadata(
            included_loci,
            mappings,
            args.output_dir,
        )

    print("\n" + "=" * 80)
    print("Model training and inference complete!")
    print("Use plot_gd_cnv_output.py to call CNVs and generate plots.")
    print("=" * 80)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bayesian genomic disorder CNV detection from binned read depth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output
    parser.add_argument(
        "-i", "--input",
        required=False,
        help="Input TSV file with normalized read depth (bins x samples). "
             "Not required when --preprocessed-dir is set, but still useful "
             "there for rebuilding count-anchored normalization metadata "
             "from a legacy preprocess output.",
    )
    parser.add_argument(
        "-g", "--gd-table",
        required=False,
        help="GD locus definition table (TSV). "
             "Not required when --preprocessed-dir is set.",
    )
    parser.add_argument(
        "-e", "--exclusion-intervals",
        nargs="+",
        action="append",
        default=[],
        help="One or more BED files (plain or .bed.gz) of genomic regions "
             "to mask (e.g. segmental duplications, centromeres, satellites)."
             "  Intervals from all files are merged into a single exclusion "
             "mask.  Only the first three columns (chr, start, end) are "
             "required; additional columns are ignored.  May be specified "
             "multiple times or as a space-separated list.",
    )
    parser.add_argument(
        "--hard-inclusion-intervals",
        nargs="+",
        action="append",
        default=[],
        help="BED file(s) of regions whose overlapping bins are always retained, overriding exclusion and quality filtering.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--high-res-counts",
        required=False,
        help="Optional bgzipped, tabix-indexed high-resolution read count "
             "file (.tsv.gz + .tbi).  When provided, loci with any body "
             "interval below the target bin count (--min-bins-per-interval) "
             "are re-queried at this finer resolution before the hard check "
             "is enforced.  The file must have the same sample columns as "
             "the low-res input and contain raw (un-normalised) counts.",
    )
    parser.add_argument(
        "--preprocessed-dir",
        required=False,
        help="Directory produced by 'gatk-sv-gd preprocess'.  When set, "
             "bins and mappings are loaded from preprocessed_bins.tsv.gz "
             "and bin_mappings.tsv.gz instead of re-running preprocessing. "
             "The -i/--input, -g/--gd-table and -e/--exclusion-intervals "
             "flags are not required in this mode.",
    )

    # Locus processing
    parser.add_argument(
        "--locus-padding",
        type=int,
        default=10000,
        help="Padding around locus boundaries (bp)",
    )
    parser.add_argument(
        "--exclusion-threshold",
        type=float,
        default=0.5,
        help="Minimum overlap fraction with exclusion regions to mask a bin",
    )
    parser.add_argument(
        "--exclusion-bypass-threshold",
        type=float,
        default=0.6,
        help="If a body interval (region between adjacent breakpoints) is "
             "at least this fraction overlapped by exclusion regions, "
             "masking is skipped for bins in that interval.  This "
             "prevents intervals that are entirely within excluded "
             "regions from losing all their bins.  Set to 1.0 to "
             "disable bypass.",
    )
    parser.add_argument(
        "--min-bins-per-interval",
        type=int,
        default=10,
        help="Hard-failure minimum bins per body interval.  Intervals "
             "below this count after all processing cause a hard failure.",
    )
    parser.add_argument(
        "--max-bins-per-interval",
        type=int,
        default=20,
        help="Maximum bins per body interval after rebinning "
             "(0 = no rebinning)",
    )
    parser.add_argument(
        "--min-rebin-coverage",
        type=float,
        default=0.5,
        help="Minimum fraction of each new rebinned bin's width that must "
             "be covered by original bins (0.0–1.0, default 0.5).  Putative "
             "bins with less coverage are discarded to avoid biased estimates.",
    )
    parser.add_argument(
        "--min-flank-bases",
        type=int,
        default=50000,
        help="Minimum cumulative base pairs each flank must cover, "
             "regardless of locus size.  For small loci (e.g. < 1 kb) this "
             "ensures flanks are wide enough to establish a reliable "
             "baseline depth.  Flanks keep growing outward until this AND "
             "the locus-size target AND --min-flank-bins are all satisfied.",
    )
    parser.add_argument(
        "--min-flank-bins",
        type=int,
        default=10,
        help="Minimum number of bins each flank must contain, regardless "
             "of the base-pair thresholds.  Flanks keep growing outward "
             "until this AND the base-pair targets are all satisfied.",
    )
    parser.add_argument(
        "--min-flank-coverage",
        type=float,
        default=0.5,
        help="Minimum fraction of the effective bp target that a flank's "
             "accumulated bin coverage should reach.  The flank generator "
             "keeps extending outward until this threshold (and the other "
             "criteria) are met; it only stops when it runs out of bins "
             "at the chromosome boundary.  When the threshold cannot be "
             "met, a warning is logged but the flank is kept "
             "(default 0.5 = 50%%).",
    )

    # Model parameters
    parser.add_argument(
        "--alpha-ref",
        type=float,
        default=1.0,
        help="Dirichlet concentration for reference CN state (CN=2)",
    )
    parser.add_argument(
        "--alpha-non-ref",
        type=float,
        default=1.0,
        help="Dirichlet concentration for non-reference CN states",
    )
    parser.add_argument(
        "--baf-temperature",
        type=float,
        default=25.0,
        help="Global multiplicative BAF variance scale. Larger values soften "
             "BAF evidence and make contradictory BAF/depth evidence reduce "
             "posterior confidence rather than switching BAF off. In learned "
             "mode this is the LogNormal prior median; with "
             "--fixed-baf-temperature it is the fixed scale. Set to 0 to "
             "disable BAF evidence.",
    )
    parser.add_argument(
        "--fixed-baf-temperature",
        action="store_true",
        default=False,
        help="Keep the global BAF variance temperature fixed at "
             "--baf-temperature instead of learning it.",
    )
    parser.add_argument(
        "--baf-temperature-prior-scale",
        type=float,
        default=0.5,
        help="LogNormal prior scale for the learned global BAF variance temperature.",
    )
    parser.add_argument(
        "--baf-outlier-rate",
        type=float,
        default=0.0,
        help="Mixture weight for a uniform minor-allele BAF noise component. "
             "Positive values cap the penalty from contradictory off-model "
             "BAF bins; 0 disables the noise component.",
    )
    parser.add_argument(
        "--null-state-prior",
        type=float,
        default=1e-3,
        help="Prior probability assigned to an outer null state during exact "
             "discrete inference. The null state contributes neutral 1:1 "
             "event-vs-non-event odds for extreme off-model bins. Set to 0 "
             "to disable it.",
    )
    parser.add_argument(
        "--var-bias-bin",
        type=float,
        default=0.01,
        help="Variance for per-bin mean bias",
    )
    parser.add_argument(
        "--var-sample",
        type=float,
        default=0.01,
        help="Mean of the Exponential prior on per-sample excess variance above the count-anchored Poisson baseline",
    )
    parser.add_argument(
        "--var-bin",
        type=float,
        default=0,
        help="Deprecated compatibility option; per-bin excess variance is no longer used by the spatial count-anchored model",
    )
    parser.add_argument(
        "--freeze-bin-bias",
        action="store_true",
        default=True, # TODO
        help="Fix per-bin mean bias at 1.0 instead of inferring it",
    )
    parser.add_argument(
        "--unfreeze-bin-var",
        action="store_false",
        dest="freeze_bin_var",
        default=True,
        help="Deprecated compatibility flag; per-bin excess variance is no longer inferred",
    )
    parser.add_argument(
        "--freeze-pair-state-priors",
        action="store_true",
        default=True, # TODO 
        help="Fix per-bin pair-state priors to the Dirichlet prior mean instead of inferring them",
    )
    parser.add_argument(
        "--bin-size-factor",
        type=float,
        default=10000.0,
        help="Reference bin size (bp) for variance scaling.  The total "
             "variance is multiplied by (bin_size_factor / interval_size) "
             "* (expected_depth / 2.0) so that smaller bins and higher "
             "expected normalized depth have proportionally higher variance.  "
             "Note that this is redundant with the other scale factors "
             "and is only exposed for debugging. Set to 0 to disable "
             "bin-size variance scaling.",
    )
    parser.add_argument(
           "--var-length-scale",
        type=float,
              default=20000.0,
           help="Mean of the Exponential prior on the shared physical "
               "correlation length-scale (in bp) used by the count-anchored "
               "spatial aggregation variance model. The excess variance is "
               "modeled as d**2 * sample_var * f(L; length_scale_var), where "
               "f saturates for small bins and decays approximately as 2*ell/L "
               "for bins much larger than the correlation length.",
    )
    parser.add_argument(
        "--guide-type",
        type=str,
        default="delta",
        choices=["delta", "diagonal"],
        help="Type of variational guide",
    )
    parser.add_argument(
        "--guide-warmup-iter",
        type=int,
        default=250,
        help="AutoDelta MAP warmup iterations before diagonal-guide training; set to 0 to disable and ignored for --guide-type delta",
    )
    parser.add_argument(
        "--clamp-threshold",
        type=float,
        default=5.0,
        help="Maximum value for depth clamping",
    )

    # Training parameters
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Maximum training iterations per locus",
    )
    parser.add_argument(
        "--lr-init",
        type=float,
        default=0.02,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=0.01,
        help="Minimum learning rate",
    )
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=500,
        help="Learning rate decay constant",
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=100,
        help="Logging frequency (iterations)",
    )
    parser.add_argument(
        "--disable-jit",
        action="store_true",
        default=False,
        help="Disable JIT compilation",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=True,
        help="Enable early stopping",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_false",
        dest="early_stopping",
        help="Disable early stopping",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Consecutive rolling ELBO-window checks below tolerance before stopping",
    )
    parser.add_argument(
        "--elbo-window",
        type=int,
        default=50,
        help="Iterations per rolling ELBO window for early stopping",
    )
    parser.add_argument(
        "--elbo-rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance between successive rolling ELBO windows",
    )

    # Inference parameters
    parser.add_argument(
        "--n-discrete-samples",
        type=int,
        default=1000,
        help="Number of samples for discrete inference",
    )
    parser.add_argument(
        "--median-min",
        type=float,
        default=1.0,
        help="Minimum median depth for bins",
    )
    parser.add_argument(
        "--median-max",
        type=float,
        default=3.0,
        help="Maximum median depth for bins",
    )
    parser.add_argument(
        "--mad-max",
        type=float,
        default=2.0,
        help="Maximum MAD for bins",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for computation",
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable detailed per-bin diagnostic logging for normalisation, "
             "quality filtering, exclusion masking, high-res replacement, and "
             "rebinning.  Useful for investigating discrepancies between "
             "raw and normalised depth signals.",
    )

    args = parser.parse_args()
    if args.baf_temperature < 0:
        parser.error("--baf-temperature must be non-negative.")
    if args.baf_temperature_prior_scale <= 0:
        parser.error("--baf-temperature-prior-scale must be positive.")
    if not 0.0 <= args.baf_outlier_rate < 1.0:
        parser.error("--baf-outlier-rate must be in [0, 1).")
    if not 0.0 <= args.null_state_prior < 1.0:
        parser.error("--null-state-prior must be in [0, 1).")
    if args.var_length_scale <= 0:
        parser.error("--var-length-scale must be positive.")
    if args.guide_warmup_iter < 0:
        parser.error("--guide-warmup-iter must be non-negative.")

    return args


def _setup_pyro(args: argparse.Namespace) -> None:
    """Configure deterministic Pyro state for infer runs.

    Validation stays off for JIT because distribution checks convert tensor
    masks to Python booleans while tracing.
    """
    validate = bool(args.disable_jit)
    pyro.enable_validation(validate)
    pyro.distributions.enable_validation(validate)
    pyro.set_rng_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)


def main():
    """Main function to run GD CNV detection pipeline."""
    args = parse_args()
    args.exclusion_intervals = _flatten_multi_args(args.exclusion_intervals)
    args.hard_inclusion_intervals = _flatten_multi_args(args.hard_inclusion_intervals)

    _util.VERBOSE = args.verbose

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    setup_logging(
        args.output_dir,
        verbose=args.verbose,
        command="infer",
        args=args,
        seed_info={"pyro": 42, "torch": 42, "numpy": 42},
    )

    print("Output directory configured")

    # ------------------------------------------------------------------
    # Branch: load preprocessed data or run full preprocessing
    # ------------------------------------------------------------------
    if args.preprocessed_dir:
        print("\nLoading preprocessed data")
        (
            preprocessed_bins,
            preprocessed_mappings,
            preprocessed_baf_summary,
            normalization_metadata,
        ) = load_preprocessed_data(args.preprocessed_dir)

        if (normalization_metadata is None or normalization_metadata.empty) and args.input:
            print("  Recomputing normalization metadata from --input for count-anchored variance")
            raw_df = read_data(args.input)
            sample_cols = get_sample_columns(raw_df)
            autosome_mask = ~raw_df["Chr"].isin(["chrX", "chrY"])
            if autosome_mask.any():
                column_medians = np.median(raw_df.loc[autosome_mask, sample_cols], axis=0)
            else:
                column_medians = np.median(raw_df[sample_cols], axis=0)
            lowres_bin_sizes = (raw_df["End"] - raw_df["Start"]).values
            lowres_median_bin_size = float(np.median(lowres_bin_sizes))
            normalization_metadata = build_normalization_metadata(
                sample_cols,
                column_medians,
                lowres_median_bin_size,
            )
            write_normalization_metadata(normalization_metadata, args.output_dir)
        elif normalization_metadata is None or normalization_metadata.empty:
            print(
                "Error: normalization metadata is required when loading "
                "preprocessed data. Re-run preprocess or also provide --input "
                "so infer can recompute it."
            )
            raise SystemExit(1)
        else:
            write_normalization_metadata(normalization_metadata, args.output_dir)

        # Set up Pyro
        _setup_pyro(args)

        # Run GD analysis with preprocessed data (no bin collection)
        run_gd_analysis(
            pd.DataFrame(), GDTable.__new__(GDTable), None, None, args,
            device=args.device,
            preprocessed_bins=preprocessed_bins,
            preprocessed_mappings=preprocessed_mappings,
            preprocessed_baf_summary=preprocessed_baf_summary,
            normalization_metadata=normalization_metadata,
        )  # ploidy_map not needed — bins already collected
    else:
        # Validate that required args are present
        if not args.input:
            print("Error: --input is required unless --preprocessed-dir is set.")
            raise SystemExit(1)
        if not args.gd_table:
            print("Error: --gd-table is required unless --preprocessed-dir is set.")
            raise SystemExit(1)

        # Load GD table
        print("\nLoading GD table")
        gd_table = GDTable(args.gd_table)
        loci_with_breakpoints = sum(1 for locus in gd_table.loci.values() if locus.breakpoints)
        total_breakpoints = sum(locus.n_breakpoints for locus in gd_table.loci.values())
        print(
            f"Loaded {len(gd_table.loci)} loci; "
            f"{loci_with_breakpoints} with breakpoints; "
            f"{total_breakpoints} total breakpoint intervals"
        )

        # Load exclusion mask — all BED files are concatenated first so
        # that cross-file overlapping intervals are merged before the
        # index is built.
        exclusion_mask = None
        if args.exclusion_intervals:
            print(f"\nLoading {len(args.exclusion_intervals)} exclusion interval file(s)")
            exclusion_mask = ExclusionMask(
                args.exclusion_intervals,
                label="exclusion regions",
            )

        hard_inclusion_mask = None
        if args.hard_inclusion_intervals:
            print(f"\nLoading {len(args.hard_inclusion_intervals)} hard inclusion interval file(s)")
            hard_inclusion_mask = ExclusionMask(
                args.hard_inclusion_intervals,
                label="hard inclusion regions",
            )

        # Load read depth data
        df = read_data(args.input)

        # Normalize by sample median over autosomal bins
        sample_cols = get_sample_columns(df)
        autosome_mask = ~df["Chr"].isin(["chrX", "chrY"])
        if autosome_mask.any():
            column_medians = np.median(df.loc[autosome_mask, sample_cols], axis=0)
        else:
            column_medians = np.median(df[sample_cols], axis=0)

        print(f"Column medians: min={column_medians.min():.3f}, "
              f"max={column_medians.max():.3f}, mean={column_medians.mean():.3f}")

        if _util.VERBOSE:
            print(
                "\n  [verbose] Per-sample autosomal median raw-count summary: "
                f"samples={len(sample_cols)}, min={column_medians.min():.3f}, "
                f"median={np.median(column_medians):.3f}, max={column_medians.max():.3f}"
            )
            raw_depths = df[sample_cols].values
            print(f"\n  [verbose] Pre-normalisation depth summary "
                  f"({len(df)} bins x {len(sample_cols)} samples):")
            print(f"    global mean = {np.nanmean(raw_depths):.4f}")
            print(f"    global median = {np.nanmedian(raw_depths):.4f}")
            print(f"    per-sample means: min={np.nanmean(raw_depths, axis=0).min():.4f}, "
                  f"max={np.nanmean(raw_depths, axis=0).max():.4f}")

        lowres_bin_sizes = (df["End"] - df["Start"]).values
        lowres_median_bin_size = float(np.median(lowres_bin_sizes))
        print(f"Low-res median bin size: {lowres_median_bin_size:,.0f} bp")

        normalization_metadata = build_normalization_metadata(
            sample_cols,
            column_medians,
            lowres_median_bin_size,
        )
        write_normalization_metadata(normalization_metadata, args.output_dir)

        # Normalize such that CN=2 corresponds to depth of 2.0
        df[sample_cols] = 2.0 * df[sample_cols] / column_medians[np.newaxis, :]

        if _util.VERBOSE:
            norm_depths = df[sample_cols].values
            print("\n  [verbose] Post-normalisation depth summary:")
            print(f"    global mean = {np.nanmean(norm_depths):.4f}")
            print(f"    global median = {np.nanmedian(norm_depths):.4f}")
            print(f"    per-sample means: min={np.nanmean(norm_depths, axis=0).min():.4f}, "
                  f"max={np.nanmean(norm_depths, axis=0).max():.4f}")

        # Estimate ploidy (before quality filtering so the ploidy map is
        # available for ploidy-adjusted median/MAD computation)
        ploidy_df = estimate_ploidy(df, args.output_dir)
        ploidy_map = build_ploidy_map(ploidy_df)

        # Filter low quality bins
        df = filter_low_quality_bins(
            df,
            median_min=args.median_min,
            median_max=args.median_max,
            mad_max=args.mad_max,
            ploidy_map=ploidy_map,
            hard_inclusion_mask=hard_inclusion_mask,
        )

        # Set up Pyro
        _setup_pyro(args)

        # Log high-res counts file if provided
        if args.high_res_counts:
            print("\nHigh-resolution counts enabled")
        else:
            print("\nNo high-resolution counts file provided (--high-res-counts)")

        # Run GD analysis (training and inference only)
        run_gd_analysis(
            df, gd_table, exclusion_mask, hard_inclusion_mask, args, device=args.device,
            column_medians=column_medians,
            lowres_median_bin_size=lowres_median_bin_size,
            normalization_metadata=normalization_metadata,
            ploidy_map=ploidy_map,
        )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nOutput tables written: 7")
    print("\nNext step: run call and plot on the inference outputs.")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
