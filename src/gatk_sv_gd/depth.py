"""
Depth data containers and Bayesian CNV model.

Contains:
    - ExclusionMask: genomic exclusion region handling
    - DepthData: depth matrix container for the Pyro model
    - CNVModel: hierarchical Bayesian CNV detection model
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from intervaltree import IntervalTree
from scipy.special import gammaln

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.ops.indexing import Vindex
from pyro.infer import config_enumerate, infer_discrete
from pyro.infer.autoguide import AutoDiagonalNormal, AutoDelta, AutoGuideList
from pyro.infer.autoguide.initialization import init_to_value
from pyro.infer import JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.infer.svi import SVI

from gatk_sv_gd._util import get_logger, get_sample_columns


_MINOR_BAF_UNIFORM_LOG_DENSITY = math.log(2.0)
_NORMALIZED_DIPLOID_DEPTH = 2.0
_MIN_DEPTH_VARIANCE_SCALE = 1e-6
_MIN_SPATIAL_AGGREGATE_RATIO = 1e-8
_SPATIAL_AGGREGATE_TAYLOR_THRESHOLD = 1e-2
_DEFAULT_MIN_VARIANCE_EXPECTED_DEPTH = 0.10


def _lognormal_location_from_mean(mean: float, scale: float) -> float:
    """Return the LogNormal location matching a natural-scale mean."""
    return math.log(float(mean)) - 0.5 * float(scale) ** 2


def _depth_variance_scale_torch(
    expected_depth: torch.Tensor,
    power: Union[float, torch.Tensor] = 1.0,
) -> torch.Tensor:
    """Scale variance with expected normalized depth relative to diploid depth."""
    scaled_depth = torch.clamp(
        expected_depth / _NORMALIZED_DIPLOID_DEPTH,
        min=_MIN_DEPTH_VARIANCE_SCALE,
    )
    return torch.clamp(
        torch.pow(scaled_depth, power),
        min=_MIN_DEPTH_VARIANCE_SCALE,
    )


def _depth_variance_scale_numpy(
    expected_depth: np.ndarray,
    power: Union[float, np.ndarray] = 1.0,
) -> np.ndarray:
    """NumPy counterpart of :func:`_depth_variance_scale_torch`."""
    expected_depth = np.asarray(expected_depth, dtype=np.float64)
    scaled_depth = np.maximum(
        expected_depth / _NORMALIZED_DIPLOID_DEPTH,
        _MIN_DEPTH_VARIANCE_SCALE,
    )
    return np.maximum(
        np.power(scaled_depth, power),
        _MIN_DEPTH_VARIANCE_SCALE,
    )


def _variance_expected_depth_torch(
    expected_depth: torch.Tensor,
    min_expected_depth: float,
) -> torch.Tensor:
    """Lower-bound expected depth when assembling variance terms."""
    if float(min_expected_depth) <= 0.0:
        return expected_depth
    return torch.clamp(expected_depth, min=float(min_expected_depth))


def _variance_expected_depth_numpy(
    expected_depth: np.ndarray,
    min_expected_depth: float,
) -> np.ndarray:
    """NumPy counterpart of :func:`_variance_expected_depth_torch`."""
    expected_depth = np.asarray(expected_depth, dtype=np.float64)
    if float(min_expected_depth) <= 0.0:
        return expected_depth
    return np.maximum(expected_depth, float(min_expected_depth))


def _count_anchored_reference_variance_torch(
    sample_raw_count_medians: torch.Tensor,
    reference_bin_size: float,
    bin_size_factor: float,
) -> torch.Tensor:
    """Return diploid Poisson variance in normalized-depth units."""
    variance = 4.0 / sample_raw_count_medians
    if bin_size_factor > 0:
        variance = variance * (reference_bin_size / bin_size_factor)
    return variance.view(1, -1)


def _count_anchored_reference_variance_numpy(
    sample_raw_count_medians: np.ndarray,
    reference_bin_size: float,
    bin_size_factor: float,
) -> np.ndarray:
    """NumPy counterpart of :func:`_count_anchored_reference_variance_torch`."""
    sample_raw_count_medians = np.asarray(sample_raw_count_medians, dtype=np.float64).reshape(-1)
    variance = 4.0 / sample_raw_count_medians
    if bin_size_factor > 0:
        variance = variance * (reference_bin_size / bin_size_factor)
    return variance.reshape(1, -1)


def _size_modifier_torch(interval_sizes: torch.Tensor, bin_size_factor: float) -> torch.Tensor:
    """Return the active bin-size modifier for depth variance."""
    if bin_size_factor > 0:
        return bin_size_factor / interval_sizes
    return torch.ones_like(interval_sizes)


def _size_modifier_numpy(interval_sizes: np.ndarray, bin_size_factor: float) -> np.ndarray:
    """NumPy counterpart of :func:`_size_modifier_torch`."""
    interval_sizes = np.asarray(interval_sizes, dtype=np.float64)
    if bin_size_factor > 0:
        return bin_size_factor / interval_sizes
    return np.ones_like(interval_sizes, dtype=np.float64)


def _spatial_aggregate_variance_scale_torch(
    interval_sizes: torch.Tensor,
    length_scale: torch.Tensor,
) -> torch.Tensor:
    """Return the continuous-AR(1) spatial aggregation factor for interval size."""
    safe_length_scale = torch.clamp(length_scale, min=_MIN_SPATIAL_AGGREGATE_RATIO)
    x = torch.clamp(interval_sizes / safe_length_scale, min=_MIN_SPATIAL_AGGREGATE_RATIO)
    one_minus_exp_neg_x = -torch.expm1(-x)
    small_x_val = 1.0 - (x / 3.0)
    large_x_val = (2.0 / x) * (1.0 - (one_minus_exp_neg_x / x))
    return torch.clamp(
        torch.where(x < _SPATIAL_AGGREGATE_TAYLOR_THRESHOLD, small_x_val, large_x_val),
        min=0.0,
    )


def _spatial_aggregate_variance_scale_numpy(
    interval_sizes: np.ndarray,
    length_scale: Union[float, np.ndarray],
) -> np.ndarray:
    """NumPy counterpart of :func:`_spatial_aggregate_variance_scale_torch`."""
    interval_sizes = np.asarray(interval_sizes, dtype=np.float64)
    safe_length_scale = np.clip(np.asarray(length_scale, dtype=np.float64), _MIN_SPATIAL_AGGREGATE_RATIO, None)
    x = np.clip(interval_sizes / safe_length_scale, _MIN_SPATIAL_AGGREGATE_RATIO, None)
    one_minus_exp_neg_x = -np.expm1(-x)
    small_x_val = 1.0 - (x / 3.0)
    large_x_val = (2.0 / x) * (1.0 - (one_minus_exp_neg_x / x))
    return np.clip(
        np.where(x < _SPATIAL_AGGREGATE_TAYLOR_THRESHOLD, small_x_val, large_x_val),
        0.0,
        None,
    )


def build_diploid_pair_states(max_hap_cn: int = 2) -> List[Tuple[int, int]]:
    """Return canonical unordered diploid pair states with h1 <= h2."""
    return [
        (h1, h2)
        for h1 in range(max_hap_cn + 1)
        for h2 in range(h1, max_hap_cn + 1)
    ]


def pair_state_minor_baf(pair_states: List[Tuple[int, int]]) -> np.ndarray:
    """Expected minor-allele BAF for each diploid pair state."""
    values = []
    for h1, h2 in pair_states:
        total = h1 + h2
        if total <= 0:
            values.append(0.0)
        else:
            values.append(min(h1, h2) / total)
    return np.asarray(values, dtype=np.float32)


def pair_state_total_cn(pair_states: List[Tuple[int, int]]) -> np.ndarray:
    """Total CN implied by each diploid pair state."""
    return np.asarray([h1 + h2 for h1, h2 in pair_states], dtype=np.int64)


def _center_state_log_likelihood_table_torch(
    log_lik_table: torch.Tensor,
    reference_probs: torch.Tensor,
) -> torch.Tensor:
    """Center per-state log-likelihoods against a fixed reference mixture.

    The returned table is a relative-evidence table with the property that

    ``logsumexp(log(reference_probs) + centered, axis=state) == 0``

    for every downstream bin/sample cell. This removes any state-independent
    density offset, which makes learned evidence temperatures well-posed.
    """
    reference_probs = torch.clamp(reference_probs, min=1e-10)
    if reference_probs.dim() != 1:
        raise ValueError("reference_probs must be a 1D tensor.")
    view_shape = (reference_probs.shape[0],) + (1,) * (log_lik_table.dim() - 1)
    reference_log_probs = torch.log(reference_probs).view(view_shape)
    baseline = torch.logsumexp(reference_log_probs + log_lik_table, dim=0, keepdim=True)
    centered = log_lik_table - baseline
    return torch.where(
        torch.isfinite(baseline).expand_as(centered),
        centered,
        torch.zeros_like(centered),
    )


def _center_state_log_likelihood_table_numpy(
    log_lik_table: np.ndarray,
    reference_probs: np.ndarray,
) -> np.ndarray:
    """NumPy counterpart of :func:`_center_state_log_likelihood_table_torch`."""
    reference_probs = np.asarray(reference_probs, dtype=np.float64)
    if reference_probs.ndim != 1:
        raise ValueError("reference_probs must be a 1D array.")
    reference_probs = np.maximum(reference_probs, 1e-10)
    view_shape = (reference_probs.shape[0],) + (1,) * (log_lik_table.ndim - 1)
    raw = np.log(reference_probs).reshape(view_shape) + log_lik_table
    max_val = np.max(raw, axis=0, keepdims=True)
    finite_max = np.where(np.isfinite(max_val), max_val, 0.0)
    shifted = np.where(np.isfinite(max_val), raw - finite_max, -np.inf)
    baseline = np.where(
        np.isfinite(max_val),
        finite_max + np.log(np.sum(np.exp(shifted), axis=0, keepdims=True) + 1e-30),
        -np.inf,
    )
    centered = np.zeros_like(log_lik_table)
    np.subtract(log_lik_table, baseline, out=centered, where=np.isfinite(baseline))
    return centered


def _select_state_log_likelihood_torch(
    log_lik_table: torch.Tensor,
    state_index: torch.Tensor,
) -> torch.Tensor:
    """Select per-cell state log-likelihoods from a state-major tensor."""
    data_ndim = log_lik_table.dim() - 1
    enum_shape = state_index.shape[:-data_ndim]
    target_shape = enum_shape + log_lik_table.shape[1:]
    gather_dim = len(enum_shape)

    expanded_state_index = state_index.expand(target_shape)
    expanded_log_lik_table = log_lik_table.view((1,) * gather_dim + log_lik_table.shape).expand(
        enum_shape + log_lik_table.shape
    )
    return torch.gather(
        expanded_log_lik_table,
        dim=gather_dim,
        index=expanded_state_index.unsqueeze(gather_dim),
    ).squeeze(gather_dim)


def _clip_baf_variance_torch(value: torch.Tensor) -> torch.Tensor:
    """Clamp scaled BAF variances to a finite range before Normal log-probs."""
    return torch.clamp(
        torch.nan_to_num(value, nan=1e6, posinf=1e6, neginf=1e-6),
        min=1e-6,
        max=1e6,
    )


def _safe_scaled_baf_variance_torch(
    baf_var: torch.Tensor,
    valid_mask: torch.Tensor,
    baf_temperature: torch.Tensor,
) -> torch.Tensor:
    """Scale BAF variance without letting masked NaNs affect gradients."""
    finite_baf_var = torch.nan_to_num(baf_var, nan=1.0, posinf=1.0, neginf=1.0)
    positive_baf_var = torch.where(
        finite_baf_var > 0,
        finite_baf_var,
        torch.ones_like(finite_baf_var),
    )
    scaled_baf_var = torch.where(
        valid_mask,
        positive_baf_var * baf_temperature,
        torch.ones_like(positive_baf_var),
    )
    return _clip_baf_variance_torch(scaled_baf_var)


def _clip_baf_variance_numpy(value: np.ndarray) -> np.ndarray:
    """NumPy counterpart of :func:`_clip_baf_variance_torch`."""
    return np.clip(
        np.nan_to_num(value, nan=1e6, posinf=1e6, neginf=1e-6),
        1e-6,
        1e6,
    )


def _robust_baf_log_likelihood_torch(
    log_lik_table: torch.Tensor,
    outlier_rate: float,
) -> torch.Tensor:
    """Mix Gaussian BAF evidence with a uniform minor-allele noise floor."""
    if outlier_rate <= 0.0:
        return log_lik_table
    if outlier_rate >= 1.0:
        return torch.full_like(log_lik_table, _MINOR_BAF_UNIFORM_LOG_DENSITY)
    inlier_log_weight = torch.tensor(
        math.log1p(-outlier_rate),
        device=log_lik_table.device,
        dtype=log_lik_table.dtype,
    )
    outlier_log_weight = torch.tensor(
        math.log(outlier_rate),
        device=log_lik_table.device,
        dtype=log_lik_table.dtype,
    )
    uniform_log_lik = torch.full_like(log_lik_table, _MINOR_BAF_UNIFORM_LOG_DENSITY)
    return torch.logaddexp(
        log_lik_table + inlier_log_weight,
        uniform_log_lik + outlier_log_weight,
    )


def _robust_baf_log_likelihood_numpy(
    log_lik_table: np.ndarray,
    outlier_rate: float,
) -> np.ndarray:
    """NumPy counterpart of :func:`_robust_baf_log_likelihood_torch`."""
    if outlier_rate <= 0.0:
        return log_lik_table
    if outlier_rate >= 1.0:
        return np.full_like(log_lik_table, _MINOR_BAF_UNIFORM_LOG_DENSITY)
    return np.logaddexp(
        log_lik_table + math.log1p(-outlier_rate),
        _MINOR_BAF_UNIFORM_LOG_DENSITY + math.log(outlier_rate),
    )


def _windowed_relative_elbo_change(
    loss_history: Sequence[float],
    window: int,
) -> Optional[float]:
    """Return the relative ELBO shift between the two latest rolling windows."""
    if window < 1:
        raise ValueError("convergence_window must be at least 1.")
    if len(loss_history) < 2 * window:
        return None

    previous_window = np.asarray(loss_history[-2 * window:-window], dtype=np.float64)
    current_window = np.asarray(loss_history[-window:], dtype=np.float64)
    previous_mean = float(previous_window.mean())
    current_mean = float(current_window.mean())
    baseline = max(abs(previous_mean), np.finfo(np.float64).eps)
    return abs(current_mean - previous_mean) / baseline


def _logit_clipped(value: float, eps: float = 1e-6) -> float:
    """Return a finite logit after clipping to the open unit interval."""
    clipped = min(max(float(value), eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


def _positive_clipped_log(value: float, eps: float = 1e-6) -> float:
    """Return a finite log after clipping to a positive value."""
    return math.log(max(float(value), eps))


class ExclusionMask:
    """
    Handles genomic exclusion regions for masking unreliable depth signal.

    Uses :class:`intervaltree.IntervalTree` per chromosome for exact
    interval-overlap queries (no midpoint heuristics).

    Exclusion regions can include segmental duplications, centromeres,
    satellite repeats, or any other intervals the user wishes to mask.
    Multiple BED files are accepted; all intervals are concatenated and
    merged *once* before the index is built, ensuring cross-file overlaps
    are collapsed properly.

    Any BED file with at least three columns (chr, start, end) is accepted.
    Additional columns are ignored.
    """

    def __init__(self, filepaths: Union[str, List[str]], label: str = "exclusion regions"):
        """
        Load genomic regions from one or more BED files.

        All files are read and concatenated first, then overlapping
        intervals are merged across the combined set before the lookup
        index is built.  This guarantees that cross-file overlaps are
        collapsed correctly and the index is only constructed once.

        Only the first three columns (chr, start, end) are required.  Extra
        BED columns (name, score, strand, …) are automatically ignored.

        Args:
            filepaths: Path to a BED file (plain or bgzipped .bed.gz),
                or a list of such paths.  All files are merged before
                the index is built.
            label: Human-readable label used in log messages.
        """
        self.label = label
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        dfs = []
        n_input_files = len(filepaths)
        for fp in filepaths:
            df = pd.read_csv(
                fp,
                sep="\t",
                header=None,
                usecols=[0, 1, 2],
                names=["chr", "start", "end"],
                compression="gzip" if fp.endswith(".gz") else None,
            )
            dfs.append(df)
        self.df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
            columns=["chr", "start", "end"]
        )
        self._build_interval_index()
        print(f"Exclusion mask: {len(self.df):,} total {label} regions "
              f"({n_input_files} file(s)) → "
              f"{sum(len(t) for t in self.trees.values())} merged intervals")

    def _build_interval_index(self):
        """Build per-chromosome :class:`IntervalTree` instances.

        Intervals are merged (via ``tree.merge_overlaps()``) so that
        overlap-fraction queries can sum non-overlapping pieces directly.
        """
        self.trees: Dict[str, IntervalTree] = {}
        for chrom, group in self.df.groupby("chr"):
            tree = IntervalTree()
            for s, e in zip(group["start"].values, group["end"].values):
                s, e = int(s), int(e)
                if e > s:
                    tree.addi(s, e)
            tree.merge_overlaps()
            self.trees[chrom] = tree

    def has_any_overlap(self, chrom: str, start: int, end: int) -> bool:
        """Return *True* if ``[start, end)`` overlaps any exclusion interval.

        This is a fast O(log N + k) query with no fraction computation.
        """
        if chrom not in self.trees:
            return False
        return bool(self.trees[chrom].overlap(start, end))

    def get_overlap_fraction(self, chrom: str, start: int, end: int) -> float:
        """
        Calculate the fraction of ``[start, end)`` overlapping exclusion intervals.

        The tree is pre-merged so every hit is non-overlapping; the total
        overlap is the sum of per-interval intersections.

        Args:
            chrom: Chromosome name
            start: Region start position
            end: Region end position

        Returns:
            Fraction of region overlapping with exclusion intervals (0.0 to 1.0)
        """
        region_length = end - start
        if region_length <= 0 or chrom not in self.trees:
            return 0.0
        hits = self.trees[chrom].overlap(start, end)
        if not hits:
            return 0.0
        total = sum(min(iv.end, end) - max(iv.begin, start) for iv in hits)
        return min(total / region_length, 1.0)

    def get_overlap_fractions_batch(
        self, chrom: str, starts: np.ndarray, ends: np.ndarray
    ) -> np.ndarray:
        """
        Compute overlap fractions for multiple bins at once.

        Args:
            chrom: Chromosome name
            starts: Array of bin start positions
            ends: Array of bin end positions

        Returns:
            Array of overlap fractions (0.0 to 1.0) for each bin
        """
        n = len(starts)
        if n == 0 or chrom not in self.trees:
            return np.zeros(n)
        tree = self.trees[chrom]
        starts = np.asarray(starts)
        ends = np.asarray(ends)
        result = np.zeros(n)
        for i in range(n):
            s, e = int(starts[i]), int(ends[i])
            length = e - s
            if length <= 0:
                continue
            hits = tree.overlap(s, e)
            if hits:
                total = sum(min(iv.end, e) - max(iv.begin, s) for iv in hits)
                result[i] = min(total / length, 1.0)
        return result

    def is_masked(self, chrom: str, start: int, end: int,
                  threshold: float = 0.5) -> bool:
        """
        Check if a region should be masked due to exclusion overlap.

        Args:
            chrom: Chromosome name
            start: Region start position
            end: Region end position
            threshold: Minimum overlap fraction to trigger masking

        Returns:
            True if region should be masked
        """
        return self.get_overlap_fraction(chrom, start, end) >= threshold

    def add_beds(self, filepaths: Union[str, List[str]]) -> None:
        """Add intervals from one or more additional BED files.

        All new intervals are appended to the existing set and the
        internal index is rebuilt once so that cross-file overlaps are
        merged before any query is made.

        Args:
            filepaths: A single BED file path or a list of paths.
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        dfs = [self.df]
        n_input_files = len(filepaths)
        for fp in filepaths:
            df = pd.read_csv(
                fp,
                sep="\t",
                header=None,
                usecols=[0, 1, 2],
                names=["chr", "start", "end"],
                compression="gzip" if fp.endswith(".gz") else None,
            )
            dfs.append(df)
        self.df = pd.concat(dfs, ignore_index=True)
        self._build_interval_index()
        print(
            f"Exclusion mask updated from {n_input_files} additional file(s): "
            f"{len(self.df):,} total regions -> "
            f"{sum(len(t) for t in self.trees.values())} merged intervals"
        )


class DepthData:
    """Data container for CNV detection model"""

    def __init__(
        self,
        df: pd.DataFrame,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        subsample_bins: int = None,
        subsample_samples: int = None,
        seed: int = 42,
        clamp_threshold: float = 5.0,
    ):
        """
        Args:
            df: DataFrame with bins as rows and samples as columns
            device: torch device
            dtype: torch data type
            subsample_bins: If specified, randomly subsample this many bins
            subsample_samples: If specified, randomly subsample this many samples
            seed: Random seed for subsampling
            clamp_threshold: Maximum value for depth; values above this are clamped
        """
        # Store original dataframe
        self.original_df = df

        # Get sample columns (exclude metadata)
        sample_cols = get_sample_columns(df)

        # Subsample if requested
        if subsample_bins is not None or subsample_samples is not None:
            np.random.seed(seed)

            # Subsample bins (rows)
            if subsample_bins is not None and subsample_bins < len(df):
                print(f"Subsampling {subsample_bins} bins from {len(df)} total bins")
                bin_indices = np.random.choice(
                    len(df), size=subsample_bins, replace=False
                )
                bin_indices = np.sort(bin_indices)  # Keep sorted for interpretability
                df = df.iloc[bin_indices].copy()

            # Subsample samples (columns)
            if subsample_samples is not None and subsample_samples < len(sample_cols):
                print(
                    f"Subsampling {subsample_samples} samples from {len(sample_cols)} total samples"
                )
                sample_indices = np.random.choice(
                    len(sample_cols), size=subsample_samples, replace=False
                )
                selected_samples = [sample_cols[i] for i in sample_indices]
                sample_cols = selected_samples

        # NOTE: Do NOT sort the DataFrame here.  The caller
        # (collect_all_locus_bins) builds a parallel `mappings` list whose
        # order must stay aligned with the rows of this DataFrame.  Sorting
        # would silently rearrange depth values relative to those mappings,
        # causing every downstream table (cn_posteriors, bin_posteriors, etc.)
        # to attribute values to the wrong genomic bins.

        # Extract metadata
        self.chr = df["Chr"].values
        self.start = df["Start"].values
        self.end = df["End"].values
        self.sample_ids = sample_cols

        # Extract normalized read depth matrix (bins x samples)
        depth_matrix = df[sample_cols].values

        # Clamp values above threshold
        if clamp_threshold is not None:
            n_clamped = np.sum(depth_matrix > clamp_threshold)
            if n_clamped > 0:
                print(f"Clamping {n_clamped} values above threshold {clamp_threshold}")
                depth_matrix = np.clip(depth_matrix, None, clamp_threshold)

        # Convert to torch tensors
        self.depth = torch.tensor(depth_matrix, dtype=dtype, device=device)
        self.n_bins = self.depth.shape[0]
        self.n_samples = self.depth.shape[1]

        # Compute interval sizes (bp) for each bin
        interval_sizes = (self.end - self.start).astype(float)
        self.interval_sizes = torch.tensor(
            interval_sizes, dtype=dtype, device=device
        ).unsqueeze(-1)  # (n_bins, 1) for broadcasting over samples

        print(f"Loaded data: {self.n_bins} bins x {self.n_samples} samples")
        print(f"Depth range: [{self.depth.min():.3f}, {self.depth.max():.3f}]")
        print(f"Depth mean: {self.depth.mean():.3f}, median: {self.depth.median():.3f}")
        print(f"Interval sizes: [{interval_sizes.min():.0f}, {interval_sizes.max():.0f}] bp, "
              f"median={np.median(interval_sizes):.0f} bp")

        # Optional SNP/BAF summaries can be attached later once the
        # preprocess bin mappings are available.
        self.baf_median = None
        self.minor_baf_median = None
        self.baf_variance = None
        self.baf_n_sites = None
        self.baf_effective_variance = None
        self.baf_effective_n_sites = None
        self.has_baf = False
        self.has_baf_effective_count = False
        self.sample_ploidy = None
        self.gc_content = None

    def attach_baf_summary(self, baf_summary_df: pd.DataFrame, mappings) -> None:
        """Attach per-bin, per-sample BAF summaries to this data object.

        The summary table is expected to contain rows keyed by preprocess
        ``array_idx`` and sample identifier, with columns such as
        ``baf_median``, ``minor_baf_median``, ``baf_variance``, and
        ``baf_n_sites``. Optional occupancy-adjusted columns
        ``baf_effective_variance`` and ``baf_effective_n_sites`` are
        attached when present. Only exact sample-id matches are attached.
        """
        if baf_summary_df is None or len(baf_summary_df) == 0:
            return

        required_columns = {
            "array_idx",
            "sample",
            "baf_median",
            "minor_baf_median",
            "baf_variance",
            "baf_n_sites",
        }
        missing_columns = required_columns.difference(baf_summary_df.columns)
        if missing_columns:
            raise ValueError(
                "BAF summary is missing required columns for inference: "
                f"{sorted(missing_columns)}"
            )
        has_effective_variance = "baf_effective_variance" in baf_summary_df.columns
        has_effective_n_sites = "baf_effective_n_sites" in baf_summary_df.columns
        if has_effective_variance != has_effective_n_sites:
            raise ValueError(
                "BAF summary must provide both baf_effective_variance and "
                "baf_effective_n_sites when either effective-count column is present."
            )

        sample_to_idx = {str(sample_id): idx for idx, sample_id in enumerate(self.sample_ids)}
        n_bins = self.n_bins
        n_samples = self.n_samples

        baf_median = np.full((n_bins, n_samples), np.nan, dtype=np.float32)
        minor_baf_median = np.full((n_bins, n_samples), np.nan, dtype=np.float32)
        baf_variance = np.full((n_bins, n_samples), np.nan, dtype=np.float32)
        baf_n_sites = np.zeros((n_bins, n_samples), dtype=np.int32)
        baf_effective_variance = None
        baf_effective_n_sites = None
        if has_effective_variance:
            baf_effective_variance = np.full((n_bins, n_samples), np.nan, dtype=np.float32)
            baf_effective_n_sites = np.zeros((n_bins, n_samples), dtype=np.int32)

        n_attached = 0
        for row in baf_summary_df.itertuples(index=False):
            try:
                bin_idx = int(row.array_idx)
            except (TypeError, ValueError):
                continue
            sample_idx = sample_to_idx.get(str(row.sample))
            if sample_idx is None or not (0 <= bin_idx < n_bins):
                continue

            baf_median[bin_idx, sample_idx] = float(row.baf_median)
            minor_baf_median[bin_idx, sample_idx] = float(row.minor_baf_median)
            baf_variance[bin_idx, sample_idx] = float(row.baf_variance)
            baf_n_sites[bin_idx, sample_idx] = int(row.baf_n_sites)
            if has_effective_variance:
                baf_effective_variance[bin_idx, sample_idx] = float(row.baf_effective_variance)
                baf_effective_n_sites[bin_idx, sample_idx] = int(row.baf_effective_n_sites)
            n_attached += 1

        self.baf_median = torch.tensor(baf_median, dtype=self.depth.dtype, device=self.depth.device)
        self.minor_baf_median = torch.tensor(minor_baf_median, dtype=self.depth.dtype, device=self.depth.device)
        self.baf_variance = torch.tensor(baf_variance, dtype=self.depth.dtype, device=self.depth.device)
        self.baf_n_sites = torch.tensor(baf_n_sites, dtype=torch.int32, device=self.depth.device)
        if has_effective_variance:
            self.baf_effective_variance = torch.tensor(
                baf_effective_variance,
                dtype=self.depth.dtype,
                device=self.depth.device,
            )
            self.baf_effective_n_sites = torch.tensor(
                baf_effective_n_sites,
                dtype=torch.int32,
                device=self.depth.device,
            )
            self.has_baf_effective_count = True
        else:
            self.baf_effective_variance = None
            self.baf_effective_n_sites = None
            self.has_baf_effective_count = False
        self.has_baf = n_attached > 0

        print(f"Attached BAF summaries: {n_attached:,} matched bin × sample rows")

    def attach_sample_ploidy(self, ploidy_map: Dict[Tuple[str, str], int]) -> None:
        """Attach per-bin, per-sample contig ploidy for exact inference."""
        if not ploidy_map:
            return

        ploidy_matrix = np.zeros((self.n_bins, self.n_samples), dtype=np.int16)
        missing_pairs = []
        for bin_idx, chrom in enumerate(self.chr):
            chrom_key = str(chrom)
            for sample_idx, sample_id in enumerate(self.sample_ids):
                key = (str(sample_id), chrom_key)
                ploidy_value = ploidy_map.get(key)
                if ploidy_value is None:
                    if len(missing_pairs) < 5:
                        missing_pairs.append(f"{sample_id}/{chrom_key}")
                    continue
                ploidy_matrix[bin_idx, sample_idx] = int(ploidy_value)

        if missing_pairs:
            raise ValueError(
                "Ploidy map is missing sample/contig pairs required for exact inference: "
                + ", ".join(missing_pairs)
            )

        self.sample_ploidy = torch.tensor(
            ploidy_matrix,
            dtype=torch.int16,
            device=self.depth.device,
        )
        print(
            "Attached sample ploidy for exact inference: "
            f"{self.n_bins:,} bins x {self.n_samples:,} samples"
        )

    def attach_gc_content(self, gc_fraction: np.ndarray) -> None:
        """Attach per-bin GC fraction and mean-center it.

        The GC fraction array is converted to a torch tensor of shape
        ``(n_bins, 1)`` for broadcasting over samples.  Values are
        mean-centered by subtracting the global mean so that the GC
        bias slope acts as a zero-centered perturbation around the
        population average.

        Args:
            gc_fraction: 1-D numpy array of GC fractions in [0, 1],
                one value per bin, aligned to ``self.depth`` rows.
        """
        if gc_fraction is None:
            return

        gc_fraction = np.asarray(gc_fraction, dtype=np.float32)
        if gc_fraction.shape[0] != self.n_bins:
            raise ValueError(
                f"gc_fraction length ({gc_fraction.shape[0]}) does not match "
                f"number of bins ({self.n_bins})"
            )

        # Mean-center so that the bias slope is a zero-centered perturbation
        mean_gc = float(np.mean(gc_fraction))
        centered = gc_fraction - mean_gc
        self.gc_content = torch.tensor(
            centered, dtype=self.depth.dtype, device=self.depth.device
        ).unsqueeze(-1)  # (n_bins, 1)
        print(
            f"Attached GC content: {self.n_bins:,} bins, "
            f"raw mean={mean_gc:.4f}, centered range="
            f"[{self.gc_content.min():.4f}, {self.gc_content.max():.4f}]"
        )


class CNVModel:
    """
    Hierarchical Bayesian model for CNV detection from normalized read depth.

    Model structure:
    - Copy number (CN): discrete latent variable [0,1,2,3,4,5] with prior favoring CN=2
    - Per-bin mean bias: modulates expected depth at each bin
    - Per-sample variance: controls noise in each sample
    - Per-bin variance: controls noise at each bin
    - Observed depth ~ StudentT(df, CN * sample_mean * bin_bias, variance)
    """

    def __init__(
        self,
        n_states: int = 6,
        alpha_ref: float = 50.0,
        alpha_non_ref: float = 1.0,
        baf_temperature: float = 25.0,
        learn_baf_temperature: bool = True,
        baf_temperature_prior_scale: float = 0.5,
        baf_outlier_rate: float = 0.0,
        use_baf_effective_count: bool = True,
        null_state_prior: float = 1e-4,
        var_bias_bin: float = 0.1,
        var_sample: float = 0.2,
        var_bin: float = 0.2,
        freeze_bin_bias: bool = False,
        freeze_bin_var: bool = False,
        freeze_pair_state_priors: bool = False,
        bin_size_factor: float = 10000.0,
        min_variance_expected_depth: float = _DEFAULT_MIN_VARIANCE_EXPECTED_DEPTH,
        sample_raw_count_medians: Optional[Sequence[float]] = None,
        reference_bin_size: Optional[float] = None,
        var_length_scale: float = 20000.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        debug: bool = False,
        guide_type: str = "diagonal",
    ):
        """
        Args:
            n_states: Number of latent states. Default 6 corresponds to the
                canonical unordered diploid pair states over per-haplotype CN
                values [0, 1, 2]: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2).
            alpha_ref: Dirichlet concentration for reference state ((1,1))
            alpha_non_ref: Dirichlet concentration for non-reference states
            baf_temperature: Fixed global multiplicative BAF variance scale
                when ``learn_baf_temperature`` is disabled; otherwise the
                LogNormal prior median for the learned global BAF variance
                scale. Larger values soften BAF evidence without removing it.
                Set to 0 to disable BAF evidence entirely in fixed mode.
            learn_baf_temperature: Whether to learn one global BAF variance
                temperature instead of keeping ``baf_temperature`` fixed.
            baf_temperature_prior_scale: LogNormal prior scale for the learned
                global BAF variance temperature.
            baf_outlier_rate: Mixture weight for a uniform minor-allele BAF
                noise component. Positive values cap the penalty from
                contradictory off-model BAF bins without weakening coherent
                BAF evidence everywhere.
            use_baf_effective_count: When true, use occupancy-adjusted BAF
                effective-count summaries when attached to the data object;
                otherwise fall back to the original raw-site-count path.
            null_state_prior: Prior probability assigned to an outer null
                state during exact discrete inference. The null state carries
                no depth or BAF evidence and contributes neutral 1:1 event
                odds for off-model bins. Set to 0 to disable it.
            var_bias_bin: Variance for per-bin mean bias (log-normal)
            var_sample: Mean of the Exponential prior on per-sample excess
                overdispersion above the count-anchored Poisson baseline.
            var_bin: Deprecated compatibility parameter. The spatial
                count-anchored model does not use per-bin excess variance and
                this value is ignored.
            freeze_bin_bias: If true, fix per-bin mean bias at 1.0 instead
                of inferring it.
            freeze_bin_var: Deprecated compatibility flag retained for API
                stability. Per-bin excess variance is no longer inferred.
            freeze_pair_state_priors: If true, fix per-bin pair-state priors
                to the Dirichlet prior mean implied by ``alpha_ref`` and
                ``alpha_non_ref``.
            bin_size_factor: Reference bin size (bp) for variance scaling.
                The total variance is multiplied by
                ``(bin_size_factor / interval_size) * (expected_depth / 2.0)``
                so that smaller bins and higher expected normalized depth
                both increase variance relative to a diploid depth baseline
                of 2.0.
            min_variance_expected_depth: Lower bound applied to expected
                depth only when assembling variance terms. This keeps copy-0
                states from collapsing to near-zero variance while leaving
                the state mean unchanged.
            sample_raw_count_medians: Optional per-sample autosomal median raw
                counts at the reference low-resolution bin size. Training and
                exact inference require these values together with
                ``reference_bin_size``; callers may omit them only when
                instantiating the model for configuration-only use.
            reference_bin_size: Reference bin size in bp corresponding to the
                supplied ``sample_raw_count_medians``.
            var_length_scale: Mean of the Exponential prior on the shared
                physical correlation length-scale ``length_scale_var`` in bp.
                In the count-anchored path the excess variance is modeled as
                ``d**2 * sample_var * f(L; length_scale_var)`` where
                ``f`` is the continuous-AR(1) spatial aggregation factor.
                This saturates to ``d**2 * sample_var`` for small bins and
                decays approximately as ``d**2 * sample_var * 2 * ell / L``
                for bins much larger than the correlation length.
            device: torch device
            dtype: torch data type
            debug: Whether to print debug statements in model()
            guide_type: Type of variational guide ('diagonal' for AutoDiagonalNormal, 'delta' for AutoDelta)
        """
        self.n_states = n_states
        self.alpha_ref = alpha_ref
        self.alpha_non_ref = alpha_non_ref
        self.baf_temperature = float(baf_temperature)
        self.learn_baf_temperature = learn_baf_temperature
        self.baf_temperature_prior_scale = baf_temperature_prior_scale
        self.baf_outlier_rate = float(baf_outlier_rate)
        self.use_baf_effective_count = bool(use_baf_effective_count)
        self.null_state_prior = float(null_state_prior)
        self.var_bias_bin = var_bias_bin
        self.var_sample = var_sample
        self.var_bin = var_bin
        self.freeze_bin_bias = freeze_bin_bias
        self.freeze_bin_var = freeze_bin_var
        self.freeze_pair_state_priors = freeze_pair_state_priors
        self.bin_size_factor = bin_size_factor
        self.min_variance_expected_depth = float(min_variance_expected_depth)
        self.sample_raw_count_medians = None
        self.reference_bin_size = None
        self.var_length_scale = float(var_length_scale)
        self.learn_length_scale_var = True
        self.device = device
        self.dtype = dtype
        self.debug = debug
        self.guide_type = guide_type
        if self.baf_temperature < 0:
            raise ValueError("baf_temperature must be non-negative.")
        if self.baf_temperature_prior_scale <= 0:
            raise ValueError("baf_temperature_prior_scale must be positive.")
        if not 0.0 <= self.baf_outlier_rate < 1.0:
            raise ValueError("baf_outlier_rate must be in [0, 1).")
        if not 0.0 <= self.null_state_prior < 1.0:
            raise ValueError("null_state_prior must be in [0, 1).")
        if self.min_variance_expected_depth < 0.0:
            raise ValueError("min_variance_expected_depth must be non-negative.")
        if self.var_length_scale <= 0:
            raise ValueError("var_length_scale must be positive.")
        if self.learn_baf_temperature and self.baf_temperature <= 0:
            raise ValueError("learn_baf_temperature requires baf_temperature > 0.")
        self.pair_states = build_diploid_pair_states(max_hap_cn=2)
        if n_states != len(self.pair_states):
            raise ValueError(
                f"n_states={n_states} does not match diploid pair-state count "
                f"{len(self.pair_states)}"
            )
        self.total_cn_by_state = torch.tensor(
            pair_state_total_cn(self.pair_states),
            device=self.device,
            dtype=self.dtype,
        )
        self.minor_baf_by_state = torch.tensor(
            pair_state_minor_baf(self.pair_states),
            device=self.device,
            dtype=self.dtype,
        )
        self.ref_state_idx = self.pair_states.index((1, 1))
        self.max_total_cn = int(max(sum(p) for p in self.pair_states))
        self._zero_t = torch.zeros(1, device=self.device, dtype=self.dtype)
        self._one_t = torch.ones(1, device=self.device, dtype=self.dtype)
        self._sample_var_rate_t = torch.tensor(
            1.0 / self.var_sample,
            device=self.device,
            dtype=self.dtype,
        )
        self._length_scale_var_rate_t = torch.tensor(
            1.0 / self.var_length_scale,
            device=self.device,
            dtype=self.dtype,
        )
        self._count_anchored_reference_variance_t = None
        self._count_anchored_reference_variance_np = None
        if (sample_raw_count_medians is None) != (reference_bin_size is None):
            raise ValueError(
                "sample_raw_count_medians and reference_bin_size must be provided together."
            )
        if sample_raw_count_medians is not None:
            sample_raw_count_medians = np.asarray(sample_raw_count_medians, dtype=np.float64).reshape(-1)
            if np.any(sample_raw_count_medians <= 0):
                raise ValueError("sample_raw_count_medians must all be positive.")
            if float(reference_bin_size) <= 0:
                raise ValueError("reference_bin_size must be positive.")
            self.sample_raw_count_medians = sample_raw_count_medians
            self.reference_bin_size = float(reference_bin_size)
            self._count_anchored_reference_variance_np = _count_anchored_reference_variance_numpy(
                self.sample_raw_count_medians,
                self.reference_bin_size,
                self.bin_size_factor,
            )
            self._count_anchored_reference_variance_t = torch.tensor(
                self._count_anchored_reference_variance_np,
                device=self.device,
                dtype=self.dtype,
            )
        self._baf_temperature_log_t = torch.tensor(
            _positive_clipped_log(self.baf_temperature),
            device=self.device,
            dtype=self.dtype,
        )
        self._baf_temperature_prior_scale_t = torch.tensor(
            self.baf_temperature_prior_scale,
            device=self.device,
            dtype=self.dtype,
        )
        self._var_bias_bin_t = torch.tensor(
            self.var_bias_bin,
            device=self.device,
            dtype=self.dtype,
        )
        self._pair_state_prior_mean_np = self._pair_state_prior_mean_values().astype(np.float64, copy=False)
        self._pair_state_prior_mean_t = torch.tensor(
            self._pair_state_prior_mean_np,
            device=self.device,
            dtype=self.dtype,
        )
        self._alpha_pair_t = torch.full(
            (self.n_states,),
            self.alpha_non_ref,
            device=self.device,
            dtype=self.dtype,
        )
        self._alpha_pair_t[self.ref_state_idx] = self.alpha_ref

        # Training history
        self.loss_history = {"epoch": [], "elbo": []}

        # Define which sites to expose to the guide (continuous latent variables)
        self.latent_sites = []
        if not self.freeze_bin_bias:
            self.latent_sites.append("bin_bias")
        self.latent_sites.append("sample_var")
        self.latent_sites.append("sample_df")
        self.latent_sites.append("length_scale_var")
        self.latent_sites.append("sample_gc_bias")
        if self.learn_baf_temperature:
            self.latent_sites.append("baf_temperature")
        if not self.freeze_pair_state_priors:
            self.latent_sites.append("pair_state_probs")

        # Initialize guide based on type
        self.guide = self._build_guide(guide_type)

    def _select_baf_support(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "use_baf_effective_count", True) and getattr(data, "has_baf_effective_count", False):
            return data.baf_effective_variance, data.baf_effective_n_sites
        return data.baf_variance, data.baf_n_sites

    def _build_guide(self, guide_type: str, init_loc_fn=None, model_fn=None, expose_sites=None):
        target_model = self.model if model_fn is None else model_fn
        target_sites = self.latent_sites if expose_sites is None else expose_sites
        blocked_model = poutine.block(target_model, expose=target_sites)
        if guide_type == "delta":
            return AutoDelta(blocked_model)
        if guide_type == "diagonal":
            if self.learn_baf_temperature and "baf_temperature" in target_sites:
                guide = AutoGuideList(target_model)
                diagonal_sites = [site for site in target_sites if site != "baf_temperature"]
                if diagonal_sites:
                    diagonal_model = poutine.block(target_model, expose=diagonal_sites)
                    if init_loc_fn is None:
                        guide.append(AutoDiagonalNormal(diagonal_model))
                    else:
                        guide.append(AutoDiagonalNormal(diagonal_model, init_loc_fn=init_loc_fn))
                temperature_model = poutine.block(target_model, expose=["baf_temperature"])
                if init_loc_fn is None:
                    guide.append(AutoDelta(temperature_model))
                else:
                    guide.append(AutoDelta(temperature_model, init_loc_fn=init_loc_fn))
                return guide
            if init_loc_fn is None:
                return AutoDiagonalNormal(blocked_model)
            return AutoDiagonalNormal(blocked_model, init_loc_fn=init_loc_fn)
        raise ValueError(
            f"Unknown guide_type: {guide_type}. Choose 'diagonal' or 'delta'."
        )

    def _warmup_model_and_initial_values(self):
        model_fn = self.model
        expose_sites = list(self.latent_sites)
        init_values = {}
        if self.learn_baf_temperature:
            fixed_baf_temperature = self._fixed_baf_temperature_tensor().detach().clone()
            model_fn = poutine.condition(
                self.model,
                data={"baf_temperature": fixed_baf_temperature},
            )
            expose_sites = [site for site in expose_sites if site != "baf_temperature"]
            init_values["baf_temperature"] = fixed_baf_temperature
        return model_fn, expose_sites, init_values

    def _extract_guide_latent_values(self, guide, data) -> Dict[str, torch.Tensor]:
        guide_trace = poutine.trace(guide).get_trace(
            depth=data.depth,
            interval_sizes=data.interval_sizes,
            n_bins=data.n_bins,
            n_samples=data.n_samples,
            gc_content=getattr(data, "gc_content", None),
        )
        return {
            site_name: guide_trace.nodes[site_name]["value"].detach().clone()
            for site_name in self.latent_sites
            if site_name in guide_trace.nodes
        }

    def _run_svi_training(
        self,
        data,
        guide,
        max_iter: int,
        lr_init: float,
        lr_min: float,
        lr_decay: float,
        adam_beta1: float,
        adam_beta2: float,
        log_freq: int,
        jit: bool,
        early_stopping: bool,
        patience: int,
        convergence_window: int,
        convergence_rtol: float,
        progress_desc: str,
        record_history: bool,
        model_fn=None,
    ) -> None:
        scheduler = pyro.optim.LambdaLR(
            {
                "optimizer": torch.optim.Adam,
                "optim_args": {"lr": 1.0, "betas": (adam_beta1, adam_beta2)},
                "lr_lambda": lambda k: (
                    lr_min + (lr_init - lr_min) * np.exp(-k / lr_decay)
                ),
            }
        )

        if jit:
            loss = JitTraceEnum_ELBO()
        else:
            loss = TraceEnum_ELBO()

        svi_model = self.model if model_fn is None else model_fn
        svi = SVI(svi_model, guide, optim=scheduler, loss=loss)

        logger = get_logger("training")
        logger.info(
            "%s started: max_iter=%d early_stopping=%s",
            progress_desc,
            max_iter,
            bool(early_stopping),
        )
        if early_stopping:
            logger.info(
                "Early stopping enabled: patience=%d elbo_window=%d elbo_rtol=%s",
                patience,
                convergence_window,
                convergence_rtol,
            )

        patience_counter = 0
        stopped_early = False
        stop_relative_change = None
        epoch_loss = float("nan")

        for epoch in range(max_iter):
            epoch_loss = svi.step(
                depth=data.depth,
                interval_sizes=data.interval_sizes,
                n_bins=data.n_bins,
                n_samples=data.n_samples,
                gc_content=getattr(data, "gc_content", None),
            )
            scheduler.step()

            if record_history:
                self.loss_history["epoch"].append(epoch)
                self.loss_history["elbo"].append(epoch_loss)

            relative_change = None
            if early_stopping and record_history:
                relative_change = _windowed_relative_elbo_change(
                    self.loss_history["elbo"],
                    convergence_window,
                )

            if log_freq > 0 and (epoch + 1) % log_freq == 0:
                if relative_change is None:
                    logger.info(
                        "%s progress: epoch=%d loss=%.4f",
                        progress_desc,
                        epoch + 1,
                        epoch_loss,
                    )
                else:
                    logger.info(
                        "%s progress: epoch=%d loss=%.4f rel_change=%.2e",
                        progress_desc,
                        epoch + 1,
                        epoch_loss,
                        relative_change,
                    )

            if early_stopping and relative_change is not None:
                if relative_change < convergence_rtol:
                    patience_counter += 1
                else:
                    patience_counter = 0

                if patience_counter >= patience:
                    stopped_early = True
                    stop_relative_change = relative_change
                    break

        if stopped_early:
            logger.info(
                "%s stopped early: epochs=%d rel_change=%.2e final_loss=%.4f",
                progress_desc,
                epoch + 1,
                stop_relative_change,
                epoch_loss,
            )
        else:
            logger.info(
                "%s completed: epochs=%d final_loss=%.4f",
                progress_desc,
                max_iter,
                epoch_loss,
            )

    def _fixed_bin_bias_values(self, n_bins: int) -> np.ndarray:
        return np.ones(n_bins, dtype=np.float32)

    def _fixed_bin_var_values(self, n_bins: int) -> np.ndarray:
        return np.full(n_bins, self.var_bin, dtype=np.float32)

    def _fixed_bin_bias_tensor(self, n_bins: int) -> torch.Tensor:
        return torch.ones((n_bins, 1), device=self.device, dtype=self.dtype)

    def _fixed_bin_var_tensor(self, n_bins: int) -> torch.Tensor:
        return torch.full((n_bins, 1), float(self.var_bin), device=self.device, dtype=self.dtype)

    def _fixed_baf_temperature_values(self) -> np.ndarray:
        return np.asarray(self.baf_temperature, dtype=np.float32)

    def _fixed_baf_temperature_tensor(self) -> torch.Tensor:
        return torch.tensor(float(self.baf_temperature), device=self.device, dtype=self.dtype)

    def _pair_state_alpha_values(self) -> np.ndarray:
        alpha = np.full(self.n_states, self.alpha_non_ref, dtype=np.float32)
        alpha[self.ref_state_idx] = self.alpha_ref
        return alpha

    def _pair_state_prior_mean_values(self) -> np.ndarray:
        if hasattr(self, "_pair_state_prior_mean_np"):
            return np.asarray(self._pair_state_prior_mean_np, dtype=np.float32)
        alpha = self._pair_state_alpha_values()
        return alpha / float(alpha.sum())

    def _reference_state_idx_for_ploidy(self, sample_ploidy: int) -> Optional[int]:
        candidate_indices = [
            idx
            for idx, (h1, h2) in enumerate(self.pair_states)
            if (h1 + h2) == int(sample_ploidy)
        ]
        if not candidate_indices:
            return None
        return min(
            candidate_indices,
            key=lambda idx: (
                abs(self.pair_states[idx][1] - self.pair_states[idx][0]),
                -min(self.pair_states[idx]),
                self.pair_states[idx][0],
                self.pair_states[idx][1],
            ),
        )

    def _pair_state_prior_mean_values_for_ploidy(self, sample_ploidy: int) -> np.ndarray:
        reference_state_idx = self._reference_state_idx_for_ploidy(int(sample_ploidy))
        if reference_state_idx is None:
            return np.asarray(self._pair_state_prior_mean_values(), dtype=np.float64)
        state_count = len(self.pair_states)
        alpha = np.full(state_count, self.alpha_non_ref, dtype=np.float64)
        alpha[reference_state_idx] = self.alpha_ref
        return alpha / float(alpha.sum())

    def _fixed_pair_state_probs_values(self, n_bins: int) -> np.ndarray:
        base_probs = self._pair_state_prior_mean_values()
        return np.broadcast_to(base_probs, (n_bins, self.n_states)).copy()

    def _fixed_pair_state_probs_tensor(self, n_bins: int) -> torch.Tensor:
        if hasattr(self, "_pair_state_prior_mean_t"):
            return self._pair_state_prior_mean_t.view(1, 1, self.n_states).expand(n_bins, 1, self.n_states).clone()
        base_probs = self._pair_state_prior_mean_values()
        repeated_probs = np.broadcast_to(base_probs, (n_bins, 1, self.n_states)).copy()
        return torch.tensor(repeated_probs, device=self.device, dtype=self.dtype)

    def _null_state_prior_value(self) -> float:
        return float(getattr(self, "null_state_prior", 0.0))

    def _effective_pair_state_prior_values(self, pair_state_probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(pair_state_probs, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        row_sums = probs.sum(axis=1, keepdims=True)
        normalized = np.divide(
            probs,
            np.maximum(row_sums, 1e-30),
            out=np.zeros_like(probs, dtype=np.float64),
            where=row_sums > 0,
        )
        return normalized * (1.0 - self._null_state_prior_value())

    def _sample_ploidy_matrix_numpy(self, data) -> Optional[np.ndarray]:
        sample_ploidy = getattr(data, "sample_ploidy", None)
        if sample_ploidy is None:
            return None
        if hasattr(sample_ploidy, "detach"):
            sample_ploidy = sample_ploidy.detach().cpu().numpy()
        sample_ploidy = np.asarray(sample_ploidy, dtype=np.int64)
        if sample_ploidy.ndim == 0:
            sample_ploidy = np.full(
                (data.n_bins, data.n_samples),
                int(sample_ploidy),
                dtype=np.int64,
            )
        elif sample_ploidy.shape == (data.n_samples,):
            sample_ploidy = np.broadcast_to(
                sample_ploidy.reshape(1, -1),
                (data.n_bins, data.n_samples),
            ).copy()
        elif sample_ploidy.shape != (data.n_bins, data.n_samples):
            raise ValueError(
                "sample_ploidy must have shape (n_bins, n_samples), (n_samples,), "
                f"or be scalar; got {sample_ploidy.shape}"
            )
        return sample_ploidy

    def _effective_pair_state_priors_by_ploidy_numpy(
        self,
        pair_state_probs: np.ndarray,
        sample_ploidy: Optional[np.ndarray],
        n_samples: int,
    ) -> np.ndarray:
        base_probs = np.asarray(pair_state_probs, dtype=np.float64)
        if base_probs.ndim == 1:
            base_probs = base_probs.reshape(1, -1)
        state_count = base_probs.shape[1]
        expanded = np.broadcast_to(
            base_probs[:, np.newaxis, :],
            (base_probs.shape[0], n_samples, state_count),
        ).copy()
        if sample_ploidy is None:
            return expanded

        default_prior_mean = np.asarray(self._pair_state_prior_mean_values(), dtype=np.float64)
        for ploidy_value in np.unique(sample_ploidy):
            target_prior_mean = self._pair_state_prior_mean_values_for_ploidy(int(ploidy_value))
            ratio = np.divide(
                target_prior_mean,
                np.maximum(default_prior_mean, 1e-30),
                out=np.ones_like(target_prior_mean, dtype=np.float64),
                where=default_prior_mean > 0,
            )
            mask = sample_ploidy == int(ploidy_value)
            if np.any(mask):
                expanded[mask] *= ratio

        target_mass = base_probs.sum(axis=1)[:, np.newaxis, np.newaxis]
        expanded_sums = expanded.sum(axis=2, keepdims=True)
        return np.divide(
            expanded,
            np.maximum(expanded_sums, 1e-30),
            out=np.zeros_like(expanded),
            where=expanded_sums > 0,
        ) * target_mass

    def _baf_scale_numpy(self, maps: dict, n_samples: int = 0) -> float:
        if "baf_temperature" in maps:
            values = np.asarray(maps["baf_temperature"]).squeeze()
            return float(np.asarray(values, dtype=np.float64).mean())
        return float(self.baf_temperature)

    def _length_scale_var_numpy(self, maps: dict) -> float:
        if "length_scale_var" in maps:
            values = np.asarray(maps["length_scale_var"]).squeeze()
            return float(np.asarray(values, dtype=np.float64).mean())
        return float(getattr(self, "var_length_scale", 1.0))

    def _baf_reference_probs_tensor(self) -> torch.Tensor:
        return self._pair_state_prior_mean_t

    def _baf_reference_probs_numpy(
        self,
        sample_ploidy: Optional[np.ndarray] = None,
        n_bins: Optional[int] = None,
        n_samples: Optional[int] = None,
    ) -> np.ndarray:
        state_count = len(self.pair_states)
        if sample_ploidy is None:
            return self._pair_state_prior_mean_np
        if n_bins is None or n_samples is None:
            raise ValueError("n_bins and n_samples are required when sample_ploidy is provided.")
        reference_probs = np.broadcast_to(
            np.asarray(self._pair_state_prior_mean_values(), dtype=np.float64).reshape(1, 1, state_count),
            (n_bins, n_samples, state_count),
        ).copy()
        for ploidy_value in np.unique(sample_ploidy):
            reference_probs[sample_ploidy == int(ploidy_value)] = self._pair_state_prior_mean_values_for_ploidy(
                int(ploidy_value)
            )
        return np.transpose(reference_probs, (2, 0, 1))

    @config_enumerate(default="parallel")
    def model(self, depth: torch.Tensor, interval_sizes: torch.Tensor, n_bins: int = None, n_samples: int = None, gc_content: torch.Tensor = None):
        """
        Probabilistic model for CNV detection.

        Args:
            depth: Observed normalized read depth (n_bins x n_samples)
            interval_sizes: Bin sizes in bp (n_bins x 1) for variance scaling
        """

        if self.debug:
            print("\n=== MODEL DEBUG ===")
            print(f"depth.shape: {depth.shape}")

        zero_t = self._zero_t
        if self._count_anchored_reference_variance_t is None:
            raise RuntimeError(
                "sample_raw_count_medians and reference_bin_size are required "
                "to train the count-anchored spatial variance model."
            )

        # Plates for bins and samples
        plate_bins = pyro.plate("bins", n_bins, dim=-2, device=self.device)
        plate_samples = pyro.plate("samples", n_samples, dim=-1, device=self.device)

        # Per-sample excess variance or overdispersion factor.
        with plate_samples:
            sample_var = pyro.sample(
                "sample_var", dist.Exponential(self._sample_var_rate_t)
            )
            sample_df = pyro.sample(
                "sample_df", dist.Gamma(torch.tensor(2.0, device=self.device), torch.tensor(0.2, device=self.device))
            )
            # Sample-specific GC bias slope — modulates expected depth via GC content
            sample_gc_bias = pyro.sample(
                "sample_gc_bias", dist.Normal(0.0, 1.0)
            )
        length_scale_var = pyro.sample(
            "length_scale_var",
            dist.Exponential(self._length_scale_var_rate_t),
        )
        if self.learn_baf_temperature:
            baf_temperature = pyro.sample(
                "baf_temperature",
                dist.LogNormal(
                    self._baf_temperature_log_t,
                    self._baf_temperature_prior_scale_t,
                ),
            )
        else:
            baf_temperature = self._fixed_baf_temperature_tensor()
        if self.debug:
            print(f"sample_var.shape: {sample_var.shape}")

        # Per-bin latent variables
        with plate_bins:
            # Per-bin mean bias factor (log-normal prior, centered at 1.0)
            if self.freeze_bin_bias:
                bin_bias = self._fixed_bin_bias_tensor(n_bins)
            else:
                bin_bias = pyro.sample(
                    "bin_bias", dist.LogNormal(zero_t, self._var_bias_bin_t)
                )
            if self.debug:
                print(f"bin_bias.shape: {bin_bias.shape}")

            bin_var = torch.zeros((n_bins, 1), device=self.device, dtype=self.dtype)
            if self.debug:
                print(f"bin_var.shape: {bin_var.shape}")

            # Pair-state prior (Dirichlet-Categorical)
            # Heavily weight the diploid reference state (1,1)
            if self.freeze_pair_state_priors:
                pair_state_probs = self._fixed_pair_state_probs_tensor(n_bins)
            else:
                pair_state_probs = pyro.sample("pair_state_probs", dist.Dirichlet(self._alpha_pair_t))
        if self.debug:
            print(f"pair_state_probs.shape: {pair_state_probs.shape}")

        # Per-bin, per-sample pair state and observation
        with plate_bins, plate_samples:
            # Sample pair state (discrete latent variable)
            # Shape: (n_bins, n_samples)
            pair_state = pyro.sample("pair_state", dist.Categorical(pair_state_probs))
            if self.debug:
                print(f"pair_state.shape: {pair_state.shape}")

            # Expected depth depends on total CN implied by the pair state.
            if self.debug:
                print(f"bin_bias.shape: {bin_bias.shape}")
            expected_total_cn = Vindex(self.total_cn_by_state)[pair_state]
            expected_depth = expected_total_cn * bin_bias
            # GC bias modulation: depth *= exp(gc_content * sample_gc_bias)
            if gc_content is not None:
                gc_mod = torch.exp(gc_content * sample_gc_bias)
                expected_depth = expected_depth * gc_mod
            min_variance_expected_depth = getattr(self, "min_variance_expected_depth", 0.0)
            if self.debug:
                print(f"expected_depth.shape: {expected_depth.shape}")

            # Variance: combination of sample and bin variance,
            # scaled by both bin size and expected normalized depth.
            if self.debug:
                print(f"bin_var).shape: {bin_var.shape}")
            size_modifier = _size_modifier_torch(interval_sizes, self.bin_size_factor)
            variance_expected_depth = _variance_expected_depth_torch(
                expected_depth,
                min_variance_expected_depth,
            )
            linear_depth_modifier = _depth_variance_scale_torch(variance_expected_depth)
            if self._count_anchored_reference_variance_t.shape[-1] != n_samples:
                raise ValueError(
                    "sample_raw_count_medians length does not match the modeled sample count."
                )
            poisson_variance = (
                self._count_anchored_reference_variance_t
                * size_modifier
                * linear_depth_modifier
            )
            spatial_factor = _spatial_aggregate_variance_scale_torch(
                interval_sizes,
                length_scale_var,
            )
            excess_variance = (variance_expected_depth ** 2) * sample_var * spatial_factor
            variance = poisson_variance + excess_variance
            if self.debug:
                print(f"variance.shape: {variance.shape}")
            std = torch.sqrt(variance)
            if self.debug:
                print(f"std.shape: {std.shape}")

            # Observed depth
            if self.debug:
                print(
                    f"About to sample obs with expected_depth.shape={expected_depth.shape}, std.shape={std.shape}, depth.shape={depth.shape}"
                )
            pyro.sample("obs", dist.StudentT(sample_df, expected_depth, std), obs=depth)

            # Optional BAF observation on the minor-allele fraction.
            # The observed variance is estimated upstream from the number of
            # SNP sites per bin, so bins with more sites contribute a tighter
            # likelihood. Missing / unsupported bins are masked out.
            if self.baf_temperature > 0 and hasattr(self, "current_data") and getattr(self.current_data, "has_baf", False):
                baf_obs = self.current_data.minor_baf_median
                baf_var, baf_sites = self._select_baf_support(self.current_data)

                valid_mask = ((torch.isfinite(baf_obs)) &
                              (torch.isfinite(baf_var)) &
                              (baf_sites > 0) &
                              (baf_var > 0))
                safe_baf_var = _safe_scaled_baf_variance_torch(
                    baf_var,
                    valid_mask,
                    baf_temperature,
                )
                baf_std = torch.sqrt(safe_baf_var)
                state_mean = self.minor_baf_by_state.view(self.n_states, 1, 1)
                obs_expanded = baf_obs.unsqueeze(0)
                std_expanded = baf_std.unsqueeze(0)
                safe_baf_obs = torch.where(valid_mask.unsqueeze(0), obs_expanded, state_mean)
                raw_baf_log_lik = dist.Normal(state_mean, std_expanded).log_prob(safe_baf_obs)
                robust_baf_log_lik = _robust_baf_log_likelihood_torch(
                    raw_baf_log_lik,
                    self.baf_outlier_rate,
                )
                robust_baf_log_lik = torch.where(
                    valid_mask.unsqueeze(0),
                    robust_baf_log_lik,
                    torch.zeros_like(robust_baf_log_lik),
                )
                centered_baf_log_lik = _center_state_log_likelihood_table_torch(
                    robust_baf_log_lik,
                    self._baf_reference_probs_tensor(),
                )
                baf_rel_lik = _select_state_log_likelihood_torch(
                    centered_baf_log_lik,
                    pair_state,
                )
                pyro.factor("baf_lik", baf_rel_lik)
        if self.debug:
            print("=== END MODEL DEBUG ===\n")

    def train(
        self,
        data,
        max_iter: int = 1000,
        guide_warmup_iter: int = 250,
        lr_init: float = 0.01,
        lr_min: float = 0.001,
        lr_decay: float = 500,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        log_freq: int = 50,
        jit: bool = False,
        early_stopping: bool = True,
        patience: int = 50,
        convergence_window: int = 50,
        convergence_rtol: float = 1e-3,
    ):
        """
        Train the model using stochastic variational inference (SVI).

        Args:
            data: DepthData object
            max_iter: Maximum number of training iterations
            guide_warmup_iter: AutoDelta MAP warmup iterations before
                switching back to a non-delta guide; set to 0 to disable
            lr_init: Initial learning rate
            lr_min: Minimum learning rate
            lr_decay: Learning rate decay constant
            adam_beta1: Adam optimizer beta1 parameter
            adam_beta2: Adam optimizer beta2 parameter
            log_freq: Frequency of logging (iterations)
            jit: Whether to use JIT compilation
            early_stopping: Whether to use early stopping
            patience: Consecutive window comparisons below the relative
                tolerance before stopping
            convergence_window: Number of iterations per rolling ELBO window
            convergence_rtol: Relative tolerance between successive rolling
                ELBO windows
        """
        print("Initializing training...")
        if early_stopping:
            if patience < 1:
                raise ValueError("patience must be at least 1.")
            if convergence_window < 1:
                raise ValueError("convergence_window must be at least 1.")
            if convergence_rtol < 0:
                raise ValueError("convergence_rtol must be non-negative.")
        if guide_warmup_iter < 0:
            raise ValueError("guide_warmup_iter must be non-negative.")

        self.loss_history = {"epoch": [], "elbo": []}

        try:
            self.current_data = data
            pyro.clear_param_store()

            if self.guide_type != "delta" and guide_warmup_iter > 0:
                print(
                    f"Running AutoDelta MAP warmup for {guide_warmup_iter} iterations "
                    f"before {self.guide_type} guide training..."
                )
                warmup_model, warmup_sites, warmup_values = self._warmup_model_and_initial_values()
                warmup_guide = self._build_guide(
                    "delta",
                    model_fn=warmup_model,
                    expose_sites=warmup_sites,
                )
                self._run_svi_training(
                    data,
                    guide=warmup_guide,
                    model_fn=warmup_model,
                    max_iter=guide_warmup_iter,
                    lr_init=lr_init,
                    lr_min=lr_min,
                    lr_decay=lr_decay,
                    adam_beta1=adam_beta1,
                    adam_beta2=adam_beta2,
                    log_freq=log_freq,
                    jit=jit,
                    early_stopping=False,
                    patience=patience,
                    convergence_window=convergence_window,
                    convergence_rtol=convergence_rtol,
                    progress_desc="MAP warmup",
                    record_history=False,
                )
                warmup_values.update(self._extract_guide_latent_values(warmup_guide, data))
                pyro.clear_param_store()
                self.guide = self._build_guide(
                    self.guide_type,
                    init_loc_fn=init_to_value(values=warmup_values),
                )
            else:
                self.guide = self._build_guide(self.guide_type)

            self._run_svi_training(
                data,
                guide=self.guide,
                max_iter=max_iter,
                lr_init=lr_init,
                lr_min=lr_min,
                lr_decay=lr_decay,
                adam_beta1=adam_beta1,
                adam_beta2=adam_beta2,
                log_freq=log_freq,
                jit=jit,
                early_stopping=early_stopping,
                patience=patience,
                convergence_window=convergence_window,
                convergence_rtol=convergence_rtol,
                progress_desc="Training",
                record_history=True,
            )

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        finally:
            self.current_data = None

    def get_map_estimates(self, data):
        """
        Get MAP (maximum a posteriori) estimates of all latent variables.

        Returns:
            Dictionary with MAP estimates
        """
        print("Computing MAP estimates...")

        # SVI mode: use guide parameters
        # Get guide trace (contains MAP estimates of continuous variables)
        guide_trace = poutine.trace(self.guide).get_trace(
            depth=data.depth,
            interval_sizes=data.interval_sizes,
            n_bins=data.n_bins,
            n_samples=data.n_samples,
            gc_content=getattr(data, "gc_content", None),
        )

        # Get model trace conditioned on guide
        trained_model = poutine.replay(self.model, trace=guide_trace)

        # Get discrete MAP using infer_discrete
        inferred_model = infer_discrete(
            trained_model, temperature=0, first_available_dim=-3
        )
        trace = poutine.trace(inferred_model).get_trace(
            depth=data.depth,
            interval_sizes=data.interval_sizes,
            n_bins=data.n_bins,
            n_samples=data.n_samples,
            gc_content=getattr(data, "gc_content", None),
        )

        # Extract all latent variables
        map_estimates = {}

        # Continuous variables from guide
        if self.freeze_bin_bias:
            map_estimates["bin_bias"] = self._fixed_bin_bias_values(data.n_bins)
        else:
            map_estimates["bin_bias"] = (
                guide_trace.nodes["bin_bias"]["value"].detach().cpu().numpy()
            )
        map_estimates["sample_var"] = (
            guide_trace.nodes["sample_var"]["value"].detach().cpu().numpy()
        )
        map_estimates["sample_df"] = (
            guide_trace.nodes["sample_df"]["value"].detach().cpu().numpy()
        )
        map_estimates["length_scale_var"] = (
            guide_trace.nodes["length_scale_var"]["value"].detach().cpu().numpy()
        )
        map_estimates["sample_gc_bias"] = (
            guide_trace.nodes["sample_gc_bias"]["value"].detach().cpu().numpy()
        )
        if self.learn_baf_temperature:
            map_estimates["baf_temperature"] = (
                guide_trace.nodes["baf_temperature"]["value"].detach().cpu().numpy()
            )
        else:
            map_estimates["baf_temperature"] = self._fixed_baf_temperature_values()
        map_estimates["bin_var"] = np.zeros(data.n_bins, dtype=np.float32)
        if self.freeze_pair_state_priors:
            map_estimates["pair_state_probs"] = self._fixed_pair_state_probs_values(data.n_bins)
        else:
            map_estimates["pair_state_probs"] = (
                guide_trace.nodes["pair_state_probs"]["value"].detach().cpu().numpy()
            )

        # Discrete variable (pair state), plus compatibility total CN.
        pair_state_idx = trace.nodes["pair_state"]["value"].detach().cpu().numpy().squeeze()
        map_estimates["pair_state"] = pair_state_idx
        total_cn_lookup = pair_state_total_cn(self.pair_states)
        map_estimates["cn"] = total_cn_lookup[pair_state_idx]
        map_estimates["pair_state_labels"] = self.pair_states

        pair_state_probs = np.asarray(map_estimates["pair_state_probs"]).squeeze()
        effective_pair_state_probs = self._effective_pair_state_prior_values(pair_state_probs)
        map_estimates["effective_pair_state_probs"] = effective_pair_state_probs.astype(
            np.float32,
            copy=False,
        )
        map_estimates["null_state_prior"] = np.full(
            effective_pair_state_probs.shape[0],
            self._null_state_prior_value(),
            dtype=np.float32,
        )
        cn_probs = np.zeros(
            (effective_pair_state_probs.shape[0], self.max_total_cn + 1),
            dtype=np.float32,
        )
        for pair_idx, total_cn in enumerate(total_cn_lookup):
            cn_probs[:, total_cn] += effective_pair_state_probs[:, pair_idx]
        map_estimates["cn_probs"] = cn_probs

        return map_estimates

    def run_discrete_inference(self, data, **kwargs):
        """
        Compute exact posterior probabilities analytically using Bayes' rule.

        Because the model uses a discrete Categorical hidden state with a
        small state space (K=6) and a Student-T observation likelihood, the
        posterior can be computed exactly once the MAP estimates for the
        continuous latent variables are available.  For each bin *b* and
        sample *s*:

        .. math::

            P(CN_{b,s} = k \\mid x_{b,s})
            = \\frac{P(x_{b,s} \\mid CN_{b,s}=k) \\cdot P(CN_b=k)}
                    {\\sum_{j} P(x_{b,s} \\mid CN_{b,s}=j) \\cdot P(CN_b=j)}

        This replaces the previous Monte Carlo sampling approach
        (``infer_discrete`` with thousands of samples) and is both faster
        and perfectly accurate.

        Args:
            data: DepthData object.
            **kwargs: Accepted for backward compatibility (``n_samples``,
                ``log_freq``) but ignored.

        Returns:
            Dictionary containing pair-state posteriors and total-CN
            marginals.
        """
        print("Computing exact discrete marginal posteriors analytically...")

        # 1. Get MAP estimates of continuous variables
        maps = self.get_map_estimates(data)
        bin_bias = maps["bin_bias"]       # (n_bins,) or (n_bins, 1)
        sample_var = maps["sample_var"]   # (n_samples,) or (1, n_samples)
        sample_df = np.asarray(maps["sample_df"]).squeeze()  # (n_samples,)
        pair_state_probs = maps.get(
            "effective_pair_state_probs",
            self._effective_pair_state_prior_values(maps["pair_state_probs"]),
        )
        null_state_prior = np.asarray(
            maps.get("null_state_prior", self._null_state_prior_value()),
            dtype=np.float64,
        ).squeeze()
        baf_temperature = self._baf_scale_numpy(maps, data.n_samples)
        length_scale_var = self._length_scale_var_numpy(maps)

        # Flatten to 1-D / 2-D where needed (guide may add singleton
        # plate dimensions).
        bin_bias = bin_bias.squeeze()
        sample_var = sample_var.squeeze()
        pair_state_probs = np.asarray(pair_state_probs, dtype=np.float64).squeeze()
        if pair_state_probs.ndim == 1:
            pair_state_probs = pair_state_probs.reshape(1, -1)

        # 2. Prepare data matrices
        obs = data.depth.detach().cpu().numpy()  # (n_bins, n_samples)
        interval_sizes = data.interval_sizes.detach().cpu().numpy().squeeze()  # (n_bins,)
        sample_ploidy = self._sample_ploidy_matrix_numpy(data)
        pair_total_cn = pair_state_total_cn(self.pair_states)
        pair_minor_baf = pair_state_minor_baf(self.pair_states)
        min_variance_expected_depth = getattr(self, "min_variance_expected_depth", 0.0)
        if np.asarray(null_state_prior).ndim == 0:
            null_state_prior = np.full(pair_state_probs.shape[0], float(null_state_prior), dtype=np.float64)
        else:
            null_state_prior = np.asarray(null_state_prior, dtype=np.float64).reshape(-1)
        effective_pair_state_priors = self._effective_pair_state_priors_by_ploidy_numpy(
            pair_state_probs,
            sample_ploidy,
            data.n_samples,
        )

        # 3. Compute state-specific expected depth and depth-aware variance.
        # states: (n_states, 1, 1);  bin_bias: (1, n_bins, 1)
        states_total_cn = pair_total_cn.reshape(-1, 1, 1)
        expected_depth = states_total_cn * bin_bias[np.newaxis, :, np.newaxis]
        # GC bias modulation: expected_depth *= exp(gc_content * sample_gc_bias)
        gc_content = getattr(data, "gc_content", None)
        if gc_content is not None:
            sample_gc_bias = np.asarray(maps["sample_gc_bias"]).squeeze()  # (n_samples,)
            gc_content_np = gc_content.detach().cpu().numpy().squeeze()  # (n_bins,)
            gc_mod = np.exp(gc_content_np[:, np.newaxis] * sample_gc_bias[np.newaxis, :])
            # Broadcast from (n_bins, n_samples) to (n_states, n_bins, n_samples)
            gc_mod = gc_mod[np.newaxis, :, :]
            expected_depth = expected_depth * gc_mod
        size_modifier = _size_modifier_numpy(interval_sizes[:, np.newaxis], self.bin_size_factor)
        variance_expected_depth = _variance_expected_depth_numpy(
            expected_depth,
            min_variance_expected_depth,
        )
        linear_depth_modifier = _depth_variance_scale_numpy(variance_expected_depth)
        count_anchored_reference_variance = getattr(self, "_count_anchored_reference_variance_np", None)
        if count_anchored_reference_variance is None:
            raise RuntimeError(
                "sample_raw_count_medians and reference_bin_size are required "
                "to run exact discrete inference with the count-anchored "
                "spatial variance model."
            )
        if count_anchored_reference_variance.shape[-1] != sample_var.shape[0]:
            raise ValueError(
                "sample_raw_count_medians length does not match the inferred sample count."
            )
        poisson_variance = (
            count_anchored_reference_variance[np.newaxis, :, :]
            * size_modifier[np.newaxis, :, :]
            * linear_depth_modifier
        )
        spatial_factor = _spatial_aggregate_variance_scale_numpy(
            interval_sizes[:, np.newaxis],
            length_scale_var,
        )
        excess_variance = (
            (variance_expected_depth ** 2)
            * sample_var[np.newaxis, :]
            * spatial_factor[np.newaxis, :, :]
        )
        variance = poisson_variance + excess_variance
        std = np.sqrt(variance)

        # Broadcast obs and std to (1, n_bins, n_samples)
        obs_b = obs[np.newaxis, :, :]
        std_b = std

        # Broadcast df to (1, 1, n_samples) to match obs_b and std_b
        df_b = sample_df[np.newaxis, np.newaxis, :]

        # Student-T log-PDF
        log_lik = (
            gammaln((df_b + 1) / 2)
            - gammaln(df_b / 2)
            - 0.5 * np.log(df_b * np.pi * std_b ** 2)
            - ((df_b + 1) / 2) * np.log(1 + ((obs_b - expected_depth) ** 2) / (df_b * std_b ** 2))
        )

        # 4. Add log-prior over pair states.
        log_prior = np.log(
            np.maximum(np.transpose(effective_pair_state_priors, (2, 0, 1)), 1e-10)
        )
        log_unnormalized = log_lik + log_prior

        # 4b. Optional BAF log-likelihood contribution.
        if getattr(data, "has_baf", False) and baf_temperature > 0:
            minor_baf = data.minor_baf_median.detach().cpu().numpy()
            baf_var_t, baf_sites_t = self._select_baf_support(data)
            baf_var = baf_var_t.detach().cpu().numpy()
            baf_sites = baf_sites_t.detach().cpu().numpy()

            valid = ((np.isfinite(minor_baf)) &
                     (np.isfinite(baf_var)) &
                     (baf_sites > 0) &
                     (baf_var > 0))
            if np.any(valid):
                exp_minor_baf = pair_minor_baf.reshape(-1, 1, 1)
                baf_obs = minor_baf[np.newaxis, :, :]
                scaled_baf_var = _clip_baf_variance_numpy(
                    baf_var[np.newaxis, :, :] * baf_temperature
                )
                baf_std = np.sqrt(scaled_baf_var)
                raw_baf_log_lik = -0.5 * np.log(2 * np.pi * baf_std ** 2) - (
                    (baf_obs - exp_minor_baf) ** 2
                ) / (2 * baf_std ** 2)
                robust_baf_log_lik = _robust_baf_log_likelihood_numpy(
                    raw_baf_log_lik,
                    self.baf_outlier_rate,
                )
                robust_baf_log_lik = np.where(valid[np.newaxis, :, :], robust_baf_log_lik, 0.0)
                centered_baf_log_lik = _center_state_log_likelihood_table_numpy(
                    robust_baf_log_lik,
                    self._baf_reference_probs_numpy(
                        sample_ploidy=sample_ploidy,
                        n_bins=pair_state_probs.shape[0],
                        n_samples=data.n_samples,
                    ) if sample_ploidy is not None else self._baf_reference_probs_numpy(),
                )
                log_unnormalized += centered_baf_log_lik

        if np.any(null_state_prior > 0.0):
            null_log_unnormalized = np.log(
                np.maximum(null_state_prior.reshape(1, -1, 1), 1e-10)
            ) + np.zeros((1, pair_state_probs.shape[0], obs.shape[1]), dtype=np.float64)
            log_unnormalized = np.concatenate(
                [log_unnormalized, null_log_unnormalized],
                axis=0,
            )

        # 5. Log-sum-exp softmax across the state dimension (axis 0)
        max_log = np.max(log_unnormalized, axis=0, keepdims=True)
        exp_vals = np.exp(log_unnormalized - max_log)
        posterior = exp_vals / np.sum(exp_vals, axis=0, keepdims=True)

        if np.any(null_state_prior > 0.0):
            null_posterior = np.transpose(posterior[-1], (0, 1)).astype(np.float32, copy=False)
            posterior = posterior[:-1]
        else:
            null_posterior = np.zeros((obs.shape[0], obs.shape[1]), dtype=np.float32)

        # Transpose from (n_states, n_bins, n_samples) → (n_bins, n_samples, n_states)
        pair_posterior = np.transpose(posterior, (1, 2, 0)).astype(np.float32, copy=False)

        cn_posterior = np.zeros(
            (pair_posterior.shape[0], pair_posterior.shape[1], self.max_total_cn + 1),
            dtype=np.float32,
        )
        for pair_idx, total_cn in enumerate(pair_total_cn):
            cn_posterior[:, :, total_cn] += pair_posterior[:, :, pair_idx]

        print("Exact analytical inference complete.")
        return {
            "cn_posterior": cn_posterior,
            "pair_state_posterior": pair_posterior,
            "null_posterior": null_posterior,
            "pair_state_labels": self.pair_states,
        }


