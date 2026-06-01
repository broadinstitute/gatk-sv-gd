"""
GD CNV calling from model posteriors.

Supports two calling modes:
- ``viterbi``: smooth segmentation using transition matrices
- ``posterior-marginal``: direct scoring from pair-state posterior marginals
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gatk_sv_gd import _util
from gatk_sv_gd._util import setup_logging
from gatk_sv_gd.models import GDTable
from gatk_sv_gd.viterbi import (
    load_transition_matrix,
    viterbi_call_gd_cnv,
)
DEFAULT_MIN_POSTERIOR_INTERVAL_CONFIDENCE = 10.
DEFAULT_MIN_FLANK_NON_EVENT_CONFIDENCE = 10.
DEFAULT_POSTERIOR_INTERVAL_BIN_CORRELATION = 0.5
DEFAULT_NULL_ANOMALY_THRESHOLD = 0.2
_EMPTY_INT_ARRAY = np.array([], dtype=int)


def _effective_independent_bin_count(
    n_bins: int,
    neighbor_bin_correlation: float,
) -> float:
    """Return the effective independent bin count for correlated interval bins."""
    if n_bins < 0:
        raise ValueError("n_bins must be non-negative.")
    if not 0.0 <= neighbor_bin_correlation <= 1.0:
        raise ValueError("neighbor_bin_correlation must be in [0, 1].")
    if n_bins == 0:
        return 0.0
    return float(n_bins) / (1.0 + float(n_bins - 1) * neighbor_bin_correlation)


def _aggregate_interval_qual(
    interval_quals: np.ndarray,
    neighbor_bin_correlation: float,
) -> float:
    """Aggregate per-bin QUAL with a conservative effective-bin penalty."""
    interval_quals = np.asarray(interval_quals, dtype=float)
    if interval_quals.size == 0:
        return 0.0
    n_eff = _effective_independent_bin_count(
        interval_quals.size,
        neighbor_bin_correlation,
    )
    return float(interval_quals.mean() * n_eff)


def get_locus_interval_bins(
    bin_mappings_df: pd.DataFrame,
    cluster: str,
) -> Dict[str, List[int]]:
    """Return array_idx values grouped by interval for one locus."""
    locus_bins = bin_mappings_df[bin_mappings_df["cluster"] == cluster]
    interval_bins: Dict[str, List[int]] = {}
    for interval_name, group in locus_bins.groupby("interval"):
        interval_bins[interval_name] = group["array_idx"].tolist()
    return interval_bins


def get_call_confidence(call: dict) -> float:
    """Return the preferred confidence score for a call."""
    confidence = call.get("confidence_score", np.nan)
    if not pd.isna(confidence):
        return float(confidence)
    log_prob_score = call.get("log_prob_score", np.nan)
    if not pd.isna(log_prob_score):
        return float(log_prob_score)
    return float("nan")


def determine_best_breakpoints(
    calls: List[dict],
    calling_mode: str = "viterbi",
    carrier_only: bool = True,
) -> Dict[str, Optional[str]]:
    """Pick the best GD_ID per svtype."""
    best_by_svtype: Dict[str, Optional[str]] = {}
    for svtype in ["DEL", "DUP"]:
        sv_calls = [c for c in calls if c.get("svtype") == svtype]
        if carrier_only:
            sv_calls = [c for c in sv_calls if bool(c.get("is_carrier", False))]

        if not sv_calls:
            best_by_svtype[svtype] = None
            continue

        if calling_mode == "posterior-marginal":
            best = max(
                sv_calls,
                key=lambda c: (
                    get_call_confidence(c),
                    c.get("matched_interval_bp", 0),
                    c.get("end", 0) - c.get("start", 0),
                    str(c.get("GD_ID", "")),
                ),
            )
        else:
            best = max(
                sv_calls,
                key=lambda c: (
                    c.get("matched_interval_bp", 0),
                    c.get("interval_coverage", c.get("reciprocal_overlap", 0.0)),
                    get_call_confidence(c),
                    c.get("end", 0) - c.get("start", 0),
                    str(c.get("GD_ID", "")),
                ),
            )

        best_by_svtype[svtype] = best.get("GD_ID")
    return best_by_svtype


def determine_posterior_carrier_breakpoints(
    calls: List[dict],
    min_interval_confidence: float,
    min_flank_non_event_confidence: float,
) -> Dict[str, Optional[str]]:
    """Pick at most one qualifying posterior-marginal GD_ID per SV type.

    A call qualifies only when each covered interval's correlation-adjusted
    called-state QUAL meets the minimum threshold and each available flank's
    median non-event QUAL meets the minimum threshold. Among qualifying calls,
    choose the largest one.
    """
    selected_by_svtype: Dict[str, Optional[str]] = {}
    for svtype in ["DEL", "DUP"]:
        qualifying_calls: List[dict] = []
        for call in calls:
            if call.get("svtype") != svtype:
                continue
            interval_confidences = call.get("interval_confidences", [])
            flank_non_event_medians = [
                call.get("left_flank_non_event_median", np.nan),
                call.get("right_flank_non_event_median", np.nan),
            ]
            flank_pass = all(
                ((not pd.notna(flank_median)) or (
                    float(flank_median) >= min_flank_non_event_confidence
                ))
                for flank_median in flank_non_event_medians
            )
            if interval_confidences and all(
                float(confidence) >= min_interval_confidence
                for confidence in interval_confidences
            ) and flank_pass:
                qualifying_calls.append(call)

        if not qualifying_calls:
            selected_by_svtype[svtype] = None
            continue

        best_call = max(
            qualifying_calls,
            key=lambda call: (
                call.get("matched_interval_bp", 0),
                call.get("end", 0) - call.get("start", 0),
                call.get("n_bins", 0),
                get_call_confidence(call),
                str(call.get("GD_ID", "")),
            ),
        )
        selected_by_svtype[svtype] = str(best_call.get("GD_ID", ""))
    return selected_by_svtype


def _annotate_best_and_confident_calls(
    calls: List[dict],
    *,
    best_by_svtype: Dict[str, Optional[str]],
    confident_by_svtype: Dict[str, Optional[str]],
) -> None:
    """Mark best-match and confident carrier state on one sample/locus call list."""
    for call in calls:
        svtype = str(call.get("svtype", ""))
        gd_id = str(call.get("GD_ID", ""))
        call["is_best_match"] = gd_id == best_by_svtype.get(svtype)
        call["is_carrier"] = gd_id == confident_by_svtype.get(svtype)


def get_pair_state_columns(
    cn_posteriors_df: pd.DataFrame,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Return pair-state columns and canonicalized labels."""
    pair_cols = [
        column for column in cn_posteriors_df.columns
        if column.startswith("prob_pair_")
    ]
    if not pair_cols:
        raise ValueError(
            "cn_posteriors.tsv.gz is missing pair-state posterior columns "
            "(expected columns like prob_pair_0_1). Re-run infer first."
        )

    canonical_seen: Dict[Tuple[int, int], str] = {}
    canonical_labels: List[Tuple[int, int]] = []
    for column in pair_cols:
        match = re.fullmatch(r"prob_pair_(\d+)_(\d+)", column)
        if match is None:
            raise ValueError(f"Unrecognized pair-state column name: {column}")
        pair = tuple(sorted((int(match.group(1)), int(match.group(2)))))
        if pair in canonical_seen:
            raise ValueError(
                f"Duplicate canonical pair-state labels: {canonical_seen[pair]} and "
                f"{column} both map to {pair}"
            )
        canonical_seen[pair] = column
        canonical_labels.append(pair)

    return pair_cols, canonical_labels


def build_event_pair_mask(
    pair_states: List[Tuple[int, int]],
    svtype: str,
    sample_ploidy: int,
) -> np.ndarray:
    """Return a boolean mask of pair states supporting the event class."""
    if svtype == "DEL":
        return np.array(
            [(h1 + h2) < sample_ploidy for h1, h2 in pair_states],
            dtype=bool,
        )
    if svtype == "DUP":
        return np.array(
            [(h1 + h2) > sample_ploidy for h1, h2 in pair_states],
            dtype=bool,
        )
    raise ValueError(f"Unsupported svtype: {svtype}")


def build_flank_non_event_pair_mask(
    pair_states: List[Tuple[int, int]],
    svtype: str,
    sample_ploidy: int,
) -> np.ndarray:
    """Return a boolean mask of pair states consistent with no event in a flank."""
    if svtype == "DEL":
        return np.array(
            [(h1 + h2) >= sample_ploidy for h1, h2 in pair_states],
            dtype=bool,
        )
    if svtype == "DUP":
        return np.array(
            [(h1 + h2) <= sample_ploidy for h1, h2 in pair_states],
            dtype=bool,
        )
    raise ValueError(f"Unsupported svtype: {svtype}")


def compute_event_marginal_probabilities(
    pair_prob_matrix: np.ndarray,
    pair_states: List[Tuple[int, int]],
    sample_ploidy: int,
    null_probability: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Return per-bin event marginal probabilities for DEL and DUP."""
    pair_prob_matrix = np.asarray(pair_prob_matrix, dtype=float)
    if pair_prob_matrix.ndim == 1:
        pair_prob_matrix = pair_prob_matrix.reshape(1, -1)
    n_bins = pair_prob_matrix.shape[0]
    if null_probability is None:
        neutral_null_probability = np.zeros(n_bins, dtype=float)
    else:
        neutral_null_probability = np.asarray(null_probability, dtype=float).squeeze()
        if neutral_null_probability.ndim == 0:
            neutral_null_probability = np.full(n_bins, float(neutral_null_probability), dtype=float)
        if neutral_null_probability.shape != (n_bins,):
            raise ValueError(
                "null_probability must have shape (n_bins,) or be scalar, "
                f"got {neutral_null_probability.shape}"
            )
        neutral_null_probability = 0.5 * np.clip(neutral_null_probability, 0.0, 1.0)
    event_probs: Dict[str, np.ndarray] = {}
    for svtype in ("DEL", "DUP"):
        event_mask = build_event_pair_mask(pair_states, svtype, sample_ploidy)
        if pair_prob_matrix.size == 0 or not np.any(event_mask):
            pair_event_probability = np.zeros(n_bins, dtype=float)
        else:
            pair_event_probability = pair_prob_matrix[:, event_mask].sum(axis=1)
        event_probs[svtype] = np.clip(
            pair_event_probability + neutral_null_probability,
            0.0,
            1.0,
        )
    return event_probs


def compute_informative_event_support_probabilities(
    pair_prob_matrix: np.ndarray,
    pair_states: List[Tuple[int, int]],
    sample_ploidy: int,
) -> Dict[str, np.ndarray]:
    """Return per-bin event support conditional on informative pair-state mass.

    This keeps null posterior mass neutral for called-state QUAL while still
    allowing the informative pair-state posterior to express DEL/DUP support.
    """
    pair_prob_matrix = np.asarray(pair_prob_matrix, dtype=float)
    if pair_prob_matrix.ndim == 1:
        pair_prob_matrix = pair_prob_matrix.reshape(1, -1)
    n_bins = pair_prob_matrix.shape[0]
    informative_mass = np.clip(pair_prob_matrix.sum(axis=1), 0.0, None)
    event_support_probs: Dict[str, np.ndarray] = {}
    for svtype in ("DEL", "DUP"):
        event_mask = build_event_pair_mask(pair_states, svtype, sample_ploidy)
        if pair_prob_matrix.size == 0 or not np.any(event_mask):
            pair_event_probability = np.zeros(n_bins, dtype=float)
        else:
            pair_event_probability = pair_prob_matrix[:, event_mask].sum(axis=1)
        support_probabilities = np.full(n_bins, 0.5, dtype=float)
        informative = informative_mass > 0.0
        if np.any(informative):
            support_probabilities[informative] = np.clip(
                pair_event_probability[informative] / informative_mass[informative],
                0.0,
                1.0,
            )
        event_support_probs[svtype] = support_probabilities
    return event_support_probs


def _build_posterior_entry_spec(
    locus,
    entry: dict,
    interval_bin_arrays: Dict[str, np.ndarray],
) -> dict:
    """Precompute interval coverage metadata for one posterior-scored GD entry."""
    bp1 = str(entry["BP1"])
    bp2 = str(entry["BP2"])
    covered_tuples = locus.get_intervals_between(bp1, bp2)
    covered_intervals = [name for _, _, name in covered_tuples]
    covered_index_arrays = [
        interval_bin_arrays.get(interval_name, _EMPTY_INT_ARRAY)
        for interval_name in covered_intervals
    ]
    non_empty_index_arrays = [
        interval_indices
        for interval_indices in covered_index_arrays
        if interval_indices.size > 0
    ]
    if non_empty_index_arrays:
        covered_bin_indices = np.concatenate(non_empty_index_arrays)
    else:
        covered_bin_indices = _EMPTY_INT_ARRAY

    covered_bp_total = int(
        sum(max(0, int(end) - int(start)) for start, end, _ in covered_tuples)
    )

    return {
        "entry": entry,
        "GD_ID": entry["GD_ID"],
        "start": int(entry["start_GRCh38"]),
        "end": int(entry["end_GRCh38"]),
        "svtype": str(entry["svtype"]),
        "BP1": bp1,
        "BP2": bp2,
        "is_terminal": locus.is_terminal,
        "covered_intervals": covered_intervals,
        "covered_bin_indices": covered_bin_indices,
        "covered_bp_total": covered_bp_total,
    }


def _compute_interval_confidence_lookup(
    interval_names: List[str],
    interval_bin_arrays: Dict[str, np.ndarray],
    event_probabilities: np.ndarray,
    neighbor_bin_correlation: float = DEFAULT_POSTERIOR_INTERVAL_BIN_CORRELATION,
) -> Dict[str, float]:
    """Return correlation-adjusted called-state QUAL per interval for one event class."""
    interval_confidences: Dict[str, float] = {}
    for interval_name in interval_names:
        interval_bin_indices = interval_bin_arrays.get(interval_name, _EMPTY_INT_ARRAY)
        if interval_bin_indices.size == 0:
            interval_confidences[interval_name] = 0.0
            continue
        interval_quals = _util.posterior_called_state_to_qual(
            event_probabilities[interval_bin_indices],
            True,
        )
        interval_confidences[interval_name] = _aggregate_interval_qual(
            interval_quals,
            neighbor_bin_correlation,
        )
    return interval_confidences


def _compute_flank_confidence_stats(
    interval_bin_arrays: Dict[str, np.ndarray],
    event_probabilities: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return flank non-event median and mean QUAL for one event class."""
    flank_non_event_medians: Dict[str, float] = {}
    flank_confidences: Dict[str, float] = {}
    for flank_name in ("left_flank", "right_flank"):
        flank_bin_indices = interval_bin_arrays.get(flank_name, _EMPTY_INT_ARRAY)
        if flank_bin_indices.size == 0:
            flank_non_event_medians[flank_name] = np.nan
            flank_confidences[flank_name] = np.nan
            continue
        flank_quals = _util.posterior_called_state_to_qual(
            event_probabilities[flank_bin_indices],
            False,
        )
        flank_non_event_medians[flank_name] = float(np.median(flank_quals))
        flank_confidences[flank_name] = float(np.mean(flank_quals))
    return flank_non_event_medians, flank_confidences


def _score_posterior_call_from_event_probabilities(
    locus,
    entry_spec: dict,
    event_probabilities: np.ndarray,
    interval_confidence_lookup: Dict[str, float],
    flank_non_event_medians: Dict[str, float],
    flank_confidences: Dict[str, float],
    sample_ploidy: int,
    raw_interval_confidence_lookup: Optional[Dict[str, float]] = None,
    raw_flank_non_event_medians: Optional[Dict[str, float]] = None,
    raw_flank_confidences: Optional[Dict[str, float]] = None,
    cluster_depth: Optional[np.ndarray] = None,
) -> dict:
    """Score one GD entry from precomputed event probabilities for its locus."""
    interval_confidences = [
        float(interval_confidence_lookup.get(interval_name, 0.0))
        for interval_name in entry_spec["covered_intervals"]
    ]
    raw_interval_confidences = [
        float(raw_interval_confidence_lookup.get(interval_name, 0.0))
        for interval_name in entry_spec["covered_intervals"]
    ] if raw_interval_confidence_lookup is not None else list(interval_confidences)
    covered_bin_indices = entry_spec["covered_bin_indices"]
    if covered_bin_indices.size > 0:
        log_prob_score = float(np.mean(event_probabilities[covered_bin_indices]))
    else:
        log_prob_score = 0.0

    valid_flank_medians = [
        value for value in flank_non_event_medians.values()
        if pd.notna(value)
    ]
    raw_left_flank_non_event_median = (
        raw_flank_non_event_medians.get("left_flank", np.nan)
        if raw_flank_non_event_medians is not None else flank_non_event_medians.get("left_flank", np.nan)
    )
    raw_right_flank_non_event_median = (
        raw_flank_non_event_medians.get("right_flank", np.nan)
        if raw_flank_non_event_medians is not None else flank_non_event_medians.get("right_flank", np.nan)
    )
    raw_valid_flank_medians = [
        value for value in [raw_left_flank_non_event_median, raw_right_flank_non_event_median]
        if pd.notna(value)
    ]

    confidence_components: List[float] = list(interval_confidences)
    for flank_name in ("left_flank", "right_flank"):
        flank_confidence = flank_confidences.get(flank_name, np.nan)
        if pd.notna(flank_confidence):
            confidence_components.append(float(flank_confidence))
    confidence = (
        float(min(confidence_components))
        if confidence_components else 0.0
    )
    raw_confidence_components: List[float] = list(raw_interval_confidences)
    if raw_flank_confidences is None:
        raw_flank_confidences = flank_confidences
    for flank_name in ("left_flank", "right_flank"):
        flank_confidence = raw_flank_confidences.get(flank_name, np.nan)
        if pd.notna(flank_confidence):
            raw_confidence_components.append(float(flank_confidence))
    raw_confidence = (
        float(min(raw_confidence_components))
        if raw_confidence_components else 0.0
    )

    if cluster_depth is not None and covered_bin_indices.size > 0:
        mean_depth = float(np.mean(cluster_depth[covered_bin_indices]))
    else:
        mean_depth = np.nan

    return {
        "GD_ID": entry_spec["GD_ID"],
        "chrom": locus.chrom,
        "start": entry_spec["start"],
        "end": entry_spec["end"],
        "svtype": entry_spec["svtype"],
        "BP1": entry_spec["BP1"],
        "BP2": entry_spec["BP2"],
        "is_terminal": entry_spec["is_terminal"],
        "n_bins": int(covered_bin_indices.size),
        "sample_ploidy": sample_ploidy,
        "haplotype": np.nan,
        "hap_cn_state": np.nan,
        "matched_seg_start": np.nan,
        "matched_seg_end": np.nan,
        "matched_seg_n_bins": 0,
        "matched_interval_bp": entry_spec["covered_bp_total"],
        "interval_coverage": log_prob_score,
        "reciprocal_overlap": log_prob_score,
        "intervals": list(entry_spec["covered_intervals"]),
        "interval_confidences": interval_confidences,
        "raw_interval_confidences": raw_interval_confidences,
        "min_interval_confidence": (
            float(min(interval_confidences)) if interval_confidences else 0.0
        ),
        "raw_min_interval_confidence": (
            float(min(raw_interval_confidences)) if raw_interval_confidences else 0.0
        ),
        "left_flank_non_event_median": flank_non_event_medians.get("left_flank", np.nan),
        "right_flank_non_event_median": flank_non_event_medians.get("right_flank", np.nan),
        "raw_left_flank_non_event_median": raw_left_flank_non_event_median,
        "raw_right_flank_non_event_median": raw_right_flank_non_event_median,
        "min_flank_non_event_confidence": (
            float(min(valid_flank_medians)) if valid_flank_medians else np.nan
        ),
        "raw_min_flank_non_event_confidence": (
            float(min(raw_valid_flank_medians)) if raw_valid_flank_medians else np.nan
        ),
        "log_prob_score": log_prob_score,
        "confidence_score": confidence,
        "raw_confidence_score": raw_confidence,
        "qual_score": confidence,
        "raw_qual_score": raw_confidence,
        "mean_depth": mean_depth,
        "is_carrier": False,
    }


def _build_locus_call_cache(
    cluster_rows: pd.DataFrame,
    locus,
) -> dict:
    """Precompute per-locus structures used repeatedly across samples."""
    cluster_bin_rows = (
        cluster_rows.drop_duplicates(subset=["array_idx"])
        .sort_values("array_idx")
    )
    cluster_bin_indices = cluster_bin_rows["array_idx"].astype(int).to_numpy()
    local_idx_by_global = {
        int(array_idx): local_idx
        for local_idx, array_idx in enumerate(cluster_bin_indices)
    }

    interval_bin_arrays_global: Dict[str, np.ndarray] = {}
    for interval_name, group in cluster_rows.groupby("interval", sort=False):
        interval_bin_arrays_global[interval_name] = group["array_idx"].astype(int).to_numpy()
    breakpoint_masked_bins = interval_bin_arrays_global.pop(
        "breakpoint_ranges",
        _EMPTY_INT_ARRAY,
    )

    interval_bin_arrays_local = {
        interval_name: np.array(
            [local_idx_by_global[int(bin_idx)] for bin_idx in bin_indices],
            dtype=int,
        )
        for interval_name, bin_indices in interval_bin_arrays_global.items()
    }

    all_cluster_bins = sorted(
        set(cluster_rows["array_idx"].astype(int))
        - set(int(bin_idx) for bin_idx in breakpoint_masked_bins)
    )

    posterior_entry_specs = [
        _build_posterior_entry_spec(locus, entry, interval_bin_arrays_local)
        for entry in locus.gd_entries
    ]
    interval_names_for_entries = list(dict.fromkeys(
        interval_name
        for entry_spec in posterior_entry_specs
        for interval_name in entry_spec["covered_intervals"]
    ))

    return {
        "cluster_bin_rows": cluster_bin_rows,
        "cluster_bin_indices": cluster_bin_indices,
        "cluster_starts": cluster_bin_rows["start"].astype(int).to_numpy(),
        "cluster_ends": cluster_bin_rows["end"].astype(int).to_numpy(),
        "interval_bin_arrays_global": interval_bin_arrays_global,
        "interval_bin_arrays_local": interval_bin_arrays_local,
        "breakpoint_masked_bins": breakpoint_masked_bins,
        "all_cluster_bins": all_cluster_bins,
        "posterior_entry_specs": posterior_entry_specs,
        "interval_names_for_entries": interval_names_for_entries,
    }


def _get_covered_bin_indices_for_call(
    call: dict,
    interval_bin_arrays: Dict[str, np.ndarray],
) -> np.ndarray:
    """Return concatenated covered body-bin indices for a call."""
    covered_index_arrays = [
        interval_bin_arrays.get(interval_name, _EMPTY_INT_ARRAY)
        for interval_name in call.get("intervals", [])
    ]
    non_empty_index_arrays = [
        interval_indices
        for interval_indices in covered_index_arrays
        if interval_indices.size > 0
    ]
    if not non_empty_index_arrays:
        return _EMPTY_INT_ARRAY
    return np.concatenate(non_empty_index_arrays)


def _get_mean_depth_for_call(
    call: dict,
    interval_bin_arrays: Dict[str, np.ndarray],
    cluster_depth: np.ndarray,
) -> float:
    """Return mean depth across the bins covered by a call."""
    covered_bin_indices = _get_covered_bin_indices_for_call(call, interval_bin_arrays)
    if covered_bin_indices.size == 0:
        return np.nan
    return float(np.mean(cluster_depth[covered_bin_indices]))


def _get_mean_null_probability_for_call(
    call: dict,
    interval_bin_arrays: Dict[str, np.ndarray],
    cluster_null_probability: np.ndarray,
) -> float:
    """Return mean null posterior across the covered body bins for a call."""
    covered_bin_indices = _get_covered_bin_indices_for_call(call, interval_bin_arrays)
    if covered_bin_indices.size == 0:
        return 0.0
    return float(np.clip(np.mean(cluster_null_probability[covered_bin_indices]), 0.0, 1.0))


def score_call_from_posterior_marginals(
    locus,
    entry: dict,
    sample_pair_probs: np.ndarray,
    pair_states: List[Tuple[int, int]],
    interval_bin_arrays: Dict[str, np.ndarray],
    sample_ploidy: int,
    posterior_interval_bin_correlation: float = DEFAULT_POSTERIOR_INTERVAL_BIN_CORRELATION,
    null_probability: Optional[np.ndarray] = None,
    event_probabilities: Optional[np.ndarray] = None,
    entry_spec: Optional[dict] = None,
    interval_confidence_lookup: Optional[Dict[str, float]] = None,
    flank_non_event_medians: Optional[Dict[str, float]] = None,
    flank_confidences: Optional[Dict[str, float]] = None,
    raw_event_probabilities: Optional[np.ndarray] = None,
    raw_interval_confidence_lookup: Optional[Dict[str, float]] = None,
    raw_flank_non_event_medians: Optional[Dict[str, float]] = None,
    raw_flank_confidences: Optional[Dict[str, float]] = None,
    cluster_depth: Optional[np.ndarray] = None,
) -> dict:
    """Score one GD entry directly from pair-state posterior marginals."""
    if entry_spec is None:
        entry_spec = _build_posterior_entry_spec(locus, entry, interval_bin_arrays)
    if raw_event_probabilities is None:
        raw_event_probabilities = compute_informative_event_support_probabilities(
            sample_pair_probs,
            pair_states,
            sample_ploidy,
        )[entry_spec["svtype"]]
    if (
        interval_confidence_lookup is None
        or flank_non_event_medians is None
        or flank_confidences is None
    ):
        if event_probabilities is None:
            event_probabilities = compute_event_marginal_probabilities(
                sample_pair_probs,
                pair_states,
                sample_ploidy,
                null_probability=null_probability,
            )[entry_spec["svtype"]]
    if event_probabilities is None:
        event_probabilities = compute_event_marginal_probabilities(
            sample_pair_probs,
            pair_states,
            sample_ploidy,
            null_probability=null_probability,
        )[entry_spec["svtype"]]
    if interval_confidence_lookup is None:
        interval_confidence_lookup = _compute_interval_confidence_lookup(
            entry_spec["covered_intervals"],
            interval_bin_arrays,
            event_probabilities,
            neighbor_bin_correlation=posterior_interval_bin_correlation,
        )
    if flank_non_event_medians is None or flank_confidences is None:
        flank_non_event_medians, flank_confidences = _compute_flank_confidence_stats(
            interval_bin_arrays,
            event_probabilities,
        )
    if raw_interval_confidence_lookup is None:
        raw_interval_confidence_lookup = _compute_interval_confidence_lookup(
            entry_spec["covered_intervals"],
            interval_bin_arrays,
            raw_event_probabilities,
            neighbor_bin_correlation=posterior_interval_bin_correlation,
        )
    if raw_flank_non_event_medians is None or raw_flank_confidences is None:
        raw_flank_non_event_medians, raw_flank_confidences = _compute_flank_confidence_stats(
            interval_bin_arrays,
            raw_event_probabilities,
        )
    return _score_posterior_call_from_event_probabilities(
        locus=locus,
        entry_spec=entry_spec,
        event_probabilities=event_probabilities,
        interval_confidence_lookup=interval_confidence_lookup,
        flank_non_event_medians=flank_non_event_medians,
        flank_confidences=flank_confidences,
        sample_ploidy=sample_ploidy,
        raw_interval_confidence_lookup=raw_interval_confidence_lookup,
        raw_flank_non_event_medians=raw_flank_non_event_medians,
        raw_flank_confidences=raw_flank_confidences,
        cluster_depth=cluster_depth,
    )


def call_cnvs_from_posteriors(
    cn_posteriors_df: pd.DataFrame,
    bin_mappings_df: pd.DataFrame,
    gd_table: GDTable,
    transition_matrix: Optional[np.ndarray] = None,
    ploidy_df: Optional[pd.DataFrame] = None,
    verbose: bool = False,
    min_mean_coverage: float = 0.90,
    breakpoint_transition_matrix: Optional[np.ndarray] = None,
    calling_mode: str = "viterbi",
    min_posterior_interval_confidence: float = DEFAULT_MIN_POSTERIOR_INTERVAL_CONFIDENCE,
    min_flank_non_event_confidence: float = DEFAULT_MIN_FLANK_NON_EVENT_CONFIDENCE,
    posterior_interval_bin_correlation: float = DEFAULT_POSTERIOR_INTERVAL_BIN_CORRELATION,
    null_anomaly_threshold: float = DEFAULT_NULL_ANOMALY_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Call GD CNVs from posterior probabilities."""
    if calling_mode not in {"viterbi", "posterior-marginal"}:
        raise ValueError(
            f"Unsupported calling_mode: {calling_mode}. "
            "Expected 'viterbi' or 'posterior-marginal'."
        )
    if not 0.0 <= posterior_interval_bin_correlation <= 1.0:
        raise ValueError("posterior_interval_bin_correlation must be in [0, 1].")
    if not 0.0 <= null_anomaly_threshold <= 1.0:
        raise ValueError("null_anomaly_threshold must be in [0, 1].")
    if calling_mode == "viterbi" and transition_matrix is None:
        raise ValueError("transition_matrix is required for calling_mode='viterbi'")

    print("\n" + "=" * 80)
    print("CALLING CNVs FROM POSTERIORS")
    if calling_mode == "viterbi":
        bp_str = " + breakpoint matrix" if breakpoint_transition_matrix is not None else ""
        print(
            f"  Calling mode: Viterbi segmentation{bp_str}  "
            f"(minimum per-interval coverage={min_mean_coverage:.0%})"
        )
    else:
        print(
            "  Calling mode: posterior-marginal scoring  "
            "(minimum per-interval QUAL="
            f"{min_posterior_interval_confidence:.2f}, "
            "interval bin correlation="
            f"{posterior_interval_bin_correlation:.2f}, "
            "minimum flank non-event QUAL="
            f"{min_flank_non_event_confidence:.2f}, "
            "null anomaly threshold="
            f"{null_anomaly_threshold:.2f})"
        )
    print("=" * 80)

    all_results: List[dict] = []
    all_path_records: List[dict] = []
    all_event_records: List[dict] = []
    sample_ids = cn_posteriors_df["sample"].unique()

    if ploidy_df is None:
        raise ValueError("ploidy_df is required for calling; provide --ploidy-table")
    required_ploidy_columns = {"sample", "contig", "ploidy"}
    missing_ploidy_columns = sorted(required_ploidy_columns.difference(ploidy_df.columns))
    if missing_ploidy_columns:
        raise ValueError(
            "Ploidy table is missing required columns: "
            f"{missing_ploidy_columns}"
        )
    duplicate_ploidy = ploidy_df.duplicated(subset=["sample", "contig"])
    if duplicate_ploidy.any():
        raise ValueError("Ploidy table contains duplicate sample/contig rows")

    ploidy_lookup: Dict[Tuple[str, str], int] = {
        (str(row.sample), str(row.contig)): int(row.ploidy)
        for row in ploidy_df.itertuples(index=False)
    }

    required_ploidy_pairs = {
        (str(sample_id), str(locus.chrom))
        for sample_id in sample_ids
        for locus in gd_table.loci.values()
    }
    missing_ploidy_pairs = sorted(required_ploidy_pairs.difference(ploidy_lookup))
    if missing_ploidy_pairs:
        examples = ", ".join(f"{sample}/{contig}" for sample, contig in missing_ploidy_pairs[:5])
        raise ValueError(
            f"Ploidy table is missing {len(missing_ploidy_pairs)} required "
            f"sample/contig pair(s), for example: {examples}"
        )
    print(f"  Loaded ploidy for {len(ploidy_lookup)} sample/contig pairs")

    n_bins = len(bin_mappings_df)
    n_samples = len(sample_ids)
    expected_rows = n_bins * n_samples
    if len(cn_posteriors_df) != expected_rows:
        raise ValueError(
            f"cn_posteriors has {len(cn_posteriors_df)} rows, expected "
            f"{expected_rows} ({n_bins} bins × {n_samples} samples)"
        )
    if bin_mappings_df["array_idx"].duplicated().any():
        raise ValueError("bin_mappings contains duplicate array_idx values")

    map_chroms = bin_mappings_df["chr"].to_numpy()
    map_starts = bin_mappings_df["start"].to_numpy()
    map_ends = bin_mappings_df["end"].to_numpy()
    sample_row_indices = cn_posteriors_df.groupby("sample", sort=False).indices
    post_chroms = cn_posteriors_df["chr"].to_numpy()
    post_starts = cn_posteriors_df["start"].to_numpy()
    post_ends = cn_posteriors_df["end"].to_numpy()
    for sample_id in sample_ids:
        sample_indices = np.asarray(sample_row_indices[str(sample_id)], dtype=int)
        if sample_indices.size != n_bins:
            raise ValueError(
                f"cn_posteriors has {sample_indices.size} rows for sample {sample_id}, "
                f"expected {n_bins}"
            )
        if not (
            np.array_equal(post_chroms[sample_indices], map_chroms)
            and np.array_equal(post_starts[sample_indices], map_starts)
            and np.array_equal(post_ends[sample_indices], map_ends)
        ):
            raise ValueError(
                "Bin coordinates in cn_posteriors do not match bin_mappings. "
                "Please re-run infer to regenerate both files."
            )
    print(f"  Validated: bin coordinates match between posteriors and mappings ({n_bins} bins)")

    print("  Organizing data for fast access...")
    pair_prob_cols, pair_state_labels = get_pair_state_columns(cn_posteriors_df)
    n_pair_states = len(pair_prob_cols)
    pair_prob_3d = np.empty((n_samples, n_bins, n_pair_states))
    depth_2d = np.empty((n_samples, n_bins))
    null_prob_2d = np.zeros((n_samples, n_bins), dtype=float)
    pair_prob_values = cn_posteriors_df[pair_prob_cols].to_numpy()
    depth_values = cn_posteriors_df["depth"].to_numpy()
    null_prob_values = None
    if "prob_null" in cn_posteriors_df.columns:
        null_prob_values = cn_posteriors_df["prob_null"].to_numpy(dtype=float)
    for s_idx, sample_id in enumerate(sample_ids):
        sample_indices = np.asarray(sample_row_indices[str(sample_id)], dtype=int)
        pair_prob_3d[s_idx] = pair_prob_values[sample_indices]
        depth_2d[s_idx] = depth_values[sample_indices]
        if null_prob_values is not None:
            null_prob_2d[s_idx] = null_prob_values[sample_indices]
    print(
        f"    Extracted {n_samples} x {n_bins} x {n_pair_states} "
        "pair-state probability array"
    )

    bin_coords_by_idx: Dict[int, Tuple[int, int]] = dict(
        zip(
            bin_mappings_df["array_idx"].astype(int),
            zip(
                bin_mappings_df["start"].astype(int),
                bin_mappings_df["end"].astype(int),
            ),
        )
    )
    cluster_rows_by_cluster = {
        str(cluster): group
        for cluster, group in bin_mappings_df.groupby("cluster", sort=False)
    }
    empty_cluster_rows = bin_mappings_df.iloc[0:0]
    locus_call_caches = {
        cluster: _build_locus_call_cache(
            cluster_rows_by_cluster.get(cluster, empty_cluster_rows),
            locus,
        )
        for cluster, locus in gd_table.loci.items()
    }

    processed_loci = 0
    skipped_loci = 0
    breakpoint_masked_bins = 0
    missing_flank_sets = 0
    interval_sets = 0
    for cluster, locus in gd_table.loci.items():
        processed_loci += 1
        locus_cache = locus_call_caches[cluster]
        cluster_bin_indices = locus_cache["cluster_bin_indices"]
        interval_bin_arrays = locus_cache["interval_bin_arrays_global"]
        interval_bin_arrays_local = locus_cache["interval_bin_arrays_local"]
        breakpoint_masked_bins += int(locus_cache["breakpoint_masked_bins"].size)

        if not interval_bin_arrays:
            skipped_loci += 1
            continue

        interval_sets += len(interval_bin_arrays)

        for flank_name in ("left_flank", "right_flank"):
            flank_bin_indices = interval_bin_arrays_local.get(flank_name, _EMPTY_INT_ARRAY)
            if flank_bin_indices.size == 0:
                missing_flank_sets += 1

        for s_idx, sample_id in enumerate(sample_ids):
            sample_ploidy = ploidy_lookup[(str(sample_id), locus.chrom)]

            cluster_pair_probs = pair_prob_3d[s_idx, cluster_bin_indices, :]
            cluster_null_probs = null_prob_2d[s_idx, cluster_bin_indices]
            cluster_depth = depth_2d[s_idx, cluster_bin_indices]
            cluster_event_probs = compute_event_marginal_probabilities(
                cluster_pair_probs,
                pair_state_labels,
                sample_ploidy,
                null_probability=cluster_null_probs,
            )
            cluster_event_support_probs = compute_informative_event_support_probabilities(
                cluster_pair_probs,
                pair_state_labels,
                sample_ploidy,
            )
            qual_del_event = _util.posterior_probability_to_qual(
                cluster_event_probs["DEL"]
            )
            qual_dup_event = _util.posterior_probability_to_qual(
                cluster_event_probs["DUP"]
            )
            raw_qual_del_event = _util.posterior_probability_to_qual(
                cluster_event_support_probs["DEL"]
            )
            raw_qual_dup_event = _util.posterior_probability_to_qual(
                cluster_event_support_probs["DUP"]
            )
            all_event_records.extend(
                {
                    "sample": sample_id,
                    "cluster": cluster,
                    "chrom": locus.chrom,
                    "start": int(start),
                    "end": int(end),
                    "prob_null": float(null_prob),
                    "prob_del_event": float(del_prob),
                    "prob_dup_event": float(dup_prob),
                    "qual_del_event": float(del_qual),
                    "qual_dup_event": float(dup_qual),
                    "raw_qual_del_event": float(raw_del_qual),
                    "raw_qual_dup_event": float(raw_dup_qual),
                }
                for start, end, null_prob, del_prob, dup_prob, del_qual, dup_qual, raw_del_qual, raw_dup_qual in zip(
                    locus_cache["cluster_starts"],
                    locus_cache["cluster_ends"],
                    cluster_null_probs,
                    cluster_event_probs["DEL"],
                    cluster_event_probs["DUP"],
                    qual_del_event,
                    qual_dup_event,
                    raw_qual_del_event,
                    raw_qual_dup_event,
                )
            )

            if calling_mode == "viterbi":
                calls, path_records = viterbi_call_gd_cnv(
                    locus,
                    pair_prob_3d[s_idx],
                    pair_state_labels,
                    transition_matrix,
                    interval_bin_arrays,
                    ploidy=sample_ploidy,
                    min_mean_coverage=min_mean_coverage,
                    verbose=False,
                    sample_id=str(sample_id),
                    breakpoint_transition_matrix=breakpoint_transition_matrix,
                    bin_coords=bin_coords_by_idx,
                    all_cluster_bins=locus_cache["all_cluster_bins"],
                )
                for start, end, cn_state, category, haplotype in path_records:
                    all_path_records.append(
                        {
                            "sample": sample_id,
                            "cluster": cluster,
                            "start": start,
                            "end": end,
                            "cn_state": cn_state,
                            "category": category,
                            "haplotype": haplotype,
                        }
                    )
                confident_by_svtype = determine_best_breakpoints(
                    calls,
                    calling_mode="viterbi",
                    carrier_only=True,
                )
                best_by_svtype = determine_best_breakpoints(
                    calls,
                    calling_mode="viterbi",
                    carrier_only=False,
                )
                _annotate_best_and_confident_calls(
                    calls,
                    best_by_svtype=best_by_svtype,
                    confident_by_svtype=confident_by_svtype,
                )
            else:
                interval_confidence_lookup_by_svtype = {
                    svtype: _compute_interval_confidence_lookup(
                        locus_cache["interval_names_for_entries"],
                        interval_bin_arrays_local,
                        cluster_event_probs[svtype],
                        neighbor_bin_correlation=posterior_interval_bin_correlation,
                    )
                    for svtype in ("DEL", "DUP")
                }
                flank_stats_by_svtype = {
                    svtype: _compute_flank_confidence_stats(
                        interval_bin_arrays_local,
                        cluster_event_probs[svtype],
                    )
                    for svtype in ("DEL", "DUP")
                }
                raw_interval_confidence_lookup_by_svtype = {
                    svtype: _compute_interval_confidence_lookup(
                        locus_cache["interval_names_for_entries"],
                        interval_bin_arrays_local,
                        cluster_event_support_probs[svtype],
                        neighbor_bin_correlation=posterior_interval_bin_correlation,
                    )
                    for svtype in ("DEL", "DUP")
                }
                raw_flank_stats_by_svtype = {
                    svtype: _compute_flank_confidence_stats(
                        interval_bin_arrays_local,
                        cluster_event_support_probs[svtype],
                    )
                    for svtype in ("DEL", "DUP")
                }
                calls = [
                    score_call_from_posterior_marginals(
                        locus=locus,
                        entry=entry_spec["entry"],
                        sample_pair_probs=pair_prob_3d[s_idx],
                        pair_states=pair_state_labels,
                        interval_bin_arrays=interval_bin_arrays_local,
                        sample_ploidy=sample_ploidy,
                        posterior_interval_bin_correlation=posterior_interval_bin_correlation,
                        event_probabilities=cluster_event_probs[entry_spec["svtype"]],
                        raw_event_probabilities=cluster_event_support_probs[entry_spec["svtype"]],
                        entry_spec=entry_spec,
                        interval_confidence_lookup=interval_confidence_lookup_by_svtype[
                            entry_spec["svtype"]
                        ],
                        flank_non_event_medians=flank_stats_by_svtype[entry_spec["svtype"]][0],
                        flank_confidences=flank_stats_by_svtype[entry_spec["svtype"]][1],
                        raw_interval_confidence_lookup=raw_interval_confidence_lookup_by_svtype[
                            entry_spec["svtype"]
                        ],
                        raw_flank_non_event_medians=raw_flank_stats_by_svtype[entry_spec["svtype"]][0],
                        raw_flank_confidences=raw_flank_stats_by_svtype[entry_spec["svtype"]][1],
                        cluster_depth=cluster_depth,
                    )
                    for entry_spec in locus_cache["posterior_entry_specs"]
                ]
                confident_by_svtype = determine_posterior_carrier_breakpoints(
                    calls,
                    min_interval_confidence=min_posterior_interval_confidence,
                    min_flank_non_event_confidence=min_flank_non_event_confidence,
                )
                best_by_svtype = determine_best_breakpoints(
                    calls,
                    calling_mode="posterior-marginal",
                    carrier_only=False,
                )
                _annotate_best_and_confident_calls(
                    calls,
                    best_by_svtype=best_by_svtype,
                    confident_by_svtype=confident_by_svtype,
                )

            for call in calls:
                mean_depth = call.get("mean_depth", np.nan)
                if pd.isna(mean_depth):
                    mean_depth = _get_mean_depth_for_call(
                        call,
                        interval_bin_arrays_local,
                        cluster_depth,
                    )
                null_anomaly_score = _get_mean_null_probability_for_call(
                    call,
                    interval_bin_arrays_local,
                    cluster_null_probs,
                )
                confidence_score = get_call_confidence(call)
                result = {
                    "sample": sample_id,
                    "cluster": cluster,
                    "GD_ID": call["GD_ID"],
                    "chrom": call["chrom"],
                    "start": call["start"],
                    "end": call["end"],
                        "svtype": call["svtype"],
                    "BP1": call["BP1"],
                    "BP2": call["BP2"],
                    "is_terminal": call["is_terminal"],
                    "n_bins": call["n_bins"],
                    "mean_depth": mean_depth,
                    "sample_ploidy": call.get("sample_ploidy", sample_ploidy),
                    "matched_haplotype": call.get("haplotype", np.nan),
                    "hap_cn_state": call.get("hap_cn_state", np.nan),
                    "matched_seg_start": call.get("matched_seg_start", np.nan),
                    "matched_seg_end": call.get("matched_seg_end", np.nan),
                    "matched_seg_n_bins": call.get("matched_seg_n_bins", 0),
                    "matched_interval_bp": call.get("matched_interval_bp", 0),
                    "interval_coverage": call.get("interval_coverage", np.nan),
                    "reciprocal_overlap": call.get("reciprocal_overlap", np.nan),
                    "min_interval_confidence": call.get("min_interval_confidence", np.nan),
                    "raw_min_interval_confidence": call.get("raw_min_interval_confidence", np.nan),
                    "left_flank_non_event_median": call.get(
                        "left_flank_non_event_median",
                        np.nan,
                    ),
                    "raw_left_flank_non_event_median": call.get(
                        "raw_left_flank_non_event_median",
                        np.nan,
                    ),
                    "right_flank_non_event_median": call.get(
                        "right_flank_non_event_median",
                        np.nan,
                    ),
                    "raw_right_flank_non_event_median": call.get(
                        "raw_right_flank_non_event_median",
                        np.nan,
                    ),
                    "min_flank_non_event_confidence": call.get(
                        "min_flank_non_event_confidence",
                        np.nan,
                    ),
                    "raw_min_flank_non_event_confidence": call.get(
                        "raw_min_flank_non_event_confidence",
                        np.nan,
                    ),
                    "is_carrier": bool(call.get("is_carrier", False)),
                    "is_best_match": bool(call.get("is_best_match", False)),
                    "log_prob_score": call.get("log_prob_score", confidence_score),
                    "confidence_score": confidence_score,
                    "raw_confidence_score": call.get("raw_confidence_score", np.nan),
                    "qual_score": call.get("qual_score", np.nan),
                    "raw_qual_score": call.get("raw_qual_score", np.nan),
                    "null_anomaly_score": null_anomaly_score,
                    "is_null_anomalous": bool(null_anomaly_score > null_anomaly_threshold),
                    "calling_method": calling_mode,
                    "call_criteria_mean_coverage": (
                        float(min_mean_coverage)
                        if calling_mode == "viterbi"
                        else np.nan
                    ),
                    "call_criteria_interval_confidence": (
                        float(min_posterior_interval_confidence)
                        if calling_mode == "posterior-marginal"
                        else np.nan
                    ),
                    "call_criteria_flank_non_event_confidence": (
                        float(min_flank_non_event_confidence)
                        if calling_mode == "posterior-marginal"
                        else np.nan
                    ),
                    "call_criteria_null_anomaly_score": float(null_anomaly_threshold),
                }
                all_results.append(result)

    print(
        "  Calling summary: "
        f"loci={processed_loci}, skipped_no_bins={skipped_loci}, "
        f"interval_sets={interval_sets}, breakpoint_bins_masked={breakpoint_masked_bins}, "
        f"missing_flank_sets={missing_flank_sets}"
    )

    calls_df = pd.DataFrame(
        all_results,
        columns=[
            "sample",
            "cluster",
            "GD_ID",
            "chrom",
            "start",
            "end",
            "svtype",
            "BP1",
            "BP2",
            "is_terminal",
            "n_bins",
            "mean_depth",
            "sample_ploidy",
            "matched_haplotype",
            "hap_cn_state",
            "matched_seg_start",
            "matched_seg_end",
            "matched_seg_n_bins",
            "matched_interval_bp",
            "interval_coverage",
            "reciprocal_overlap",
            "min_interval_confidence",
            "raw_min_interval_confidence",
            "left_flank_non_event_median",
            "raw_left_flank_non_event_median",
            "right_flank_non_event_median",
            "raw_right_flank_non_event_median",
            "min_flank_non_event_confidence",
            "raw_min_flank_non_event_confidence",
            "is_carrier",
            "is_best_match",
            "log_prob_score",
            "confidence_score",
            "raw_confidence_score",
            "qual_score",
            "raw_qual_score",
            "null_anomaly_score",
            "is_null_anomalous",
            "calling_method",
            "call_criteria_mean_coverage",
            "call_criteria_interval_confidence",
            "call_criteria_flank_non_event_confidence",
            "call_criteria_null_anomaly_score",
        ],
    )
    paths_df = pd.DataFrame(
        all_path_records,
        columns=[
            "sample",
            "cluster",
            "start",
            "end",
            "cn_state",
            "category",
            "haplotype",
        ],
    )
    event_marginals_df = pd.DataFrame(
        all_event_records,
        columns=[
            "sample",
            "cluster",
            "chrom",
            "start",
            "end",
            "prob_null",
            "prob_del_event",
            "prob_dup_event",
            "qual_del_event",
            "qual_dup_event",
            "raw_qual_del_event",
            "raw_qual_dup_event",
        ],
    )
    return calls_df, paths_df, event_marginals_df


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Call GD CNVs from model posterior probabilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cn-posteriors",
        required=True,
        help="CN posteriors file (cn_posteriors.tsv.gz) with depth values",
    )
    parser.add_argument(
        "--bin-mappings",
        required=True,
        help="Bin mappings file (bin_mappings.tsv.gz) from gd_cnv_pyro.py",
    )
    parser.add_argument(
        "--gd-table", "-g",
        required=True,
        help="GD locus definition table (TSV)",
    )
    parser.add_argument(
        "--ploidy-table",
           required=True,
        help="Ploidy estimates TSV (ploidy_estimates.tsv) from gd_cnv_pyro.py. "
               "Columns: sample, contig, median_depth, ploidy.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Output directory for calls",
    )
    parser.add_argument(
        "--transition-matrix",
        required=False,
        help="CN-state transition probability matrix (TSV) for Viterbi calling only. "
             "Required when --calling-mode=viterbi.",
    )
    parser.add_argument(
        "--breakpoint-transition-matrix",
        required=False,
        help="CN-state transition probability matrix (TSV) applied at known "
             "breakpoint boundaries during Viterbi calling only.",
    )
    parser.add_argument(
        "--min-mean-coverage",
        type=float,
        default=0.50,
        help="Minimum per-interval coverage required for a breakpoint pair "
             "to participate in the selected contiguous run in Viterbi mode.",
    )
    parser.add_argument(
        "--calling-mode",
        choices=["viterbi", "posterior-marginal"],
        default="posterior-marginal",
        help="Calling strategy to use.",
    )
    parser.add_argument(
        "--min-posterior-interval-confidence",
        type=float,
           default=DEFAULT_MIN_POSTERIOR_INTERVAL_CONFIDENCE,
           help="In posterior-marginal mode, mark a breakpoint combination as a "
               "carrier only when each covered interval's correlation-adjusted "
               "called-state QUAL meets or exceeds this threshold.",
    )
    parser.add_argument(
        "--posterior-interval-bin-correlation",
        type=float,
        default=DEFAULT_POSTERIOR_INTERVAL_BIN_CORRELATION,
        help="In posterior-marginal mode, aggregate per-bin interval QUAL using "
             "N_eff = N / (1 + (N - 1) * rho), where rho is this neighboring-bin "
             "correlation coefficient. 0 treats bins as independent; 1 reduces to "
             "the per-bin mean.",
    )
    parser.add_argument(
        "--min-flank-non-event-confidence",
        type=float,
           default=DEFAULT_MIN_FLANK_NON_EVENT_CONFIDENCE,
        help="In posterior-marginal mode, reject a breakpoint combination when any "
               "available flank has median non-event QUAL below this threshold on the "
               "0-99 scale.",
    )
    parser.add_argument(
        "--null-anomaly-threshold",
        type=float,
        default=DEFAULT_NULL_ANOMALY_THRESHOLD,
        help="Flag calls whose covered-body mean null posterior exceeds this threshold.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed per-sample scores for all GD entries at every locus.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.posterior_interval_bin_correlation <= 1.0:
        parser.error("--posterior-interval-bin-correlation must be in [0, 1].")
    if not 0.0 <= args.null_anomaly_threshold <= 1.0:
        parser.error("--null-anomaly-threshold must be in [0, 1].")
    return args


def main():
    """Main function."""
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(
        args.output_dir,
        filename="call_log.txt",
        verbose=args.verbose,
        command="call",
        args=args,
    )

    print("Output directory configured")
    print(f"Calling mode: {args.calling_mode}")

    print("\nLoading data...")

    print("  Loading CN posteriors")
    cn_posteriors_df = pd.read_csv(args.cn_posteriors, sep="\t", compression="infer")
    print(f"    {len(cn_posteriors_df)} bin-sample records")

    print("  Loading bin mappings")
    bin_mappings_df = pd.read_csv(args.bin_mappings, sep="\t", compression="infer")
    print(f"    {len(bin_mappings_df)} bin mappings")

    print("  Loading GD table")
    gd_table = GDTable(args.gd_table)
    print(f"    {len(gd_table.loci)} loci")

    print("  Loading ploidy table")
    ploidy_df = pd.read_csv(args.ploidy_table, sep="\t")
    print(f"    {len(ploidy_df)} sample/contig ploidy records")

    transition_matrix = None
    breakpoint_transition_matrix = None
    if args.calling_mode == "viterbi":
        if not args.transition_matrix:
            raise ValueError("--transition-matrix is required when --calling-mode=viterbi")
        print("\n  Loading transition matrix")
        transition_matrix = load_transition_matrix(args.transition_matrix)

        if args.breakpoint_transition_matrix:
            print("  Loading breakpoint transition matrix")
            breakpoint_transition_matrix = load_transition_matrix(
                args.breakpoint_transition_matrix
            )

    calls_df, paths_df, event_marginals_df = call_cnvs_from_posteriors(
        cn_posteriors_df,
        bin_mappings_df,
        gd_table,
        transition_matrix=transition_matrix,
        ploidy_df=ploidy_df,
        verbose=args.verbose,
        min_mean_coverage=args.min_mean_coverage,
        breakpoint_transition_matrix=breakpoint_transition_matrix,
        calling_mode=args.calling_mode,
        min_posterior_interval_confidence=args.min_posterior_interval_confidence,
        min_flank_non_event_confidence=args.min_flank_non_event_confidence,
        posterior_interval_bin_correlation=args.posterior_interval_bin_correlation,
        null_anomaly_threshold=args.null_anomaly_threshold,
    )

    output_file = os.path.join(args.output_dir, "gd_cnv_calls.tsv.gz")
    calls_df.to_csv(output_file, sep="\t", index=False, compression="gzip")
    print("\n  Saved calls table")
    print(f"    {len(calls_df)} call records")

    paths_file = os.path.join(args.output_dir, "viterbi_paths.tsv.gz")
    paths_df.to_csv(paths_file, sep="\t", index=False, compression="gzip")
    print("  Saved Viterbi paths table")
    print(f"    {len(paths_df)} path records")

    event_marginals_file = os.path.join(args.output_dir, "event_marginals.tsv.gz")
    event_marginals_df.to_csv(
        event_marginals_file,
        sep="\t",
        index=False,
        compression="gzip",
    )
    print("  Saved event marginals table")
    print(f"    {len(event_marginals_df)} bin-sample records")

    if len(calls_df) > 0:
        carriers = calls_df[calls_df["is_carrier"]]
        n_carriers = carriers["sample"].nunique()
        n_sites = carriers["GD_ID"].nunique()
        print(f"\n  {n_carriers} carrier samples across {n_sites} GD sites")
    else:
        print(
            "\n  No calls produced — check that the GD table, bin mappings, "
            "and CN posteriors refer to the same loci."
        )

    print("\n" + "=" * 80)
    print("Calling complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
