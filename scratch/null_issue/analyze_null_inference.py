from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gatk_sv_gd.depth import (
    _count_anchored_reference_variance_numpy,
    _depth_variance_scale_numpy,
    _size_modifier_numpy,
    _spatial_aggregate_variance_scale_numpy,
    build_diploid_pair_states,
    pair_state_total_cn,
)


OUTPUT_DIR = Path(__file__).resolve().parent
PAIR_STATES = build_diploid_pair_states(max_hap_cn=2)
STATE_LABELS = [f"({h1},{h2})" for h1, h2 in PAIR_STATES]
STATE_TOTAL_CN = pair_state_total_cn(PAIR_STATES).astype(float)
HEMIZYGOUS_DEL_IDX = PAIR_STATES.index((0, 0))
HAPLOID_NEUTRAL_IDX = PAIR_STATES.index((0, 1))
HAPLOID_DUP_IDX = PAIR_STATES.index((1, 1))

NULL_PRIOR = 1e-3
ALPHA_REF = 50.0
ALPHA_NON_REF = 1.0
SAMPLE_VAR = 0.2
BIN_BIAS = 1.0
RAW_COUNT_MEDIAN = 200.0
REFERENCE_BIN_SIZE = 10_000.0
BIN_SIZE_FACTOR = 10_000.0
INTERVAL_SIZE = 10_000.0
LENGTH_SCALE_VAR = 20_000.0


def default_pair_priors() -> np.ndarray:
    priors = np.full(len(PAIR_STATES), ALPHA_NON_REF, dtype=float)
    priors[PAIR_STATES.index((1, 1))] = ALPHA_REF
    priors /= priors.sum()
    return priors


def format_markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False):
        values: List[str] = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def compute_state_geometry(raw_count_median: float = RAW_COUNT_MEDIAN) -> pd.DataFrame:
    expected_depth = STATE_TOTAL_CN * BIN_BIAS
    count_ref_variance = float(
        _count_anchored_reference_variance_numpy(
            np.array([raw_count_median], dtype=float),
            REFERENCE_BIN_SIZE,
            BIN_SIZE_FACTOR,
        ).squeeze()
    )
    size_modifier = float(
        _size_modifier_numpy(np.array([INTERVAL_SIZE], dtype=float), BIN_SIZE_FACTOR).squeeze()
    )
    spatial_factor = float(
        _spatial_aggregate_variance_scale_numpy(
            np.array([[INTERVAL_SIZE]], dtype=float),
            LENGTH_SCALE_VAR,
        ).squeeze()
    )
    poisson_variance = count_ref_variance * size_modifier * _depth_variance_scale_numpy(expected_depth)
    excess_variance = (expected_depth ** 2) * SAMPLE_VAR * spatial_factor
    total_variance = poisson_variance + excess_variance
    return pd.DataFrame(
        {
            "state": STATE_LABELS,
            "total_cn": STATE_TOTAL_CN,
            "expected_depth": expected_depth,
            "poisson_variance": poisson_variance,
            "excess_variance": excess_variance,
            "total_variance": total_variance,
            "std": np.sqrt(total_variance),
        }
    )


def compute_posteriors(
    observed_depth: float,
    pair_priors: np.ndarray,
    raw_count_median: float = RAW_COUNT_MEDIAN,
) -> Dict[str, np.ndarray]:
    geometry = compute_state_geometry(raw_count_median)
    expected_depth = geometry["expected_depth"].to_numpy(dtype=float)
    std = geometry["std"].to_numpy(dtype=float)
    pair_priors = np.asarray(pair_priors, dtype=float)
    pair_priors = pair_priors / pair_priors.sum()
    effective_pair_priors = pair_priors * (1.0 - NULL_PRIOR)

    log_lik = -0.5 * np.log(2.0 * math.pi * std ** 2) - (
        (float(observed_depth) - expected_depth) ** 2
    ) / (2.0 * std ** 2)
    log_prior = np.log(np.maximum(effective_pair_priors, 1e-300))
    log_unnormalized = log_lik + log_prior
    null_log_unnormalized = math.log(NULL_PRIOR)

    combined = np.concatenate([log_unnormalized, np.array([null_log_unnormalized], dtype=float)])
    max_log = float(np.max(combined))
    probs = np.exp(combined - max_log)
    probs /= probs.sum()

    return {
        "geometry": geometry,
        "log_lik": log_lik,
        "log_prior": log_prior,
        "log_unnormalized": log_unnormalized,
        "state_posterior": probs[:-1],
        "null_posterior": float(probs[-1]),
    }


def build_depth_sweep() -> pd.DataFrame:
    sweeps = []
    prior_sets = {
        "default_ref_biased": default_pair_priors(),
        "uniform_pair_priors": np.full(len(PAIR_STATES), 1.0 / len(PAIR_STATES), dtype=float),
    }
    observed_depths = np.linspace(0.0, 0.30, 31)
    for prior_name, pair_priors in prior_sets.items():
        for observed_depth in observed_depths:
            posterior = compute_posteriors(float(observed_depth), pair_priors)
            state_posterior = posterior["state_posterior"]
            sweeps.append(
                {
                    "prior_regime": prior_name,
                    "observed_depth": float(observed_depth),
                    "posterior_del_state_00": float(state_posterior[HEMIZYGOUS_DEL_IDX]),
                    "posterior_haploid_neutral_01": float(state_posterior[HAPLOID_NEUTRAL_IDX]),
                    "posterior_dup_state_11": float(state_posterior[HAPLOID_DUP_IDX]),
                    "posterior_null": float(posterior["null_posterior"]),
                }
            )
    return pd.DataFrame(sweeps)


def build_raw_count_sweep(observed_depth: float = 0.10) -> pd.DataFrame:
    rows = []
    pair_priors = default_pair_priors()
    for raw_count_median in [25.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]:
        posterior = compute_posteriors(observed_depth, pair_priors, raw_count_median=raw_count_median)
        geometry = posterior["geometry"]
        cn0_row = geometry.iloc[HEMIZYGOUS_DEL_IDX]
        state_posterior = posterior["state_posterior"]
        rows.append(
            {
                "raw_count_median": float(raw_count_median),
                "cn0_std": float(cn0_row["std"]),
                "posterior_del_state_00": float(state_posterior[HEMIZYGOUS_DEL_IDX]),
                "posterior_haploid_neutral_01": float(state_posterior[HAPLOID_NEUTRAL_IDX]),
                "posterior_null": float(posterior["null_posterior"]),
            }
        )
    return pd.DataFrame(rows)


def build_one_depth_breakdown(observed_depth: float = 0.10) -> pd.DataFrame:
    pair_priors = default_pair_priors()
    posterior = compute_posteriors(observed_depth, pair_priors)
    breakdown = posterior["geometry"].copy()
    breakdown["pair_prior"] = pair_priors
    breakdown["effective_pair_prior"] = pair_priors * (1.0 - NULL_PRIOR)
    breakdown["log_likelihood"] = posterior["log_lik"]
    breakdown["log_unnormalized"] = posterior["log_unnormalized"]
    breakdown["posterior"] = posterior["state_posterior"]
    return breakdown


def write_diagnosis(
    geometry: pd.DataFrame,
    depth_sweep: pd.DataFrame,
    raw_count_sweep: pd.DataFrame,
    breakdown: pd.DataFrame,
) -> None:
    default_depth_010 = depth_sweep[
        (depth_sweep["prior_regime"] == "default_ref_biased")
        & np.isclose(depth_sweep["observed_depth"], 0.10)
    ].iloc[0]
    uniform_depth_010 = depth_sweep[
        (depth_sweep["prior_regime"] == "uniform_pair_priors")
        & np.isclose(depth_sweep["observed_depth"], 0.10)
    ].iloc[0]
    cn0_row = geometry.iloc[HEMIZYGOUS_DEL_IDX]
    cn1_row = geometry.iloc[HAPLOID_NEUTRAL_IDX]
    cn2_row = geometry.iloc[HAPLOID_DUP_IDX]
    lines = [
        "# Synthetic null-inference diagnosis",
        "",
        "Setup:",
        "- One synthetic haploid chrX bin with no BAF contribution.",
        "- Normalized depths use the model's convention that CN=2 corresponds to depth 2.0, so haploid neutral CN=1 sits at depth 1.0.",
        "- The exact inference model itself is not sample-ploidy-aware; ploidy is estimated upstream for filtering and then only reintroduced later during calling.",
        f"- Default null prior = {NULL_PRIOR:.4g}; default pair-state prior is reference-biased with alpha_ref={ALPHA_REF:.1f} and alpha_non_ref={ALPHA_NON_REF:.1f}.",
        f"- Count-anchored depth config: raw_count_median={RAW_COUNT_MEDIAN:.1f}, interval_size={INTERVAL_SIZE:.0f}, sample_var={SAMPLE_VAR:.3f}, length_scale_var={LENGTH_SCALE_VAR:.0f}.",
        "",
        "Key mechanism:",
        "- The explicit CN0 deletion state has expected depth 0.",
        "- Its Poisson term is multiplied by depth_scale(expected_depth), and expected_depth=0 is clamped only to a tiny floor of 1e-6.",
        "- Its excess variance term is expected_depth^2 * sample_var * spatial_factor, which is exactly 0 when expected_depth=0.",
        "- The outer null state has no depth or BAF likelihood penalty at all; it contributes only its prior mass.",
        "",
        "Representative state geometry:",
        f"- CN0 state (0,0): expected_depth={cn0_row['expected_depth']:.3f}, std={cn0_row['std']:.6f}",
        f"- Haploid-neutral state (0,1): expected_depth={cn1_row['expected_depth']:.3f}, std={cn1_row['std']:.6f}",
        f"- Diploid-reference state (1,1): expected_depth={cn2_row['expected_depth']:.3f}, std={cn2_row['std']:.6f}",
        "",
        "Consequence at observed normalized depth 0.10:",
        f"- Default prior: posterior(CN0)={default_depth_010['posterior_del_state_00']:.6f}, posterior(CN1)={default_depth_010['posterior_haploid_neutral_01']:.6f}, posterior(CN2 ref)={default_depth_010['posterior_dup_state_11']:.6f}, posterior(null)={default_depth_010['posterior_null']:.6f}",
        f"- Uniform priors: posterior(CN0)={uniform_depth_010['posterior_del_state_00']:.6f}, posterior(CN1)={uniform_depth_010['posterior_haploid_neutral_01']:.6f}, posterior(CN2 ref)={uniform_depth_010['posterior_dup_state_11']:.6f}, posterior(null)={uniform_depth_010['posterior_null']:.6f}",
        "- So the immediate effect of residual depth is to destroy the explicit CN0 state. Where that mass goes next depends on the rest of the model; in this depth-only toy it mostly spills into broader nonzero-CN states, with null remaining nontrivial rather than dominant.",
        "",
        "Interpretation:",
        "- A real hemizygous deletion can still have residual normalized depth around 0.05-0.20 from background counts, mismapping, repeats, or imperfect normalization.",
        "- Under this model, those residual depths are catastrophically unlikely under the explicit CN0 state because its variance is nearly zero.",
        "- The inference model is also ploidy-agnostic, so haploid bins compete against the same diploid pair-state family and reference-biased prior used everywhere else.",
        "- In the pure depth-only toy above, that makes broad higher-CN states surprisingly competitive. In a real deletion region, any additional evidence that weakens those alternatives will push more of the off-model mass into null.",
        "- That means high prob_null is best understood as a symptom that the explicit haploid-deletion state is too brittle and the upstream inference model is not calibrated for haploid copy-0 bins.",
        "",
        "State geometry table:",
        "",
        format_markdown_table(geometry),
        "",
        "Observed-depth sweep excerpt:",
        "",
        format_markdown_table(depth_sweep[depth_sweep['prior_regime'] == 'default_ref_biased'].iloc[0:11]),
        "",
        "Observed-depth 0.10 per-state breakdown:",
        "",
        format_markdown_table(breakdown[["state", "expected_depth", "std", "pair_prior", "log_likelihood", "posterior"]]),
        "",
        "Raw-count-median sweep at observed depth 0.10:",
        "",
        format_markdown_table(raw_count_sweep),
        "",
    ]
    (OUTPUT_DIR / "inference_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    geometry = compute_state_geometry()
    depth_sweep = build_depth_sweep()
    raw_count_sweep = build_raw_count_sweep()
    breakdown = build_one_depth_breakdown()

    geometry.to_csv(OUTPUT_DIR / "state_depth_geometry.tsv", sep="\t", index=False)
    depth_sweep.to_csv(OUTPUT_DIR / "observed_depth_sweep.tsv", sep="\t", index=False)
    raw_count_sweep.to_csv(OUTPUT_DIR / "raw_count_median_sweep.tsv", sep="\t", index=False)
    breakdown.to_csv(OUTPUT_DIR / "observed_depth_010_breakdown.tsv", sep="\t", index=False)
    write_diagnosis(geometry, depth_sweep, raw_count_sweep, breakdown)

    default_depth_010 = depth_sweep[
        (depth_sweep["prior_regime"] == "default_ref_biased")
        & np.isclose(depth_sweep["observed_depth"], 0.10)
    ].iloc[0]
    print("Wrote synthetic null-inference analysis to", OUTPUT_DIR)
    print("\nKey summary at observed depth 0.10:")
    print(
        "  posterior(CN0)="
        f"{default_depth_010['posterior_del_state_00']:.6f}  "
        "posterior(null)="
        f"{default_depth_010['posterior_null']:.6f}"
    )


if __name__ == "__main__":
    main()