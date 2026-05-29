from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gatk_sv_gd.depth import CNVModel, _count_anchored_reference_variance_numpy, build_diploid_pair_states


OUTPUT_DIR = Path(__file__).resolve().parent
PAIR_STATES = build_diploid_pair_states(2)
SCENARIO_PRIORS = {
    "haploid_reference_bin": np.asarray(
        [1.0 / 55.0, 1.0 / 55.0, 1.0 / 55.0, 50.0 / 55.0, 1.0 / 55.0, 1.0 / 55.0],
        dtype=np.float32,
    ),
    "deletion_biased_lowdepth_bin": np.asarray(
        [0.6, 0.1, 0.05, 0.1, 0.1, 0.05],
        dtype=np.float32,
    ),
}
SCENARIO_DEPTHS = {
    "haploid_reference_bin": 1.5,
    "deletion_biased_lowdepth_bin": 0.1,
}
REGIMES = [
    ("baseline", 0.0, None),
    ("ploidy_only", 0.0, 1),
    ("variance_floor_only", 0.1, None),
    ("ploidy_and_variance_floor", 0.1, 1),
]


class _FakeTensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


def format_markdown_table(df: pd.DataFrame) -> str:
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


def make_model(min_variance_expected_depth: float, pair_priors: np.ndarray) -> CNVModel:
    model = object.__new__(CNVModel)
    model.pair_states = PAIR_STATES
    model.n_states = len(PAIR_STATES)
    model.max_total_cn = 4
    model.bin_size_factor = 1.0
    model.baf_temperature = 0.0
    model.baf_outlier_rate = 0.0
    model.null_state_prior = 1e-3
    model.var_length_scale = 1_000.0
    model.alpha_ref = 50.0
    model.alpha_non_ref = 1.0
    model.ref_state_idx = PAIR_STATES.index((1, 1))
    model.min_variance_expected_depth = float(min_variance_expected_depth)
    model._count_anchored_reference_variance_np = _count_anchored_reference_variance_numpy(
        np.asarray([200.0, 200.0], dtype=np.float32),
        reference_bin_size=1.0,
        bin_size_factor=1.0,
    )
    pair_priors = np.asarray(pair_priors, dtype=np.float32)
    model.get_map_estimates = lambda data, pair_priors=pair_priors: {
        "bin_bias": np.asarray([1.0, 1.0], dtype=np.float32),
        "sample_var": np.asarray([0.2, 0.2], dtype=np.float32),
        "pair_state_probs": np.asarray([pair_priors, pair_priors], dtype=np.float32),
        "length_scale_var": np.asarray(1_000.0, dtype=np.float32),
    }
    return model


def run_regime(
    observed_depth: float,
    pair_priors: np.ndarray,
    regime_name: str,
    min_variance_expected_depth: float,
    sample_ploidy: Optional[int],
) -> dict:
    model = make_model(min_variance_expected_depth, pair_priors)
    payload = {
        "depth": _FakeTensor([[observed_depth, observed_depth], [observed_depth, observed_depth]]),
        "interval_sizes": _FakeTensor([[1.0], [1.0]]),
        "n_bins": 2,
        "n_samples": 2,
        "has_baf": False,
    }
    if sample_ploidy is not None:
        payload["sample_ploidy"] = _FakeTensor(
            [[sample_ploidy, sample_ploidy], [sample_ploidy, sample_ploidy]]
        )
    posterior = CNVModel.run_discrete_inference(model, SimpleNamespace(**payload))
    pair_posterior = posterior["pair_state_posterior"][0, 0]
    return {
        "regime": regime_name,
        "observed_depth": float(observed_depth),
        "min_variance_expected_depth": float(min_variance_expected_depth),
        "sample_ploidy": float(sample_ploidy) if sample_ploidy is not None else np.nan,
        "posterior_cn0": float(pair_posterior[0]),
        "posterior_cn1": float(pair_posterior[1]),
        "posterior_cn2_ref": float(pair_posterior[3]),
        "posterior_null": float(posterior["null_posterior"][0, 0]),
    }


def build_comparison_table() -> pd.DataFrame:
    rows = []
    for scenario_name, pair_priors in SCENARIO_PRIORS.items():
        observed_depth = SCENARIO_DEPTHS[scenario_name]
        for regime_name, min_variance_expected_depth, sample_ploidy in REGIMES:
            row = run_regime(
                observed_depth,
                pair_priors,
                regime_name,
                min_variance_expected_depth,
                sample_ploidy,
            )
            row["scenario"] = scenario_name
            rows.append(row)
    return pd.DataFrame(rows)[
        [
            "scenario",
            "regime",
            "observed_depth",
            "min_variance_expected_depth",
            "sample_ploidy",
            "posterior_cn0",
            "posterior_cn1",
            "posterior_cn2_ref",
            "posterior_null",
        ]
    ]


def write_diagnosis(comparison: pd.DataFrame) -> None:
    ref_rows = comparison[comparison["scenario"] == "haploid_reference_bin"].copy()
    del_rows = comparison[comparison["scenario"] == "deletion_biased_lowdepth_bin"].copy()
    lines = [
        "# Exact-inference intervention comparison",
        "",
        "Synthetic scenarios:",
        "- `haploid_reference_bin`: observed normalized depth 1.5 with the default diploid-reference-biased pair-state prior.",
        "- `deletion_biased_lowdepth_bin`: observed normalized depth 0.1 with an upstream prior that already leans toward CN0 but still includes substantial neutral mass.",
        "",
        "Interventions compared:",
        "- `baseline`: current old exact-inference behavior (no ploidy input, no CN0 variance floor).",
        "- `ploidy_only`: sample ploidy attached to exact inference, but no CN0 variance floor.",
        "- `variance_floor_only`: CN0 residual-depth tolerance only.",
        "- `ploidy_and_variance_floor`: both changes together.",
        "",
        "Interpretation:",
        "- On the haploid-reference synthetic bin, the ploidy-aware exact-inference path does what it is supposed to do: it moves posterior mass from diploid-reference CN2 toward haploid-neutral CN1.",
        "- On the deletion-biased low-depth bin, the CN0 variance floor is the intervention that directly recovers deletion posterior and reduces null/off-model behavior.",
        "- The combined regime shows that these two interventions solve different problems and can trade off against each other depending on the learned pair-state prior for the bin.",
        "",
        "Haploid-reference comparison:",
        "",
        format_markdown_table(ref_rows),
        "",
        "Deletion-biased low-depth comparison:",
        "",
        format_markdown_table(del_rows),
        "",
    ]
    (OUTPUT_DIR / "intervention_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    comparison = build_comparison_table()
    comparison.to_csv(OUTPUT_DIR / "intervention_comparison.tsv", sep="\t", index=False)
    write_diagnosis(comparison)
    print("Wrote exact-inference intervention comparison to", OUTPUT_DIR)
    print("\nComparison summary:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()