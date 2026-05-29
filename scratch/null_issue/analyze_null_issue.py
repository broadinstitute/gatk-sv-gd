from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gatk_sv_gd._util import posterior_called_state_to_qual
from gatk_sv_gd.call import (
    _build_posterior_entry_spec,
    _compute_flank_confidence_stats,
    _compute_interval_confidence_lookup,
    _score_posterior_call_from_event_probabilities,
    compute_event_marginal_probabilities,
    compute_informative_event_support_probabilities,
    score_call_from_posterior_marginals,
)
from gatk_sv_gd.models import GDLocus


OUTPUT_DIR = Path(__file__).resolve().parent
PAIR_STATES = [(0, 0), (0, 1), (1, 1)]
SAMPLE_PLOIDY = 1
BIN_LABELS = ["body_A-B", "body_B-C", "left_flank", "right_flank"]
CALLED_EVENT_MASK = np.array([True, True, False, False], dtype=bool)
EXPECTED_STATE = np.where(CALLED_EVENT_MASK, "DEL", "non-DEL")


def build_locus() -> Tuple[GDLocus, dict, Dict[str, np.ndarray]]:
    locus = GDLocus(
        cluster="null_issue",
        chrom="chrX",
        breakpoints=[(100, 100), (200, 200), (300, 300)],
        breakpoint_names=["A", "B", "C"],
        gd_entries=[],
        is_nahr=True,
        is_terminal=False,
    )
    entry = {
        "GD_ID": "synthetic_chrX_del",
        "start_GRCh38": 100,
        "end_GRCh38": 300,
        "svtype": "DEL",
        "BP1": "A",
        "BP2": "C",
    }
    interval_bin_arrays = {
        "A-B": np.array([0], dtype=int),
        "B-C": np.array([1], dtype=int),
        "left_flank": np.array([2], dtype=int),
        "right_flank": np.array([3], dtype=int),
    }
    return locus, entry, interval_bin_arrays


def make_pair_prob_matrix(body_null: float, flank_null: float) -> Tuple[np.ndarray, np.ndarray]:
    body_informative_mass = 1.0 - float(body_null)
    flank_informative_mass = 1.0 - float(flank_null)
    pair_prob_matrix = np.array(
        [
            [0.90 * body_informative_mass, 0.10 * body_informative_mass, 0.0],
            [0.90 * body_informative_mass, 0.10 * body_informative_mass, 0.0],
            [0.10 * flank_informative_mass, 0.90 * flank_informative_mass, 0.0],
            [0.10 * flank_informative_mass, 0.90 * flank_informative_mass, 0.0],
        ],
        dtype=float,
    )
    null_probability = np.array(
        [body_null, body_null, flank_null, flank_null],
        dtype=float,
    )
    return pair_prob_matrix, null_probability


def score_old_style(
    locus: GDLocus,
    entry: dict,
    interval_bin_arrays: Dict[str, np.ndarray],
    pair_prob_matrix: np.ndarray,
    null_probability: np.ndarray,
) -> dict:
    entry_spec = _build_posterior_entry_spec(locus, entry, interval_bin_arrays)
    old_event_probabilities = compute_event_marginal_probabilities(
        pair_prob_matrix,
        PAIR_STATES,
        SAMPLE_PLOIDY,
        null_probability=null_probability,
    )[entry["svtype"]]
    interval_confidence_lookup = _compute_interval_confidence_lookup(
        entry_spec["covered_intervals"],
        interval_bin_arrays,
        old_event_probabilities,
    )
    flank_non_event_medians, flank_confidences = _compute_flank_confidence_stats(
        interval_bin_arrays,
        old_event_probabilities,
    )
    return _score_posterior_call_from_event_probabilities(
        locus=locus,
        entry_spec=entry_spec,
        event_probabilities=old_event_probabilities,
        interval_confidence_lookup=interval_confidence_lookup,
        flank_non_event_medians=flank_non_event_medians,
        flank_confidences=flank_confidences,
        sample_ploidy=SAMPLE_PLOIDY,
    )


def score_case(body_null: float, flank_null: float) -> dict:
    locus, entry, interval_bin_arrays = build_locus()
    pair_prob_matrix, null_probability = make_pair_prob_matrix(body_null, flank_null)
    event_probabilities = compute_event_marginal_probabilities(
        pair_prob_matrix,
        PAIR_STATES,
        SAMPLE_PLOIDY,
        null_probability=null_probability,
    )["DEL"]
    support_probabilities = compute_informative_event_support_probabilities(
        pair_prob_matrix,
        PAIR_STATES,
        SAMPLE_PLOIDY,
    )["DEL"]
    old_call = score_old_style(
        locus,
        entry,
        interval_bin_arrays,
        pair_prob_matrix,
        null_probability,
    )
    new_call = score_call_from_posterior_marginals(
        locus=locus,
        entry=entry,
        sample_pair_probs=pair_prob_matrix,
        pair_states=PAIR_STATES,
        interval_bin_arrays=interval_bin_arrays,
        sample_ploidy=SAMPLE_PLOIDY,
        null_probability=null_probability,
    )
    old_expected_state_qual = posterior_called_state_to_qual(
        event_probabilities,
        CALLED_EVENT_MASK,
    )
    new_expected_state_qual = posterior_called_state_to_qual(
        support_probabilities,
        CALLED_EVENT_MASK,
    )
    return {
        "body_null": float(body_null),
        "flank_null": float(flank_null),
        "pair_prob_matrix": pair_prob_matrix,
        "null_probability": null_probability,
        "event_probabilities": event_probabilities,
        "support_probabilities": support_probabilities,
        "old_expected_state_qual": old_expected_state_qual,
        "new_expected_state_qual": new_expected_state_qual,
        "old_call": old_call,
        "new_call": new_call,
    }


def build_bin_rows(scenario: str, result: dict) -> Iterable[dict]:
    del_informative_mass = result["pair_prob_matrix"][:, 0]
    informative_mass = result["pair_prob_matrix"].sum(axis=1)
    non_del_informative_mass = informative_mass - del_informative_mass
    for idx, label in enumerate(BIN_LABELS):
        yield {
            "scenario": scenario,
            "bin_index": idx,
            "bin_label": label,
            "expected_state": EXPECTED_STATE[idx],
            "null_probability": float(result["null_probability"][idx]),
            "informative_mass": float(informative_mass[idx]),
            "del_informative_mass": float(del_informative_mass[idx]),
            "non_del_informative_mass": float(non_del_informative_mass[idx]),
            "old_event_probability_used_for_qual": float(result["event_probabilities"][idx]),
            "new_informative_support_used_for_qual": float(result["support_probabilities"][idx]),
            "old_expected_state_qual": float(result["old_expected_state_qual"][idx]),
            "new_expected_state_qual": float(result["new_expected_state_qual"][idx]),
        }


def build_score_row(scenario: str, result: dict) -> dict:
    old_call = result["old_call"]
    new_call = result["new_call"]
    return {
        "scenario": scenario,
        "body_null": result["body_null"],
        "flank_null": result["flank_null"],
        "old_log_prob_score": float(old_call["log_prob_score"]),
        "new_log_prob_score": float(new_call["log_prob_score"]),
        "old_min_interval_confidence": float(old_call["min_interval_confidence"]),
        "new_min_interval_confidence": float(new_call["min_interval_confidence"]),
        "old_min_flank_non_event_confidence": float(old_call["min_flank_non_event_confidence"]),
        "new_min_flank_non_event_confidence": float(new_call["min_flank_non_event_confidence"]),
        "old_qual_score": float(old_call["qual_score"]),
        "new_qual_score": float(new_call["qual_score"]),
    }


def build_body_null_sweep() -> pd.DataFrame:
    rows = []
    for body_null in np.linspace(0.0, 0.95, 20):
        result = score_case(float(body_null), flank_null=0.0)
        rows.append(
            {
                "body_null": float(body_null),
                "old_body_event_probability": float(result["event_probabilities"][0]),
                "new_body_support_probability": float(result["support_probabilities"][0]),
                "old_body_expected_state_qual": float(result["old_expected_state_qual"][0]),
                "new_body_expected_state_qual": float(result["new_expected_state_qual"][0]),
                "old_final_qual_score": float(result["old_call"]["qual_score"]),
                "new_final_qual_score": float(result["new_call"]["qual_score"]),
            }
        )
    return pd.DataFrame(rows)


def format_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_diagnosis(scenario_scores: pd.DataFrame, bin_details: pd.DataFrame) -> None:
    body_null_only = scenario_scores.set_index("scenario").loc["body_null_only"]
    example_rows = bin_details[bin_details["scenario"] == "body_null_only"].copy()
    lines = [
        "# Synthetic null-heavy posterior diagnosis",
        "",
        "Synthetic setup:",
        "- Haploid chrX locus with two deleted body bins and two flanking non-event bins.",
        "- Among informative pair states, body bins are 90% DEL and flank bins are 90% non-DEL.",
        "- The body_null_only scenario sets body prob_null=0.90 and flank prob_null=0.00.",
        "",
        "Key result:",
        "- High null in the deleted body alone is sufficient to collapse the old final QUAL.",
        "- Clean flanks do not rescue the old score because final qual_score is limited by the minimum interval/flank confidence component.",
        "- The fixed score keeps log_prob_score unchanged but computes body/flank confidence from informative support only.",
        "",
        "Analytic body-bin comparison for this setup:",
        "- Old DEL probability fed into body QUAL = 0.9 * (1 - body_null) + 0.5 * body_null = 0.9 - 0.4 * body_null.",
        "- New DEL support fed into body QUAL = 0.9, independent of body_null.",
        "- At body_null = 0.90, old body input is 0.54 while new body input remains 0.90.",
        "",
        "Representative numbers for body_null_only:",
        f"- Old min_interval_confidence = {body_null_only['old_min_interval_confidence']:.3f}",
        f"- New min_interval_confidence = {body_null_only['new_min_interval_confidence']:.3f}",
        f"- Old min_flank_non_event_confidence = {body_null_only['old_min_flank_non_event_confidence']:.3f}",
        f"- New min_flank_non_event_confidence = {body_null_only['new_min_flank_non_event_confidence']:.3f}",
        f"- Old qual_score = {body_null_only['old_qual_score']:.3f}",
        f"- New qual_score = {body_null_only['new_qual_score']:.3f}",
        f"- Old log_prob_score = {body_null_only['old_log_prob_score']:.3f}",
        f"- New log_prob_score = {body_null_only['new_log_prob_score']:.3f}",
        "",
        "Interpretation:",
        "- The old bug was not that the event summary was wrong; log_prob_score stays the same before and after the fix.",
        "- The bug was that null-neutralized event probabilities were reused for called-state confidence.",
        "- In a null-heavy hemizygous deletion, that drags body-bin QUAL toward zero even when the informative posterior strongly favors deletion.",
        "",
        "Per-bin body_null_only details:",
        "",
        format_markdown_table(example_rows),
        "",
    ]
    (OUTPUT_DIR / "diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    scenarios = {
        "clean_haploid_del": score_case(body_null=0.0, flank_null=0.0),
        "body_null_only": score_case(body_null=0.90, flank_null=0.0),
        "body_and_flank_null": score_case(body_null=0.90, flank_null=0.90),
    }

    bin_details = pd.DataFrame(
        row
        for scenario, result in scenarios.items()
        for row in build_bin_rows(scenario, result)
    )
    scenario_scores = pd.DataFrame(
        build_score_row(scenario, result)
        for scenario, result in scenarios.items()
    )
    body_null_sweep = build_body_null_sweep()

    bin_details.to_csv(OUTPUT_DIR / "scenario_bin_posteriors.tsv", sep="\t", index=False)
    scenario_scores.to_csv(OUTPUT_DIR / "scenario_scores.tsv", sep="\t", index=False)
    body_null_sweep.to_csv(OUTPUT_DIR / "body_null_sweep.tsv", sep="\t", index=False)
    write_diagnosis(scenario_scores, bin_details)

    print("Wrote synthetic null-issue analysis to", OUTPUT_DIR)
    print("\nScenario score summary:")
    print(scenario_scores.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()