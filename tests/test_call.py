import numpy as np
import pandas as pd
import pytest

from gatk_sv_gd._util import (
    posterior_called_state_to_qual,
    posterior_probability_to_qual,
)
from gatk_sv_gd.call import call_cnvs_from_posteriors, score_call_from_posterior_marginals
from gatk_sv_gd.models import GDLocus


def test_posterior_probability_to_qual_phred_scales_and_caps():
    assert posterior_probability_to_qual(0.0) == pytest.approx(0.0)
    assert posterior_probability_to_qual(0.50) == pytest.approx(0.0)
    assert posterior_probability_to_qual(0.90) == pytest.approx(10.0 * np.log10(9.0))
    assert posterior_probability_to_qual(0.99) == pytest.approx(10.0 * np.log10(99.0))
    assert posterior_probability_to_qual(1.0) == pytest.approx(99.0)


def test_posterior_called_state_to_qual_flips_for_non_event_bins():
    event_probabilities = np.array([0.90, 0.10, 0.50, 1.00, 0.00])
    called_event = np.array([True, False, True, True, False])

    assert posterior_called_state_to_qual(event_probabilities, called_event).tolist() == pytest.approx([
        10.0 * np.log10(9.0),
        10.0 * np.log10(9.0),
        0.0,
        99.0,
        99.0,
    ])


def test_score_call_from_posterior_marginals_reports_qual_score():
    locus = GDLocus(
        cluster="test_cluster",
        chrom="chr1",
        breakpoints=[(100, 100), (200, 200), (300, 300)],
        breakpoint_names=["A", "B", "C"],
        gd_entries=[],
        is_nahr=True,
        is_terminal=False,
    )
    entry = {
        "GD_ID": "gd_del",
        "start_GRCh38": 100,
        "end_GRCh38": 300,
        "svtype": "DEL",
        "BP1": "A",
        "BP2": "C",
    }
    pair_states = [(1, 1), (0, 1), (1, 2)]
    sample_pair_probs = np.array([
        [0.10, 0.90, 0.00],
        [0.10, 0.90, 0.00],
        [0.95, 0.05, 0.00],
        [0.90, 0.10, 0.00],
    ])
    interval_bin_arrays = {
        "A-B": np.array([0], dtype=int),
        "B-C": np.array([1], dtype=int),
        "left_flank": np.array([2], dtype=int),
        "right_flank": np.array([3], dtype=int),
    }

    call = score_call_from_posterior_marginals(
        locus=locus,
        entry=entry,
        sample_pair_probs=sample_pair_probs,
        pair_states=pair_states,
        interval_bin_arrays=interval_bin_arrays,
        sample_ploidy=2,
    )

    body_qual = 10.0 * np.log10(9.0)
    expected_confidence = body_qual

    assert call["log_prob_score"] == pytest.approx(0.90)
    assert call["confidence_score"] == pytest.approx(expected_confidence)
    assert call["qual_score"] == pytest.approx(expected_confidence)
    assert call["min_interval_confidence"] == pytest.approx(body_qual)
    assert call["min_flank_non_event_confidence"] == pytest.approx(body_qual)


def _minimal_call_inputs():
    cn_posteriors_df = pd.DataFrame(
        {
            "sample": ["S1"],
            "cluster": ["test_cluster"],
            "chr": ["chr1"],
            "start": [100],
            "end": [200],
            "depth": [2.0],
            "prob_pair_1_1": [1.0],
        }
    )
    bin_mappings_df = pd.DataFrame(
        {
            "array_idx": [0],
            "cluster": ["test_cluster"],
            "interval": ["A-B"],
            "chr": ["chr1"],
            "start": [100],
            "end": [200],
        }
    )
    gd_table = type("GDTableStub", (), {})()
    gd_table.loci = {
        "test_cluster": GDLocus(
            cluster="test_cluster",
            chrom="chr1",
            breakpoints=[(100, 100), (200, 200)],
            breakpoint_names=["A", "B"],
            gd_entries=[],
            is_nahr=True,
            is_terminal=False,
        )
    }
    return cn_posteriors_df, bin_mappings_df, gd_table


def test_call_cnvs_requires_ploidy_table():
    cn_posteriors_df, bin_mappings_df, gd_table = _minimal_call_inputs()

    with pytest.raises(ValueError, match="ploidy_df is required"):
        call_cnvs_from_posteriors(
            cn_posteriors_df,
            bin_mappings_df,
            gd_table,
            ploidy_df=None,
            calling_mode="posterior-marginal",
        )


def test_call_cnvs_rejects_missing_ploidy_pairs():
    cn_posteriors_df, bin_mappings_df, gd_table = _minimal_call_inputs()
    ploidy_df = pd.DataFrame(
        {"sample": ["S2"], "contig": ["chr1"], "ploidy": [2]}
    )

    with pytest.raises(ValueError, match="missing 1 required"):
        call_cnvs_from_posteriors(
            cn_posteriors_df,
            bin_mappings_df,
            gd_table,
            ploidy_df=ploidy_df,
            calling_mode="posterior-marginal",
        )


def test_call_cnvs_rejects_posterior_mapping_row_mismatch():
    cn_posteriors_df, bin_mappings_df, gd_table = _minimal_call_inputs()
    bin_mappings_df = pd.concat([bin_mappings_df, bin_mappings_df], ignore_index=True)
    bin_mappings_df.loc[1, "array_idx"] = 1
    bin_mappings_df.loc[1, "start"] = 200
    bin_mappings_df.loc[1, "end"] = 300
    ploidy_df = pd.DataFrame(
        {"sample": ["S1"], "contig": ["chr1"], "ploidy": [2]}
    )

    with pytest.raises(ValueError, match="cn_posteriors has 1 rows"):
        call_cnvs_from_posteriors(
            cn_posteriors_df,
            bin_mappings_df,
            gd_table,
            ploidy_df=ploidy_df,
            calling_mode="posterior-marginal",
        )


def test_call_cnvs_marks_best_match_without_confident_carrier(monkeypatch):
    cn_posteriors_df, bin_mappings_df, gd_table = _minimal_call_inputs()
    gd_table.loci["test_cluster"].gd_entries = [
        {
            "GD_ID": "GD1",
            "start_GRCh38": 100,
            "end_GRCh38": 200,
            "svtype": "DEL",
            "BP1": "A",
            "BP2": "B",
        },
        {
            "GD_ID": "GD2",
            "start_GRCh38": 100,
            "end_GRCh38": 200,
            "svtype": "DEL",
            "BP1": "A",
            "BP2": "B",
        },
    ]
    ploidy_df = pd.DataFrame(
        {"sample": ["S1"], "contig": ["chr1"], "ploidy": [2]}
    )

    def _fake_score_call_from_posterior_marginals(**kwargs):
        entry = kwargs["entry"]
        gd_id = entry["GD_ID"]
        score = 40.0 if gd_id == "GD1" else 20.0
        return {
            "GD_ID": gd_id,
            "chrom": "chr1",
            "start": 100,
            "end": 200,
            "svtype": "DEL",
            "BP1": "A",
            "BP2": "B",
            "is_terminal": False,
            "n_bins": 1,
            "sample_ploidy": 2,
            "haplotype": np.nan,
            "hap_cn_state": np.nan,
            "matched_seg_start": np.nan,
            "matched_seg_end": np.nan,
            "matched_seg_n_bins": 0,
            "matched_interval_bp": 100,
            "interval_coverage": score,
            "reciprocal_overlap": score,
            "intervals": ["A-B"],
            "interval_confidences": [score],
            "min_interval_confidence": score,
            "left_flank_non_event_median": score,
            "right_flank_non_event_median": score,
            "min_flank_non_event_confidence": score,
            "log_prob_score": score,
            "confidence_score": score,
            "qual_score": score,
            "is_carrier": False,
        }

    monkeypatch.setattr(
        "gatk_sv_gd.call.score_call_from_posterior_marginals",
        _fake_score_call_from_posterior_marginals,
    )

    calls_df, _, _ = call_cnvs_from_posteriors(
        cn_posteriors_df,
        bin_mappings_df,
        gd_table,
        ploidy_df=ploidy_df,
        calling_mode="posterior-marginal",
        min_posterior_interval_confidence=60.0,
        min_flank_non_event_confidence=60.0,
    )

    assert calls_df["is_carrier"].sum() == 0
    assert calls_df["is_best_match"].sum() == 1
    assert calls_df.loc[calls_df["GD_ID"] == "GD1", "is_best_match"].item()
    assert calls_df["call_criteria_interval_confidence"].tolist() == [60.0, 60.0]
    assert calls_df["call_criteria_flank_non_event_confidence"].tolist() == [60.0, 60.0]