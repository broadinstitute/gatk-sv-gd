import sys

import numpy as np
import pandas as pd
import pytest

from gatk_sv_gd._util import (
    posterior_called_state_to_qual,
    posterior_probability_to_qual,
)
from gatk_sv_gd.call import (
    _build_posterior_entry_spec,
    _compute_flank_confidence_stats,
    _compute_interval_confidence_lookup,
    call_cnvs_from_posteriors,
    compute_informative_event_support_probabilities,
    compute_event_marginal_probabilities,
    parse_args,
    score_call_from_posterior_marginals,
)
from gatk_sv_gd.models import GDLocus


def test_posterior_probability_to_qual_phred_scales_and_caps():
    assert posterior_probability_to_qual(0.0) == pytest.approx(-99.0)
    assert posterior_probability_to_qual(0.50) == pytest.approx(0.0)
    assert posterior_probability_to_qual(0.10) == pytest.approx(-10.0 * np.log10(9.0))
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


def test_posterior_called_state_to_qual_returns_negative_support_for_wrong_state():
    event_probabilities = np.array([0.10, 0.90, 0.0, 1.0])
    called_event = np.array([True, False, True, False])

    assert posterior_called_state_to_qual(event_probabilities, called_event).tolist() == pytest.approx([
        -10.0 * np.log10(9.0),
        -10.0 * np.log10(9.0),
        -99.0,
        -99.0,
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


def test_score_call_from_posterior_marginals_preserves_negative_interval_qual():
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
        [0.90, 0.10, 0.00],
        [0.90, 0.10, 0.00],
        [0.10, 0.90, 0.00],
        [0.10, 0.90, 0.00],
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

    body_qual = -10.0 * np.log10(9.0)

    assert call["log_prob_score"] == pytest.approx(0.10)
    assert call["min_interval_confidence"] == pytest.approx(body_qual)
    assert call["confidence_score"] == pytest.approx(body_qual)
    assert call["qual_score"] == pytest.approx(body_qual)
    assert call["min_flank_non_event_confidence"] == pytest.approx(body_qual)


def test_compute_event_marginal_probabilities_treats_null_mass_as_neutral():
    pair_states = [(1, 1), (0, 1), (1, 2)]
    pair_prob_matrix = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.1, 0.0],
        ],
        dtype=float,
    )

    observed = compute_event_marginal_probabilities(
        pair_prob_matrix,
        pair_states,
        2,
        null_probability=np.array([1.0, 0.2], dtype=float),
    )

    assert observed["DEL"] == pytest.approx([0.5, 0.2])
    assert observed["DUP"] == pytest.approx([0.5, 0.1])


def test_compute_informative_event_support_probabilities_ignores_null_mass():
    pair_states = [(0, 0), (0, 1), (1, 1)]
    pair_prob_matrix = np.array(
        [
            [0.09, 0.01, 0.0],
            [0.01, 0.09, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    observed = compute_informative_event_support_probabilities(
        pair_prob_matrix,
        pair_states,
        sample_ploidy=1,
    )

    assert observed["DEL"] == pytest.approx([0.9, 0.1, 0.5])
    assert observed["DUP"] == pytest.approx([0.0, 0.0, 0.5])


def test_compute_interval_confidence_lookup_uses_effective_bin_count():
    event_probabilities = np.array([0.90, 0.90, 0.10], dtype=float)
    interval_bin_arrays = {
        "A-B": np.array([0, 1], dtype=int),
        "left_flank": np.array([2], dtype=int),
    }

    observed = _compute_interval_confidence_lookup(
        ["A-B"],
        interval_bin_arrays,
        event_probabilities,
        neighbor_bin_correlation=0.5,
    )

    body_qual = posterior_called_state_to_qual(np.array([0.90, 0.90]), True)
    expected = float(np.mean(body_qual) * (2.0 / 1.5))

    assert observed["A-B"] == pytest.approx(expected)


def test_score_call_from_posterior_marginals_accumulates_multi_bin_interval_qual():
    locus = GDLocus(
        cluster="test_cluster",
        chrom="chr1",
        breakpoints=[(100, 100), (200, 200)],
        breakpoint_names=["A", "B"],
        gd_entries=[],
        is_nahr=True,
        is_terminal=False,
    )
    entry = {
        "GD_ID": "gd_del",
        "start_GRCh38": 100,
        "end_GRCh38": 200,
        "svtype": "DEL",
        "BP1": "A",
        "BP2": "B",
    }
    pair_states = [(1, 1), (0, 1), (1, 2)]
    sample_pair_probs = np.array([
        [0.10, 0.90, 0.00],
        [0.10, 0.90, 0.00],
        [0.95, 0.05, 0.00],
        [0.90, 0.10, 0.00],
    ])
    interval_bin_arrays = {
        "A-B": np.array([0, 1], dtype=int),
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
        posterior_interval_bin_correlation=0.5,
    )

    body_qual = 10.0 * np.log10(9.0)
    expected_interval_confidence = body_qual * (2.0 / 1.5)

    assert call["interval_confidences"] == pytest.approx([expected_interval_confidence])
    assert call["min_interval_confidence"] == pytest.approx(expected_interval_confidence)


def test_score_call_from_posterior_marginals_uses_non_null_support_for_haploid_qual():
    locus = GDLocus(
        cluster="test_cluster",
        chrom="chrX",
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
    pair_states = [(0, 0), (0, 1), (1, 1)]
    sample_pair_probs = np.array([
        [0.09, 0.01, 0.00],
        [0.09, 0.01, 0.00],
        [0.01, 0.09, 0.00],
        [0.01, 0.09, 0.00],
    ])
    null_probability = np.array([0.90, 0.90, 0.90, 0.90], dtype=float)
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
        sample_ploidy=1,
        null_probability=null_probability,
    )

    body_qual = posterior_probability_to_qual(0.90)

    assert call["log_prob_score"] == pytest.approx(0.54)
    assert call["min_interval_confidence"] == pytest.approx(body_qual)
    assert call["min_flank_non_event_confidence"] == pytest.approx(body_qual)
    assert call["confidence_score"] == pytest.approx(body_qual)
    assert call["qual_score"] == pytest.approx(body_qual)


def test_parse_args_accepts_posterior_interval_bin_correlation(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gatk-sv-gd call",
            "--cn-posteriors",
            "cn.tsv.gz",
            "--bin-mappings",
            "bins.tsv.gz",
            "--gd-table",
            "gd.tsv",
            "--ploidy-table",
            "ploidy.tsv",
            "--output-dir",
            "out",
            "--posterior-interval-bin-correlation",
            "0.25",
        ],
    )

    args = parse_args()

    assert args.posterior_interval_bin_correlation == pytest.approx(0.25)


def test_score_call_from_posterior_marginals_fast_path_matches_default():
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

    expected_call = score_call_from_posterior_marginals(
        locus=locus,
        entry=entry,
        sample_pair_probs=sample_pair_probs,
        pair_states=pair_states,
        interval_bin_arrays=interval_bin_arrays,
        sample_ploidy=2,
    )

    entry_spec = _build_posterior_entry_spec(locus, entry, interval_bin_arrays)
    event_probabilities = compute_event_marginal_probabilities(
        sample_pair_probs,
        pair_states,
        2,
    )["DEL"]
    interval_confidence_lookup = _compute_interval_confidence_lookup(
        entry_spec["covered_intervals"],
        interval_bin_arrays,
        event_probabilities,
    )
    flank_non_event_medians, flank_confidences = _compute_flank_confidence_stats(
        interval_bin_arrays,
        event_probabilities,
    )
    observed_call = score_call_from_posterior_marginals(
        locus=locus,
        entry=entry,
        sample_pair_probs=sample_pair_probs,
        pair_states=pair_states,
        interval_bin_arrays=interval_bin_arrays,
        sample_ploidy=2,
        event_probabilities=event_probabilities,
        entry_spec=entry_spec,
        interval_confidence_lookup=interval_confidence_lookup,
        flank_non_event_medians=flank_non_event_medians,
        flank_confidences=flank_confidences,
    )

    assert observed_call["intervals"] == expected_call["intervals"]
    assert observed_call["interval_confidences"] == pytest.approx(
        expected_call["interval_confidences"]
    )
    for field in [
        "n_bins",
        "matched_interval_bp",
        "interval_coverage",
        "reciprocal_overlap",
        "min_interval_confidence",
        "left_flank_non_event_median",
        "right_flank_non_event_median",
        "min_flank_non_event_confidence",
        "log_prob_score",
        "confidence_score",
        "qual_score",
    ]:
        assert observed_call[field] == pytest.approx(expected_call[field], nan_ok=True)


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


def test_call_cnvs_from_posteriors_uses_null_mass_as_neutral_event_evidence():
    cn_posteriors_df, bin_mappings_df, gd_table = _minimal_call_inputs()
    cn_posteriors_df["prob_pair_1_1"] = 0.0
    cn_posteriors_df["prob_null"] = 1.0
    ploidy_df = pd.DataFrame(
        {"sample": ["S1"], "contig": ["chr1"], "ploidy": [2]}
    )

    _, _, event_marginals_df = call_cnvs_from_posteriors(
        cn_posteriors_df,
        bin_mappings_df,
        gd_table,
        ploidy_df=ploidy_df,
        calling_mode="posterior-marginal",
    )

    event_row = event_marginals_df.iloc[0]
    assert event_row["prob_null"] == pytest.approx(1.0)
    assert event_row["prob_del_event"] == pytest.approx(0.5)
    assert event_row["prob_dup_event"] == pytest.approx(0.5)
    assert event_row["qual_del_event"] == pytest.approx(0.0)
    assert event_row["qual_dup_event"] == pytest.approx(0.0)


def test_call_cnvs_from_posteriors_event_qual_uses_informative_support():
    cn_posteriors_df = pd.DataFrame(
        {
            "sample": ["S1"],
            "cluster": ["test_cluster"],
            "chr": ["chrX"],
            "start": [100],
            "end": [200],
            "depth": [0.25],
            "prob_null": [0.90],
            "prob_pair_0_0": [0.09],
            "prob_pair_0_1": [0.01],
            "prob_pair_1_1": [0.0],
        }
    )
    bin_mappings_df = pd.DataFrame(
        {
            "array_idx": [0],
            "cluster": ["test_cluster"],
            "interval": ["A-B"],
            "chr": ["chrX"],
            "start": [100],
            "end": [200],
        }
    )
    gd_table = type("GDTableStub", (), {})()
    gd_table.loci = {
        "test_cluster": GDLocus(
            cluster="test_cluster",
            chrom="chrX",
            breakpoints=[(100, 100), (200, 200)],
            breakpoint_names=["A", "B"],
            gd_entries=[],
            is_nahr=True,
            is_terminal=False,
        )
    }
    ploidy_df = pd.DataFrame(
        {"sample": ["S1"], "contig": ["chrX"], "ploidy": [1]}
    )

    _, _, event_marginals_df = call_cnvs_from_posteriors(
        cn_posteriors_df,
        bin_mappings_df,
        gd_table,
        ploidy_df=ploidy_df,
        calling_mode="posterior-marginal",
    )

    event_row = event_marginals_df.iloc[0]
    assert event_row["prob_del_event"] == pytest.approx(0.54)
    assert event_row["prob_dup_event"] == pytest.approx(0.45)
    assert event_row["qual_del_event"] == pytest.approx(
        posterior_probability_to_qual(0.90)
    )
    assert event_row["qual_dup_event"] == pytest.approx(-99.0)


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