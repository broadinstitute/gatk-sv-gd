import numpy as np
import pytest

from gatk_sv_gd._util import (
    posterior_called_state_to_qual,
    posterior_probability_to_qual,
)
from gatk_sv_gd.call import score_call_from_posterior_marginals
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