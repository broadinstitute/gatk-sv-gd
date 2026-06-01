import pandas as pd
import numpy as np
import pytest

from gatk_sv_gd.bins import compute_flank_regions_from_bins
from gatk_sv_gd.bins import filter_low_quality_bins
from gatk_sv_gd.bins import get_flank_filter_params
from gatk_sv_gd.bins import rebin_locus_intervals
from gatk_sv_gd.bins import _allocate_bins_across_segments
from gatk_sv_gd.bins import _compute_quality_stats_single_chrom
from gatk_sv_gd.bins import _ploidy_adjust_depths_single_chrom
from gatk_sv_gd.bins import _split_region_into_supported_segments
from gatk_sv_gd.models import GDLocus


class DummyParMask:
    def get_overlap_fractions_batch(self, chrom, starts, ends):
        if chrom != "chrX":
            return np.zeros(len(starts))
        return np.ones(len(starts))


def test_fragmented_flank_rebinning_preserves_masked_gap():
    locus = GDLocus(
        cluster="test_cluster",
        chrom="chr1",
        breakpoints=[(100, 120), (180, 200)],
        breakpoint_names=["1", "2"],
        gd_entries=[],
        is_nahr=True,
        is_terminal=False,
    )
    df = pd.DataFrame(
        {
            "Chr": ["chr1"] * 4,
            "Start": [200, 220, 280, 300],
            "End": [220, 240, 300, 320],
            "sample_1": [1.0, 1.2, 0.9, 1.1],
        }
    )

    rebinned = rebin_locus_intervals(
        df,
        locus,
        max_bins_per_interval=2,
        flank_regions=[(200, 320, "right_flank")],
        min_rebin_coverage=0.5,
    ).sort_values(["Start", "End"]).reset_index(drop=True)

    observed = list(zip(rebinned["Start"].tolist(), rebinned["End"].tolist()))
    assert observed == [(200, 240), (280, 320)]
    assert all(not (start < 280 and end > 240) for start, end in observed)


def test_flank_selection_uses_all_filtered_bins():
    """Flank selection should accumulate across all available clean bins,
    not just the nearest contiguous segment.  Segment-aware rebinning
    handles gaps downstream."""
    locus = GDLocus(
        cluster="test_cluster",
        chrom="chr1",
        breakpoints=[(1000, 1020), (1080, 1100)],
        breakpoint_names=["1", "2"],
        gd_entries=[],
        is_nahr=True,
        is_terminal=False,
    )
    # Two groups of bins left of the locus with a gap between them:
    #   group A: 600-640, 700-740  (distant)
    #   group B: 960-1000          (near locus)
    df = pd.DataFrame(
        {
            "Chr": ["chr1"] * 6,
            "Start": [600, 620, 700, 720, 960, 980],
            "End": [620, 640, 720, 740, 980, 1000],
            "sample_1": [1.0] * 6,
        }
    )

    flanks = compute_flank_regions_from_bins(
        df,
        locus,
        target_size=80,
        min_flank_bases=80,
        min_flank_bins=4,
        min_flank_coverage=0.1,
    )

    # All 6 bins should participate — the left flank extends to 600
    left_flanks = [(s, e, n) for s, e, n in flanks if n == "left_flank"]
    assert len(left_flanks) == 1
    lf_start, lf_end, _ = left_flanks[0]
    # With 4-bin / 80-bp targets, accumulation reaches the 700-740 group
    # (across the gap from the 960-1000 group).  Previously the nearest-
    # segment restriction would have stopped at 960.
    assert lf_start == 700
    assert lf_end == 1000


def test_filter_low_quality_bins_excludes_ploidy_zero_samples():
    df = pd.DataFrame(
        {
            "Chr": ["chrY", "chrY"],
            "Start": [100, 200],
            "End": [150, 250],
            "sample_male": [1.0, 1.0],
            "sample_female": [0.0, 0.0],
        }
    )
    ploidy_map = {
        ("sample_male", "chrY"): 1,
        ("sample_female", "chrY"): 0,
    }

    filtered = filter_low_quality_bins(
        df,
        median_min=1.5,
        median_max=2.5,
        mad_max=0.1,
        ploidy_map=ploidy_map,
    )

    assert len(filtered) == 2


def test_filter_low_quality_bins_ignores_par_bins():
    par_mask = DummyParMask()
    df = pd.DataFrame(
        {
            "Chr": ["chrX"],
            "Start": [120],
            "End": [180],
            "sample_1": [0.1],
            "sample_2": [4.0],
        }
    )

    filtered = filter_low_quality_bins(
        df,
        median_min=1.0,
        median_max=3.0,
        mad_max=0.2,
        par_mask=par_mask,
    )

    assert len(filtered) == 1


def test_get_flank_filter_params_tightens_autosomes_only():
    base = {"median_min": 1.0, "median_max": 3.0, "mad_max": 1.0}

    assert get_flank_filter_params(base, "chr1") == {
        "median_min": 1.5,
        "median_max": 2.5,
        "mad_max": 0.3,
    }
    assert get_flank_filter_params(base, "chrX") == base


def test_get_flank_filter_params_preserves_stricter_inputs():
    base = {"median_min": 1.8, "median_max": 2.2, "mad_max": 0.1}

    assert get_flank_filter_params(base, "chr1") == base


def test_split_region_into_supported_segments_clamps_and_preserves_internal_gaps():
    region_df = pd.DataFrame(
        {
            "Chr": ["chr1"] * 5,
            "Start": [90, 100, 140, 170, 220],
            "End": [100, 120, 160, 200, 240],
            "sample_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    segments = _split_region_into_supported_segments(
        region_df,
        clamp_start=95,
        clamp_end=210,
    )

    assert [(start, end, len(segment_df)) for segment_df, start, end in segments] == [
        (95, 120, 2),
        (140, 160, 1),
        (170, 200, 1),
    ]
    assert segments[0][0]["Start"].tolist() == [90, 100]
    assert segments[1][0]["Start"].tolist() == [140]
    assert segments[2][0]["Start"].tolist() == [170]


def test_allocate_bins_across_segments_preserves_each_fragment_and_favors_larger_ones():
    segments = [
        (pd.DataFrame({"Start": [0], "End": [10]}), 0, 10),
        (pd.DataFrame({"Start": [20, 30, 40], "End": [30, 40, 50]}), 20, 50),
        (pd.DataFrame({"Start": [60, 70], "End": [70, 80]}), 60, 80),
    ]

    assert _allocate_bins_across_segments([], max_bins_per_interval=3) == []
    assert _allocate_bins_across_segments(segments, max_bins_per_interval=2) == [1, 1, 1]
    assert _allocate_bins_across_segments(segments, max_bins_per_interval=5) == [1, 2, 2]


def test_ploidy_adjust_depths_single_chrom_scales_non_diploid_samples_only():
    depths = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    adjusted = _ploidy_adjust_depths_single_chrom(
        depths,
        "chrX",
        ["s1", "s2", "s3"],
        {("s1", "chrX"): 1, ("s2", "chrX"): 0, ("s3", "chrX"): 2},
    )

    assert np.allclose(adjusted, np.array([[2.0, 2.0, 3.0], [8.0, 5.0, 6.0]]))
    assert np.allclose(depths, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_compute_quality_stats_single_chrom_excludes_ploidy_zero_samples_and_marks_empty_rows():
    depths = np.array(
        [
            [1.0, 10.0, 3.0],
            [2.0, 20.0, 4.0],
        ]
    )
    medians, mads, informative_counts = _compute_quality_stats_single_chrom(
        depths,
        "chrX",
        ["s1", "s2", "s3"],
        {("s1", "chrX"): 1, ("s2", "chrX"): 0, ("s3", "chrX"): 2},
    )

    assert medians.tolist() == pytest.approx([2.5, 4.0])
    assert mads.tolist() == pytest.approx([0.5, 0.0])
    assert informative_counts.tolist() == [2, 2]

    empty_medians, empty_mads, empty_counts = _compute_quality_stats_single_chrom(
        np.array([[1.0, 2.0]], dtype=float),
        "chrY",
        ["s1", "s2"],
        {("s1", "chrY"): 0, ("s2", "chrY"): 0},
    )

    assert np.isnan(empty_medians).tolist() == [True]
    assert np.isnan(empty_mads).tolist() == [True]
    assert empty_counts.tolist() == [0]
