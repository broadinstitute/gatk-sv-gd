import pandas as pd
import numpy as np

from gatk_sv_gd.bins import compute_flank_regions_from_bins
from gatk_sv_gd.bins import filter_low_quality_bins
from gatk_sv_gd.bins import get_flank_filter_params
from gatk_sv_gd.bins import rebin_locus_intervals
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
