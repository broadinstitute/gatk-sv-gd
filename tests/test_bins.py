import pandas as pd
import numpy as np
import pytest

import gatk_sv_gd.bins as bins_module
from gatk_sv_gd.bins import compute_flank_regions_from_bins
from gatk_sv_gd.bins import compute_bin_quality_mask
from gatk_sv_gd.bins import compute_interval_cn_stats
from gatk_sv_gd.bins import call_gd_cnv
from gatk_sv_gd.bins import determine_best_breakpoints
from gatk_sv_gd.bins import read_data
from gatk_sv_gd.bins import filter_low_quality_bins
from gatk_sv_gd.bins import extract_locus_bins
from gatk_sv_gd.bins import assign_bins_to_intervals
from gatk_sv_gd.bins import get_flank_filter_params
from gatk_sv_gd.bins import rebin_locus_intervals
from gatk_sv_gd.bins import _allocate_bins_across_segments
from gatk_sv_gd.bins import _compute_quality_stats_single_chrom
from gatk_sv_gd.bins import _mask_overlap_bool
from gatk_sv_gd.bins import _ploidy_adjust_depths
from gatk_sv_gd.bins import _ploidy_adjust_depths_single_chrom
from gatk_sv_gd.bins import _split_region_into_supported_segments
from gatk_sv_gd.models import GDLocus


class DummyParMask:
    def get_overlap_fractions_batch(self, chrom, starts, ends):
        if chrom != "chrX":
            return np.zeros(len(starts))
        return np.ones(len(starts))


class DummyOverlapMask:
    def get_overlap_fractions_batch(self, chrom, starts, ends):
        starts = np.asarray(starts)
        ends = np.asarray(ends)
        if chrom != "chr1":
            return np.zeros(len(starts))
        overlap = np.maximum(0, np.minimum(ends, 25) - np.maximum(starts, 15))
        return overlap / np.maximum(1, ends - starts)


class IndexedOverlapMask:
    def __init__(self, fractions_by_start):
        self.fractions_by_start = dict(fractions_by_start)

    def get_overlap_fractions_batch(self, chrom, starts, ends):
        starts = np.asarray(starts)
        return np.asarray([
            self.fractions_by_start.get((chrom, int(start)), 0.0)
            for start in starts
        ], dtype=float)


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


def test_mask_overlap_bool_handles_none_empty_and_positive_overlap_cases():
    starts = np.array([0, 10, 20, 30])
    ends = np.array([10, 20, 30, 40])

    assert _mask_overlap_bool(None, "chr1", starts, ends).tolist() == [False, False, False, False]
    assert _mask_overlap_bool(DummyOverlapMask(), "chr1", np.array([], dtype=int), np.array([], dtype=int)).tolist() == []
    assert _mask_overlap_bool(DummyOverlapMask(), "chr1", starts, ends).tolist() == [False, True, True, False]
    assert _mask_overlap_bool(DummyOverlapMask(), "chr2", starts, ends).tolist() == [False, False, False, False]


def test_extract_locus_bins_returns_empty_when_region_has_no_overlap():
    locus = GDLocus(
        cluster="cluster1",
        chrom="chr1",
        breakpoints=[(100, 100), (200, 200)],
        breakpoint_names=["1", "2"],
        gd_entries=[],
        is_nahr=True,
        is_terminal=False,
    )
    df = pd.DataFrame(
        {
            "Chr": ["chr2"],
            "Start": [0],
            "End": [50],
            "sample1": [1.0],
        }
    )

    result = extract_locus_bins(df, locus)

    assert result.empty


def test_extract_locus_bins_applies_exclusion_bypass_and_hard_include():
    locus = GDLocus(
        cluster="cluster1",
        chrom="chr1",
        breakpoints=[(100, 100), (140, 140)],
        breakpoint_names=["1", "2"],
        gd_entries=[],
        is_nahr=True,
        is_terminal=False,
    )
    df = pd.DataFrame(
        {
            "Chr": ["chr1", "chr1", "chr1", "chr1"],
            "Start": [90, 100, 110, 120],
            "End": [100, 110, 120, 130],
            "sample1": [1.0, 1.1, 1.2, 1.3],
        }
    )
    exclusion_mask = IndexedOverlapMask(
        {
            ("chr1", 90): 0.0,
            ("chr1", 100): 0.7,
            ("chr1", 110): 0.8,
            ("chr1", 120): 0.6,
        }
    )
    hard_inclusion_mask = IndexedOverlapMask({("chr1", 120): 1.0})

    result = extract_locus_bins(
        df,
        locus,
        exclusion_mask=exclusion_mask,
        exclusion_threshold=0.5,
        padding=10,
        exclusion_bypass_regions=[(110, 115)],
        hard_inclusion_mask=hard_inclusion_mask,
    )

    assert result["Start"].tolist() == [90, 110, 120]


def test_assign_bins_to_intervals_prefers_greatest_overlap_and_tracks_breakpoint_ranges():
    locus = GDLocus(
        cluster="cluster1",
        chrom="chr1",
        breakpoints=[(100, 110), (150, 160), (210, 220)],
        breakpoint_names=["1", "2", "3"],
        gd_entries=[],
        is_nahr=True,
        is_terminal=False,
    )
    df = pd.DataFrame(
        {
            "Start": [85, 112, 100, 220, 500],
            "End": [100, 140, 110, 240, 510],
        }
    )

    interval_bins = assign_bins_to_intervals(
        df,
        locus,
        flank_regions=[(80, 100, "left_flank"), (220, 240, "right_flank")],
    )

    assert interval_bins["left_flank"] == [0]
    assert interval_bins["1-2"] == [1]
    assert interval_bins["2-3"] == []
    assert interval_bins["right_flank"] == [3]
    assert interval_bins["breakpoint_ranges"] == [2, 4]


def test_assign_bins_to_intervals_falls_back_to_locus_flanking_regions():
    locus = GDLocus(
        cluster="cluster1",
        chrom="chr1",
        breakpoints=[(100, 110), (150, 160)],
        breakpoint_names=["1", "2"],
        gd_entries=[],
        is_nahr=True,
        is_terminal=False,
    )
    df = pd.DataFrame(
        {
            "Start": [90, 112, 160],
            "End": [100, 148, 170],
        }
    )

    interval_bins = assign_bins_to_intervals(df, locus)

    assert interval_bins["left_flank"] == [0]
    assert interval_bins["1-2"] == [1]
    assert interval_bins["right_flank"] == [2]
    assert interval_bins["breakpoint_ranges"] == []


def test_ploidy_adjust_depths_scales_each_sample_by_chromosome_specific_ploidy():
    depths = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    df = pd.DataFrame({"Chr": ["chr1", "chrX", "chr1", "chrX"]})

    adjusted = _ploidy_adjust_depths(
        depths,
        df,
        ["s1", "s2"],
        {
            ("s1", "chr1"): 1,
            ("s1", "chrX"): 2,
            ("s2", "chr1"): 4,
            ("s2", "chrX"): 0,
        },
    )

    assert np.allclose(
        adjusted,
        np.array(
            [
                [2.0, 5.0],
                [2.0, 20.0],
                [6.0, 15.0],
                [4.0, 40.0],
            ]
        ),
    )
    assert np.allclose(
        depths,
        np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
            ]
        ),
    )


def test_compute_bin_quality_mask_covers_empty_par_hard_include_and_no_informative_cases():
    empty_mask, empty_stats = compute_bin_quality_mask(
        pd.DataFrame(columns=["Chr", "Start", "End", "s1"]),
        median_min=1.0,
        median_max=3.0,
        mad_max=0.2,
    )

    assert empty_mask.tolist() == []
    assert empty_stats == {"filtered": 0, "par_ignored": 0, "no_informative": 0}

    df = pd.DataFrame(
        {
            "Chr": ["chr1", "chrX", "chrY", "chr1"],
            "Start": [10, 100, 50, 30],
            "End": [20, 120, 60, 40],
            "s1": [0.1, 0.1, 1.0, 2.0],
            "s2": [4.0, 4.0, 2.0, 2.1],
        }
    )

    keep_mask, stats = compute_bin_quality_mask(
        df,
        median_min=1.0,
        median_max=3.0,
        mad_max=0.2,
        ploidy_map={("s1", "chrY"): 0, ("s2", "chrY"): 0},
        par_mask=DummyParMask(),
        hard_inclusion_mask=DummyOverlapMask(),
    )

    assert keep_mask.tolist() == [True, True, False, True]
    assert stats == {
        "filtered": 1,
        "par_ignored": 1,
        "hard_included": 1,
        "no_informative": 1,
    }


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


def test_compute_flank_regions_from_bins_applies_quality_filter_before_accumulation(monkeypatch):
    locus = GDLocus(
        cluster="test_cluster",
        chrom="chr1",
        breakpoints=[(100, 100), (140, 140)],
        breakpoint_names=["1", "2"],
        gd_entries=[],
        is_nahr=True,
        is_terminal=False,
    )
    df = pd.DataFrame(
        {
            "Chr": ["chr1", "chr1", "chr1", "chr1"],
            "Start": [60, 80, 140, 160],
            "End": [80, 100, 160, 180],
            "sample_1": [1.0, 1.1, 1.2, 1.3],
        }
    )

    def fake_compute_bin_quality_mask(locus_df, **kwargs):
        assert kwargs["median_min"] == 1.0
        assert kwargs["median_max"] == 3.0
        assert kwargs["mad_max"] == 0.2
        return np.asarray([True, False, True, True]), {"filtered": 1, "par_ignored": 1}

    monkeypatch.setattr(bins_module, "compute_bin_quality_mask", fake_compute_bin_quality_mask)

    flanks = compute_flank_regions_from_bins(
        df,
        locus,
        target_size=20,
        min_flank_bases=20,
        min_flank_bins=2,
        min_flank_coverage=0.9,
        filter_params={"median_min": 1.0, "median_max": 3.0, "mad_max": 0.2},
    )

    assert flanks == [
        (60, 100, "left_flank"),
        (140, 180, "right_flank"),
    ]


def test_compute_interval_cn_stats_averages_probabilities_and_handles_empty_intervals():
    cn_posterior = np.asarray(
        [
            [[[0.7, 0.2, 0.1, 0.0, 0.0, 0.0]]],
            [[[0.1, 0.4, 0.5, 0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.2, 0.3, 0.3, 0.2]]],
        ],
        dtype=float,
    ).reshape(3, 1, 6)

    stats = compute_interval_cn_stats(
        cn_posterior,
        {"body": [0, 1], "empty": [], "flank": [2]},
        sample_idx=0,
    )

    assert stats["body"]["n_bins"] == 2
    assert stats["body"]["cn_probs"].tolist() == pytest.approx([0.4, 0.3, 0.3, 0.0, 0.0, 0.0])
    assert stats["empty"]["n_bins"] == 0
    assert stats["empty"]["cn_probs"].tolist() == pytest.approx([0.0] * 6)
    assert stats["flank"]["cn_probs"].tolist() == pytest.approx([0.0, 0.0, 0.2, 0.3, 0.3, 0.2])


def test_call_gd_cnv_distinguishes_supported_spanning_and_unsupported_events():
    class FakeLocus:
        cluster = "cluster1"
        chrom = "chr1"
        is_terminal = False
        gd_entries = [
            {
                "GD_ID": "del_supported",
                "svtype": "DEL",
                "start_GRCh38": 100,
                "end_GRCh38": 200,
                "BP1": "1",
                "BP2": "2",
            },
            {
                "GD_ID": "dup_spanning",
                "svtype": "DUP",
                "start_GRCh38": 200,
                "end_GRCh38": 300,
                "BP1": "2",
                "BP2": "3",
            },
            {
                "GD_ID": "del_unsupported",
                "svtype": "DEL",
                "start_GRCh38": 300,
                "end_GRCh38": 400,
                "BP1": "3",
                "BP2": "4",
            },
        ]

        def get_intervals_between(self, bp1, bp2):
            mapping = {
                ("1", "2"): [(100, 200, "A")],
                ("2", "3"): [(200, 300, "B")],
                ("3", "4"): [(300, 400, "C")],
            }
            return mapping.get((bp1, bp2), [])

        def get_flanking_regions(self):
            return [(0, 100, "left_flank"), (400, 500, "right_flank")]

    interval_stats = {
        "A": {"n_bins": 2, "cn_probs": np.asarray([0.5, 0.4, 0.1, 0.0, 0.0, 0.0])},
        "B": {"n_bins": 3, "cn_probs": np.asarray([0.0, 0.0, 0.1, 0.4, 0.3, 0.2])},
        "C": {"n_bins": 0, "cn_probs": np.zeros(6)},
        "left_flank": {"n_bins": 2, "cn_probs": np.asarray([0.0, 0.0, 0.1, 0.4, 0.3, 0.2])},
        "right_flank": {"n_bins": 2, "cn_probs": np.asarray([0.0, 0.0, 0.1, 0.4, 0.3, 0.2])},
    }

    calls = call_gd_cnv(FakeLocus(), interval_stats, log_prob_threshold=-0.5)
    by_id = {call["GD_ID"]: call for call in calls}

    assert bool(by_id["del_supported"]["is_carrier"])
    assert not bool(by_id["del_supported"]["is_spanning"])
    assert by_id["del_supported"]["n_bins"] == 2
    assert by_id["del_supported"]["intervals"] == ["A"]

    assert not bool(by_id["dup_spanning"]["is_carrier"])
    assert bool(by_id["dup_spanning"]["is_spanning"])
    assert by_id["dup_spanning"]["flanking_log_prob_score"] > -0.5

    assert np.isnan(by_id["del_unsupported"]["log_prob_score"])
    assert not bool(by_id["del_unsupported"]["is_carrier"])
    assert by_id["del_unsupported"]["n_bins"] == 0


def test_determine_best_breakpoints_handles_none_single_and_multiple_carriers():
    class FakeLocus:
        def get_intervals(self):
            return [
                (100, 200, "A"),
                (200, 300, "B"),
                (300, 400, "C"),
            ]

    interval_stats = {
        "A": {"n_bins": 3, "cn_probs": np.asarray([0.55, 0.30, 0.15, 0.0, 0.0, 0.0])},
        "B": {"n_bins": 2, "cn_probs": np.asarray([0.05, 0.10, 0.85, 0.0, 0.0, 0.0])},
        "C": {"n_bins": 4, "cn_probs": np.asarray([0.02, 0.03, 0.15, 0.30, 0.25, 0.25])},
    }
    calls = [
        {"GD_ID": "del_best", "svtype": "DEL", "is_carrier": True, "intervals": ["A"], "start": 100, "end": 200},
        {"GD_ID": "del_worse", "svtype": "DEL", "is_carrier": True, "intervals": ["A", "B"], "start": 100, "end": 300},
        {"GD_ID": "dup_only", "svtype": "DUP", "is_carrier": True, "intervals": ["C"], "start": 300, "end": 400},
        {"GD_ID": "dup_noncarrier", "svtype": "DUP", "is_carrier": False, "intervals": ["B"], "start": 200, "end": 300},
    ]

    best = determine_best_breakpoints(FakeLocus(), interval_stats, calls)

    assert best == {"DEL": "del_best", "DUP": "dup_only"}


def test_determine_best_breakpoints_prefers_larger_variant_on_exact_tie():
    class FakeLocus:
        def get_intervals(self):
            return [
                (100, 200, "A"),
                (200, 300, "B"),
            ]

    interval_stats = {
        "A": {"n_bins": 1, "cn_probs": np.asarray([0.4, 0.4, 0.2, 0.0, 0.0, 0.0])},
        "B": {"n_bins": 1, "cn_probs": np.asarray([0.4, 0.4, 0.2, 0.0, 0.0, 0.0])},
    }
    calls = [
        {"GD_ID": "short", "svtype": "DEL", "is_carrier": True, "intervals": ["A"], "start": 100, "end": 200},
        {"GD_ID": "long", "svtype": "DEL", "is_carrier": True, "intervals": ["B"], "start": 100, "end": 300},
    ]

    best = determine_best_breakpoints(FakeLocus(), interval_stats, calls)

    assert best == {"DEL": "long", "DUP": None}


def test_determine_best_breakpoints_scores_multiple_dup_carriers_by_affected_and_unaffected_intervals():
    class FakeLocus:
        def get_intervals(self):
            return [
                (100, 200, "A"),
                (200, 300, "B"),
                (300, 400, "C"),
            ]

    interval_stats = {
        "A": {"n_bins": 2, "cn_probs": np.asarray([0.1, 0.2, 0.65, 0.03, 0.01, 0.01])},
        "B": {"n_bins": 3, "cn_probs": np.asarray([0.02, 0.03, 0.10, 0.35, 0.25, 0.25])},
        "C": {"n_bins": 2, "cn_probs": np.asarray([0.05, 0.10, 0.80, 0.03, 0.01, 0.01])},
    }
    calls = [
        {"GD_ID": "dup_best", "svtype": "DUP", "is_carrier": True, "intervals": ["B"], "start": 200, "end": 300},
        {"GD_ID": "dup_worse", "svtype": "DUP", "is_carrier": True, "intervals": ["A", "B"], "start": 100, "end": 300},
    ]

    best = determine_best_breakpoints(FakeLocus(), interval_stats, calls)

    assert best == {"DEL": None, "DUP": "dup_best"}


def test_read_data_renames_chr_builds_bin_index_and_sets_source_file(tmp_path):
    source_dir = tmp_path / "panel_run" / "counts"
    source_dir.mkdir(parents=True)
    input_path = source_dir / "bins.tsv.gz"
    pd.DataFrame(
        {
            "#Chr": ["chr1", "chr2"],
            "Start": [100, 200],
            "End": [150, 260],
            "sample1": [1.0, 2.0],
        }
    ).to_csv(input_path, sep="\t", index=False, compression="gzip")

    loaded = read_data(str(input_path))

    assert loaded.index.tolist() == ["chr1:100-150", "chr2:200-260"]
    assert loaded["Chr"].tolist() == ["chr1", "chr2"]
    assert loaded["source_file"].tolist() == ["panel_run", "panel_run"]


def test_read_data_prints_error_and_reraises_on_invalid_input(tmp_path, capsys):
    input_path = tmp_path / "bins.tsv.gz"
    input_path.write_text("not\ta\tvalid\tgzip\n", encoding="utf-8")

    with pytest.raises(Exception):
        read_data(str(input_path))

    out = capsys.readouterr().out
    assert "Loading binned read-count data" in out
    assert "Error loading binned read-count data" in out


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


def test_filter_low_quality_bins_reports_no_informative_and_verbose_breakdown(monkeypatch, capsys):
    df = pd.DataFrame(
        {
            "Chr": ["chr1", "chr1"],
            "Start": [100, 200],
            "End": [150, 250],
            "sample_1": [0.1, 0.2],
        }
    )

    monkeypatch.setattr(
        bins_module,
        "compute_bin_quality_mask",
        lambda *args, **kwargs: (
            np.asarray([True, False], dtype=bool),
            {
                "filtered": 1,
                "par_ignored": 0,
                "hard_included": 0,
                "no_informative": 2,
            },
        ),
    )
    monkeypatch.setattr(bins_module._util, "VERBOSE", True)

    filtered = filter_low_quality_bins(df)

    out = capsys.readouterr().out
    assert "Bins with no informative samples: 2" in out
    assert "[verbose] Filter breakdown:" in out
    assert "bins without informative samples: 2" in out
    assert filtered["Start"].tolist() == [100]


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
