import sys

import pandas as pd
import matplotlib.pyplot as plt
import pytest
import numpy as np
from matplotlib.collections import LineCollection

from gatk_sv_gd.models import GDLocus
from gatk_sv_gd.annotations import FlankCompressor
from gatk_sv_gd._util import posterior_called_state_to_qual
from gatk_sv_gd.plot import (
    ViterbiOverlayData,
    _apply_carrier_pdf_x_axis_layout,
    _build_anomalous_pdf_specs,
    _build_eval_pdf_specs,
    _build_raw_region_df,
    _coarsen_pdf_page_signals,
    _plot_baf_signal_panel,
    _plot_event_marginal_panel,
    _rebin_aligned_region_dfs_for_display,
    parse_args,
)


def _make_locus() -> GDLocus:
    return GDLocus(
        cluster="10q11.2",
        chrom="chr10",
        breakpoints=[(46005406, 46005406), (48181660, 48181660), (49845537, 49845537), (50651802, 50651802)],
        breakpoint_names=["A", "C", "D", "E"],
        gd_entries=[],
        is_nahr=True,
        is_terminal=False,
    )


def _make_eval_locus() -> GDLocus:
    locus = _make_locus()
    locus.gd_entries = [
        {
            "GD_ID": "GD1",
            "svtype": "DEL",
            "start_GRCh38": 46005406,
            "end_GRCh38": 49845537,
        }
    ]
    return locus


def test_parse_args_allows_skipping_locus_plots(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gatk-sv-gd plot",
            "--calls",
            "calls.tsv.gz",
            "--cn-posteriors",
            "cn.tsv.gz",
            "--gd-table",
            "gd.tsv",
            "--output-dir",
            "out",
            "--skip-locus-plots",
        ],
    )

    args = parse_args()

    assert args.skip_locus_plots is True


def test_build_eval_pdf_specs_rejects_missing_eval_sample_columns():
    calls_df = pd.DataFrame(
        {
            "sample": ["S1"],
            "cluster": ["10q11.2"],
            "GD_ID": ["GD1"],
            "is_carrier": [True],
            "is_best_match": [True],
            "qual_score": [20.0],
        }
    )
    eval_report_df = pd.DataFrame(
        {
            "GD_ID": ["GD1"],
            "TP": [1],
            "FP_samples": [""],
            "FN_samples": [""],
        }
    )

    with pytest.raises(ValueError, match="TP_samples"):
        _build_eval_pdf_specs(
            eval_report_df,
            calls_df,
            {"10q11.2": _make_eval_locus()},
        )


def test_build_eval_pdf_specs_uses_reported_tp_samples(capsys):
    calls_df = pd.DataFrame(
        {
            "sample": ["S1"],
            "cluster": ["10q11.2"],
            "GD_ID": ["GD1"],
            "is_carrier": [True],
            "is_best_match": [True],
            "qual_score": [20.0],
        }
    )
    eval_report_df = pd.DataFrame(
        {
            "GD_ID": ["GD1"],
            "TP": [1],
            "TP_samples": ["S1"],
            "FP_samples": [""],
            "FN_samples": [""],
        }
    )

    specs = _build_eval_pdf_specs(
        eval_report_df,
        calls_df,
        {"10q11.2": _make_eval_locus()},
    )

    assert [spec["sample"] for spec in specs["true_positives"]] == ["S1"]
    assert "derived TP count" not in capsys.readouterr().out


def test_build_eval_pdf_specs_uses_flagged_anomalous_calls():
    calls_df = pd.DataFrame(
        {
            "sample": ["S1"],
            "cluster": ["10q11.2"],
            "GD_ID": ["GD1"],
            "is_carrier": [True],
            "is_best_match": [True],
            "is_null_anomalous": [True],
            "qual_score": [20.0],
        }
    )
    eval_report_df = pd.DataFrame(
        {
            "GD_ID": ["GD1"],
            "TP": [1],
            "TP_samples": ["S1"],
            "FP_samples": [""],
            "FN_samples": [""],
        }
    )

    specs = _build_eval_pdf_specs(
        eval_report_df,
        calls_df,
        {"10q11.2": _make_eval_locus()},
    )

    assert [
        spec["sample"] for spec in specs["anomalous_discrepancies"]
    ] == ["S1"]


def test_build_anomalous_pdf_specs_works_without_eval_report_rows():
    calls_df = pd.DataFrame(
        {
            "sample": ["S1", "S2"],
            "cluster": ["10q11.2", "10q11.2"],
            "GD_ID": ["GD1", "GD1"],
            "is_best_match": [True, False],
            "is_null_anomalous": [True, True],
            "qual_score": [20.0, 20.0],
        }
    )

    specs = _build_anomalous_pdf_specs(
        calls_df,
        {"10q11.2": _make_eval_locus()},
    )

    assert [spec["sample"] for spec in specs] == ["S1"]


def test_viterbi_overlay_data_rejects_legacy_schema():
    paths_df = pd.DataFrame(
        {
            "sample": ["S1"],
            "cluster": ["10q11.2"],
            "start": [1],
            "end": [2],
            "mean_cn": [2.0],
        }
    )

    with pytest.raises(ValueError, match="cn_state"):
        ViterbiOverlayData(paths_df)


def test_build_raw_region_df_keeps_lowres_when_processed_counts_do_not_exceed_it():
    locus = _make_locus()
    raw_counts_df = pd.DataFrame(
        {
            "Chr": ["chr10"] * 4,
            "Start": [46005406, 47000000, 48181660, 49000000],
            "End": [47000000, 48181660, 49000000, 49845537],
            "sample_1": [20.0, 22.0, 18.0, 21.0],
        }
    )
    processed_region_df = pd.DataFrame(
        {
            "Chr": ["chr10"] * 4,
            "Start": [46005406, 47000000, 48181660, 49000000],
            "End": [47000000, 48181660, 49000000, 49845537],
            "sample_1": [2.0, 2.1, 1.9, 2.0],
        }
    )

    result = _build_raw_region_df(
        locus,
        46005406,
        49845537,
        raw_counts_df,
        ["sample_1"],
        {"sample_1": 20.0},
        100000.0,
        processed_region_df=processed_region_df,
        highres_path=None,
    )

    assert result is not None
    assert len(result) == 4
    assert result["Start"].tolist() == raw_counts_df["Start"].tolist()


def test_build_raw_region_df_uses_full_region_highres_when_processed_bins_exceed_lowres(monkeypatch):
    locus = _make_locus()
    raw_counts_df = pd.DataFrame(
        {
            "Chr": ["chr10"] * 3,
            "Start": [46005406, 47000000, 48181660],
            "End": [47000000, 48181660, 49845537],
            "sample_1": [20.0, 22.0, 18.0],
        }
    )
    processed_region_df = pd.DataFrame(
        {
            "Chr": ["chr10"] * 5,
            "Start": [46005406, 46500000, 47000000, 48181660, 49000000],
            "End": [46500000, 47000000, 48181660, 49000000, 49845537],
            "sample_1": [2.0, 2.0, 2.1, 1.9, 2.0],
        }
    )

    calls = []

    def fake_query_highres_bins(highres_path, chrom, start, end, sample_cols, max_bins=None):
        calls.append((highres_path, chrom, start, end, tuple(sample_cols), max_bins))
        return pd.DataFrame(
            {
                "Chr": [chrom, chrom],
                "Start": [start, start + 100],
                "End": [start + 100, end],
                "source_file": ["highres", "highres"],
                "sample_1": [10.0, 12.0],
            },
            index=[f"{chrom}:{start}-{start + 100}", f"{chrom}:{start + 100}-{end}"],
        )

    def fake_normalize_highres_bins(highres_df, sample_cols, column_medians, lowres_median_bin_size):
        df = highres_df.copy()
        df["sample_1"] = [2.5, 2.7]
        return df

    monkeypatch.setattr("gatk_sv_gd.plot.query_highres_bins", fake_query_highres_bins)
    monkeypatch.setattr("gatk_sv_gd.plot.normalize_highres_bins", fake_normalize_highres_bins)

    result = _build_raw_region_df(
        locus,
        46005406,
        49845537,
        raw_counts_df,
        ["sample_1"],
        {"sample_1": 20.0},
        100000.0,
        processed_region_df=processed_region_df,
        highres_path="/tmp/highres.tsv.gz",
    )

    assert calls == [("/tmp/highres.tsv.gz", "chr10", 46005406, 49845537, ("sample_1",), None)]
    assert result is not None
    assert result["Start"].tolist() == [46005406, 46005506]
    assert result["sample_1"].tolist() == [2.5, 2.7]


def test_build_raw_region_df_coarsens_highres_before_normalization(monkeypatch):
    locus = _make_locus()
    raw_counts_df = pd.DataFrame(
        {
            "Chr": ["chr10"],
            "Start": [46005406],
            "End": [49845537],
            "sample_1": [20.0],
        }
    )
    processed_region_df = pd.DataFrame(
        {
            "Chr": ["chr10"] * 6,
            "Start": [46005406, 46400000, 46800000, 48181660, 49000000, 49400000],
            "End": [46400000, 46800000, 48181660, 49000000, 49400000, 49845537],
            "sample_1": [2.0] * 6,
        }
    )

    observed = []

    def fake_query_highres_bins(highres_path, chrom, start, end, sample_cols, max_bins=None):
        observed.append((chrom, start, end, tuple(sample_cols), max_bins))
        return pd.DataFrame(
            {
                "Chr": [chrom] * 3,
                "Start": [start, start + 300, start + 600],
                "End": [start + 300, start + 600, end],
                "source_file": ["highres"] * 3,
                "sample_1": [30.0, 40.0, 50.0],
            },
            index=[f"{chrom}:{start}-{start + 300}", f"{chrom}:{start + 300}-{start + 600}", f"{chrom}:{start + 600}-{end}"],
        )

    def fake_normalize_highres_bins(highres_df, sample_cols, column_medians, lowres_median_bin_size):
        df = highres_df.copy()
        df["sample_1"] = df["sample_1"] / 10.0
        return df

    monkeypatch.setattr("gatk_sv_gd.plot.query_highres_bins", fake_query_highres_bins)
    monkeypatch.setattr("gatk_sv_gd.plot.normalize_highres_bins", fake_normalize_highres_bins)

    result = _build_raw_region_df(
        locus,
        46005406,
        49845537,
        raw_counts_df,
        ["sample_1"],
        {"sample_1": 20.0},
        100000.0,
        processed_region_df=processed_region_df,
        highres_path="/tmp/highres.tsv.gz",
        highres_max_bins=3,
    )

    assert observed == [("chr10", 46005406, 49845537, ("sample_1",), 3)]
    assert result is not None
    assert len(result) == 3


def test_plot_baf_signal_panel_draws_confidence_intervals_for_valid_points():
    locus = _make_locus()
    x_positions = pd.Series([46050000, 48190000, 49800000], dtype=float).to_numpy()
    bar_widths = pd.Series([0.001, 0.001, 0.001], dtype=float).to_numpy()
    minor_baf_values = pd.Series([0.45, 0.40, 0.49], dtype=float).to_numpy()
    baf_variances = pd.Series([0.0004, 0.0009, 0.0001], dtype=float).to_numpy()
    baf_site_counts = pd.Series([20, 15, 30], dtype=int).to_numpy()
    xform = FlankCompressor(46005406, 49845537, locus.start, locus.end, flank_scale=0.2)

    fig, ax = plt.subplots()
    try:
        _plot_baf_signal_panel(
            ax,
            x_positions,
            bar_widths,
            minor_baf_values,
            baf_site_counts,
            xform,
            locus,
            locus.chrom,
            baf_variances=baf_variances,
        )

        line_collections = [
            collection for collection in ax.collections
            if isinstance(collection, LineCollection)
        ]
        interval_segments = None
        for collection in line_collections:
            segments = collection.get_segments()
            if len(segments) == 3:
                interval_segments = np.asarray(segments, dtype=float)
                break

        assert len(ax.patches) == 3
        assert interval_segments is not None
        expected_lower = np.clip(
            minor_baf_values - (1.959963984540054 * np.sqrt(baf_variances)),
            0.0,
            0.5,
        )
        expected_upper = np.clip(
            minor_baf_values + (1.959963984540054 * np.sqrt(baf_variances)),
            0.0,
            0.5,
        )
        assert interval_segments[:, 0, 1].tolist() == pytest.approx(expected_lower.tolist())
        assert interval_segments[:, 1, 1].tolist() == pytest.approx(expected_upper.tolist())
        assert expected_upper[-1] == pytest.approx(0.5)
    finally:
        plt.close(fig)


def test_plot_baf_signal_panel_uses_fixed_minor_baf_reference_lines():
    locus = _make_locus()
    x_positions = pd.Series([46050000, 48190000, 49800000], dtype=float).to_numpy()
    bar_widths = pd.Series([0.001, 0.001, 0.001], dtype=float).to_numpy()
    minor_baf_values = pd.Series([0.45, 0.40, 0.49], dtype=float).to_numpy()
    baf_variances = pd.Series([0.0004, 0.0009, 0.0001], dtype=float).to_numpy()
    baf_site_counts = pd.Series([20, 15, 30], dtype=int).to_numpy()
    xform = FlankCompressor(46005406, 49845537, locus.start, locus.end, flank_scale=0.2)

    fig, ax = plt.subplots()
    try:
        _plot_baf_signal_panel(
            ax,
            x_positions,
            bar_widths,
            minor_baf_values,
            baf_site_counts,
            xform,
            locus,
            locus.chrom,
            baf_temperature=25.0,
            baf_variances=baf_variances,
        )

        observed_reference_levels = [
            float(line.get_ydata()[0])
            for line in ax.lines
            if line.get_linestyle() == ":"
        ]

        assert observed_reference_levels == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    finally:
        plt.close(fig)


def test_coarsen_pdf_page_signals_aggregates_baf_intervals_with_site_weights():
    locus = GDLocus(
        cluster="left-flank-only",
        chrom="chr1",
        breakpoints=[(100, 100), (200, 200)],
        breakpoint_names=["A", "B"],
        gd_entries=[],
        is_nahr=False,
        is_terminal=False,
    )
    region_df = pd.DataFrame(
        {
            "Cluster": [locus.cluster] * 4,
            "Chr": [locus.chrom] * 4,
            "Start": [0, 10, 20, 30],
            "End": [10, 20, 30, 40],
        }
    )

    plot_region_df, plot_depth, plot_minor_baf, plot_baf_variance, plot_baf_sites, plot_event_probs = _coarsen_pdf_page_signals(
        locus,
        region_df,
        np.array([2.0, 2.0, 2.0, 2.0], dtype=float),
        np.array([0.2, 0.4, 0.3, 0.5], dtype=float),
        np.array([0.04, 0.01, 0.09, 0.16], dtype=float),
        np.array([2.0, 6.0, 3.0, 1.0], dtype=float),
        None,
        max_total_bins=2,
    )

    assert len(plot_region_df) == 2
    assert plot_depth.tolist() == pytest.approx([2.0, 2.0])
    assert plot_minor_baf.tolist() == pytest.approx([0.35, 0.35])
    assert plot_baf_variance.tolist() == pytest.approx([0.008125, 0.060625])
    assert plot_baf_sites.tolist() == pytest.approx([8.0, 4.0])
    assert plot_event_probs is None


def test_plot_baf_signal_panel_omits_baf_temperature_title():
    locus = _make_locus()
    x_positions = pd.Series([46050000, 48190000], dtype=float).to_numpy()
    bar_widths = pd.Series([0.001, 0.001], dtype=float).to_numpy()
    minor_baf_values = pd.Series([0.45, 0.40], dtype=float).to_numpy()
    baf_site_counts = pd.Series([20, 15], dtype=int).to_numpy()
    xform = FlankCompressor(46005406, 49845537, locus.start, locus.end, flank_scale=0.2)

    fig, ax = plt.subplots()
    try:
        _plot_baf_signal_panel(
            ax,
            x_positions,
            bar_widths,
            minor_baf_values,
            baf_site_counts,
            xform,
            locus,
            locus.chrom,
            baf_temperature=25.0,
        )

        assert ax.get_title() == ""
    finally:
        plt.close(fig)


def test_apply_carrier_pdf_x_axis_layout_moves_labels_to_annotation_panel():
    locus = _make_locus()
    xform = FlankCompressor(46005406, 49845537, locus.start, locus.end, flank_scale=0.2)

    fig, axes = plt.subplots(4, 1)
    try:
        _apply_carrier_pdf_x_axis_layout(axes, xform, locus)

        assert axes[0].xaxis.get_ticks_position() == "top"
        assert any(label.get_text() for label in axes[0].get_xticklabels())

        for ax in axes[1:]:
            assert ax.get_xlabel() == ""
            assert list(ax.get_xticks()) == []
    finally:
        plt.close(fig)


def test_flank_compressor_limits_minor_ticks_for_tiny_body_regions():
    xform = FlankCompressor(0, 100000, 50000, 50001, flank_scale=0.2)

    fig, ax = plt.subplots()
    try:
        xform.format_genomic_ticks(ax, breakpoints=[(50000, 50001)], label_x=True)

        assert len(ax.get_xticks(minor=True)) <= 50
    finally:
        plt.close(fig)


def test_plot_event_marginal_panel_uses_called_state_qual_scale():
    locus = _make_locus()
    x_positions = pd.Series([46050000, 48190000, 49800000], dtype=float).to_numpy()
    bar_widths = pd.Series([0.001, 0.001, 0.001], dtype=float).to_numpy()
    event_probabilities = pd.Series([0.90, 0.10, 1.0], dtype=float).to_numpy()
    called_event_mask = np.array([True, False, True], dtype=bool)
    xform = FlankCompressor(46005406, 49845537, locus.start, locus.end, flank_scale=0.2)

    fig, ax = plt.subplots()
    try:
        _plot_event_marginal_panel(
            ax,
            x_positions,
            bar_widths,
            event_probabilities,
            xform,
            locus,
            locus.chrom,
            "DEL",
            called_event_mask=called_event_mask,
        )

        assert ax.get_ylim() == (-99.0, 99.0)
        assert ax.get_ylabel() == "QUAL(DEL)"
        assert len(ax.lines) == 1
        assert list(ax.lines[0].get_ydata()) == pytest.approx(
            posterior_called_state_to_qual(
                event_probabilities,
                called_event_mask,
            ).tolist()
        )
    finally:
        plt.close(fig)


def test_plot_event_marginal_panel_accepts_precomputed_event_qual_values():
    locus = _make_locus()
    x_positions = pd.Series([46050000, 48190000, 49800000], dtype=float).to_numpy()
    bar_widths = pd.Series([0.001, 0.001, 0.001], dtype=float).to_numpy()
    event_quals = pd.Series(
        [10.0 * np.log10(9.0), -10.0 * np.log10(9.0), 99.0],
        dtype=float,
    ).to_numpy()
    called_event_mask = np.array([True, False, True], dtype=bool)
    xform = FlankCompressor(46005406, 49845537, locus.start, locus.end, flank_scale=0.2)

    fig, ax = plt.subplots()
    try:
        _plot_event_marginal_panel(
            ax,
            x_positions,
            bar_widths,
            event_quals,
            xform,
            locus,
            locus.chrom,
            "DEL",
            called_event_mask=called_event_mask,
            values_are_qual=True,
        )

        assert ax.get_ylim() == (-99.0, 99.0)
        assert ax.get_ylabel() == "QUAL(DEL)"
        assert len(ax.lines) == 1
        assert list(ax.lines[0].get_ydata()) == pytest.approx([
            10.0 * np.log10(9.0),
            10.0 * np.log10(9.0),
            99.0,
        ])
    finally:
        plt.close(fig)


def test_rebin_aligned_region_dfs_for_display_caps_total_bins_and_preserves_alignment():
    locus = GDLocus(
        cluster="1q21",
        chrom="chr1",
        breakpoints=[(145700000, 145700000), (146000000, 146000000), (147000000, 147000000), (148400000, 148400000)],
        breakpoint_names=["1", "2", "3", "4"],
        gd_entries=[],
        is_nahr=True,
        is_terminal=False,
    )

    starts = list(range(145600000, 145600000 + 509 * 1000, 1000))
    ends = [start + 1000 for start in starts]
    region_df = pd.DataFrame(
        {
            "Cluster": [locus.cluster] * len(starts),
            "Chr": [locus.chrom] * len(starts),
            "Start": starts,
            "End": ends,
            "sample_1": [2.0] * len(starts),
        }
    )
    baf_df = region_df.copy()
    baf_df["sample_1"] = [0.4] * len(starts)
    event_df = region_df.copy()
    event_df["sample_1"] = [0.8] * len(starts)

    rebinned_region, rebinned_frames = _rebin_aligned_region_dfs_for_display(
        locus,
        region_df,
        [baf_df, event_df],
        max_total_bins=300,
    )

    rebinned_baf, rebinned_event = rebinned_frames

    assert len(rebinned_region) <= 300
    assert len(rebinned_region) == len(rebinned_baf) == len(rebinned_event)
    assert rebinned_region[["Cluster", "Chr", "Start", "End"]].equals(
        rebinned_baf[["Cluster", "Chr", "Start", "End"]]
    )
    assert rebinned_region[["Cluster", "Chr", "Start", "End"]].equals(
        rebinned_event[["Cluster", "Chr", "Start", "End"]]
    )
