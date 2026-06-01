import sys
from types import SimpleNamespace

import pandas as pd
import matplotlib.pyplot as plt
import pytest
import numpy as np
from matplotlib.collections import LineCollection

from gatk_sv_gd.models import GDLocus
from gatk_sv_gd.annotations import FlankCompressor
from gatk_sv_gd._util import posterior_called_state_to_qual
from gatk_sv_gd import plot as plot_module
from gatk_sv_gd.plot import (
    _apply_carrier_pdf_x_axis_layout,
    _aligned_region_sample_vector,
    _build_anomalous_pdf_specs,
    _build_eval_pdf_specs,
    _build_called_event_mask,
    _build_carrier_best_match_mask,
    _build_gd_to_cluster_map,
    _build_raw_region_df,
    _allocate_segment_bin_targets,
    _coarsen_pdf_page_signals,
    _compute_raw_sample_medians,
    _create_review_category_pdf,
    create_anomalous_pdf,
    create_carrier_pdf,
    create_eval_category_pdfs,
    _extract_sample_event_probabilities,
    _get_event_probability_column,
    _get_confidence_column,
    _get_confidence_label,
    _get_locus_gd_entry,
    _minor_baf_reference_levels,
    _parse_eval_sample_list,
    _plot_depth_bars_with_baf,
    _plot_baf_signal_panel,
    _plot_event_marginal_panel,
    _rebin_aligned_region_dfs_for_display,
    _render_pdf_sample_page,
    _rebin_region_df,
    _select_baf_plot_support_columns,
    _sanitize_plot_label,
    estimate_lowres_bin_size,
    main as plot_main,
    parse_args,
    plot_carrier_summary,
    plot_confidence_distribution,
    plot_locus_overview,
    plot_sample_at_locus,
)


class _StubPdfPages:
    def __init__(self, path):
        self.path = path
        self.saved_figures = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def savefig(self, fig):
        self.saved_figures.append(fig)


class _RecordingAxis:
    def __init__(self):
        self.calls = []

    def bar(self, *args, **kwargs):
        self.calls.append((args, kwargs))


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


@pytest.mark.parametrize(
    ("minor_baf_values", "baf_site_counts"),
    [
        (None, None),
        ([np.nan, 0.2], [1, 0]),
    ],
)
def test_plot_depth_bars_with_baf_falls_back_to_single_depth_series(
    minor_baf_values,
    baf_site_counts,
):
    ax = _RecordingAxis()

    _plot_depth_bars_with_baf(
        ax,
        np.array([10.0, 20.0]),
        np.array([4.0, 6.0]),
        np.array([2.0, 3.0]),
        minor_baf_values=minor_baf_values,
        baf_site_counts=baf_site_counts,
        zorder=7,
        rasterized=True,
    )

    assert len(ax.calls) == 1
    args, kwargs = ax.calls[0]
    assert np.array_equal(args[0], np.array([10.0, 20.0]))
    assert np.array_equal(args[1], np.array([2.0, 3.0]))
    assert np.array_equal(kwargs["width"], np.array([3.6, 5.4]))
    assert kwargs["color"] == "steelblue"
    assert kwargs["label"] == "Normalized depth"
    assert kwargs["zorder"] == 7
    assert kwargs["rasterized"] is True


def test_plot_depth_bars_with_baf_splits_valid_bins_and_keeps_invalid_depth_gray():
    ax = _RecordingAxis()

    _plot_depth_bars_with_baf(
        ax,
        np.array([10.0, 20.0, 30.0]),
        np.array([10.0, 10.0, 10.0]),
        np.array([4.0, 6.0, 8.0]),
        minor_baf_values=np.array([0.25, 0.8, np.nan]),
        baf_site_counts=np.array([5, 2, 0]),
    )

    assert len(ax.calls) == 3

    invalid_args, invalid_kwargs = ax.calls[0]
    assert np.array_equal(invalid_args[0], np.array([30.0]))
    assert np.array_equal(invalid_args[1], np.array([8.0]))
    assert np.array_equal(invalid_kwargs["width"], np.array([9.0]))
    assert invalid_kwargs["color"] == "gray"
    assert invalid_kwargs["label"] == "Depth (no BAF)"

    minor_args, minor_kwargs = ax.calls[1]
    assert np.array_equal(minor_args[0], np.array([10.0, 20.0]))
    assert np.allclose(minor_args[1], np.array([1.0, 3.0]))
    assert np.array_equal(minor_kwargs["width"], np.array([9.0, 9.0]))
    assert minor_kwargs["color"] == "#F4A261"
    assert minor_kwargs["label"] == "Minor-allele depth"

    major_args, major_kwargs = ax.calls[2]
    assert np.array_equal(major_args[0], np.array([10.0, 20.0]))
    assert np.allclose(major_args[1], np.array([3.0, 3.0]))
    assert np.allclose(major_kwargs["bottom"], np.array([1.0, 3.0]))
    assert np.array_equal(major_kwargs["width"], np.array([9.0, 9.0]))
    assert major_kwargs["color"] == "#4C78A8"
    assert major_kwargs["label"] == "Major-allele depth"


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


def test_plot_helpers_cover_confidence_labels_masks_and_medians():
    qual_df = pd.DataFrame({"qual_score": [10.0]})
    confidence_df = pd.DataFrame({"confidence_score": [5.0]})
    log_prob_df = pd.DataFrame({"log_prob_score": [1.0]})

    assert _get_confidence_column(qual_df) == "qual_score"
    assert _get_confidence_column(confidence_df) == "confidence_score"
    assert _get_confidence_column(log_prob_df) == "log_prob_score"
    assert _get_confidence_label("qual_score") == "Call QUAL"
    assert _get_confidence_label("confidence_score") == "Confidence Score"
    assert _get_confidence_label("log_prob_score") == "Log Probability Score"
    assert _sanitize_plot_label("chr1:10-20/sample") == "chr1_10_20_sample"

    with pytest.raises(ValueError, match="missing 'qual_score'"):
        _get_confidence_column(pd.DataFrame({"other": [1]}))

    calls_df = pd.DataFrame(
        {
            "is_carrier": [True, True, False],
            "is_best_match": [True, False, True],
        }
    )
    assert _build_carrier_best_match_mask(calls_df).tolist() == [True, False, False]
    assert _build_carrier_best_match_mask(pd.DataFrame({"is_carrier": [True, False]})).tolist() == [True, False]

    raw_counts_df = pd.DataFrame(
        {
            "Chr": ["chr1", "chr2", "chrX"],
            "Start": [0, 10, 20],
            "End": [10, 20, 30],
            "S1": [2.0, 6.0, 100.0],
            "S2": [0.0, 0.0, 50.0],
        }
    )
    assert _compute_raw_sample_medians(raw_counts_df) == {"S1": 4.0}
    assert _compute_raw_sample_medians(None) == {}


def test_plot_helpers_cover_eval_parsing_gd_lookup_and_bin_size():
    assert _parse_eval_sample_list(np.nan) == []
    assert _parse_eval_sample_list("") == []
    assert _parse_eval_sample_list(" S1, S2 ,,S3 ") == ["S1", "S2", "S3"]

    locus1 = _make_eval_locus()
    locus2 = GDLocus(
        cluster="16p11.2",
        chrom="chr16",
        breakpoints=[(29000000, 29000000), (30100000, 30100000)],
        breakpoint_names=["BP4", "BP5"],
        gd_entries=[
            {
                "GD_ID": "GD2",
                "svtype": "DUP",
                "start_GRCh38": 29000000,
                "end_GRCh38": 30100000,
            }
        ],
        is_nahr=True,
        is_terminal=False,
    )

    gd_to_cluster = _build_gd_to_cluster_map({"10q11.2": locus1, "16p11.2": locus2})
    assert gd_to_cluster == {"GD1": "10q11.2", "GD2": "16p11.2"}
    assert _get_locus_gd_entry(locus1, "GD1")["svtype"] == "DEL"
    assert _get_locus_gd_entry(locus1, None) is None
    assert _get_locus_gd_entry(locus1, "missing") is None

    df = pd.DataFrame({"Start": [0, 10, 25], "End": [10, 25, 55]})
    assert estimate_lowres_bin_size(df) == 15.0


def test_plot_helpers_cover_event_probability_alignment_and_called_masks():
    assert _get_event_probability_column("DEL") == "prob_del_event"
    assert _get_event_probability_column("DUP") == "prob_dup_event"
    with pytest.raises(ValueError, match="Unsupported svtype"):
        _get_event_probability_column("CNV")

    region_df = pd.DataFrame(
        {
            "Cluster": ["c1", "c1", "c1"],
            "Chr": ["chr1", "chr1", "chr1"],
            "Start": [100, 200, 300],
            "End": [150, 250, 350],
        }
    )
    value_df = pd.DataFrame(
        {
            "Cluster": ["c1", "c1", "c1"],
            "Chr": ["chr1", "chr1", "chr1"],
            "Start": [100, 200, 300],
            "End": [150, 250, 350],
            "S1": [0.1, 0.2, 0.3],
        }
    )
    shuffled_value_df = value_df.iloc[[2, 0, 1]].reset_index(drop=True)

    assert np.allclose(_aligned_region_sample_vector(region_df, value_df, "S1"), [0.1, 0.2, 0.3])
    assert np.allclose(_aligned_region_sample_vector(region_df, shuffled_value_df, "S1"), [0.1, 0.2, 0.3])
    assert _aligned_region_sample_vector(region_df, None, "S1") is None
    assert _aligned_region_sample_vector(region_df, value_df, "S9") is None
    assert np.allclose(_extract_sample_event_probabilities(region_df, value_df, "S1"), [0.1, 0.2, 0.3])

    assert np.allclose(_minor_baf_reference_levels(), [0.1, 0.2, 0.3, 0.4, 0.5])
    assert _build_called_event_mask(region_df.iloc[0:0], 100, 200) is None
    assert _build_called_event_mask(region_df, None, 200) is None
    assert _build_called_event_mask(region_df, 150, None) is None
    assert _build_called_event_mask(region_df, 150, 320).tolist() == [False, True, True]
    assert _build_called_event_mask(region_df, 150, 200).tolist() == [False, False, False]


def test_plot_helpers_cover_baf_support_column_selection():
    assert _select_baf_plot_support_columns(
        pd.DataFrame(columns=["baf_effective_variance", "baf_effective_n_sites"])
    ) == ("baf_effective_variance", "baf_effective_n_sites")
    assert _select_baf_plot_support_columns(
        pd.DataFrame(columns=["baf_variance", "baf_n_sites"])
    ) == ("baf_variance", "baf_n_sites")
    assert _select_baf_plot_support_columns(pd.DataFrame(columns=["baf_n_sites"])) == (None, "baf_n_sites")
    assert _select_baf_plot_support_columns(pd.DataFrame(columns=["other"])) == (None, None)


def test_plot_depth_bars_with_baf_handles_missing_and_split_baf_cases():
    fig, axes = plt.subplots(1, 2)
    try:
        x_positions = np.array([1.0, 2.0, 3.0], dtype=float)
        bar_widths = np.array([0.8, 0.8, 0.8], dtype=float)
        depth_values = np.array([2.0, 4.0, 6.0], dtype=float)

        _plot_depth_bars_with_baf(axes[0], x_positions, bar_widths, depth_values)
        assert len(axes[0].patches) == 3

        _plot_depth_bars_with_baf(
            axes[1],
            x_positions,
            bar_widths,
            depth_values,
            minor_baf_values=np.array([0.25, np.nan, 0.5], dtype=float),
            baf_site_counts=np.array([10, 0, 8], dtype=float),
        )
        assert len(axes[1].patches) == 5
    finally:
        plt.close(fig)


def test_plot_event_marginal_panel_handles_empty_invalid_and_mismatched_inputs():
    locus = _make_locus()
    x_positions = np.array([46050000.0, 48190000.0], dtype=float)
    bar_widths = np.array([0.001, 0.001], dtype=float)
    xform = FlankCompressor(46005406, 49845537, locus.start, locus.end, flank_scale=0.2)

    fig, axes = plt.subplots(1, 3)
    try:
        _plot_event_marginal_panel(
            axes[0],
            x_positions,
            bar_widths,
            None,
            xform,
            locus,
            locus.chrom,
            "DEL",
        )
        assert axes[0].texts[0].get_text() == "No event marginals available"
        assert axes[0].get_ylabel() == "Called-state QUAL"

        _plot_event_marginal_panel(
            axes[1],
            x_positions,
            bar_widths,
            np.array([np.nan, np.nan]),
            xform,
            locus,
            locus.chrom,
            "DUP",
            show_trace=False,
        )
        assert axes[1].texts[0].get_text() == "No finite event marginals"
        assert axes[1].get_ylabel() == "QUAL(DUP on ≥1 hap)"

        with pytest.raises(ValueError, match="called_event_mask must align"):
            _plot_event_marginal_panel(
                axes[2],
                x_positions,
                bar_widths,
                np.array([0.8, 0.2]),
                xform,
                locus,
                locus.chrom,
                "DEL",
                called_event_mask=np.array([True]),
            )
    finally:
        plt.close(fig)


def test_create_review_category_pdf_returns_early_without_pages(monkeypatch):
    opened_paths = []
    monkeypatch.setattr(plot_module, "PdfPages", lambda path: opened_paths.append(path) or _StubPdfPages(path))

    _create_review_category_pdf(
        category_name="empty",
        page_specs=[],
        pdf_path="out.pdf",
        calls_df=pd.DataFrame(),
        depth_df=pd.DataFrame(),
        loci_by_cluster={},
        gtf=None,
        segdup=None,
        event_del_df=None,
        event_dup_df=None,
        minor_baf_df=None,
        baf_variance_df=None,
        baf_sites_df=None,
        gaps=None,
        raw_counts_df=None,
        lowres_median_bin_size=None,
        highres_path=None,
        flank_scale=0.2,
        min_gene_label_spacing=0.05,
        event_values_are_qual=False,
        baf_temperature_by_sample=None,
    )

    assert opened_paths == []


def test_create_review_category_pdf_builds_cluster_pages_and_raw_region(monkeypatch, tmp_path):
    render_calls = []
    monkeypatch.setattr(plot_module, "PdfPages", _StubPdfPages)
    monkeypatch.setattr(
        plot_module,
        "_render_pdf_sample_page",
        lambda pdf, sample_id, cluster, locus, region_df, cluster_calls_df, confidence_column, *args, **kwargs: render_calls.append(
            {
                "pdf": pdf,
                "sample": sample_id,
                "cluster": cluster,
                "locus": locus.cluster,
                "region_rows": len(region_df),
                "cluster_call_rows": len(cluster_calls_df),
                "confidence_column": confidence_column,
                "raw_region_present": args[4] is not None,
                "target_gd_id": kwargs.get("target_gd_id"),
                "title_suffix": kwargs.get("title_suffix"),
            }
        ) or True,
    )
    monkeypatch.setattr(
        plot_module,
        "_build_raw_region_df",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "Cluster": ["10q11.2", "10q11.2"],
                "Chr": ["chr10", "chr10"],
                "Start": [46005406, 48181660],
                "End": [48181660, 49845537],
                "S1": [2.0, 2.1],
            }
        ),
    )
    monkeypatch.setattr(plot_module, "_rebin_region_df", lambda df, locus: df)

    locus = _make_eval_locus()
    page_specs = [{"cluster": "10q11.2", "sample": "S1", "gd_id": "GD1", "title_suffix": "TP: GD1"}]
    calls_df = pd.DataFrame(
        {
            "cluster": ["10q11.2"],
            "sample": ["S1"],
            "GD_ID": ["GD1"],
            "qual_score": [25.0],
        }
    )
    depth_df = pd.DataFrame(
        {
            "Cluster": ["10q11.2", "10q11.2"],
            "Chr": ["chr10", "chr10"],
            "Start": [46005406, 48181660],
            "End": [48181660, 49845537],
            "S1": [2.0, 1.1],
        }
    )
    raw_counts_df = pd.DataFrame(
        {
            "Chr": ["chr1", "chr2"],
            "Start": [0, 10],
            "End": [10, 20],
            "S1": [10.0, 12.0],
        }
    )
    baf_df = depth_df.copy()
    baf_df["S1"] = [0.4, 0.3]
    event_df = depth_df.copy()
    event_df["S1"] = [0.8, 0.9]

    _create_review_category_pdf(
        "true_positives",
        page_specs,
        str(tmp_path / "review.pdf"),
        calls_df,
        depth_df,
        {"10q11.2": locus},
        None,
        None,
        event_df,
        event_df,
        baf_df,
        baf_df,
        baf_df,
        None,
        raw_counts_df,
        None,
        None,
        0.2,
        0.05,
        False,
        {"S1": 20.0},
    )

    assert len(render_calls) == 1
    assert render_calls[0]["sample"] == "S1"
    assert render_calls[0]["cluster"] == "10q11.2"
    assert render_calls[0]["locus"] == "10q11.2"
    assert render_calls[0]["region_rows"] == 2
    assert render_calls[0]["cluster_call_rows"] == 1
    assert render_calls[0]["confidence_column"] == "qual_score"
    assert render_calls[0]["raw_region_present"] is True
    assert render_calls[0]["target_gd_id"] == "GD1"
    assert render_calls[0]["title_suffix"] == "TP: GD1"


def test_render_pdf_sample_page_handles_missing_sample_and_target_entry_path():
    locus = _make_eval_locus()
    xform = FlankCompressor(46005406, 49845537, locus.start, locus.end, flank_scale=0.2)
    region_df = pd.DataFrame(
        {
            "Cluster": ["10q11.2", "10q11.2"],
            "Chr": ["chr10", "chr10"],
            "Start": [46005406, 48181660],
            "End": [48181660, 49845537],
            "S1": [2.0, 1.2],
        }
    )

    assert _render_pdf_sample_page(
        _StubPdfPages("unused.pdf"),
        "MISSING",
        "10q11.2",
        locus,
        region_df,
        pd.DataFrame(columns=["sample", "cluster", "GD_ID", "svtype", "start", "end", "is_best_match", "qual_score", "mean_depth"]),
        "qual_score",
        None,
        None,
        0.05,
        xform,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ) is False

    pdf = _StubPdfPages("page.pdf")
    result = _render_pdf_sample_page(
        pdf,
        "S1",
        "10q11.2",
        locus,
        region_df,
        pd.DataFrame(columns=["sample", "cluster", "GD_ID", "svtype", "start", "end", "is_best_match", "qual_score", "mean_depth"]),
        "qual_score",
        None,
        None,
        0.05,
        xform,
        region_df.copy(),
        region_df.assign(S1=[0.4, 0.3]),
        region_df.assign(S1=[0.01, 0.02]),
        region_df.assign(S1=[10, 12]),
        region_df.assign(S1=[0.8, 0.9]),
        None,
        None,
        event_values_are_qual=False,
        target_gd_id="GD1",
        title_suffix="Target GD",
    )

    assert result is True
    assert len(pdf.saved_figures) == 1


def test_plot_locus_overview_skips_invalid_loci(capsys, tmp_path):
    no_bp_locus = _make_eval_locus()
    no_bp_locus.breakpoints = []
    plot_locus_overview(
        no_bp_locus,
        pd.DataFrame(),
        pd.DataFrame(),
        None,
        None,
        str(tmp_path),
    )

    plot_locus_overview(
        _make_eval_locus(),
        pd.DataFrame(columns=["cluster", "sample", "is_carrier"]),
        pd.DataFrame(columns=["Cluster", "Chr", "Start", "End", "S1"]),
        None,
        None,
        str(tmp_path),
    )

    out = capsys.readouterr().out
    assert "Warning: no breakpoints defined for one locus; skipping" in out
    assert "Warning: no depth data found for one locus; skipping plot" in out


def test_plot_locus_overview_draws_raw_and_processed_columns(monkeypatch, tmp_path):
    locus = _make_eval_locus()
    draw_calls = []
    timing_calls = []
    saved_paths = []

    class _DummyFigure:
        def subplots_adjust(self, **kwargs):
            return None

        def savefig(self, path, **kwargs):
            saved_paths.append(path)

    def fake_subplots(n_rows, n_cols, **kwargs):
        axes = np.empty((n_rows, n_cols), dtype=object)
        axes[:] = object()
        return _DummyFigure(), axes

    monkeypatch.setattr(plot_module.plt, "subplots", fake_subplots)
    monkeypatch.setattr(plot_module.plt, "tight_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr(plot_module.plt, "close", lambda *args, **kwargs: None)
    monkeypatch.setattr(plot_module, "get_sample_columns", lambda df: ["S1", "S2"])
    monkeypatch.setattr(plot_module, "_build_ploidy_lookup", lambda df: {("S1", "chr10"): 2, ("S2", "chr10"): 3})
    monkeypatch.setattr(plot_module, "_sort_samples_by_ploidy", lambda samples, chrom, lookup: list(samples))
    monkeypatch.setattr(
        plot_module,
        "_build_raw_region_df",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "Cluster": ["10q11.2"],
                "Chr": ["chr10"],
                "Start": [46005406],
                "End": [49845537],
                "S1": [2.1],
                "S2": [1.9],
            }
        ),
    )
    monkeypatch.setattr(plot_module, "_rebin_region_df", lambda df, locus: df)
    monkeypatch.setattr(
        plot_module,
        "_draw_overview_column",
        lambda axes, region_df, locus, calls_df, region_start, region_end, carrier_cols, non_carrier_cols, all_ploidies, ploidy_lookup, sample_cols, carriers, gtf, segdup, **kwargs: draw_calls.append(
            {
                "title": kwargs["col_title"],
                "carrier_cols": carrier_cols,
                "non_carrier_cols": non_carrier_cols,
                "all_ploidies": all_ploidies,
                "sample_cols": sample_cols,
                "carriers": carriers,
                "rows": len(axes),
            }
        ),
    )
    monkeypatch.setattr(plot_module, "_print_timing", lambda label, start: timing_calls.append(label))

    calls_df = pd.DataFrame(
        {
            "cluster": ["10q11.2"],
            "sample": ["S1"],
            "is_carrier": [True],
        }
    )
    depth_df = pd.DataFrame(
        {
            "Cluster": ["10q11.2", "10q11.2"],
            "Chr": ["chr10", "chr10"],
            "Start": [46005406, 48181660],
            "End": [48181660, 49845537],
            "S1": [2.0, 1.2],
            "S2": [2.1, 2.0],
        }
    )

    plot_locus_overview(
        locus,
        calls_df,
        depth_df,
        None,
        None,
        str(tmp_path),
        raw_counts_df=pd.DataFrame({"Chr": ["chr1"], "Start": [0], "End": [10], "S1": [10.0], "S2": [12.0]}),
        raw_sample_medians={"S1": 10.0, "S2": 12.0},
        lowres_median_bin_size=100000.0,
    )

    assert [call["title"] for call in draw_calls] == [
        "Raw normalised — 10q11.2 (chr10:46,005,406-50,651,802)",
        "Processed — 10q11.2 (chr10:46,005,406-50,651,802)",
    ]
    assert draw_calls[0]["carrier_cols"] == ["S1"]
    assert draw_calls[0]["non_carrier_cols"] == ["S2"]
    assert draw_calls[0]["all_ploidies"] == [2, 3]
    assert draw_calls[0]["sample_cols"] == ["S1", "S2"]
    assert draw_calls[0]["carriers"] == {"S1"}
    assert draw_calls[0]["rows"] == 6
    assert saved_paths == [str(tmp_path / "locus_plots" / "10q11.2_overview.png")]
    assert any(label.endswith("overview total") for label in timing_calls)


def test_plot_sample_at_locus_skips_missing_sample_and_saves_carrier_plot(monkeypatch, tmp_path):
    locus = _make_eval_locus()
    annotation_titles = []
    saved_paths = []
    monkeypatch.setattr(
        plot_module,
        "draw_annotations_panel",
        lambda ax, locus, region_start, region_end, chrom, title, gtf, segdup, **kwargs: annotation_titles.append(title),
    )
    monkeypatch.setattr(plot_module.plt, "savefig", lambda path, **kwargs: saved_paths.append(path))

    calls_df = pd.DataFrame(
        {
            "cluster": ["10q11.2"],
            "sample": ["S1"],
            "start": [46005406],
            "end": [49845537],
            "svtype": ["DEL"],
            "is_carrier": [True],
            "is_best_match": [True],
            "matched_haplotype": [1],
            "hap_cn_state": [1],
        }
    )
    depth_df = pd.DataFrame(
        {
            "Cluster": ["10q11.2", "10q11.2"],
            "Chr": ["chr10", "chr10"],
            "Start": [46005406, 48181660],
            "End": [48181660, 49845537],
            "S1": [2.0, 1.0],
        }
    )

    plot_sample_at_locus("missing", locus, calls_df, depth_df, None, None, str(tmp_path))
    plot_sample_at_locus("S1", locus, calls_df, depth_df, None, None, str(tmp_path))

    assert annotation_titles == ["S1 at 10q11.2 [CARRIER]"]
    assert saved_paths == [str(tmp_path / "sample_plots" / "10q11.2" / "S1.png")]


def test_plot_summary_helpers_cover_empty_and_nonempty_paths(monkeypatch, tmp_path, capsys):
    saved_paths = []
    monkeypatch.setattr(plot_module.plt, "savefig", lambda path, **kwargs: saved_paths.append(path))

    empty_calls = pd.DataFrame(
        columns=["is_carrier", "cluster", "svtype", "sample", "qual_score", "mean_depth"]
    )
    plot_carrier_summary(empty_calls, str(tmp_path))
    plot_confidence_distribution(empty_calls, str(tmp_path))
    out = capsys.readouterr().out
    assert "No carriers to plot." in out
    assert "No calls to plot confidence distribution." in out

    calls_df = pd.DataFrame(
        {
            "is_carrier": [True, True, False],
            "cluster": ["10q11.2", "16p11.2", "10q11.2"],
            "svtype": ["DEL", "DUP", "DEL"],
            "sample": ["S1", "S2", "S3"],
            "qual_score": [25.0, 30.0, 5.0],
            "mean_depth": [1.0, 3.0, 2.0],
        }
    )
    plot_carrier_summary(calls_df, str(tmp_path))
    plot_confidence_distribution(calls_df, str(tmp_path))

    assert saved_paths == [
        str(tmp_path / "carrier_summary.png"),
        str(tmp_path / "confidence_distribution.png"),
    ]


def test_create_carrier_pdf_handles_empty_and_nonempty_carrier_sets(monkeypatch, tmp_path, capsys):
    locus = _make_eval_locus()
    monkeypatch.setattr(plot_module, "PdfPages", _StubPdfPages)

    empty_calls = pd.DataFrame(
        {
            "cluster": ["10q11.2"],
            "sample": ["S1"],
            "is_carrier": [False],
            "is_best_match": [False],
            "qual_score": [5.0],
        }
    )
    create_carrier_pdf(empty_calls, pd.DataFrame(), {"10q11.2": locus}, None, None, str(tmp_path))
    assert "No confident carrier calls were emitted by the call step." in capsys.readouterr().out

    render_calls = []
    monkeypatch.setattr(
        plot_module,
        "_render_pdf_sample_page",
        lambda pdf, sample_id, cluster, locus, region_df, cluster_calls_df, confidence_column, *args, **kwargs: render_calls.append(
            (sample_id, cluster, len(region_df), len(cluster_calls_df), confidence_column, args[4] is not None)
        ) or True,
    )
    monkeypatch.setattr(
        plot_module,
        "_build_raw_region_df",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "Cluster": ["10q11.2"],
                "Chr": ["chr10"],
                "Start": [46005406],
                "End": [49845537],
                "S1": [2.0],
            }
        ),
    )
    monkeypatch.setattr(plot_module, "_rebin_region_df", lambda df, locus: df)

    calls_df = pd.DataFrame(
        {
            "cluster": ["10q11.2"],
            "sample": ["S1"],
            "is_carrier": [True],
            "is_best_match": [True],
            "qual_score": [25.0],
            "svtype": ["DEL"],
        }
    )
    depth_df = pd.DataFrame(
        {
            "Cluster": ["10q11.2"],
            "Chr": ["chr10"],
            "Start": [46005406],
            "End": [49845537],
            "S1": [1.1],
        }
    )
    raw_counts_df = pd.DataFrame(
        {
            "Chr": ["chr1", "chr2"],
            "Start": [0, 10],
            "End": [10, 20],
            "S1": [10.0, 12.0],
        }
    )

    create_carrier_pdf(
        calls_df,
        depth_df,
        {"10q11.2": locus},
        None,
        None,
        str(tmp_path),
        raw_counts_df=raw_counts_df,
    )

    assert render_calls == [("S1", "10q11.2", 1, 1, "qual_score", True)]


def test_eval_and_anomalous_pdf_wrappers_delegate_to_review_builder(monkeypatch, tmp_path):
    review_calls = []
    monkeypatch.setattr(
        plot_module,
        "_build_eval_pdf_specs",
        lambda *args, **kwargs: {
            "true_positives": [{"sample": "S1"}],
            "false_positives": [],
            "false_negatives": [{"sample": "S2"}],
            "anomalous_discrepancies": [{"sample": "S3"}],
        },
    )
    monkeypatch.setattr(
        plot_module,
        "_build_anomalous_pdf_specs",
        lambda *args, **kwargs: [{"sample": "S9"}],
    )
    monkeypatch.setattr(
        plot_module,
        "_create_review_category_pdf",
        lambda category_name, page_specs, pdf_path, *args, **kwargs: review_calls.append((category_name, page_specs, pdf_path)),
    )

    create_eval_category_pdfs(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {},
        None,
        None,
        str(tmp_path),
    )
    create_anomalous_pdf(
        pd.DataFrame(),
        pd.DataFrame(),
        {},
        None,
        None,
        str(tmp_path),
    )

    assert review_calls == [
        ("true_positives", [{"sample": "S1"}], str(tmp_path / "true_positives.pdf")),
        ("false_positives", [], str(tmp_path / "false_positives.pdf")),
        ("false_negatives", [{"sample": "S2"}], str(tmp_path / "false_negatives.pdf")),
        ("anomalous_discrepancies", [{"sample": "S3"}], str(tmp_path / "anomalous_discrepancies.pdf")),
        ("anomalous_discrepancies", [{"sample": "S9"}], str(tmp_path / "anomalous_discrepancies.pdf")),
    ]


def test_plot_main_routes_eval_mode_and_highres_validation(monkeypatch, tmp_path, capsys):
    locus = _make_eval_locus()
    out_dir = tmp_path / "out"
    summary_calls = []
    eval_calls = []
    carrier_calls = []
    anomalous_calls = []
    locus_calls = []
    monkeypatch.setattr(plot_module, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(plot_module.os.path, "exists", lambda path: False)
    monkeypatch.setattr(plot_module, "GDTable", lambda path: SimpleNamespace(loci={"10q11.2": locus}))
    monkeypatch.setattr(plot_module, "plot_carrier_summary", lambda calls_df, output_dir: summary_calls.append(("carrier", len(calls_df), output_dir)))
    monkeypatch.setattr(plot_module, "plot_confidence_distribution", lambda calls_df, output_dir: summary_calls.append(("confidence", len(calls_df), output_dir)))
    monkeypatch.setattr(plot_module, "create_eval_category_pdfs", lambda *args, **kwargs: eval_calls.append((args[1], args[2], args[3])))
    monkeypatch.setattr(plot_module, "create_carrier_pdf", lambda *args, **kwargs: carrier_calls.append(True))
    monkeypatch.setattr(plot_module, "create_anomalous_pdf", lambda *args, **kwargs: anomalous_calls.append(True))
    monkeypatch.setattr(plot_module, "plot_locus_overview", lambda *args, **kwargs: locus_calls.append(True))

    def fake_read_csv(path, sep="\t", compression=None):
        if path == "calls.tsv":
            raise pd.errors.EmptyDataError("empty")
        if path == "cn.tsv":
            return pd.DataFrame(
                {
                    "cluster": ["10q11.2"],
                    "chr": ["chr10"],
                    "start": [46005406],
                    "end": [49845537],
                    "sample": ["S1"],
                    "depth": [2.0],
                    "minor_baf_median": [0.4],
                    "baf_n_sites": [10],
                }
            )
        if path == "ploidy.tsv":
            return pd.DataFrame({"sample": ["S1"], "contig": ["chr10"], "ploidy": [2]})
        if path == "eval.tsv":
            return pd.DataFrame({"GD_ID": ["GD1"], "TP_samples": ["S1"], "FP_samples": [""], "FN_samples": [""]})
        raise AssertionError(path)

    monkeypatch.setattr(plot_module.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(
        plot_module,
        "parse_args",
        lambda: SimpleNamespace(
            calls="calls.tsv",
            cn_posteriors="cn.tsv",
            sample_posteriors=None,
            gd_table="gd.tsv",
            ploidy_table="ploidy.tsv",
            gtf=None,
            segdup_bed=None,
            gaps_bed=None,
            raw_counts=None,
            high_res_counts=None,
            output_dir=str(out_dir),
            padding=50000,
            plot_all_samples=False,
            skip_locus_plots=True,
            sample=None,
            min_gene_label_spacing=0.05,
            loci=["10q11.2"],
            flank_scale=0.2,
            event_marginals=None,
            eval_report="eval.tsv",
        ),
    )

    plot_main()

    stdout = capsys.readouterr().out
    assert "Calls file is empty; continuing with a no-calls DataFrame" in stdout
    assert "Skipping locus overview plots (--skip-locus-plots)" in stdout
    assert summary_calls == [
        ("carrier", 0, str(out_dir)),
        ("confidence", 0, str(out_dir)),
    ]
    assert len(eval_calls) == 1
    assert carrier_calls == []
    assert anomalous_calls == []
    assert locus_calls == []

    monkeypatch.setattr(
        plot_module,
        "parse_args",
        lambda: SimpleNamespace(
            calls="calls.tsv",
            cn_posteriors="cn.tsv",
            sample_posteriors=None,
            gd_table="gd.tsv",
            ploidy_table=None,
            gtf=None,
            segdup_bed=None,
            gaps_bed=None,
            raw_counts=None,
            high_res_counts="hires.tsv.gz",
            output_dir=str(out_dir),
            padding=50000,
            plot_all_samples=False,
            skip_locus_plots=True,
            sample=None,
            min_gene_label_spacing=0.05,
            loci=None,
            flank_scale=0.2,
            event_marginals=None,
            eval_report=None,
        ),
    )

    with pytest.raises(SystemExit) as excinfo:
        plot_main()
    assert excinfo.value.code == 1


def test_plot_main_routes_carrier_mode_with_sibling_inputs(monkeypatch, tmp_path, capsys):
    locus = _make_eval_locus()
    summary_calls = []
    overview_calls = []
    sample_calls = []
    carrier_pdf_calls = []
    anomalous_pdf_calls = []

    def fake_read_csv(path, sep="\t", compression=None):
        if path == "calls.tsv":
            return pd.DataFrame(
                {
                    "cluster": ["10q11.2"],
                    "sample": ["S1"],
                    "GD_ID": ["GD1"],
                    "svtype": ["DEL"],
                    "is_carrier": [True],
                    "is_best_match": [True],
                    "is_null_anomalous": [False],
                    "qual_score": [20.0],
                    "mean_depth": [1.2],
                    "start": [46005406],
                    "end": [49845537],
                }
            )
        if path == "cn.tsv":
            return pd.DataFrame(
                {
                    "cluster": ["10q11.2", "10q11.2", "10q11.2"],
                    "chr": ["chr10", "chr10", "chr10"],
                    "start": [46005406, 46005406, 48181660],
                    "end": [48181660, 48181660, 49845537],
                    "sample": ["S1", "S1", "S1"],
                    "depth": [2.0, 2.0, 1.1],
                    "minor_baf_median": [0.4, 0.4, 0.3],
                    "baf_n_sites": [10, 10, 12],
                    "baf_variance": [0.01, 0.01, 0.02],
                }
            )
        if path.endswith("sample_posteriors.tsv.gz"):
            return pd.DataFrame({"sample": ["S1"], "baf_temperature_map": [2.5]})
        if path == "raw.tsv":
            return pd.DataFrame({"#Chr": ["chr1", "chr2"], "Start": [0, 10], "End": [10, 20], "S1": [10.0, 12.0]})
        if path.endswith("event_marginals.tsv.gz"):
            return pd.DataFrame(
                {
                    "cluster": ["10q11.2", "10q11.2"],
                    "chrom": ["chr10", "chr10"],
                    "start": [46005406, 48181660],
                    "end": [48181660, 49845537],
                    "sample": ["S1", "S1"],
                    "qual_del_event": [15.0, 30.0],
                    "qual_dup_event": [-15.0, -30.0],
                }
            )
        raise AssertionError(path)

    monkeypatch.setattr(plot_module, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(plot_module.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(plot_module.os.path, "exists", lambda path: path.endswith("sample_posteriors.tsv.gz") or path.endswith("event_marginals.tsv.gz"))
    monkeypatch.setattr(plot_module, "GDTable", lambda path: SimpleNamespace(loci={"10q11.2": locus}))
    monkeypatch.setattr(plot_module, "GTFParser", lambda path: "gtf")
    monkeypatch.setattr(plot_module, "SegDupAnnotation", lambda path: "segdup")
    monkeypatch.setattr(plot_module, "GapsAnnotation", lambda path: "gaps")
    monkeypatch.setattr(plot_module, "plot_carrier_summary", lambda calls_df, output_dir: summary_calls.append(("carrier", len(calls_df), output_dir)))
    monkeypatch.setattr(plot_module, "plot_confidence_distribution", lambda calls_df, output_dir: summary_calls.append(("confidence", len(calls_df), output_dir)))
    monkeypatch.setattr(plot_module, "plot_locus_overview", lambda *args, **kwargs: overview_calls.append((args, kwargs)))
    monkeypatch.setattr(plot_module, "plot_sample_at_locus", lambda *args, **kwargs: sample_calls.append((args, kwargs)))
    monkeypatch.setattr(plot_module, "create_carrier_pdf", lambda *args, **kwargs: carrier_pdf_calls.append((args, kwargs)))
    monkeypatch.setattr(plot_module, "create_anomalous_pdf", lambda *args, **kwargs: anomalous_pdf_calls.append((args, kwargs)))

    monkeypatch.setattr(
        plot_module,
        "parse_args",
        lambda: SimpleNamespace(
            calls="calls.tsv",
            cn_posteriors="cn.tsv",
            sample_posteriors=None,
            gd_table="gd.tsv",
            ploidy_table=None,
            gtf="genes.gtf",
            segdup_bed="segdup.bed",
            gaps_bed="gaps.bed",
            raw_counts="raw.tsv",
            high_res_counts=None,
            output_dir=str(tmp_path),
            padding=50000,
            plot_all_samples=False,
            skip_locus_plots=False,
            sample="S1",
            min_gene_label_spacing=0.05,
            loci=None,
            flank_scale=0.2,
            event_marginals=None,
            eval_report=None,
        ),
    )

    plot_main()

    stdout = capsys.readouterr().out
    assert "NOTE: dropped 1 duplicate bin-sample rows before pivoting" in stdout
    assert "Loaded BAF variance scale MAP values for 1 samples" in stdout
    assert "Loading event marginals" in stdout
    assert summary_calls == [
        ("carrier", 1, str(tmp_path)),
        ("confidence", 1, str(tmp_path)),
    ]
    assert len(overview_calls) == 1
    assert len(sample_calls) == 1
    assert len(carrier_pdf_calls) == 1
    assert len(anomalous_pdf_calls) == 1
    assert carrier_pdf_calls[0][1]["event_values_are_qual"] is True
    assert carrier_pdf_calls[0][1]["minor_baf_df"] is not None
    assert carrier_pdf_calls[0][1]["baf_variance_df"] is not None
    assert carrier_pdf_calls[0][1]["baf_sites_df"] is not None
    assert carrier_pdf_calls[0][1]["raw_counts_df"] is not None
    assert carrier_pdf_calls[0][1]["baf_temperature_by_sample"] == {"S1": 2.5}


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


def test_build_raw_region_df_returns_none_without_valid_sample_medians():
    locus = _make_locus()
    raw_counts_df = pd.DataFrame(
        {
            "Chr": ["chr10"],
            "Start": [46005406],
            "End": [49845537],
            "sample_1": [20.0],
        }
    )

    result = _build_raw_region_df(
        locus,
        46005406,
        49845537,
        raw_counts_df,
        ["sample_1"],
        {"sample_1": 0.0},
        100000.0,
    )

    assert result is None


def test_build_raw_region_df_falls_back_to_partial_highres_substitution(monkeypatch):
    locus = _make_locus()
    region_start = 45900000
    region_end = 50700000
    raw_counts_df = pd.DataFrame(
        {
            "Chr": ["chr10"] * 4,
            "Start": [45900000, 46050000, 48100000, 49900000],
            "End": [46050000, 48100000, 49900000, 50700000],
            "sample_1": [20.0, 22.0, 18.0, 24.0],
        }
    )
    processed_region_df = pd.DataFrame(
        {
            "Chr": ["chr10"] * 4,
            "Start": [46005406, 47000000, 48181660, 49845537],
            "End": [47000000, 48181660, 49845537, 50651802],
            "sample_1": [2.0, 2.1, 1.9, 2.0],
        }
    )
    observed_queries = []

    def make_highres_df(chrom, starts, ends, values):
        return pd.DataFrame(
            {
                "Chr": [chrom] * len(starts),
                "Start": starts,
                "End": ends,
                "source_file": ["highres"] * len(starts),
                "sample_1": values,
            },
            index=[f"{chrom}:{start}-{end}" for start, end in zip(starts, ends)],
        )

    def fake_query_highres_bins(highres_path, chrom, start, end, sample_cols, max_bins=None):
        observed_queries.append((start, end))
        if (start, end) == (region_start, region_end):
            return pd.DataFrame(columns=["Chr", "Start", "End", "source_file", "sample_1"])
        if (start, end) == (region_start, locus.start):
            return make_highres_df(chrom, [region_start], [locus.start], [10.0])
        if (start, end) == (locus.start, locus.breakpoints[1][0]):
            return make_highres_df(chrom, [locus.start, 47000000], [47000000, locus.breakpoints[1][0]], [11.0, 12.0])
        return pd.DataFrame(columns=["Chr", "Start", "End", "source_file", "sample_1"])

    def fake_normalize_highres_bins(highres_df, sample_cols, column_medians, lowres_median_bin_size):
        df = highres_df.copy()
        df["sample_1"] = df["sample_1"] / 10.0
        return df

    monkeypatch.setattr("gatk_sv_gd.plot.query_highres_bins", fake_query_highres_bins)
    monkeypatch.setattr("gatk_sv_gd.plot.normalize_highres_bins", fake_normalize_highres_bins)

    result = _build_raw_region_df(
        locus,
        region_start,
        region_end,
        raw_counts_df,
        ["sample_1"],
        {"sample_1": 20.0},
        1000000.0,
        processed_region_df=processed_region_df,
        highres_path="/tmp/highres.tsv.gz",
    )

    assert observed_queries == [
        (region_start, region_end),
        (locus.start, locus.breakpoints[1][0]),
        (locus.breakpoints[1][0], locus.breakpoints[2][0]),
        (locus.breakpoints[2][0], locus.end),
        (region_start, locus.start),
        (locus.end, region_end),
    ]
    assert result is not None
    assert result["Start"].tolist() == [45900000, 46005406, 47000000, 49900000]
    assert result["End"].tolist() == [46005406, 47000000, 48181660, 50700000]
    assert result["sample_1"].tolist() == pytest.approx([1.0, 1.1, 1.2, 2.4])


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


def test_plot_baf_signal_panel_reports_missing_baf_signal_when_inputs_absent():
    locus = _make_locus()
    x_positions = pd.Series([46050000, 48190000], dtype=float).to_numpy()
    bar_widths = pd.Series([0.001, 0.001], dtype=float).to_numpy()
    xform = FlankCompressor(46005406, 49845537, locus.start, locus.end, flank_scale=0.2)

    fig, ax = plt.subplots()
    try:
        _plot_baf_signal_panel(
            ax,
            x_positions,
            bar_widths,
            None,
            None,
            xform,
            locus,
            locus.chrom,
        )

        assert [text.get_text() for text in ax.texts] == ["No BAF signal available"]
        assert ax.get_xlabel() == f"Position on {locus.chrom}"
        assert ax.get_ylabel() == "Minor BAF"
        assert len(ax.patches) == 0
    finally:
        plt.close(fig)


def test_plot_baf_signal_panel_reports_no_supported_bins_when_all_values_invalid():
    locus = _make_locus()
    x_positions = pd.Series([46050000, 48190000], dtype=float).to_numpy()
    bar_widths = pd.Series([0.001, 0.001], dtype=float).to_numpy()
    minor_baf_values = pd.Series([np.nan, 0.4], dtype=float).to_numpy()
    baf_site_counts = pd.Series([1, 0], dtype=int).to_numpy()
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
            show_xlabel=False,
        )

        assert [text.get_text() for text in ax.texts] == ["No BAF-supported bins"]
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == "Minor BAF"
        assert len(ax.patches) == 0
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


def test_coarsen_pdf_page_signals_returns_inputs_when_already_within_budget():
    locus = _make_locus()
    region_df = pd.DataFrame(
        {
            "Cluster": [locus.cluster] * 2,
            "Chr": [locus.chrom] * 2,
            "Start": [46005406, 48181660],
            "End": [48181660, 49845537],
        }
    )
    sample_depth = np.array([2.0, 1.5], dtype=float)
    minor_baf = np.array([0.2, 0.3], dtype=float)
    baf_variance = np.array([0.01, 0.02], dtype=float)
    baf_sites = np.array([10.0, 12.0], dtype=float)
    event_probs = np.array([0.7, 0.8], dtype=float)

    result = _coarsen_pdf_page_signals(
        locus,
        region_df,
        sample_depth,
        minor_baf,
        baf_variance,
        baf_sites,
        event_probs,
        max_total_bins=3,
    )

    assert result[0] is region_df
    assert result[1] is sample_depth
    assert result[2] is minor_baf
    assert result[3] is baf_variance
    assert result[4] is baf_sites
    assert result[5] is event_probs


def test_coarsen_pdf_page_signals_handles_unweighted_grouped_values():
    locus = GDLocus(
        cluster="body-only",
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
            "Start": [110, 130, 150, 170],
            "End": [130, 150, 170, 190],
        }
    )

    plot_region_df, plot_depth, plot_minor_baf, plot_baf_variance, plot_baf_sites, plot_event_probs = _coarsen_pdf_page_signals(
        locus,
        region_df,
        np.array([2.0, 4.0, 6.0, 8.0], dtype=float),
        np.array([0.1, np.nan, 0.3, 0.5], dtype=float),
        np.array([0.04, 0.09, np.nan, 0.16], dtype=float),
        None,
        np.array([0.2, 0.4, 0.6, 0.8], dtype=float),
        max_total_bins=2,
    )

    assert plot_region_df["Start"].tolist() == [110, 150]
    assert plot_region_df["End"].tolist() == [150, 190]
    assert plot_depth.tolist() == pytest.approx([3.0, 7.0])
    assert plot_minor_baf.tolist() == pytest.approx([0.1, 0.4])
    assert plot_baf_variance.tolist() == pytest.approx([0.0325, 0.16])
    assert plot_baf_sites is None
    assert plot_event_probs.tolist() == pytest.approx([0.3, 0.7])


def test_coarsen_pdf_page_signals_emits_nan_baf_summaries_when_group_sites_are_invalid():
    locus = GDLocus(
        cluster="body-only",
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
            "Start": [110, 130, 150, 170],
            "End": [130, 150, 170, 190],
        }
    )

    plot_region_df, plot_depth, plot_minor_baf, plot_baf_variance, plot_baf_sites, plot_event_probs = _coarsen_pdf_page_signals(
        locus,
        region_df,
        np.array([2.0, 4.0, 6.0, 8.0], dtype=float),
        np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
        np.array([0.04, 0.09, 0.16, 0.25], dtype=float),
        np.array([0.0, np.nan, 0.0, np.nan], dtype=float),
        np.array([np.nan, 0.2, 0.4, np.nan], dtype=float),
        max_total_bins=2,
    )

    assert plot_region_df["Start"].tolist() == [110, 150]
    assert plot_region_df["End"].tolist() == [150, 190]
    assert plot_depth.tolist() == pytest.approx([3.0, 7.0])
    assert np.isnan(plot_minor_baf).tolist() == [True, True]
    assert np.isnan(plot_baf_variance).tolist() == [True, True]
    assert plot_baf_sites.tolist() == pytest.approx([0.0, 0.0])
    assert plot_event_probs.tolist() == pytest.approx([0.2, 0.4])


def test_allocate_segment_bin_targets_covers_empty_residual_and_shrink_paths():
    assert _allocate_segment_bin_targets([0, 0, 0], 5) == [0, 0, 0]
    assert _allocate_segment_bin_targets([5, 4, 1], 7) == [3, 3, 1]
    assert _allocate_segment_bin_targets([10, 10, 10], 2) == [1, 1, 1]


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


def test_rebin_aligned_region_dfs_for_display_falls_back_for_misaligned_frames(monkeypatch):
    locus = GDLocus(
        cluster="1q21",
        chrom="chr1",
        breakpoints=[(100, 100), (200, 200), (300, 300), (400, 400)],
        breakpoint_names=["1", "2", "3", "4"],
        gd_entries=[],
        is_nahr=True,
        is_terminal=False,
    )
    region_df = pd.DataFrame(
        {
            "Cluster": [locus.cluster] * 6,
            "Chr": [locus.chrom] * 6,
            "Start": [50, 90, 120, 220, 320, 420],
            "End": [90, 120, 220, 320, 420, 520],
            "sample_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    shorter_df = region_df.iloc[:4].copy()
    shifted_df = region_df.copy()
    shifted_df.loc[0, "Start"] = 51
    fallback_calls = []

    monkeypatch.setattr(
        plot_module,
        "_rebin_region_df",
        lambda df, locus, max_bins_per_region=100: fallback_calls.append((len(df), max_bins_per_region)) or df.iloc[[0]].copy(),
    )

    rebinned_region, rebinned_frames = _rebin_aligned_region_dfs_for_display(
        locus,
        region_df,
        [shorter_df, shifted_df, None],
        max_total_bins=3,
    )

    assert len(rebinned_region) == 3
    assert [None if frame is None else len(frame) for frame in rebinned_frames] == [1, 1, None]
    assert fallback_calls == [(4, 1), (6, 1)]


def test_rebin_region_df_returns_none_or_empty_input_unchanged():
    empty_df = pd.DataFrame(columns=["Chr", "Start", "End", "sample_1"])
    locus = GDLocus(
        cluster="1q21",
        chrom="chr1",
        breakpoints=[(100, 100), (150, 150), (200, 200), (250, 250)],
        breakpoint_names=["1", "2", "3", "4"],
        gd_entries=[],
        is_nahr=True,
        is_terminal=False,
    )

    assert _rebin_region_df(None, locus) is None
    assert _rebin_region_df(empty_df, locus).equals(empty_df)


def test_rebin_region_df_rebins_left_body_and_right_segments_independently():
    locus = GDLocus(
        cluster="1q21",
        chrom="chr1",
        breakpoints=[(100, 100), (150, 150), (200, 200), (250, 250)],
        breakpoint_names=["1", "2", "3", "4"],
        gd_entries=[],
        is_nahr=True,
        is_terminal=False,
    )
    df = pd.DataFrame(
        {
            "Chr": ["chr1"] * 9,
            "Start": [0, 10, 20, 110, 120, 130, 260, 270, 280],
            "End": [10, 20, 30, 120, 130, 140, 270, 280, 290],
            "sample_1": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0],
        }
    )

    rebinned = _rebin_region_df(df, locus, max_bins_per_region=2)

    assert rebinned["Start"].tolist() == [0, 20, 110, 130, 260, 280]
    assert rebinned["End"].tolist() == [20, 30, 130, 140, 280, 290]
    assert rebinned["sample_1"].tolist() == pytest.approx([1.5, 3.0, 15.0, 30.0, 4.5, 6.0])
