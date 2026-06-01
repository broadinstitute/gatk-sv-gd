import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from gatk_sv_gd import aggregate


def _call_row(
    sample,
    *,
    gd_id="GD1",
    cluster="cluster1",
    svtype="DEL",
    is_carrier=False,
    is_best_match=True,
    qual_score=20.0,
    calling_method="posterior-marginal",
    left_flank_non_event_median=2.0,
    right_flank_non_event_median=2.0,
    call_criteria_mean_coverage=float("nan"),
    call_criteria_interval_confidence=60.0,
    call_criteria_flank_non_event_confidence=60.0,
    null_anomaly_score=0.0,
    is_null_anomalous=False,
    call_criteria_null_anomaly_score=0.2,
):
    return {
        "sample": sample,
        "cluster": cluster,
        "GD_ID": gd_id,
        "chrom": "chr1",
        "start": 100,
        "end": 200,
        "svtype": svtype,
        "BP1": "A",
        "BP2": "B",
        "is_terminal": False,
        "n_bins": 3,
        "mean_depth": 1.5 if svtype == "DEL" else 2.5,
        "sample_ploidy": 2,
        "matched_haplotype": 1,
        "hap_cn_state": 1 if svtype == "DEL" else 3,
        "matched_seg_start": 100,
        "matched_seg_end": 200,
        "matched_seg_n_bins": 3,
        "matched_interval_bp": 100,
        "interval_coverage": 1.0,
        "reciprocal_overlap": 1.0,
        "min_interval_confidence": qual_score,
        "left_flank_non_event_median": left_flank_non_event_median,
        "right_flank_non_event_median": right_flank_non_event_median,
        "min_flank_non_event_confidence": qual_score,
        "is_carrier": is_carrier,
        "is_best_match": is_best_match,
        "log_prob_score": 0.95,
        "confidence_score": qual_score,
        "qual_score": qual_score,
        "null_anomaly_score": null_anomaly_score,
        "is_null_anomalous": is_null_anomalous,
        "calling_method": calling_method,
        "call_criteria_mean_coverage": call_criteria_mean_coverage,
        "call_criteria_interval_confidence": call_criteria_interval_confidence,
        "call_criteria_flank_non_event_confidence": call_criteria_flank_non_event_confidence,
        "call_criteria_null_anomaly_score": call_criteria_null_anomaly_score,
    }


def _write_run(
    root: Path,
    calls,
    *,
    include_eval=False,
    touch_optional=False,
):
    (root / "call").mkdir(parents=True)
    (root / "preprocess").mkdir(parents=True)
    (root / "infer").mkdir(parents=True)
    (root / "plot").mkdir(parents=True)
    pd.DataFrame(calls).to_csv(
        root / "call" / "gd_cnv_calls.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(
        {
            "sample": sorted({row["sample"] for row in calls}),
            "contig": ["chr1"] * len({row["sample"] for row in calls}),
            "median_depth": [2.0] * len({row["sample"] for row in calls}),
            "ploidy": [2] * len({row["sample"] for row in calls}),
        }
    ).to_csv(root / "preprocess" / "ploidy_estimates.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "chr": ["chr1"],
            "start_GRCh38": [100],
            "end_GRCh38": [200],
            "GD_ID": ["GD1"],
            "svtype": ["DEL"],
            "NAHR": ["yes"],
            "terminal": ["no"],
            "cluster": ["cluster1"],
            "BP1": ["A"],
            "BP2": ["B"],
        }
    ).to_csv(root / "preprocess" / "gd_table_filtered.tsv", sep="\t", index=False)
    if include_eval:
        (root / "eval").mkdir()
        pd.DataFrame(
            {
                "GD_ID": ["GD1"],
                "TP": [1],
                "FP": [0],
                "FN": [1],
                "sensitivity": [0.5],
                "precision": [1.0],
                "TP_samples": ["S1"],
                "FP_samples": [""],
                "FN_samples": ["S2"],
                "anomalous_discrepancy_samples": ["S2"],
            }
        ).to_csv(root / "eval" / "truth_evaluation_report.tsv", sep="\t", index=False)
    if touch_optional:
        for artifact, parts in aggregate._OPTIONAL_ARTIFACTS:
            if artifact == "eval_report":
                continue
            path = root.joinpath(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"placeholder")


def test_default_batch_labels_are_stable_for_duplicate_names():
    labels = aggregate._default_batch_labels(["/tmp/run", "/other/run", "/tmp/next"])

    assert labels == ["run", "run_2", "next"]


def test_validate_args_rejects_bad_output_name_and_label_count():
    args = argparse.Namespace(
        work_dirs=["run_a", "run_b"],
        output_dir="out",
        output_name="nested/report.pdf",
        min_confidence=0.5,
        batch_label=None,
    )

    with pytest.raises(ValueError, match="PDF filename"):
        aggregate._validate_args(args)

    args.output_name = "report.pdf"
    args.min_confidence = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        aggregate._validate_args(args)

    args.min_confidence = 0.5
    args.batch_label = ["only_one"]
    with pytest.raises(ValueError, match="once per work directory"):
        aggregate._validate_args(args)


def test_bool_normalization_helpers_cover_preferred_confidence_and_coercion():
    assert aggregate._get_confidence_column(pd.DataFrame({"qual_score": [1.0]})) == "qual_score"
    assert aggregate._get_confidence_column(pd.DataFrame({"confidence_score": [1.0]})) == "confidence_score"
    assert aggregate._get_confidence_column(pd.DataFrame({"log_prob_score": [1.0]})) == "log_prob_score"
    with pytest.raises(ValueError, match="Calls table is missing"):
        aggregate._get_confidence_column(pd.DataFrame({"other": [1.0]}))

    assert aggregate._to_bool_value(pd.NA) is False
    assert aggregate._to_bool_value(True) is True
    assert aggregate._to_bool_value(np.bool_(False)) is False
    assert aggregate._to_bool_value(" YES ") is True
    assert aggregate._to_bool_value("0") is False
    assert aggregate._to_bool_value(2) is True


def test_normalize_bool_columns_only_converts_supported_columns():
    calls_df = pd.DataFrame(
        {
            "is_carrier": ["yes", "no", True],
            "is_best_match": [1, 0, "TRUE"],
            "is_terminal": [pd.NA, False, "0"],
            "other_col": ["keep", "these", "values"],
        }
    )

    normalized = aggregate._normalize_bool_columns(calls_df)

    assert normalized["is_carrier"].tolist() == [True, False, True]
    assert normalized["is_best_match"].tolist() == [True, False, True]
    assert normalized["is_terminal"].tolist() == [False, False, False]
    assert normalized["other_col"].tolist() == ["keep", "these", "values"]
    assert calls_df["is_carrier"].tolist() == ["yes", "no", True]


def test_aggregate_parse_args_load_helpers_and_main_cover_cli_support(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gatk-sv-gd",
            "run_a",
            "run_b",
            "--output-dir",
            str(tmp_path),
            "--output-name",
            "report.pdf",
            "--min-confidence",
            "12.5",
            "--batch-label",
            "batch_a",
            "--batch-label",
            "batch_b",
        ],
    )

    parsed = aggregate.parse_args()

    assert parsed.work_dirs == ["run_a", "run_b"]
    assert parsed.output_dir == str(tmp_path)
    assert parsed.output_name == "report.pdf"
    assert parsed.min_confidence == pytest.approx(12.5)
    assert parsed.batch_label == ["batch_a", "batch_b"]

    run = aggregate.RunData(
        batch_id=7,
        batch_label="batch_a",
        work_dir=tmp_path,
        work_dir_input="run_a",
        calls_df=pd.DataFrame({"sample": ["S1"], "value": [1]}),
        ploidy_df=pd.DataFrame(),
        gd_table_df=pd.DataFrame(),
        eval_df=None,
        optional_artifact_status={},
        missing_artifacts=[],
    )
    with_sample = aggregate._add_batch_columns(pd.DataFrame({"sample": ["S1"]}), run)
    without_sample = aggregate._add_batch_columns(pd.DataFrame({"value": [1]}), run)
    assert with_sample.columns.tolist()[:4] == ["batch_id", "batch_label", "work_dir", "sample_key"]
    assert with_sample.loc[0, "sample_key"] == "batch_a/S1"
    assert without_sample.columns.tolist() == ["batch_id", "batch_label", "work_dir", "value"]

    loaded = []
    monkeypatch.setattr(
        aggregate,
        "_load_run_data",
        lambda work_dir, batch_id, batch_label: loaded.append((work_dir, batch_id, batch_label)) or work_dir,
    )
    args = argparse.Namespace(work_dirs=["run_a", "run_b"], batch_label=["A", "B"])
    assert aggregate._load_runs(args) == ["run_a", "run_b"]
    assert loaded == [("run_a", 1, "A"), ("run_b", 2, "B")]

    main_args = argparse.Namespace(output_dir=str(tmp_path), work_dirs=["run_a"], output_name="report.pdf", min_confidence=0.5, batch_label=None)
    calls = {}
    monkeypatch.setattr(aggregate, "parse_args", lambda: main_args)
    monkeypatch.setattr(aggregate, "setup_logging", lambda *args, **kwargs: calls.setdefault("logging", (args, kwargs)))
    monkeypatch.setattr(aggregate, "_run_aggregate", lambda passed_args: calls.setdefault("run", passed_args))

    aggregate.main()

    assert calls["run"] is main_args
    assert calls["logging"][1]["filename"] == "aggregate_log.txt"
    assert calls["logging"][1]["command"] == "aggregate"


def test_optional_eval_and_plot_matrix_helpers_cover_edge_cases(monkeypatch, tmp_path):
    missing = []
    work_dir = tmp_path / "run"
    eval_path = work_dir / "eval" / "truth_evaluation_report.tsv"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text("placeholder")

    monkeypatch.setattr(aggregate, "_read_tsv", lambda path: (_ for _ in ()).throw(ValueError("bad eval")))
    assert aggregate._load_optional_eval_report(work_dir, "batch", "run", missing) is None
    assert missing[-1]["artifact"] == "eval_report"
    assert missing[-1]["reason"] == "bad eval"

    monkeypatch.setattr(aggregate, "_read_tsv", lambda path: pd.DataFrame({"other": [1]}))
    assert aggregate._load_optional_eval_report(work_dir, "batch", "run", missing) is None
    assert missing[-1]["reason"] == "missing GD_ID column"

    valid_eval = pd.DataFrame({"GD_ID": ["GD1"], "TP": [1]})
    monkeypatch.setattr(aggregate, "_read_tsv", lambda path: valid_eval)
    assert aggregate._load_optional_eval_report(work_dir, "batch", "run", missing).equals(valid_eval)

    sample_posteriors = work_dir / "infer" / "sample_posteriors.tsv.gz"
    sample_posteriors.parent.mkdir(parents=True)
    sample_posteriors.write_text("placeholder")
    status, missing_artifacts = aggregate._scan_optional_artifacts(work_dir, "batch", "run")
    assert status["sample_posteriors"] is True
    assert any(row["artifact"] == "bin_posteriors" for row in missing_artifacts)
    assert aggregate._artifact_path(work_dir, ("plot", "carrier_plots.pdf")) == work_dir / "plot" / "carrier_plots.pdf"

    grouped = aggregate._group_plot_frames_by_cluster(
        pd.DataFrame(
            {
                "Cluster": ["c2", "c1", "c1"],
                "Start": [20, 30, 10],
                "Value": [1, 2, 3],
            }
        )
    )
    assert grouped["c1"]["Start"].tolist() == [10, 30]
    assert aggregate._group_plot_frames_by_cluster(pd.DataFrame({"other": [1]})) == {}

    cn_posteriors_df = pd.DataFrame(
        {
            "cluster": ["c1", "c1"],
            "chr": ["chr1", "chr1"],
            "start": [0, 10],
            "end": [10, 20],
            "sample": ["S1", "S1"],
            "depth": [2.0, 3.0],
        }
    )
    pivoted = aggregate._pivot_plot_matrix(cn_posteriors_df, "depth")
    assert pivoted.columns.tolist() == ["Cluster", "Chr", "Start", "End", "S1"]
    assert aggregate._pivot_plot_matrix(cn_posteriors_df, "missing") is None


def test_load_plot_run_context_handles_missing_files_and_successful_plot_artifacts(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run = aggregate.RunData(
        batch_id=1,
        batch_label="batch",
        work_dir=run_dir,
        work_dir_input=str(run_dir),
        calls_df=pd.DataFrame(),
        ploidy_df=pd.DataFrame(),
        gd_table_df=pd.DataFrame(),
        eval_df=None,
        optional_artifact_status={},
        missing_artifacts=[],
    )

    missing_context = aggregate._load_plot_run_context(run)
    assert missing_context.unavailable_reason == "missing infer/cn_posteriors.tsv.gz"

    (run_dir / "infer").mkdir(parents=True)
    (run_dir / "call").mkdir(parents=True)
    (run_dir / "preprocess").mkdir(parents=True)
    for path in [
        run_dir / "infer" / "cn_posteriors.tsv.gz",
        run_dir / "infer" / "sample_posteriors.tsv.gz",
        run_dir / "call" / "event_marginals.tsv.gz",
        run_dir / "preprocess" / "gd_table_filtered.tsv",
    ]:
        path.write_text("placeholder")

    cn_posteriors = pd.DataFrame(
        {
            "cluster": ["cluster1", "cluster1", "cluster1"],
            "chr": ["chr1", "chr1", "chr1"],
            "start": [100, 100, 200],
            "end": [200, 200, 300],
            "sample": ["S1", "S1", "S1"],
            "depth": [2.0, 2.0, 3.0],
            "minor_baf_median": [0.4, 0.4, 0.3],
            "baf_variance": [0.02, 0.02, 0.03],
            "baf_n_sites": [5, 5, 6],
        }
    )
    sample_posteriors_df = pd.DataFrame(
        {
            "sample": ["S1", "S2"],
            "baf_temperature_map": [0.25, np.nan],
        }
    )
    event_marginals_df = pd.DataFrame(
        {
            "cluster": ["cluster1"],
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
            "sample": ["S1"],
            "prob_del_event": [0.8],
            "prob_dup_event": [0.1],
        }
    )

    def fake_read_tsv(path):
        if path.name == "cn_posteriors.tsv.gz":
            return cn_posteriors.copy()
        if path.name == "sample_posteriors.tsv.gz":
            return sample_posteriors_df.copy()
        if path.name == "event_marginals.tsv.gz":
            return event_marginals_df.copy()
        raise AssertionError(path)

    class FakeGDTable:
        def __init__(self, path):
            self.loci = {"cluster1": SimpleNamespace(start=120, end=280)}

    monkeypatch.setattr(aggregate, "_read_tsv", fake_read_tsv)
    monkeypatch.setattr(aggregate, "GDTable", FakeGDTable)

    context = aggregate._load_plot_run_context(run)

    assert context.unavailable_reason is None
    assert list(context.loci_by_cluster) == ["cluster1"]
    assert context.depth_by_cluster["cluster1"]["Start"].tolist() == [100, 200]
    assert context.minor_baf_by_cluster["cluster1"]["S1"].tolist() == pytest.approx([0.4, 0.3])
    assert context.baf_variance_by_cluster["cluster1"]["S1"].tolist() == pytest.approx([0.02, 0.03])
    assert context.baf_sites_by_cluster["cluster1"]["S1"].tolist() == pytest.approx([5, 6])
    assert context.event_del_by_cluster["cluster1"]["S1"].tolist() == pytest.approx([0.8])
    assert context.event_dup_by_cluster["cluster1"]["S1"].tolist() == pytest.approx([0.1])
    assert context.baf_temperature_by_sample == {"S1": pytest.approx(0.25)}


def test_add_case_plot_page_handles_unavailable_and_missing_data_paths(monkeypatch):
    case = pd.Series({"sample": "S1", "batch_label": "batch", "GD_ID": "GD1", "cluster": "cluster1"})
    run = aggregate.RunData(
        batch_id=1,
        batch_label="batch",
        work_dir=Path("/tmp/run"),
        work_dir_input="/tmp/run",
        calls_df=pd.DataFrame({"cluster": ["cluster1"], "sample": ["S1"], "qual_score": [10.0]}),
        ploidy_df=pd.DataFrame(),
        gd_table_df=pd.DataFrame(),
        eval_df=None,
        optional_artifact_status={},
        missing_artifacts=[],
    )
    recorded_reasons = []

    monkeypatch.setattr(
        aggregate,
        "_add_case_plot_unavailable_page",
        lambda pdf, pdf_state, case_row, **kwargs: recorded_reasons.append(kwargs["reason"]),
    )

    aggregate._add_case_plot_page(
        None,
        {"page": 0},
        case,
        run,
        aggregate.PlotRunContext({}, {}, {}, {}, {}, {}, {}, {}, unavailable_reason="missing posterior data"),
        section_number=1,
        section_title="Confident GD Calls",
    )
    aggregate._add_case_plot_page(
        None,
        {"page": 0},
        case,
        run,
        aggregate.PlotRunContext({}, {}, {}, {}, {}, {}, {}, {}),
        section_number=1,
        section_title="Confident GD Calls",
    )
    aggregate._add_case_plot_page(
        None,
        {"page": 0},
        case,
        run,
        aggregate.PlotRunContext(
            {"cluster1": SimpleNamespace(start=120, end=180)},
            {"cluster1": pd.DataFrame({"Start": [100], "End": [200], "S2": [2.0]})},
            {},
            {},
            {},
            {},
            {},
            {},
        ),
        section_number=1,
        section_title="Confident GD Calls",
    )

    assert recorded_reasons == [
        "missing posterior data",
        "no locus-specific depth data were available for this cluster",
        "the sample is not present in the inferred depth matrix for this locus",
    ]


def test_add_case_plot_page_renders_and_falls_back_when_renderer_returns_false(monkeypatch):
    case = pd.Series({"sample": "S1", "batch_label": "batch", "GD_ID": "GD1", "cluster": "cluster1"})
    run = aggregate.RunData(
        batch_id=1,
        batch_label="batch",
        work_dir=Path("/tmp/run"),
        work_dir_input="/tmp/run",
        calls_df=pd.DataFrame({"cluster": ["cluster1"], "sample": ["S1"], "qual_score": [25.0]}),
        ploidy_df=pd.DataFrame(),
        gd_table_df=pd.DataFrame(),
        eval_df=None,
        optional_artifact_status={},
        missing_artifacts=[],
    )
    region_df = pd.DataFrame({"Start": [100, 200], "End": [200, 300], "S1": [2.0, 3.0]})
    plot_context = aggregate.PlotRunContext(
        {"cluster1": SimpleNamespace(start=125, end=275)},
        {"cluster1": region_df},
        {"cluster1": pd.DataFrame({"Start": [100], "End": [200], "S1": [0.4]})},
        {"cluster1": pd.DataFrame({"Start": [100], "End": [200], "S1": [0.03]})},
        {"cluster1": pd.DataFrame({"Start": [100], "End": [200], "S1": [5]})},
        {"cluster1": pd.DataFrame({"Start": [100], "End": [200], "S1": [0.8]})},
        {"cluster1": pd.DataFrame({"Start": [100], "End": [200], "S1": [0.1]})},
        {"S1": 0.25},
    )
    renders = []
    recorded_reasons = []
    render_results = iter([True, False])

    monkeypatch.setattr(aggregate, "FlankCompressor", lambda *args, **kwargs: {"args": args, "kwargs": kwargs})
    monkeypatch.setattr(
        aggregate,
        "_render_pdf_sample_page",
        lambda *args, **kwargs: renders.append((args, kwargs)) or next(render_results),
    )
    monkeypatch.setattr(
        aggregate,
        "_add_case_plot_unavailable_page",
        lambda pdf, pdf_state, case_row, **kwargs: recorded_reasons.append(kwargs["reason"]),
    )

    pdf_state = {"page": 0}
    aggregate._add_case_plot_page(
        None,
        pdf_state,
        case,
        run,
        plot_context,
        section_number=2,
        section_title="Confident GD Calls",
    )
    assert pdf_state["page"] == 1
    assert renders[0][0][1] == "S1"
    assert renders[0][0][2] == "cluster1"
    assert renders[0][0][6] == "qual_score"
    assert renders[0][1]["target_gd_id"] == "GD1"

    aggregate._add_case_plot_page(
        None,
        pdf_state,
        case,
        run,
        plot_context,
        section_number=2,
        section_title="Confident GD Calls",
    )
    assert pdf_state["page"] == 1
    assert recorded_reasons == [
        "the plot renderer could not produce a page for this sample/locus combination",
    ]


def test_draw_case_evidence_plot_handles_empty_and_metric_cases():
    empty_fig = plt.figure()
    aggregate._draw_case_evidence_plot(
        empty_fig,
        pd.Series(dtype=object),
        left=0.1,
        bottom=0.1,
        width=0.8,
        height=0.6,
    )
    assert any("No numeric evidence fields available" in text.get_text() for text in empty_fig.texts)
    assert empty_fig.axes == []

    metric_fig = plt.figure()
    aggregate._draw_case_evidence_plot(
        metric_fig,
        pd.Series(
            {
                "confidence_value": 75.0,
                "min_interval_confidence": 60.0,
                "min_flank_non_event_confidence": 55.0,
                "interval_coverage": 0.8,
                "reciprocal_overlap": 0.6,
            }
        ),
        left=0.1,
        bottom=0.1,
        width=0.8,
        height=0.6,
    )
    assert len(metric_fig.axes) == 2
    assert metric_fig.axes[0].get_xlabel() == "QUAL/confidence"
    assert metric_fig.axes[1].get_xlabel() == "Fraction"

    overlap_only_fig = plt.figure()
    aggregate._draw_case_evidence_plot(
        overlap_only_fig,
        pd.Series({"interval_coverage": 0.75}),
        left=0.1,
        bottom=0.1,
        width=0.8,
        height=0.6,
    )
    assert overlap_only_fig.axes[0].axison is False
    assert overlap_only_fig.axes[1].get_xlabel() == "Fraction"

    confidence_only_fig = plt.figure()
    aggregate._draw_case_evidence_plot(
        confidence_only_fig,
        pd.Series({"confidence_value": 85.0}),
        left=0.1,
        bottom=0.1,
        width=0.8,
        height=0.6,
    )
    assert confidence_only_fig.axes[0].get_xlabel() == "QUAL/confidence"
    assert confidence_only_fig.axes[1].axison is False

    plt.close(empty_fig)
    plt.close(metric_fig)
    plt.close(overlap_only_fig)
    plt.close(confidence_only_fig)


def test_add_case_plot_unavailable_page_writes_reason_block(monkeypatch):
    captured = {}

    def fake_new_page(pdf_state, header):
        captured["header"] = header
        return plt.figure()

    def fake_draw_paragraph(fig, text, start_y, fontsize=9):
        captured["paragraph"] = text
        return 0.65

    monkeypatch.setattr(aggregate, "_new_page", fake_new_page)
    monkeypatch.setattr(aggregate, "_section_band", lambda fig, top, title, eyebrow=None: 0.75)
    monkeypatch.setattr(aggregate, "_draw_paragraph", fake_draw_paragraph)
    monkeypatch.setattr(
        aggregate,
        "_draw_kv_block",
        lambda fig, items, **kwargs: captured.setdefault("items", items),
    )
    monkeypatch.setattr(aggregate, "_save_page", lambda pdf, fig: captured.setdefault("saved", True))

    aggregate._add_case_plot_unavailable_page(
        None,
        {"page": 0},
        pd.Series({"sample": "S1", "batch_label": "batch", "GD_ID": "GD1", "cluster": "cluster1"}),
        section_number=4,
        section_title="Confident GD Calls",
        reason="missing plot inputs",
    )

    assert captured["header"] == "Section 4 - Confident GD Calls"
    assert "could not be rendered" in captured["paragraph"]
    assert ("Reason", "missing plot inputs") in captured["items"]
    assert captured["saved"] is True


def test_add_case_pages_handles_missing_runs_and_reuses_plot_context(monkeypatch):
    cases_df = pd.DataFrame(
        [
            {"batch_id": 1, "batch_label": "batch_a", "cluster": "cluster1", "GD_ID": "GD1", "sample": "S1", "carrier_category": "high_confidence_carrier"},
            {"batch_id": 1, "batch_label": "batch_a", "cluster": "cluster1", "GD_ID": "GD1", "sample": "S2", "carrier_category": "high_confidence_carrier"},
            {"batch_id": 2, "batch_label": "batch_b", "cluster": "cluster2", "GD_ID": "GD2", "sample": "S3", "carrier_category": "low_confidence_carrier"},
        ]
    )
    run = aggregate.RunData(
        batch_id=1,
        batch_label="batch_a",
        work_dir=Path("/tmp/run_a"),
        work_dir_input="/tmp/run_a",
        calls_df=pd.DataFrame(),
        ploidy_df=pd.DataFrame(),
        gd_table_df=pd.DataFrame(),
        eval_df=None,
        optional_artifact_status={},
        missing_artifacts=[],
    )
    records = {"dividers": [], "case_pages": [], "unavailable": [], "plot_pages": [], "contexts": []}

    monkeypatch.setattr(
        aggregate,
        "_add_section_divider",
        lambda pdf, pdf_state, section_number, title, count: records["dividers"].append((section_number, title, count)),
    )
    monkeypatch.setattr(
        aggregate,
        "_add_case_page",
        lambda pdf, pdf_state, case, **kwargs: records["case_pages"].append((case["sample"], kwargs["section_number"], kwargs["section_title"])),
    )
    monkeypatch.setattr(
        aggregate,
        "_add_case_plot_unavailable_page",
        lambda pdf, pdf_state, case, **kwargs: records["unavailable"].append((case["sample"], kwargs["reason"])),
    )
    monkeypatch.setattr(
        aggregate,
        "_load_plot_run_context",
        lambda passed_run: records["contexts"].append(passed_run.batch_id) or aggregate.PlotRunContext({}, {}, {}, {}, {}, {}, {}, {}),
    )
    monkeypatch.setattr(
        aggregate,
        "_add_case_plot_page",
        lambda pdf, pdf_state, case, passed_run, plot_context, **kwargs: records["plot_pages"].append((case["sample"], passed_run.batch_id, kwargs["section_number"])),
    )

    aggregate._add_case_pages(None, {"page": 0}, cases_df, [run])

    assert records["dividers"] == [
        (4, "Confident GD Calls", 2),
        (5, "Non-confident GD Calls", 1),
    ]
    assert [sample for sample, _, _ in records["case_pages"]] == ["S1", "S2", "S3"]
    assert records["plot_pages"] == [("S1", 1, 4), ("S2", 1, 4)]
    assert records["contexts"] == [1]
    assert records["unavailable"] == [
        ("S3", "the aggregate run metadata for this batch were unavailable"),
    ]


def test_add_missing_pages_and_field_guide_cover_appendix_helpers(monkeypatch):
    captured = {}
    missing_df = pd.DataFrame(
        {
            "batch_label": ["batch_a"],
            "artifact": ["carrier_plots_pdf"],
            "reason": ["missing"],
            "path": ["/tmp/run/plot/carrier_plots.pdf"],
        }
    )

    monkeypatch.setattr(
        aggregate,
        "_draw_table_section_pages",
        lambda pdf, pdf_state, df, columns, **kwargs: captured.setdefault(
            "call",
            (df.copy(), list(columns), kwargs),
        ),
    )

    aggregate._add_missing_pages(None, {"page": 0}, missing_df)
    field_guide = aggregate._field_guide_table()

    assert captured["call"][0].equals(missing_df)
    assert captured["call"][1] == ["batch_label", "artifact", "reason", "path"]
    assert captured["call"][2]["title"] == "Missing Optional Artifacts"
    assert "Per-sample plot" in set(field_guide["displayed_label"])
    assert "Evaluation Summary" in set(field_guide["display_element"])


def test_load_run_rejects_missing_confidence_column(tmp_path):
    run_dir = tmp_path / "run"
    call = _call_row("S1")
    call.pop("qual_score")
    call.pop("confidence_score")
    call.pop("log_prob_score")
    _write_run(run_dir, [call])

    with pytest.raises(ValueError, match="qual_score"):
        aggregate._load_run_data(run_dir, batch_id=1, batch_label="batch")


def test_optional_artifacts_include_anomalous_discrepancy_pdf():
    assert (
        "anomalous_discrepancies_pdf",
        ("plot", "anomalous_discrepancies.pdf"),
    ) in aggregate._OPTIONAL_ARTIFACTS


def test_build_report_tables_classifies_carriers_and_eval(tmp_path):
    run_a = tmp_path / "run_a"
    _write_run(
        run_a,
        [
            _call_row("S1", is_carrier=True, qual_score=80.0),
            _call_row("S2", is_carrier=False, is_best_match=True, qual_score=20.0),
            _call_row("S3", is_carrier=False, is_best_match=False, qual_score=99.0),
        ],
        include_eval=True,
    )
    run_b = tmp_path / "run_b"
    low_confidence_no_best = _call_row("S4", is_carrier=False, is_best_match=False, qual_score=90.0)
    _write_run(run_b, [low_confidence_no_best])
    runs = [
        aggregate._load_run_data(str(run_a), batch_id=1, batch_label="run_a"),
        aggregate._load_run_data(str(run_b), batch_id=2, batch_label="run_b"),
    ]

    summary_df, inventory_df, calls_df, cases_df, locus_summary_df, eval_df, missing_df = aggregate._build_report_tables(
        runs,
        min_confidence=0.5,
    )

    summary = summary_df.set_index("metric")["value"].to_dict()
    assert summary["n_batches"] == 2
    assert summary["n_samples"] == 4
    assert summary["n_carrier_events"] == 2
    assert summary["n_high_confidence_carrier_events"] == 1
    assert summary["n_low_confidence_carrier_events"] == 1
    assert summary["aggregate_TP"] == 1
    assert summary["aggregate_FN"] == 1
    assert summary["aggregate_min_confidence"] == 0.5
    assert summary["call_interval_confidence_threshold"] == 60.0
    assert summary["call_flank_non_event_confidence_threshold"] == 60.0
    assert bool(inventory_df.set_index("batch_label").loc["run_a", "eval_report_present"])
    assert len(calls_df) == 4
    assert cases_df["sample"].tolist() == ["S1", "S2"]
    assert set(locus_summary_df["carrier_record_count"]) == {2}
    assert len(eval_df) == 1
    assert missing_df["artifact"].isin(["sample_posteriors", "carrier_plots_pdf"]).any()
    assert "n_call_records" not in set(summary_df["metric"])


def test_best_match_non_carriers_appear_as_non_confident_cases(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [
            _call_row(
                "S1",
                is_carrier=True,
                qual_score=80.0,
                left_flank_non_event_median=95.0,
                right_flank_non_event_median=10.0,
            ),
            _call_row(
                "S2",
                is_carrier=False,
                is_best_match=True,
                qual_score=20.0,
                left_flank_non_event_median=95.0,
                right_flank_non_event_median=95.0,
            ),
            _call_row("S3", is_carrier=False, is_best_match=False, qual_score=0.6),
        ],
    )
    run = aggregate._load_run_data(str(run_dir), batch_id=1, batch_label="run")

    _, _, calls_df, cases_df, locus_summary_df, _, _ = aggregate._build_report_tables(
        [run],
        min_confidence=0.5,
    )

    assert set(cases_df["sample"]) == {"S1", "S2"}
    assert calls_df.set_index("sample").loc["S3", "carrier_category"] == "non_carrier"
    assert locus_summary_df.iloc[0]["low_confidence_carrier_record_count"] == 1
    assert cases_df.set_index("sample").loc["S1", "left_flank_status"] == "PASS"
    assert cases_df.set_index("sample").loc["S1", "right_flank_status"] == "FAIL"
    assert cases_df.set_index("sample").loc["S2", "left_flank_status"] == "PASS"
    assert cases_df.set_index("sample").loc["S2", "right_flank_status"] == "PASS"


def test_non_confident_calls_below_min_confidence_are_hidden(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [
            _call_row("S1", is_carrier=True, qual_score=80.0),
            _call_row("S2", is_carrier=False, is_best_match=True, qual_score=0.4),
            _call_row("S3", is_carrier=False, is_best_match=True, qual_score=0.6),
        ],
    )
    run = aggregate._load_run_data(str(run_dir), batch_id=1, batch_label="run")

    _, _, calls_df, cases_df, locus_summary_df, _, _ = aggregate._build_report_tables(
        [run],
        min_confidence=0.5,
    )

    assert set(cases_df["sample"]) == {"S1", "S3"}
    assert not bool(calls_df.set_index("sample").loc["S2", "is_low_confidence_carrier"])
    assert calls_df.set_index("sample").loc["S2", "carrier_category"] == "non_carrier"
    assert locus_summary_df.iloc[0]["low_confidence_carrier_record_count"] == 1


def test_pdf_structure_includes_case_sections_and_evidence_plot(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [
            _call_row("S1", is_carrier=True, qual_score=80.0),
            _call_row("S2", is_carrier=False, is_best_match=True, qual_score=20.0),
        ],
    )
    run = aggregate._load_run_data(str(run_dir), batch_id=1, batch_label="run")
    summary_df, inventory_df, _, cases_df, locus_summary_df, eval_df, missing_df = aggregate._build_report_tables(
        [run],
        min_confidence=0.5,
    )

    toc_entries = dict(
        aggregate._build_toc_entries(
            inventory_df,
            summary_df,
            cases_df,
            locus_summary_df,
            eval_df,
            missing_df,
        )
    )
    field_guide = aggregate._field_guide_table()

    assert toc_entries["Confident GD Calls"] < toc_entries["Non-confident GD Calls"]
    assert "Missing Optional Artifacts" not in toc_entries
    assert "Per-sample plot" in set(field_guide["displayed_label"])
    assert "Missing Optional Artifacts" not in set(field_guide["display_element"])


def test_run_aggregate_writes_sidecars_and_pdf(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "aggregate"
    work_dir_arg = "{}/".format(run_dir)
    _write_run(
        run_dir,
        [
            _call_row("S1", is_carrier=True, qual_score=80.0),
            _call_row("S2", is_carrier=False, is_best_match=False, qual_score=20.0),
        ],
    )
    args = argparse.Namespace(
        work_dirs=[work_dir_arg],
        output_dir=str(output_dir),
        output_name="aggregate_report.pdf",
        min_confidence=0.5,
        batch_label=["batch"],
    )

    aggregate._run_aggregate(args)

    expected_outputs = [
        "aggregate_report.pdf",
        "aggregate_summary.tsv",
        "aggregate_inventory.tsv",
        "aggregate_calls.tsv",
        "aggregate_cases.tsv",
        "aggregate_locus_summary.tsv",
        "aggregate_eval.tsv",
        "aggregate_missing_artifacts.tsv",
    ]
    for output_name in expected_outputs:
        output_path = output_dir / output_name
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    cases_df = pd.read_csv(output_dir / "aggregate_cases.tsv", sep="\t")
    assert cases_df["sample"].tolist() == ["S1"]
    assert cases_df["work_dir"].tolist() == [work_dir_arg]
    assert "batch_id" not in cases_df.columns
    assert "batch_label" not in cases_df.columns

    inventory_df = pd.read_csv(output_dir / "aggregate_inventory.tsv", sep="\t")
    assert inventory_df["work_dir"].tolist() == [work_dir_arg]
    assert "batch_id" not in inventory_df.columns
    assert "batch_label" not in inventory_df.columns

    calls_df = pd.read_csv(output_dir / "aggregate_calls.tsv", sep="\t")
    assert set(calls_df["work_dir"]) == {work_dir_arg}
    assert "batch_id" not in calls_df.columns
    assert "batch_label" not in calls_df.columns

    locus_summary_df = pd.read_csv(output_dir / "aggregate_locus_summary.tsv", sep="\t")
    assert locus_summary_df["work_dir"].tolist() == [work_dir_arg]
    assert "batch_id" not in locus_summary_df.columns
    assert "batch_label" not in locus_summary_df.columns

    eval_df = pd.read_csv(output_dir / "aggregate_eval.tsv", sep="\t")
    assert list(eval_df.columns) == ["work_dir", "GD_ID"]

    missing_df = pd.read_csv(output_dir / "aggregate_missing_artifacts.tsv", sep="\t")
    assert set(missing_df["work_dir"]) == {work_dir_arg}
    assert "batch_label" not in missing_df.columns