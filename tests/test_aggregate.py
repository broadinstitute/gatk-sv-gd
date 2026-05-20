import argparse
from pathlib import Path

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
    call_criteria_mean_coverage=float("nan"),
    call_criteria_interval_confidence=60.0,
    call_criteria_flank_non_event_confidence=60.0,
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
        "left_flank_non_event_median": 2.0,
        "right_flank_non_event_median": 2.0,
        "min_flank_non_event_confidence": qual_score,
        "is_carrier": is_carrier,
        "is_best_match": is_best_match,
        "log_prob_score": 0.95,
        "confidence_score": qual_score,
        "qual_score": qual_score,
        "calling_method": calling_method,
        "call_criteria_mean_coverage": call_criteria_mean_coverage,
        "call_criteria_interval_confidence": call_criteria_interval_confidence,
        "call_criteria_flank_non_event_confidence": call_criteria_flank_non_event_confidence,
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
        batch_label=None,
    )

    with pytest.raises(ValueError, match="PDF filename"):
        aggregate._validate_args(args)

    args.output_name = "report.pdf"
    args.batch_label = ["only_one"]
    with pytest.raises(ValueError, match="once per work directory"):
        aggregate._validate_args(args)


def test_load_run_rejects_missing_confidence_column(tmp_path):
    run_dir = tmp_path / "run"
    call = _call_row("S1")
    call.pop("qual_score")
    call.pop("confidence_score")
    call.pop("log_prob_score")
    _write_run(run_dir, [call])

    with pytest.raises(ValueError, match="qual_score"):
        aggregate._load_run_data(run_dir, batch_id=1, batch_label="batch")


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

    summary_df, inventory_df, calls_df, cases_df, locus_summary_df, eval_df, missing_df = aggregate._build_report_tables(runs)

    summary = summary_df.set_index("metric")["value"].to_dict()
    assert summary["n_batches"] == 2
    assert summary["n_samples"] == 4
    assert summary["n_carrier_events"] == 2
    assert summary["n_high_confidence_carrier_events"] == 1
    assert summary["n_low_confidence_carrier_events"] == 1
    assert summary["aggregate_TP"] == 1
    assert summary["aggregate_FN"] == 1
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
            _call_row("S1", is_carrier=True, qual_score=80.0),
            _call_row("S2", is_carrier=False, is_best_match=True, qual_score=20.0),
            _call_row("S3", is_carrier=False, is_best_match=False, qual_score=0.6),
        ],
    )
    run = aggregate._load_run_data(str(run_dir), batch_id=1, batch_label="run")

    _, _, calls_df, cases_df, locus_summary_df, _, _ = aggregate._build_report_tables([run])

    assert set(cases_df["sample"]) == {"S1", "S2"}
    assert calls_df.set_index("sample").loc["S3", "carrier_category"] == "non_carrier"
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
    summary_df, inventory_df, _, cases_df, locus_summary_df, eval_df, missing_df = aggregate._build_report_tables([run])

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
    assert "Evidence Plot" in set(field_guide["displayed_label"])


def test_run_aggregate_writes_sidecars_and_pdf(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "aggregate"
    _write_run(
        run_dir,
        [
            _call_row("S1", is_carrier=True, qual_score=80.0),
            _call_row("S2", is_carrier=False, is_best_match=False, qual_score=20.0),
        ],
    )
    args = argparse.Namespace(
        work_dirs=[str(run_dir)],
        output_dir=str(output_dir),
        output_name="aggregate_report.pdf",
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