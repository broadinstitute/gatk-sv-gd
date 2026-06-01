import sys

import pandas as pd

from gatk_sv_gd import eval as eval_module
from gatk_sv_gd.eval import evaluate_against_truth, load_truth_table


def test_evaluate_against_truth_uses_call_emitted_carriers(tmp_path):
    calls_df = pd.DataFrame(
        {
            "GD_ID": ["GD1", "GD1", "GD1"],
            "sample": ["S1", "S2", "S3"],
            "is_carrier": [True, False, True],
            "is_best_match": [True, True, True],
            "is_null_anomalous": [False, True, True],
            "null_anomaly_score": [0.01, 0.2, 0.3],
            "qual_score": [20.0, 5.0, 20.0],
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 100, 100],
            "end": [200, 200, 200],
            "cluster": ["cluster1", "cluster1", "cluster1"],
            "svtype": ["DEL", "DEL", "DEL"],
        }
    )
    truth_df = pd.DataFrame(
        [
            {
                "GD_ID": "GD1",
                "carrier_set": {"S1", "S2"},
                "chr": "chr1",
                "start": 100,
                "end": 200,
                "cluster_ID": "cluster1",
                "SVTYPE": "DEL",
            }
        ]
    )

    report_df = evaluate_against_truth(
        calls_df,
        truth_df,
        str(tmp_path),
        batch_samples={"S1", "S2", "S3"},
    )

    row = report_df.iloc[0]
    assert row["TP_samples"] == "S1"
    assert row["FP_samples"] == "S3"
    assert row["FN_samples"] == "S2"
    assert row["anomalous_discrepancy_samples"] == "S2,S3"


def test_load_truth_table_bed_format_filters_to_canonical_nahr(tmp_path):
    truth_path = tmp_path / "truth_bed.tsv"
    truth_path.write_text(
        "#chrom\tstart\tend\tname\tsvtype\tsamples\tNAHR_GD\tNAHR_GD_atypical\n"
        "chr1\t100\t200\tGD1\tDEL\tS1,S2\tTrue\tFalse\n"
        "chr1\t100\t200\tGD2\tDEL\tS3\tFalse\tFalse\n"
        "chr1\t100\t200\tGD3\tDEL\tS4\tTrue\tTrue\n"
    )

    truth_df = load_truth_table(str(truth_path))

    assert truth_df["GD_ID"].tolist() == ["GD1"]
    assert truth_df.iloc[0]["carrier_set"] == {"S1", "S2"}
    assert truth_df.iloc[0]["SVTYPE"] == "DEL"


def test_load_truth_table_detects_synthesize_format(tmp_path):
    truth_path = tmp_path / "truth_synth.tsv"
    truth_path.write_text(
        "sample_id\tGD_ID\n"
        "S1\tGD1\n"
        "S2\tGD1\n"
        "S3\tGD2\n"
    )

    truth_df = load_truth_table(str(truth_path))
    truth_df = truth_df.sort_values("GD_ID").reset_index(drop=True)

    assert truth_df["GD_ID"].tolist() == ["GD1", "GD2"]
    assert truth_df.loc[0, "carrier_set"] == {"S1", "S2"}
    assert truth_df.loc[1, "carrier_set"] == {"S3"}


def test_eval_main_filters_unmodeled_truth_entries_and_writes_report(monkeypatch, tmp_path):
    calls_path = tmp_path / "calls.tsv"
    calls_df = pd.DataFrame(
        {
            "GD_ID": ["GD1", "GD1"],
            "sample": ["S1", "S2"],
            "is_carrier": [True, False],
            "is_best_match": [True, True],
            "is_null_anomalous": [False, False],
            "qual_score": [25.0, 5.0],
            "chrom": ["chr1", "chr1"],
            "start": [100, 100],
            "end": [200, 200],
            "cluster": ["cluster1", "cluster1"],
            "svtype": ["DEL", "DEL"],
        }
    )
    calls_df.to_csv(calls_path, sep="\t", index=False)

    ploidy_path = tmp_path / "ploidy.tsv"
    pd.DataFrame(
        {
            "sample": ["S1", "S2"],
            "contig": ["chr1", "chr1"],
            "median_depth": [2.0, 2.0],
            "ploidy": [2, 2],
        }
    ).to_csv(ploidy_path, sep="\t", index=False)

    truth_path = tmp_path / "truth.tsv"
    truth_path.write_text(
        "sample_id\tGD_ID\n"
        "S1\tGD1\n"
        "S9\tGD2\n"
    )

    gd_table_path = tmp_path / "gd_table.tsv"
    pd.DataFrame(
        {
            "GD_ID": ["GD1"],
            "chr": ["chr1"],
            "start_GRCh38": [100],
            "end_GRCh38": [200],
            "svtype": ["DEL"],
            "NAHR": ["yes"],
            "terminal": ["no"],
            "cluster": ["cluster1"],
            "BP1": ["A"],
            "BP2": ["B"],
        }
    ).to_csv(gd_table_path, sep="\t", index=False)

    output_dir = tmp_path / "eval_out"
    monkeypatch.setattr(eval_module, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gatk-sv-gd eval",
            "--calls",
            str(calls_path),
            "--truth-table",
            str(truth_path),
            "--gd-table",
            str(gd_table_path),
            "--ploidy-table",
            str(ploidy_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    eval_module.main()

    report_df = pd.read_csv(output_dir / "truth_evaluation_report.tsv", sep="\t")
    assert report_df["GD_ID"].tolist() == ["GD1"]
    assert report_df.loc[0, "TP"] == 1
    assert report_df.loc[0, "FP"] == 0
    assert report_df.loc[0, "FN"] == 0