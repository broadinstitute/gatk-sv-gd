import pandas as pd

from gatk_sv_gd.eval import evaluate_against_truth


def test_evaluate_against_truth_uses_call_emitted_carriers(tmp_path):
    calls_df = pd.DataFrame(
        {
            "GD_ID": ["GD1", "GD1", "GD1"],
            "sample": ["S1", "S2", "S3"],
            "is_carrier": [True, False, True],
            "is_best_match": [True, True, True],
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