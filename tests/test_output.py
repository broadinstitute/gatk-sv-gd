from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gatk_sv_gd.bins import LocusBinMapping
from gatk_sv_gd.models import GDLocus
from gatk_sv_gd.output import (
    build_ploidy_map,
    estimate_ploidy,
    write_locus_metadata,
    write_posterior_tables,
)


class _TensorStub:
    def __init__(self, array):
        self._array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self._array


def _make_locus() -> GDLocus:
    return GDLocus(
        cluster="cluster1",
        chrom="chr1",
        breakpoints=[(100, 110), (200, 210), (300, 310)],
        breakpoint_names=["A", "B", "C"],
        gd_entries=[
            {
                "GD_ID": "GD_DEL",
                "svtype": "DEL",
                "BP1": "A",
                "BP2": "C",
                "start_GRCh38": 110,
                "end_GRCh38": 300,
            }
        ],
        is_nahr=True,
        is_terminal=False,
    )


def _make_mapping(interval_name: str, array_idx: int, start: int, end: int, locus=None):
    return LocusBinMapping(
        cluster="cluster1",
        locus=locus,
        interval_name=interval_name,
        array_idx=array_idx,
        chrom="chr1",
        start=start,
        end=end,
    )


def test_write_posterior_tables_handles_single_bin_single_sample_and_baf(tmp_path):
    combined_data = SimpleNamespace(
        n_bins=1,
        n_samples=1,
        sample_ids=["S1"],
        depth=_TensorStub([2.25]),
        has_baf=True,
        has_baf_effective_count=True,
        baf_median=_TensorStub([[0.40]]),
        minor_baf_median=_TensorStub([[0.10]]),
        baf_variance=_TensorStub([[0.02]]),
        baf_n_sites=_TensorStub([[5]]),
        baf_effective_variance=_TensorStub([[0.03]]),
        baf_effective_n_sites=_TensorStub([[4]]),
    )
    mappings = [_make_mapping("A-B", 0, 110, 200)]
    map_estimates = {
        "cn": 1,
        "pair_state": 1,
        "sample_var": 0.125,
        "sample_df": 4.5,
        "baf_temperature": 1.5,
        "length_scale_var": 250.0,
        "bin_bias": 0.95,
        "bin_var": 0.0,
        "cn_probs": [0.3, 0.7],
        "effective_pair_state_probs": [0.4, 0.6],
        "null_state_prior": 0.05,
    }
    cn_posterior = {
        "cn_posterior": [0.2, 0.8],
        "pair_state_posterior": [0.1, 0.9],
        "pair_state_labels": [(0, 1), (1, 1)],
        "null_posterior": 0.15,
    }

    write_posterior_tables(
        combined_data,
        map_estimates,
        cn_posterior,
        mappings,
        str(tmp_path),
    )

    cn_df = pd.read_csv(tmp_path / "cn_posteriors.tsv.gz", sep="\t")
    sample_df = pd.read_csv(tmp_path / "sample_posteriors.tsv.gz", sep="\t")
    bin_df = pd.read_csv(tmp_path / "bin_posteriors.tsv.gz", sep="\t")

    cn_row = cn_df.iloc[0]
    assert cn_row["sample"] == "S1"
    assert cn_row["depth"] == pytest.approx(2.25)
    assert cn_row["prob_null"] == pytest.approx(0.15)
    assert cn_row["prob_cn_0"] == pytest.approx(0.2)
    assert cn_row["prob_cn_1"] == pytest.approx(0.8)
    assert cn_row["cn_map"] == 1
    assert cn_row["pair_state_map"] == 1
    assert cn_row["pair_h1_map"] == 1
    assert cn_row["pair_h2_map"] == 1
    assert cn_row["baf_median"] == pytest.approx(0.40)
    assert cn_row["minor_baf_median"] == pytest.approx(0.10)
    assert cn_row["baf_variance"] == pytest.approx(0.02)
    assert cn_row["baf_n_sites"] == 5
    assert cn_row["baf_effective_variance"] == pytest.approx(0.03)
    assert cn_row["baf_effective_n_sites"] == 4
    assert cn_row["prob_pair_0_1"] == pytest.approx(0.1)
    assert cn_row["prob_pair_1_1"] == pytest.approx(0.9)

    sample_row = sample_df.iloc[0]
    assert sample_row["sample_var_map"] == pytest.approx(0.125)
    assert sample_row["baf_temperature_map"] == pytest.approx(1.5)
    assert sample_row["baf_variance_scale_map"] == pytest.approx(1.5)
    assert sample_row["length_scale_var_map"] == pytest.approx(250.0)
    assert sample_row["sample_df_map"] == pytest.approx(4.5)

    bin_row = bin_df.iloc[0]
    assert bin_row["bin_bias_map"] == pytest.approx(0.95)
    assert bin_row["bin_var_map"] == pytest.approx(0.0)
    assert bin_row["null_prior"] == pytest.approx(0.05)
    assert bin_row["cn_prior_0"] == pytest.approx(0.3)
    assert bin_row["cn_prior_1"] == pytest.approx(0.7)
    assert bin_row["pair_prior_0_1"] == pytest.approx(0.4)
    assert bin_row["pair_prior_1_1"] == pytest.approx(0.6)


def test_write_posterior_tables_transposes_sample_bin_inputs(tmp_path):
    combined_data = SimpleNamespace(
        n_bins=2,
        n_samples=3,
        sample_ids=["S1", "S2", "S3"],
        depth=_TensorStub([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        has_baf=False,
        has_baf_effective_count=False,
    )
    mappings = [
        _make_mapping("A-B", 0, 110, 200),
        _make_mapping("B-C", 1, 210, 300),
    ]
    map_estimates = {
        "cn": np.array([[0, 1], [1, 0], [1, 1]]),
        "sample_var": np.array([0.1, 0.2, 0.3]),
        "bin_bias": np.array([1.1, 0.9]),
        "bin_var": np.array([0.01, 0.02]),
        "cn_probs": np.array([[0.9, 0.1], [0.2, 0.8]]),
    }
    cn_posterior = {
        "cn_posterior": np.array(
            [
                [[0.90, 0.10], [0.20, 0.80]],
                [[0.80, 0.20], [0.30, 0.70]],
                [[0.70, 0.30], [0.40, 0.60]],
            ]
        ),
        "null_posterior": np.array(
            [
                [0.01, 0.02],
                [0.03, 0.04],
                [0.05, 0.06],
            ]
        ),
    }

    write_posterior_tables(
        combined_data,
        map_estimates,
        cn_posterior,
        mappings,
        str(tmp_path),
    )

    cn_df = pd.read_csv(tmp_path / "cn_posteriors.tsv.gz", sep="\t")
    bin0_s3 = cn_df[(cn_df["start"] == 110) & (cn_df["sample"] == "S3")].iloc[0]
    bin1_s2 = cn_df[(cn_df["start"] == 210) & (cn_df["sample"] == "S2")].iloc[0]

    assert bin0_s3["prob_cn_0"] == pytest.approx(0.70)
    assert bin0_s3["prob_cn_1"] == pytest.approx(0.30)
    assert bin0_s3["prob_null"] == pytest.approx(0.05)
    assert bin0_s3["cn_map"] == 1
    assert bin1_s2["prob_cn_0"] == pytest.approx(0.30)
    assert bin1_s2["prob_cn_1"] == pytest.approx(0.70)
    assert bin1_s2["prob_null"] == pytest.approx(0.04)
    assert bin1_s2["cn_map"] == 0


def test_write_posterior_tables_rejects_incompatible_state_tensor_shape(tmp_path):
    combined_data = SimpleNamespace(
        n_bins=2,
        n_samples=2,
        sample_ids=["S1", "S2"],
        depth=_TensorStub([[1.0, 2.0], [3.0, 4.0]]),
        has_baf=False,
        has_baf_effective_count=False,
    )
    mappings = [
        _make_mapping("A-B", 0, 110, 200),
        _make_mapping("B-C", 1, 210, 300),
    ]

    with pytest.raises(ValueError, match="Expected 3D state tensor"):
        write_posterior_tables(
            combined_data,
            {
                "cn": np.array([[0, 1], [1, 0]]),
                "sample_var": np.array([0.1, 0.2]),
                "bin_bias": np.array([1.0, 1.0]),
                "bin_var": np.array([0.0, 0.0]),
                "cn_probs": np.array([[0.5, 0.5], [0.5, 0.5]]),
            },
            {"cn_posterior": np.array([0.5, 0.5])},
            mappings,
            str(tmp_path),
        )


def test_write_posterior_tables_rejects_mismatched_3d_state_tensor_shape(tmp_path):
    combined_data = SimpleNamespace(
        n_bins=3,
        n_samples=2,
        sample_ids=["S1", "S2"],
        depth=_TensorStub([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        has_baf=False,
        has_baf_effective_count=False,
    )
    mappings = [
        _make_mapping("A-B", 0, 110, 200),
        _make_mapping("B-C", 1, 210, 300),
        _make_mapping("C-D", 2, 310, 400),
    ]

    with pytest.raises(ValueError, match="State tensor shape does not match bins/samples"):
        write_posterior_tables(
            combined_data,
            {
                "cn": np.array([[0, 1], [1, 0], [1, 1]]),
                "sample_var": np.array([0.1, 0.2]),
                "bin_bias": np.array([1.0, 1.0, 1.0]),
                "bin_var": np.array([0.0, 0.0, 0.0]),
                "cn_probs": np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
            },
            {"cn_posterior": np.ones((2, 2, 2), dtype=float) / 2.0},
            mappings,
            str(tmp_path),
        )


def test_write_posterior_tables_handles_extra_dim_transposes_and_scalar_expansions(tmp_path):
    combined_data = SimpleNamespace(
        n_bins=2,
        n_samples=3,
        sample_ids=["S1", "S2", "S3"],
        depth=_TensorStub([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        has_baf=True,
        has_baf_effective_count=False,
        baf_median=_TensorStub([[0.40, 0.41, 0.42], [0.30, 0.31, 0.32]]),
        minor_baf_median=_TensorStub([[0.10, 0.11, 0.12], [0.20, 0.21, 0.22]]),
        baf_variance=_TensorStub([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]),
        baf_n_sites=_TensorStub([[5, 6, 7], [8, 9, 10]]),
    )
    mappings = [
        _make_mapping("A-B", 0, 110, 200),
        _make_mapping("B-C", 1, 210, 300),
    ]
    map_estimates = {
        "cn": np.array([[0, 1, 0], [1, 0, 1]]),
        "pair_state": np.array([[0, 1], [1, 0], [0, 1]]),
        "sample_var": np.array([0.1, 0.2, 0.3]),
        "sample_df": np.array([3.5, 4.0, 4.5]),
        "baf_temperature": np.array([2.5]),
        "length_scale_var": np.array([300.0]),
        "bin_bias": np.array([1.1, 0.9]),
        "bin_var": np.array([0.01, 0.02]),
        "cn_probs": np.array([0.6, 0.4]),
        "pair_state_probs": np.array([[0.7, 0.3], [0.4, 0.6]]),
        "null_state_prior": 0.05,
    }
    cn_posterior = {
        "cn_posterior": np.array([[0.9, 0.8, 0.7], [0.1, 0.2, 0.3]]),
        "pair_state_posterior": np.array(
            [[
                [[0.95, 0.85, 0.75], [0.05, 0.15, 0.25]],
                [[0.30, 0.20, 0.10], [0.70, 0.80, 0.90]],
            ]]
        ),
        "pair_state_labels": [(0, 0), (1, 1)],
    }

    write_posterior_tables(
        combined_data,
        map_estimates,
        cn_posterior,
        mappings,
        str(tmp_path),
    )

    cn_df = pd.read_csv(tmp_path / "cn_posteriors.tsv.gz", sep="\t")
    sample_df = pd.read_csv(tmp_path / "sample_posteriors.tsv.gz", sep="\t")
    bin_df = pd.read_csv(tmp_path / "bin_posteriors.tsv.gz", sep="\t")

    first_row = cn_df[(cn_df["start"] == 110) & (cn_df["sample"] == "S1")].iloc[0]
    second_row = cn_df[(cn_df["start"] == 210) & (cn_df["sample"] == "S3")].iloc[0]
    assert first_row["prob_cn_0"] == pytest.approx(0.9)
    assert second_row["prob_cn_0"] == pytest.approx(0.3)
    assert first_row["prob_null"] == pytest.approx(0.0)
    assert second_row["pair_state_map"] == 1
    assert first_row["prob_pair_0_0"] == pytest.approx(0.95)
    assert second_row["prob_pair_1_1"] == pytest.approx(0.90)
    assert "baf_effective_variance" not in cn_df.columns
    assert "baf_effective_n_sites" not in cn_df.columns

    assert sample_df["baf_temperature_map"].tolist() == pytest.approx([2.5, 2.5, 2.5])
    assert sample_df["length_scale_var_map"].tolist() == pytest.approx([300.0, 300.0, 300.0])
    assert sample_df["sample_df_map"].tolist() == pytest.approx([3.5, 4.0, 4.5])

    assert bin_df["cn_prior_0"].tolist() == pytest.approx([0.6, 0.4])
    assert bin_df["pair_prior_0_0"].tolist() == pytest.approx([0.7, 0.4])
    assert bin_df["pair_prior_1_1"].tolist() == pytest.approx([0.3, 0.6])
    assert bin_df["null_prior"].tolist() == pytest.approx([0.05, 0.05])


@pytest.mark.parametrize(
    ("map_estimates", "cn_posterior", "message"),
    [
        (
            {
                "cn": np.array([0, 1]),
                "sample_var": np.array([0.1, 0.2]),
                "bin_bias": np.array([1.0, 1.0]),
                "bin_var": np.array([0.0, 0.0]),
                "cn_probs": np.array([[0.5, 0.5], [0.5, 0.5]]),
            },
            {"cn_posterior": np.array([[[0.6], [0.4]], [[0.3], [0.7]]])},
            "Expected 2D bin/sample matrix",
        ),
        (
            {
                "cn": np.array([[0, 1], [1, 0]]),
                "sample_var": np.array([0.1, 0.2]),
                "bin_bias": np.array([1.0, 1.0]),
                "bin_var": np.array([0.0, 0.0]),
                "cn_probs": np.array([[[0.5, 0.5], [0.5, 0.5]], [[0.4, 0.6], [0.6, 0.4]]]),
            },
            {"cn_posterior": np.array([[[0.6], [0.4]], [[0.3], [0.7]]])},
            "Expected 2D bin/state matrix",
        ),
        (
            {
                "cn": np.array([[0, 1], [1, 0]]),
                "sample_var": np.array([0.1, 0.2]),
                "bin_bias": np.array([1.0, 1.0]),
                "bin_var": np.array([0.0, 0.0]),
                "cn_probs": np.array([[0.5], [0.5]]),
            },
            {"cn_posterior": np.array([[[[0.5, 0.5], [0.5, 0.5]], [[0.4, 0.6], [0.6, 0.4]]], [[[0.5, 0.5], [0.5, 0.5]], [[0.4, 0.6], [0.6, 0.4]]]])},
            "Expected 3D state tensor",
        ),
    ],
)
def test_write_posterior_tables_rejects_incompatible_matrix_shapes(
    tmp_path,
    map_estimates,
    cn_posterior,
    message,
):
    combined_data = SimpleNamespace(
        n_bins=2,
        n_samples=2,
        sample_ids=["S1", "S2"],
        depth=_TensorStub([[1.0, 2.0], [3.0, 4.0]]),
        has_baf=False,
        has_baf_effective_count=False,
    )
    mappings = [
        _make_mapping("A-B", 0, 110, 200),
        _make_mapping("B-C", 1, 210, 300),
    ]

    with pytest.raises(ValueError, match=message):
        write_posterior_tables(
            combined_data,
            map_estimates,
            cn_posterior,
            mappings,
            str(tmp_path),
        )


def test_write_locus_metadata_and_estimate_ploidy_round_trip(tmp_path):
    locus = _make_locus()
    mappings = [
        _make_mapping("A-B", 0, 110, 200, locus=locus),
        _make_mapping("B-C", 1, 210, 300, locus=locus),
    ]

    write_locus_metadata({"cluster1": locus}, mappings, str(tmp_path))

    bin_mappings_df = pd.read_csv(tmp_path / "bin_mappings.tsv.gz", sep="\t")
    locus_intervals_df = pd.read_csv(tmp_path / "locus_intervals.tsv.gz", sep="\t")
    gd_entry_intervals_df = pd.read_csv(tmp_path / "gd_entry_intervals.tsv.gz", sep="\t")

    assert bin_mappings_df["interval"].tolist() == ["A-B", "B-C"]
    assert locus_intervals_df["interval"].tolist() == ["A-B", "B-C"]
    assert gd_entry_intervals_df["GD_ID"].tolist() == ["GD_DEL", "GD_DEL"]
    assert gd_entry_intervals_df["interval"].tolist() == ["A-B", "B-C"]
    assert gd_entry_intervals_df["interval_start"].tolist() == [110, 210]
    assert gd_entry_intervals_df["interval_end"].tolist() == [200, 300]

    depth_df = pd.DataFrame(
        {
            "Chr": ["chr1", "chr1", "chrX", "chrX"],
            "Start": [0, 100, 0, 100],
            "End": [100, 200, 100, 200],
            "S1": [2.1, 1.9, 1.1, 0.9],
            "S2": [3.2, 2.8, 1.8, 2.2],
        }
    )

    ploidy_df = estimate_ploidy(depth_df, str(tmp_path))
    ploidy_lookup = build_ploidy_map(ploidy_df)

    assert ploidy_lookup == {
        ("S1", "chr1"): 2,
        ("S2", "chr1"): 3,
        ("S1", "chrX"): 1,
        ("S2", "chrX"): 2,
    }
    assert ploidy_df.set_index(["sample", "contig"]).loc[("S1", "chr1"), "median_depth"] == pytest.approx(2.0)
    assert ploidy_df.set_index(["sample", "contig"]).loc[("S2", "chr1"), "median_depth"] == pytest.approx(3.0)
    written_ploidy_df = pd.read_csv(tmp_path / "ploidy_estimates.tsv", sep="\t")
    pd.testing.assert_frame_equal(ploidy_df, written_ploidy_df)