import logging
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gatk_sv_gd.infer as infer_module
from gatk_sv_gd.infer import (
    _align_normalization_metadata,
    _flatten_multi_args,
    _setup_pyro,
    _write_training_loss_history,
    run_gd_analysis,
    parse_args,
)


def _make_infer_args(**overrides):
    args = dict(
        median_min=0.5,
        median_max=3.0,
        mad_max=0.5,
        high_res_counts=None,
        exclusion_threshold=0.5,
        locus_padding=100,
        min_bins_per_interval=1,
        max_bins_per_interval=10,
        exclusion_bypass_threshold=0.0,
        min_rebin_coverage=0.5,
        min_flank_bases=100,
        min_flank_bins=1,
        min_flank_coverage=0.5,
        clamp_threshold=10.0,
        alpha_ref=100.0,
        alpha_non_ref=10.0,
        null_state_prior=1e-3,
        baf_temperature=1.0,
        fixed_baf_temperature=False,
        baf_temperature_prior_scale=0.5,
        baf_outlier_rate=0.0,
        use_baf_effective_count=True,
        var_bias_bin=0.1,
        var_sample=0.2,
        var_bin=0.3,
        freeze_bin_bias=False,
        freeze_bin_var=False,
        freeze_pair_state_priors=False,
        bin_size_factor=1.0,
        var_length_scale=20000.0,
        guide_type="delta",
        max_iter=25,
        guide_warmup_iter=5,
        lr_init=0.1,
        lr_min=0.01,
        lr_decay=0.9,
        log_freq=2,
        disable_jit=True,
        early_stopping=False,
        patience=10,
        elbo_window=3,
        elbo_rtol=1e-4,
        n_discrete_samples=7,
        output_dir="./out",
        preprocessed_dir=None,
        input=None,
        gd_table=None,
        device="cpu",
        verbose=False,
        exclusion_intervals=[],
        hard_inclusion_intervals=[],
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_parse_args_uses_learned_baf_temperature_by_default(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["gatk-sv-gd infer", "--preprocessed-dir", "./preprocess", "-o", "./out"],
    )

    args = parse_args()

    assert args.fixed_baf_temperature is False
    assert args.baf_outlier_rate == pytest.approx(0.0)
    assert args.use_baf_effective_count is True
    assert args.null_state_prior == pytest.approx(1e-3)
    assert args.var_length_scale == pytest.approx(20000.0)
    assert not hasattr(args, "state_prior_weight")


def test_parse_args_allows_explicit_fixed_baf_temperature(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gatk-sv-gd infer",
            "--preprocessed-dir",
            "./preprocess",
            "-o",
            "./out",
            "--fixed-baf-temperature",
        ],
    )

    args = parse_args()

    assert args.fixed_baf_temperature is True
    assert args.null_state_prior == pytest.approx(1e-3)
    assert not hasattr(args, "state_prior_weight")


def test_parse_args_allows_explicit_baf_outlier_rate(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gatk-sv-gd infer",
            "--preprocessed-dir",
            "./preprocess",
            "-o",
            "./out",
            "--baf-outlier-rate",
            "0.05",
        ],
    )

    args = parse_args()

    assert args.baf_outlier_rate == pytest.approx(0.05)


def test_parse_args_can_disable_baf_effective_count(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gatk-sv-gd infer",
            "--preprocessed-dir",
            "./preprocess",
            "-o",
            "./out",
            "--disable-baf-effective-count",
        ],
    )

    args = parse_args()

    assert args.use_baf_effective_count is False


def test_parse_args_allows_explicit_null_state_prior(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gatk-sv-gd infer",
            "--preprocessed-dir",
            "./preprocess",
            "-o",
            "./out",
            "--null-state-prior",
            "0.02",
        ],
    )

    args = parse_args()

    assert args.null_state_prior == pytest.approx(0.02)


def test_parse_args_allows_explicit_var_length_scale(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gatk-sv-gd infer",
            "--preprocessed-dir",
            "./preprocess",
            "-o",
            "./out",
            "--var-length-scale",
            "7500",
        ],
    )

    args = parse_args()

    assert args.var_length_scale == pytest.approx(7500.0)


@pytest.mark.parametrize(
    ("extra_args", "error_fragment"),
    [
        (["--baf-temperature", "-1"], "--baf-temperature must be non-negative"),
        (["--baf-temperature-prior-scale", "0"], "--baf-temperature-prior-scale must be positive"),
        (["--baf-outlier-rate", "1.0"], "--baf-outlier-rate must be in [0, 1)"),
        (["--null-state-prior", "1.0"], "--null-state-prior must be in [0, 1)"),
        (["--var-length-scale", "0"], "--var-length-scale must be positive"),
        (["--guide-warmup-iter", "-1"], "--guide-warmup-iter must be non-negative"),
    ],
)
def test_parse_args_rejects_invalid_numeric_values(monkeypatch, extra_args, error_fragment):
    monkeypatch.setattr(
        sys,
        "argv",
        ["gatk-sv-gd infer", "--preprocessed-dir", "./preprocess", "-o", "./out", *extra_args],
    )

    with pytest.raises(SystemExit) as excinfo:
        parse_args()

    assert excinfo.value.code == 2


def test_align_normalization_metadata_requires_metadata():
    with pytest.raises(ValueError, match="Normalization metadata is required"):
        _align_normalization_metadata(None, ["S1"])


def test_flatten_multi_args_flattens_nested_argument_groups():
    assert _flatten_multi_args([]) == []
    assert _flatten_multi_args([["a", "b"], ["c"], []]) == ["a", "b", "c"]


def test_align_normalization_metadata_validates_required_columns_and_sample_membership():
    with pytest.raises(ValueError, match="missing required columns"):
        _align_normalization_metadata(pd.DataFrame({"sample": ["S1"]}), ["S1"])

    metadata = pd.DataFrame(
        {
            "sample": ["S1"],
            "raw_count_median": [1000.0],
            "reference_bin_size": [5000.0],
        }
    )
    with pytest.raises(ValueError, match="missing raw-count medians for samples"):
        _align_normalization_metadata(metadata, ["S1", "S2"])


def test_align_normalization_metadata_uses_last_duplicate_and_validates_reference_bin_size():
    metadata = pd.DataFrame(
        {
            "sample": ["S2", "S1", "S1"],
            "raw_count_median": [2000.0, 900.0, 1000.0],
            "reference_bin_size": [5000.0, 5000.0, 5000.0],
        }
    )

    aligned_medians, reference_bin_size = _align_normalization_metadata(metadata, ["S1", "S2"])

    assert aligned_medians.tolist() == pytest.approx([1000.0, 2000.0])
    assert reference_bin_size == pytest.approx(5000.0)

    mismatched_reference_sizes = metadata.copy()
    mismatched_reference_sizes.loc[0, "reference_bin_size"] = 6000.0
    with pytest.raises(ValueError, match="exactly one reference_bin_size"):
        _align_normalization_metadata(mismatched_reference_sizes, ["S1", "S2"])


def test_run_gd_analysis_uses_preprocessed_bins_and_trains_model(monkeypatch, tmp_path):
    records = {}

    class FakeDepthData:
        def __init__(self, df, device, dtype, clamp_threshold):
            records["depth_init"] = {
                "df": df.copy(),
                "device": device,
                "dtype": dtype,
                "clamp_threshold": clamp_threshold,
            }
            self.sample_ids = ["S1", "S2"]

        def attach_sample_ploidy(self, ploidy_map):
            records["ploidy_map"] = ploidy_map

        def attach_baf_summary(self, baf_summary, mappings):
            records["baf_attach"] = (baf_summary.copy(), list(mappings))

    class FakeModel:
        def __init__(self, **kwargs):
            records["model_init"] = kwargs

        def train(self, data, **kwargs):
            records["train"] = {"data": data, "kwargs": kwargs}

        def get_map_estimates(self, data):
            records["map_data"] = data
            return {"state": "map"}

        def run_discrete_inference(self, data, n_samples, log_freq):
            records["discrete"] = {
                "data": data,
                "n_samples": n_samples,
                "log_freq": log_freq,
            }
            return {"state": "posterior"}

    monkeypatch.setattr(
        infer_module,
        "collect_all_locus_bins",
        lambda *args, **kwargs: pytest.fail("collect_all_locus_bins should not run for preprocessed data"),
    )
    monkeypatch.setattr(infer_module, "DepthData", FakeDepthData)
    monkeypatch.setattr(infer_module, "CNVModel", FakeModel)
    monkeypatch.setattr(
        infer_module,
        "_align_normalization_metadata",
        lambda normalization_metadata, sample_ids: (np.array([10.0, 20.0]), 5000.0),
    )
    monkeypatch.setattr(infer_module, "_write_training_loss_history", lambda model, args: records.setdefault("loss_history", True))
    monkeypatch.setattr(
        infer_module,
        "write_posterior_tables",
        lambda data, map_estimates, cn_posterior, mappings, output_dir: records.setdefault(
            "posterior_tables",
            (data, map_estimates, cn_posterior, list(mappings), output_dir),
        ),
    )
    monkeypatch.setattr(
        infer_module,
        "write_locus_metadata",
        lambda *args, **kwargs: pytest.fail("write_locus_metadata should be skipped for preprocessed bins"),
    )

    args = _make_infer_args(output_dir=str(tmp_path), clamp_threshold=7.5, n_discrete_samples=9)
    preprocessed_bins = pd.DataFrame({"Chr": ["chr1"], "Start": [0], "End": [100], "S1": [2.0]})
    baf_summary = pd.DataFrame({"cluster": ["cluster1"], "sample": ["S1"]})
    mappings = [SimpleNamespace(cluster="cluster1")]
    ploidy_map = {("S1", "chr1"): 2}

    run_gd_analysis(
        pd.DataFrame(),
        SimpleNamespace(),
        None,
        None,
        args,
        device="cpu",
        normalization_metadata=pd.DataFrame({"sample": ["S1"]}),
        preprocessed_bins=preprocessed_bins,
        preprocessed_mappings=mappings,
        preprocessed_baf_summary=baf_summary,
        ploidy_map=ploidy_map,
    )

    assert records["depth_init"]["df"].equals(preprocessed_bins)
    assert records["depth_init"]["clamp_threshold"] == pytest.approx(7.5)
    assert records["ploidy_map"] == ploidy_map
    assert records["baf_attach"][0].equals(baf_summary)
    assert records["baf_attach"][1] == mappings
    assert records["model_init"]["sample_raw_count_medians"].tolist() == pytest.approx([10.0, 20.0])
    assert records["model_init"]["reference_bin_size"] == pytest.approx(5000.0)
    assert records["train"]["kwargs"]["max_iter"] == 25
    assert records["discrete"]["n_samples"] == 9
    assert records["posterior_tables"][3] == mappings
    assert records["posterior_tables"][4] == str(tmp_path)
    assert records["loss_history"] is True


def test_run_gd_analysis_collects_bins_and_writes_locus_metadata(monkeypatch, tmp_path):
    combined_df = pd.DataFrame({"Chr": ["chr1"], "Start": [0], "End": [50], "S1": [2.0]})
    mappings = [SimpleNamespace(cluster="cluster1")]
    included_loci = [SimpleNamespace(name="cluster1")]
    records = {}

    class FakeDepthData:
        def __init__(self, df, device, dtype, clamp_threshold):
            records["combined_df"] = df.copy()
            self.sample_ids = ["S1"]

        def attach_sample_ploidy(self, ploidy_map):
            records["unexpected_ploidy"] = ploidy_map

        def attach_baf_summary(self, baf_summary, mappings):
            records["unexpected_baf"] = (baf_summary, mappings)

    class FakeModel:
        def __init__(self, **kwargs):
            records["model_kwargs"] = kwargs

        def train(self, data, **kwargs):
            records["trained"] = True

        def get_map_estimates(self, data):
            return {"ok": True}

        def run_discrete_inference(self, data, n_samples, log_freq):
            return {"posterior": True}

    def fake_collect(df, gd_table, exclusion_mask, **kwargs):
        records["collect"] = {
            "df": df.copy(),
            "gd_table": gd_table,
            "exclusion_mask": exclusion_mask,
            "kwargs": kwargs,
        }
        return combined_df, mappings, included_loci

    monkeypatch.setattr(infer_module, "collect_all_locus_bins", fake_collect)
    monkeypatch.setattr(infer_module, "DepthData", FakeDepthData)
    monkeypatch.setattr(infer_module, "CNVModel", FakeModel)
    monkeypatch.setattr(
        infer_module,
        "_align_normalization_metadata",
        lambda normalization_metadata, sample_ids: (np.array([12.0]), 100.0),
    )
    monkeypatch.setattr(infer_module, "_write_training_loss_history", lambda *args: None)
    monkeypatch.setattr(infer_module, "write_posterior_tables", lambda *args: records.setdefault("posterior_written", True))
    monkeypatch.setattr(
        infer_module,
        "write_locus_metadata",
        lambda loci, mapping_list, output_dir: records.setdefault(
            "locus_metadata",
            (loci, list(mapping_list), output_dir),
        ),
    )

    args = _make_infer_args(output_dir=str(tmp_path), high_res_counts="highres.tsv.gz")
    df = pd.DataFrame({"Chr": ["chr1"], "Start": [0], "End": [50], "S1": [1.0]})

    run_gd_analysis(
        df,
        SimpleNamespace(name="gd_table"),
        "mask",
        "hard-mask",
        args,
        device="cpu",
        column_medians=np.array([4.0]),
        lowres_median_bin_size=50.0,
        normalization_metadata=pd.DataFrame(
            {
                "sample": ["S1"],
                "raw_count_median": [12.0],
                "reference_bin_size": [100.0],
            }
        ),
    )

    assert records["collect"]["df"].equals(df)
    assert records["collect"]["exclusion_mask"] == "mask"
    assert records["collect"]["kwargs"]["hard_inclusion_mask"] == "hard-mask"
    assert records["collect"]["kwargs"]["highres_counts_path"] == "highres.tsv.gz"
    assert records["collect"]["kwargs"]["column_medians"].tolist() == pytest.approx([4.0])
    assert records["combined_df"].equals(combined_df)
    assert records["posterior_written"] is True
    assert records["locus_metadata"] == (included_loci, mappings, str(tmp_path))
    assert records.get("unexpected_ploidy") is None
    assert records.get("unexpected_baf") is None


def test_run_gd_analysis_returns_empty_when_no_bins_are_available(monkeypatch):
    monkeypatch.setattr(
        infer_module,
        "collect_all_locus_bins",
        lambda *args, **kwargs: (pd.DataFrame(), [], []),
    )
    monkeypatch.setattr(
        infer_module,
        "DepthData",
        lambda *args, **kwargs: pytest.fail("DepthData should not be constructed when there are no bins"),
    )

    result = run_gd_analysis(
        pd.DataFrame(),
        SimpleNamespace(),
        None,
        None,
        _make_infer_args(),
        normalization_metadata=pd.DataFrame(
            {
                "sample": ["S1"],
                "raw_count_median": [1.0],
                "reference_bin_size": [100.0],
            }
        ),
    )

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_main_with_preprocessed_dir_recomputes_normalization_metadata(monkeypatch, tmp_path):
    args = _make_infer_args(
        preprocessed_dir="prep-dir",
        input="raw.tsv.gz",
        output_dir=str(tmp_path),
        verbose=True,
        exclusion_intervals=[["a.bed", "b.bed"], ["c.bed"]],
        hard_inclusion_intervals=[["keep.bed"]],
    )
    preprocessed_bins = pd.DataFrame({"Chr": ["chr1"], "Start": [0], "End": [100], "S1": [2.0], "S2": [4.0]})
    preprocessed_baf_summary = pd.DataFrame({"cluster": ["cluster1"]})
    normalization_metadata = pd.DataFrame({"sample": [], "raw_count_median": [], "reference_bin_size": []})
    raw_df = pd.DataFrame(
        {
            "Chr": ["chr1", "chrX"],
            "Start": [0, 100],
            "End": [100, 200],
            "S1": [10.0, 40.0],
            "S2": [30.0, 90.0],
        }
    )
    rebuilt_metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2"],
            "raw_count_median": [10.0, 30.0],
            "reference_bin_size": [100.0, 100.0],
        }
    )
    records = {}

    def fake_build_normalization_metadata(sample_cols, column_medians, lowres_median_bin_size):
        records["rebuilt_metadata_args"] = (
            list(sample_cols),
            column_medians.copy(),
            lowres_median_bin_size,
        )
        return rebuilt_metadata

    monkeypatch.setattr(infer_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        infer_module,
        "load_preprocessed_data",
        lambda path: (preprocessed_bins, ["m1"], preprocessed_baf_summary, normalization_metadata),
    )
    monkeypatch.setattr(infer_module, "setup_logging", lambda *call_args, **call_kwargs: records.setdefault("setup_logging", (call_args, call_kwargs)))
    monkeypatch.setattr(infer_module, "read_data", lambda path: raw_df.copy())
    monkeypatch.setattr(infer_module, "get_sample_columns", lambda df: ["S1", "S2"])
    monkeypatch.setattr(infer_module, "build_normalization_metadata", fake_build_normalization_metadata)
    monkeypatch.setattr(
        infer_module,
        "write_normalization_metadata",
        lambda metadata, output_dir: records.setdefault("written_metadata", (metadata.copy(), output_dir)),
    )
    monkeypatch.setattr(infer_module, "_setup_pyro", lambda passed_args: records.setdefault("setup_pyro", passed_args))
    monkeypatch.setattr(
        infer_module,
        "run_gd_analysis",
        lambda *call_args, **call_kwargs: records.setdefault("run_gd_analysis", (call_args, call_kwargs)),
    )

    infer_module.main()

    assert args.exclusion_intervals == ["a.bed", "b.bed", "c.bed"]
    assert args.hard_inclusion_intervals == ["keep.bed"]
    assert infer_module._util.VERBOSE is True
    assert records["rebuilt_metadata_args"][0] == ["S1", "S2"]
    assert records["rebuilt_metadata_args"][1].tolist() == pytest.approx([10.0, 30.0])
    assert records["rebuilt_metadata_args"][2] == pytest.approx(100.0)
    assert records["written_metadata"][1] == str(tmp_path)
    assert records["run_gd_analysis"][1]["preprocessed_bins"].equals(preprocessed_bins)
    assert records["run_gd_analysis"][1]["preprocessed_baf_summary"].equals(preprocessed_baf_summary)
    assert records["run_gd_analysis"][1]["normalization_metadata"].equals(records["written_metadata"][0])


def test_main_with_preprocessed_dir_recomputes_metadata_from_all_bins_when_no_autosomes(monkeypatch, tmp_path):
    args = _make_infer_args(
        preprocessed_dir="prep-dir",
        input="raw.tsv.gz",
        output_dir=str(tmp_path),
    )
    empty_metadata = pd.DataFrame({"sample": [], "raw_count_median": [], "reference_bin_size": []})
    raw_df = pd.DataFrame(
        {
            "Chr": ["chrX", "chrY"],
            "Start": [0, 100],
            "End": [100, 200],
            "S1": [10.0, 30.0],
            "S2": [20.0, 40.0],
        }
    )
    rebuilt_metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2"],
            "raw_count_median": [20.0, 30.0],
            "reference_bin_size": [100.0, 100.0],
        }
    )
    records = {}

    def fake_build_normalization_metadata(sample_cols, column_medians, lowres_median_bin_size):
        records["rebuilt"] = (list(sample_cols), column_medians.copy(), lowres_median_bin_size)
        return rebuilt_metadata

    monkeypatch.setattr(infer_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        infer_module,
        "load_preprocessed_data",
        lambda path: (pd.DataFrame(), ["m1"], pd.DataFrame(), empty_metadata),
    )
    monkeypatch.setattr(infer_module, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(infer_module, "read_data", lambda path: raw_df.copy())
    monkeypatch.setattr(infer_module, "get_sample_columns", lambda df: ["S1", "S2"])
    monkeypatch.setattr(infer_module, "build_normalization_metadata", fake_build_normalization_metadata)
    monkeypatch.setattr(infer_module, "write_normalization_metadata", lambda metadata, output_dir: records.setdefault("written", metadata.copy()))
    monkeypatch.setattr(infer_module, "_setup_pyro", lambda passed_args: None)
    monkeypatch.setattr(infer_module, "run_gd_analysis", lambda *args, **kwargs: records.setdefault("ran", kwargs))

    infer_module.main()

    assert records["rebuilt"][0] == ["S1", "S2"]
    assert records["rebuilt"][1].tolist() == pytest.approx([20.0, 30.0])
    assert records["rebuilt"][2] == pytest.approx(100.0)


def test_main_with_preprocessed_dir_rewrites_existing_normalization_metadata(monkeypatch, tmp_path):
    args = _make_infer_args(
        preprocessed_dir="prep-dir",
        input=None,
        output_dir=str(tmp_path),
    )
    normalization_metadata = pd.DataFrame(
        {
            "sample": ["S1"],
            "raw_count_median": [12.0],
            "reference_bin_size": [100.0],
        }
    )
    records = {}

    monkeypatch.setattr(infer_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        infer_module,
        "load_preprocessed_data",
        lambda path: (pd.DataFrame(), ["m1"], pd.DataFrame(), normalization_metadata),
    )
    monkeypatch.setattr(infer_module, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        infer_module,
        "write_normalization_metadata",
        lambda metadata, output_dir: records.setdefault("written", (metadata.copy(), output_dir)),
    )
    monkeypatch.setattr(infer_module, "_setup_pyro", lambda passed_args: records.setdefault("pyro", passed_args))
    monkeypatch.setattr(
        infer_module,
        "run_gd_analysis",
        lambda *args, **kwargs: records.setdefault("run_gd_analysis", kwargs),
    )

    infer_module.main()

    assert records["written"][0].equals(normalization_metadata)
    assert records["written"][1] == str(tmp_path)
    assert records["run_gd_analysis"]["normalization_metadata"].equals(normalization_metadata)


def test_main_without_preprocessed_dir_runs_full_pipeline(monkeypatch, tmp_path):
    args = _make_infer_args(
        input="raw.tsv.gz",
        gd_table="gd.tsv",
        output_dir=str(tmp_path),
        verbose=True,
        high_res_counts="highres.tsv.gz",
        exclusion_intervals=[["mask_a.bed", "mask_b.bed"]],
        hard_inclusion_intervals=[["keep.bed"]],
    )
    raw_df = pd.DataFrame(
        {
            "Chr": ["chr1", "chr1", "chrX"],
            "Start": [0, 100, 200],
            "End": [100, 200, 300],
            "S1": [10.0, 20.0, 90.0],
            "S2": [30.0, 50.0, 120.0],
        }
    )
    normalization_metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2"],
            "raw_count_median": [15.0, 40.0],
            "reference_bin_size": [100.0, 100.0],
        }
    )
    records = {"masks": []}

    def fake_build_normalization_metadata(sample_cols, column_medians, lowres_median_bin_size):
        records["normalization_inputs"] = (
            list(sample_cols),
            column_medians.copy(),
            lowres_median_bin_size,
        )
        return normalization_metadata

    def fake_filter_low_quality_bins(df, **kwargs):
        records["filter_call"] = (df.copy(), kwargs)
        return df

    class FakeGDTable:
        def __init__(self, path):
            records["gd_table_path"] = path
            self.loci = {
                "locus1": SimpleNamespace(breakpoints=[(0, 1)], n_breakpoints=2),
                "locus2": SimpleNamespace(breakpoints=[], n_breakpoints=0),
            }

    def fake_exclusion_mask(paths, label):
        records["masks"].append((list(paths), label))
        return f"{label}-mask"

    monkeypatch.setattr(infer_module, "parse_args", lambda: args)
    monkeypatch.setattr(infer_module, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(infer_module, "GDTable", FakeGDTable)
    monkeypatch.setattr(infer_module, "ExclusionMask", fake_exclusion_mask)
    monkeypatch.setattr(infer_module, "read_data", lambda path: raw_df.copy())
    monkeypatch.setattr(infer_module, "get_sample_columns", lambda df: ["S1", "S2"])
    monkeypatch.setattr(infer_module, "build_normalization_metadata", fake_build_normalization_metadata)
    monkeypatch.setattr(
        infer_module,
        "write_normalization_metadata",
        lambda metadata, output_dir: records.setdefault("written_metadata", (metadata.copy(), output_dir)),
    )
    monkeypatch.setattr(infer_module, "estimate_ploidy", lambda df, output_dir: pd.DataFrame({"sample": ["S1"], "contig": ["chr1"], "ploidy": [2]}))
    monkeypatch.setattr(infer_module, "build_ploidy_map", lambda df: {("S1", "chr1"): 2})
    monkeypatch.setattr(infer_module, "filter_low_quality_bins", fake_filter_low_quality_bins)
    monkeypatch.setattr(infer_module, "_setup_pyro", lambda passed_args: records.setdefault("setup_pyro", passed_args))
    monkeypatch.setattr(
        infer_module,
        "run_gd_analysis",
        lambda *call_args, **call_kwargs: records.setdefault("run_call", (call_args, call_kwargs)),
    )

    infer_module.main()

    assert args.exclusion_intervals == ["mask_a.bed", "mask_b.bed"]
    assert args.hard_inclusion_intervals == ["keep.bed"]
    assert records["gd_table_path"] == "gd.tsv"
    assert records["masks"] == [
        (["mask_a.bed", "mask_b.bed"], "exclusion regions"),
        (["keep.bed"], "hard inclusion regions"),
    ]
    assert records["normalization_inputs"][0] == ["S1", "S2"]
    assert records["normalization_inputs"][1].tolist() == pytest.approx([15.0, 40.0])
    assert records["normalization_inputs"][2] == pytest.approx(100.0)
    normalized_df = records["filter_call"][0]
    assert normalized_df["S1"].tolist() == pytest.approx([4.0 / 3.0, 8.0 / 3.0, 12.0])
    assert normalized_df["S2"].tolist() == pytest.approx([1.5, 2.5, 6.0])
    run_df = records["run_call"][0][0]
    assert run_df["S1"].tolist() == pytest.approx([4.0 / 3.0, 8.0 / 3.0, 12.0])
    assert run_df["S2"].tolist() == pytest.approx([1.5, 2.5, 6.0])
    assert records["run_call"][0][2] == "exclusion regions-mask"
    assert records["run_call"][0][3] == "hard inclusion regions-mask"
    assert records["run_call"][1]["column_medians"].tolist() == pytest.approx([15.0, 40.0])
    assert records["run_call"][1]["lowres_median_bin_size"] == pytest.approx(100.0)
    assert records["run_call"][1]["normalization_metadata"].equals(records["written_metadata"][0])
    assert records["run_call"][1]["ploidy_map"] == {("S1", "chr1"): 2}


def test_main_requires_normalization_metadata_for_preprocessed_data_without_input(monkeypatch, tmp_path):
    args = _make_infer_args(preprocessed_dir="prep-dir", input=None, output_dir=str(tmp_path))

    monkeypatch.setattr(infer_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        infer_module,
        "load_preprocessed_data",
        lambda path: (pd.DataFrame(), [], None, pd.DataFrame()),
    )
    monkeypatch.setattr(infer_module, "setup_logging", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit, match="1"):
        infer_module.main()


def test_main_without_preprocessed_dir_requires_input(monkeypatch, tmp_path):
    args = _make_infer_args(input=None, gd_table="gd.tsv", output_dir=str(tmp_path))

    monkeypatch.setattr(infer_module, "parse_args", lambda: args)
    monkeypatch.setattr(infer_module, "setup_logging", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit, match="1"):
        infer_module.main()


def test_main_without_preprocessed_dir_requires_gd_table(monkeypatch, tmp_path):
    args = _make_infer_args(input="raw.tsv.gz", gd_table=None, output_dir=str(tmp_path))

    monkeypatch.setattr(infer_module, "parse_args", lambda: args)
    monkeypatch.setattr(infer_module, "setup_logging", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit, match="1"):
        infer_module.main()


def test_main_without_preprocessed_dir_uses_all_bins_when_no_autosomes(monkeypatch, tmp_path, capsys):
    args = _make_infer_args(
        input="raw.tsv.gz",
        gd_table="gd.tsv",
        output_dir=str(tmp_path),
        verbose=False,
        high_res_counts=None,
        exclusion_intervals=[],
        hard_inclusion_intervals=[],
    )
    raw_df = pd.DataFrame(
        {
            "Chr": ["chrX", "chrY"],
            "Start": [0, 100],
            "End": [100, 200],
            "S1": [10.0, 20.0],
            "S2": [30.0, 50.0],
        }
    )
    normalization_metadata = pd.DataFrame(
        {
            "sample": ["S1", "S2"],
            "raw_count_median": [15.0, 40.0],
            "reference_bin_size": [100.0, 100.0],
        }
    )
    records = {}

    class FakeGDTable:
        def __init__(self, path):
            self.loci = {"locus1": SimpleNamespace(breakpoints=[], n_breakpoints=0)}

    def fake_build_normalization_metadata(sample_cols, column_medians, lowres_median_bin_size):
        records["normalization_inputs"] = (
            list(sample_cols),
            column_medians.copy(),
            lowres_median_bin_size,
        )
        return normalization_metadata

    monkeypatch.setattr(infer_module, "parse_args", lambda: args)
    monkeypatch.setattr(infer_module, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(infer_module, "GDTable", FakeGDTable)
    monkeypatch.setattr(infer_module, "read_data", lambda path: raw_df.copy())
    monkeypatch.setattr(infer_module, "get_sample_columns", lambda df: ["S1", "S2"])
    monkeypatch.setattr(infer_module, "build_normalization_metadata", fake_build_normalization_metadata)
    monkeypatch.setattr(infer_module, "write_normalization_metadata", lambda metadata, output_dir: records.setdefault("written_metadata", (metadata.copy(), output_dir)))
    monkeypatch.setattr(infer_module, "estimate_ploidy", lambda df, output_dir: pd.DataFrame({"sample": ["S1"], "contig": ["chrX"], "ploidy": [2]}))
    monkeypatch.setattr(infer_module, "build_ploidy_map", lambda df: {("S1", "chrX"): 2})
    monkeypatch.setattr(infer_module, "filter_low_quality_bins", lambda df, **kwargs: df)
    monkeypatch.setattr(infer_module, "_setup_pyro", lambda passed_args: records.setdefault("setup_pyro", passed_args))
    monkeypatch.setattr(
        infer_module,
        "run_gd_analysis",
        lambda *call_args, **call_kwargs: records.setdefault("run_call", (call_args, call_kwargs)),
    )

    infer_module.main()

    stdout = capsys.readouterr().out
    assert records["normalization_inputs"][0] == ["S1", "S2"]
    assert records["normalization_inputs"][1].tolist() == pytest.approx([15.0, 40.0])
    assert records["normalization_inputs"][2] == pytest.approx(100.0)
    assert records["run_call"][0][2] is None
    assert records["run_call"][0][3] is None
    assert records["run_call"][1]["column_medians"].tolist() == pytest.approx([15.0, 40.0])
    assert records["run_call"][1]["ploidy_map"] == {("S1", "chrX"): 2}
    assert "No high-resolution counts file provided (--high-res-counts)" in stdout


def test_setup_pyro_disables_validation_for_jit(monkeypatch):
    calls = []

    monkeypatch.setattr(
        infer_module.pyro,
        "enable_validation",
        lambda value: calls.append(("pyro", value)),
        raising=False,
    )
    monkeypatch.setattr(
        infer_module.pyro.distributions,
        "enable_validation",
        lambda value: calls.append(("dist", value)),
    )
    monkeypatch.setattr(
        infer_module.pyro,
        "set_rng_seed",
        lambda value: calls.append(("pyro_seed", value)),
    )
    monkeypatch.setattr(
        infer_module.torch,
        "manual_seed",
        lambda value: calls.append(("torch_seed", value)),
    )
    monkeypatch.setattr(
        infer_module.np.random,
        "seed",
        lambda value: calls.append(("numpy_seed", value)),
    )

    _setup_pyro(SimpleNamespace(disable_jit=False))

    assert calls[:2] == [("pyro", False), ("dist", False)]
    assert calls[2:] == [("pyro_seed", 42), ("torch_seed", 42), ("numpy_seed", 42)]


def test_setup_pyro_keeps_validation_for_non_jit(monkeypatch):
    calls = []

    monkeypatch.setattr(
        infer_module.pyro,
        "enable_validation",
        lambda value: calls.append(("pyro", value)),
        raising=False,
    )
    monkeypatch.setattr(
        infer_module.pyro.distributions,
        "enable_validation",
        lambda value: calls.append(("dist", value)),
    )
    monkeypatch.setattr(infer_module.pyro, "set_rng_seed", lambda value: None)
    monkeypatch.setattr(infer_module.torch, "manual_seed", lambda value: None)
    monkeypatch.setattr(infer_module.np.random, "seed", lambda value: None)

    _setup_pyro(SimpleNamespace(disable_jit=True))

    assert calls == [("pyro", True), ("dist", True)]


def test_write_training_loss_history_logs_convergence_summary(tmp_path, caplog):
    model = SimpleNamespace(
        loss_history={
            "epoch": [0, 1, 2, 3],
            "elbo": [100.0, 100.0, 100.01, 100.01],
        }
    )
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        elbo_window=2,
        elbo_rtol=2e-4,
    )

    caplog.set_level(logging.INFO, logger="gatk_sv_gd.infer")

    _write_training_loss_history(model, args)

    loss_df = pd.read_csv(tmp_path / "training_loss.tsv", sep="\t")
    assert loss_df["epoch"].tolist() == [0, 1, 2, 3]
    assert loss_df["elbo"].tolist() == pytest.approx([100.0, 100.0, 100.01, 100.01])

    messages = [record.getMessage() for record in caplog.records]
    assert "Wrote training loss history: epochs=4" in messages
    assert "ELBO history summary: initial=100.0000 final=100.0100 best=100.0000" in messages
    assert (
        "ELBO convergence summary: final_window_change=1.00e-04 "
        "window=2 target_rtol=0.0002 within_tolerance=True"
    ) in messages


def test_write_training_loss_history_warns_when_history_is_missing(tmp_path, caplog):
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        elbo_window=2,
        elbo_rtol=2e-4,
    )

    caplog.set_level(logging.INFO, logger="gatk_sv_gd.infer")

    _write_training_loss_history(SimpleNamespace(loss_history={}), args)

    loss_df = pd.read_csv(tmp_path / "training_loss.tsv", sep="\t")
    assert loss_df.empty
    messages = [record.getMessage() for record in caplog.records]
    assert "Wrote training loss history: epochs=0" in messages
    assert "Training produced no ELBO history." in messages


def test_write_training_loss_history_logs_unavailable_window_change(monkeypatch, tmp_path, caplog):
    model = SimpleNamespace(
        loss_history={
            "epoch": [0, 1],
            "elbo": [100.0, 99.5],
        }
    )
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        elbo_window=5,
        elbo_rtol=1e-4,
    )

    monkeypatch.setattr(infer_module, "_windowed_relative_elbo_change", lambda history, window: None)
    caplog.set_level(logging.INFO, logger="gatk_sv_gd.infer")

    _write_training_loss_history(model, args)

    messages = [record.getMessage() for record in caplog.records]
    assert "ELBO history summary: initial=100.0000 final=99.5000 best=99.5000" in messages
    assert (
        "ELBO convergence summary: final_window_change=unavailable "
        "window=5 target_rtol=0.0001"
    ) in messages