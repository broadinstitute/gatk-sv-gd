import logging
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import gatk_sv_gd.infer as infer_module
from gatk_sv_gd.infer import _align_normalization_metadata, _setup_pyro, _write_training_loss_history, parse_args


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


def test_align_normalization_metadata_requires_metadata():
    with pytest.raises(ValueError, match="Normalization metadata is required"):
        _align_normalization_metadata(None, ["S1"])


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