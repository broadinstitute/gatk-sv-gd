import sys
from types import SimpleNamespace

import gatk_sv_gd.infer as infer_module
from gatk_sv_gd.infer import _setup_pyro, parse_args


def test_parse_args_uses_learned_baf_temperature_by_default(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["gatk-sv-gd infer", "--preprocessed-dir", "./preprocess", "-o", "./out"],
    )

    args = parse_args()

    assert args.fixed_baf_temperature is False


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