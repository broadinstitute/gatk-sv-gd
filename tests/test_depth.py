from types import SimpleNamespace

import numpy as np

import pytest

from gatk_sv_gd.depth import (
    CNVModel,
    _center_state_log_likelihood_table_numpy,
    _safe_scaled_baf_variance_torch,
    _windowed_relative_elbo_change,
)
import gatk_sv_gd.depth as depth_module


def test_windowed_relative_elbo_change_uses_two_latest_windows():
    assert _windowed_relative_elbo_change([100.0, 100.0, 100.01], window=2) is None

    relative_change = _windowed_relative_elbo_change(
        [100.0, 100.0, 100.01, 100.01],
        window=2,
    )

    assert relative_change == pytest.approx(1e-4)


def test_train_early_stopping_uses_windowed_relative_elbo_change(monkeypatch):
    model = object.__new__(CNVModel)
    model.model = object()
    model.guide = object()
    model.guide_type = "delta"
    model._build_guide = lambda *args, **kwargs: model.guide
    model.loss_history = {"epoch": [999], "elbo": [123.0]}
    model.current_data = None

    losses = iter([100.0, 100.0, 100.01, 100.01, 100.01001, 100.01001, 99.0])

    class FakeScheduler:
        def step(self):
            return None

    class FakeSVI:
        def __init__(self, *args, **kwargs):
            pass

        def step(self, **kwargs):
            return next(losses)

    monkeypatch.setattr(depth_module.pyro, "clear_param_store", lambda: None)
    monkeypatch.setattr(depth_module.pyro.optim, "LambdaLR", lambda config: FakeScheduler())
    monkeypatch.setattr(depth_module, "TraceEnum_ELBO", lambda: object())
    monkeypatch.setattr(depth_module, "SVI", FakeSVI)

    data = SimpleNamespace(
        depth=object(),
        interval_sizes=object(),
        n_bins=1,
        n_samples=1,
    )

    model.train(
        data,
        max_iter=20,
        log_freq=100,
        early_stopping=True,
        patience=2,
        convergence_window=2,
        convergence_rtol=2e-4,
    )

    assert model.loss_history["elbo"] == pytest.approx(
        [100.0, 100.0, 100.01, 100.01, 100.01001]
    )
    assert model.loss_history["epoch"] == [0, 1, 2, 3, 4]
    assert model.current_data is None


def test_train_uses_conditioned_model_for_map_warmup(monkeypatch):
    model = object.__new__(CNVModel)
    base_model = object()
    warmup_model = object()
    warmup_guide = object()
    final_guide = object()
    model.model = base_model
    model.guide = final_guide
    model.guide_type = "diagonal"
    model.loss_history = {"epoch": [], "elbo": []}
    model.current_data = None
    model._warmup_model_and_initial_values = lambda: (
        warmup_model,
        ["sample_var"],
        {"baf_temperature": object()},
    )
    model._extract_guide_latent_values = lambda guide, data: {}

    def fake_build_guide(guide_type, **kwargs):
        if guide_type == "delta":
            return warmup_guide
        return final_guide

    class FakeScheduler:
        def step(self):
            return None

    svi_calls = []

    class FakeSVI:
        def __init__(self, svi_model, guide, **kwargs):
            svi_calls.append((svi_model, guide))

        def step(self, **kwargs):
            return 100.0

    model._build_guide = fake_build_guide
    monkeypatch.setattr(depth_module.pyro, "clear_param_store", lambda: None)
    monkeypatch.setattr(depth_module.pyro.optim, "LambdaLR", lambda config: FakeScheduler())
    monkeypatch.setattr(depth_module, "TraceEnum_ELBO", lambda: object())
    monkeypatch.setattr(depth_module, "SVI", FakeSVI)

    data = SimpleNamespace(
        depth=object(),
        interval_sizes=object(),
        n_bins=1,
        n_samples=1,
    )

    model.train(
        data,
        max_iter=1,
        guide_warmup_iter=1,
        early_stopping=False,
    )

    assert svi_calls == [(warmup_model, warmup_guide), (base_model, final_guide)]
    assert model.current_data is None


def test_init_omits_frozen_bin_latents_from_guide(monkeypatch):
    fake_torch = SimpleNamespace(
        tensor=lambda value, **kwargs: value,
        zeros=lambda shape, **kwargs: np.zeros(shape, dtype=np.float32),
        ones=lambda shape, **kwargs: np.ones(shape, dtype=np.float32),
        full=lambda shape, value, **kwargs: np.full(shape, value, dtype=np.float32),
        float32=object(),
    )
    block_calls = {}

    monkeypatch.setattr(depth_module, "torch", fake_torch)
    monkeypatch.setattr(
        depth_module.poutine,
        "block",
        lambda model, expose: block_calls.setdefault("expose", list(expose)),
        raising=False,
    )
    monkeypatch.setattr(depth_module, "AutoDelta", lambda blocked_model: blocked_model)

    model = CNVModel(
        guide_type="delta",
        freeze_bin_bias=True,
        freeze_bin_var=True,
    )

    assert model.latent_sites == ["sample_var", "baf_temperature", "pair_state_probs"]
    assert block_calls["expose"] == ["sample_var", "baf_temperature", "pair_state_probs"]


def test_init_omits_frozen_pair_state_priors_from_guide(monkeypatch):
    fake_torch = SimpleNamespace(
        tensor=lambda value, **kwargs: value,
        zeros=lambda shape, **kwargs: np.zeros(shape, dtype=np.float32),
        ones=lambda shape, **kwargs: np.ones(shape, dtype=np.float32),
        full=lambda shape, value, **kwargs: np.full(shape, value, dtype=np.float32),
        float32=object(),
    )
    block_calls = {}

    monkeypatch.setattr(depth_module, "torch", fake_torch)
    monkeypatch.setattr(
        depth_module.poutine,
        "block",
        lambda model, expose: block_calls.setdefault("expose", list(expose)),
        raising=False,
    )
    monkeypatch.setattr(depth_module, "AutoDelta", lambda blocked_model: blocked_model)

    model = CNVModel(
        guide_type="delta",
        freeze_pair_state_priors=True,
    )

    assert model.latent_sites == ["bin_bias", "sample_var", "baf_temperature", "bin_var"]
    assert block_calls["expose"] == ["bin_bias", "sample_var", "baf_temperature", "bin_var"]


def test_init_omits_baf_temperature_when_fixed(monkeypatch):
    fake_torch = SimpleNamespace(
        tensor=lambda value, **kwargs: value,
        zeros=lambda shape, **kwargs: np.zeros(shape, dtype=np.float32),
        ones=lambda shape, **kwargs: np.ones(shape, dtype=np.float32),
        full=lambda shape, value, **kwargs: np.full(shape, value, dtype=np.float32),
        float32=object(),
    )
    block_calls = {}

    monkeypatch.setattr(depth_module, "torch", fake_torch)
    monkeypatch.setattr(
        depth_module.poutine,
        "block",
        lambda model, expose: block_calls.setdefault("expose", list(expose)),
        raising=False,
    )
    monkeypatch.setattr(depth_module, "AutoDelta", lambda blocked_model: blocked_model)

    model = CNVModel(
        guide_type="delta",
        learn_baf_temperature=False,
    )

    assert "baf_temperature" not in model.latent_sites
    assert "baf_temperature" not in block_calls["expose"]


def test_fixed_bin_latent_values_use_centered_defaults():
    model = object.__new__(CNVModel)
    model.var_bin = 0.125

    assert np.array_equal(
        CNVModel._fixed_bin_bias_values(model, 3),
        np.ones(3, dtype=np.float32),
    )
    assert np.array_equal(
        CNVModel._fixed_bin_var_values(model, 3),
        np.full(3, 0.125, dtype=np.float32),
    )


def test_fixed_bin_latent_tensors_match_bin_plate_shape(monkeypatch):
    fake_torch = SimpleNamespace(
        ones=lambda shape, **kwargs: np.ones(shape, dtype=np.float32),
        full=lambda shape, value, **kwargs: np.full(shape, value, dtype=np.float32),
    )
    model = object.__new__(CNVModel)
    model.device = "cpu"
    model.dtype = object()
    model.var_bin = 0.125

    monkeypatch.setattr(depth_module, "torch", fake_torch)

    assert CNVModel._fixed_bin_bias_tensor(model, 3).shape == (3, 1)
    assert CNVModel._fixed_bin_var_tensor(model, 3).shape == (3, 1)


def test_fixed_baf_temperature_values_use_global_scale():
    model = object.__new__(CNVModel)
    model.baf_temperature = 25.0

    assert CNVModel._fixed_baf_temperature_values(model) == np.asarray(25.0, dtype=np.float32)


def test_init_rejects_nonpositive_learned_baf_temperature():
    with pytest.raises(ValueError, match="baf_temperature > 0"):
        CNVModel(learn_baf_temperature=True, baf_temperature=0.0)


def test_center_state_log_likelihood_table_numpy_removes_constant_offset_and_normalizes_reference():
    raw = np.array(
        [
            [[2.0, 1.0]],
            [[0.0, 4.0]],
            [[-1.0, 3.0]],
        ],
        dtype=np.float64,
    )
    shifted = raw + 17.5
    reference_probs = np.array([0.7, 0.2, 0.1], dtype=np.float64)

    centered = _center_state_log_likelihood_table_numpy(raw, reference_probs)
    centered_shifted = _center_state_log_likelihood_table_numpy(shifted, reference_probs)

    assert np.allclose(centered, centered_shifted)

    reference_logsumexp = np.log(reference_probs)[:, np.newaxis, np.newaxis] + centered
    max_val = np.max(reference_logsumexp, axis=0, keepdims=True)
    normalized = max_val + np.log(np.sum(np.exp(reference_logsumexp - max_val), axis=0, keepdims=True))
    assert np.allclose(normalized, 0.0)


def test_center_state_log_likelihood_table_numpy_returns_neutral_values_for_nonfinite_baseline():
    raw = np.full((3, 2, 2), -np.inf, dtype=np.float64)
    reference_probs = np.array([0.7, 0.2, 0.1], dtype=np.float64)

    centered = _center_state_log_likelihood_table_numpy(raw, reference_probs)

    assert np.all(np.isfinite(centered))
    assert np.array_equal(centered, np.zeros_like(raw))


def test_safe_scaled_baf_variance_torch_masks_inactive_nan_gradient():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "isfinite"):
        pytest.skip("real torch is not available")

    log_temperature = torch.tensor(3.0, dtype=torch.float32, requires_grad=True)
    baf_temperature = torch.exp(log_temperature)
    baf_var = torch.tensor([[float("nan"), 1e-4]], dtype=torch.float32)
    valid_mask = torch.tensor([[False, True]])

    scaled_baf_var = _safe_scaled_baf_variance_torch(
        baf_var,
        valid_mask,
        baf_temperature,
    )
    scaled_baf_var.sum().backward()

    assert torch.isfinite(log_temperature.grad)


def test_fixed_pair_state_priors_use_dirichlet_mean(monkeypatch):
    fake_torch = SimpleNamespace(
        tensor=lambda value, **kwargs: np.asarray(value, dtype=np.float32),
    )
    model = object.__new__(CNVModel)
    model.n_states = 6
    model.alpha_ref = 50.0
    model.alpha_non_ref = 1.0
    model.ref_state_idx = 3
    model.device = "cpu"
    model.dtype = object()

    monkeypatch.setattr(depth_module, "torch", fake_torch)

    expected = np.full(6, 1.0 / 55.0, dtype=np.float32)
    expected[3] = 50.0 / 55.0

    assert np.allclose(CNVModel._pair_state_prior_mean_values(model), expected)
    assert np.allclose(
        CNVModel._fixed_pair_state_probs_values(model, 2),
        np.vstack([expected, expected]),
    )
    tensor_values = CNVModel._fixed_pair_state_probs_tensor(model, 2)
    assert tensor_values.shape == (2, 1, 6)
    assert np.allclose(tensor_values[:, 0, :], np.vstack([expected, expected]))