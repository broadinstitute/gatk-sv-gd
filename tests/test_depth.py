import gzip
import logging
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

import pytest

from gatk_sv_gd.depth import (
    CNVModel,
    DepthData,
    ExclusionMask,
    _clip_baf_variance_numpy,
    build_diploid_pair_states,
    _center_state_log_likelihood_table_numpy,
    _count_anchored_reference_variance_numpy,
    _depth_variance_scale_numpy,
    _lognormal_location_from_mean,
    _logit_clipped,
    _positive_clipped_log,
    _robust_baf_log_likelihood_numpy,
    _size_modifier_numpy,
    _variance_expected_depth_numpy,
    pair_state_minor_baf,
    pair_state_total_cn,
    _select_state_log_likelihood_torch,
    _spatial_aggregate_variance_scale_numpy,
    _safe_scaled_baf_variance_torch,
    _windowed_relative_elbo_change,
)
import gatk_sv_gd.depth as depth_module


class _FakeTensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


def test_diploid_pair_state_helpers_return_canonical_states_and_expected_metrics():
    pair_states = build_diploid_pair_states(max_hap_cn=2)

    assert pair_states == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 2),
    ]
    assert all(h1 <= h2 for h1, h2 in pair_states)

    assert pair_state_minor_baf(pair_states).tolist() == pytest.approx([
        0.0,
        0.0,
        0.0,
        0.5,
        1.0 / 3.0,
        0.5,
    ])
    assert pair_state_total_cn(pair_states).tolist() == [0, 1, 2, 2, 3, 4]


def test_windowed_relative_elbo_change_uses_two_latest_windows():
    assert _windowed_relative_elbo_change([100.0, 100.0, 100.01], window=2) is None

    relative_change = _windowed_relative_elbo_change(
        [100.0, 100.0, 100.01, 100.01],
        window=2,
    )

    assert relative_change == pytest.approx(1e-4)


def test_baf_helper_numpy_functions_clip_and_mix_extremes():
    clipped = _clip_baf_variance_numpy(
        np.asarray([np.nan, np.inf, -np.inf, 0.0, 0.25], dtype=np.float64)
    )
    assert clipped.tolist() == pytest.approx([1e6, 1e6, 1e-6, 1e-6, 0.25])

    log_lik = np.asarray([[-2.0, -0.5]], dtype=np.float64)
    unchanged = _robust_baf_log_likelihood_numpy(log_lik, outlier_rate=0.0)
    fully_uniform = _robust_baf_log_likelihood_numpy(log_lik, outlier_rate=1.0)
    mixed = _robust_baf_log_likelihood_numpy(log_lik, outlier_rate=0.25)

    assert np.allclose(unchanged, log_lik)
    assert np.allclose(fully_uniform, np.full_like(log_lik, math.log(2.0)))
    assert np.allclose(
        mixed,
        np.logaddexp(log_lik + math.log(0.75), math.log(2.0) + math.log(0.25)),
    )


def test_finite_logit_and_log_helpers_clip_to_safe_domain():
    assert _logit_clipped(0.0) == pytest.approx(math.log(1e-6 / (1.0 - 1e-6)))
    assert _logit_clipped(1.0) == pytest.approx(math.log((1.0 - 1e-6) / 1e-6))
    assert _logit_clipped(0.25) == pytest.approx(math.log(0.25 / 0.75))

    assert _positive_clipped_log(0.0) == pytest.approx(math.log(1e-6))
    assert _positive_clipped_log(-5.0) == pytest.approx(math.log(1e-6))
    assert _positive_clipped_log(3.0) == pytest.approx(math.log(3.0))


def test_exclusion_mask_merges_across_files_and_updates_queries(tmp_path, monkeypatch):
    class FakeInterval:
        def __init__(self, begin, end):
            self.begin = int(begin)
            self.end = int(end)

    class FakeIntervalTree:
        def __init__(self):
            self._intervals = []

        def addi(self, start, end):
            self._intervals.append(FakeInterval(start, end))

        def merge_overlaps(self):
            merged = []
            for interval in sorted(self._intervals, key=lambda iv: (iv.begin, iv.end)):
                if not merged or interval.begin > merged[-1].end:
                    merged.append(FakeInterval(interval.begin, interval.end))
                else:
                    merged[-1].end = max(merged[-1].end, interval.end)
            self._intervals = merged

        def overlap(self, start, end):
            return [
                iv for iv in self._intervals
                if iv.begin < int(end) and int(start) < iv.end
            ]

        def __len__(self):
            return len(self._intervals)

    monkeypatch.setattr(depth_module, "IntervalTree", FakeIntervalTree)

    bed_a = tmp_path / "mask_a.bed"
    bed_a.write_text("chr1\t10\t20\nchr1\t18\t25\nchr2\t5\t10\n")

    bed_b = tmp_path / "mask_b.bed.gz"
    with gzip.open(bed_b, "wt") as handle:
        handle.write("chr1\t24\t30\nchr2\t8\t12\textra\n")

    mask = ExclusionMask([str(bed_a), str(bed_b)], label="test regions")

    assert len(mask.df) == 5
    assert sum(len(tree) for tree in mask.trees.values()) == 2
    assert mask.has_any_overlap("chr1", 0, 12) is True
    assert mask.has_any_overlap("chr3", 0, 10) is False
    assert mask.get_overlap_fraction("chr1", 0, 40) == pytest.approx(20.0 / 40.0)
    assert mask.get_overlap_fraction("chr2", 6, 11) == pytest.approx(1.0)
    assert mask.get_overlap_fraction("chr1", 10, 10) == 0.0

    batch = mask.get_overlap_fractions_batch(
        "chr1",
        np.asarray([0, 10, 26, 35], dtype=np.int64),
        np.asarray([10, 20, 29, 35], dtype=np.int64),
    )
    assert batch.tolist() == pytest.approx([0.0, 1.0, 1.0, 0.0])
    assert mask.get_overlap_fractions_batch(
        "chr3",
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([2, 3], dtype=np.int64),
    ).tolist() == pytest.approx([0.0, 0.0])

    assert mask.is_masked("chr1", 0, 40, threshold=0.5) is True
    assert mask.is_masked("chr1", 0, 40, threshold=0.6) is False

    bed_c = tmp_path / "mask_c.bed"
    bed_c.write_text("chr1\t40\t50\nchr2\t0\t6\n")
    mask.add_beds(str(bed_c))

    assert len(mask.df) == 7
    assert sum(len(tree) for tree in mask.trees.values()) == 3
    assert mask.get_overlap_fraction("chr2", 0, 12) == pytest.approx(1.0)
    assert mask.has_any_overlap("chr1", 45, 46) is True


def test_depth_data_attach_baf_summary_populates_matching_cells_and_effective_counts():
    df = pd.DataFrame([
        {"Chr": "chr1", "Start": 100, "End": 200, "sample1": 2.5, "sample2": 6.0},
        {"Chr": "chr2", "Start": 200, "End": 260, "sample1": 1.5, "sample2": 3.0},
    ])
    data = DepthData(df, clamp_threshold=5.0)

    summary_df = pd.DataFrame([
        {
            "array_idx": 0,
            "sample": "sample1",
            "baf_median": 0.40,
            "minor_baf_median": 0.40,
            "baf_variance": 0.02,
            "baf_n_sites": 3,
            "baf_effective_variance": 0.03,
            "baf_effective_n_sites": 2,
        },
        {
            "array_idx": 1,
            "sample": "sample2",
            "baf_median": 0.60,
            "minor_baf_median": 0.40,
            "baf_variance": 0.05,
            "baf_n_sites": 4,
            "baf_effective_variance": 0.08,
            "baf_effective_n_sites": 1,
        },
        {
            "array_idx": 4,
            "sample": "sample1",
            "baf_median": 0.20,
            "minor_baf_median": 0.20,
            "baf_variance": 0.10,
            "baf_n_sites": 2,
            "baf_effective_variance": 0.10,
            "baf_effective_n_sites": 2,
        },
        {
            "array_idx": 0,
            "sample": "missing_sample",
            "baf_median": 0.20,
            "minor_baf_median": 0.20,
            "baf_variance": 0.10,
            "baf_n_sites": 2,
            "baf_effective_variance": 0.10,
            "baf_effective_n_sites": 2,
        },
    ])

    data.attach_baf_summary(summary_df, mappings=None)

    assert np.allclose(
        data.depth.detach().cpu().numpy(),
        np.asarray([[2.5, 5.0], [1.5, 3.0]], dtype=np.float32),
    )
    assert data.has_baf is True
    assert data.has_baf_effective_count is True
    assert data.baf_n_sites.detach().cpu().numpy().tolist() == [[3, 0], [0, 4]]
    assert data.baf_effective_n_sites.detach().cpu().numpy().tolist() == [[2, 0], [0, 1]]

    baf_median = data.baf_median.detach().cpu().numpy()
    assert baf_median[0, 0] == pytest.approx(0.40)
    assert np.isnan(baf_median[0, 1])
    assert baf_median[1, 1] == pytest.approx(0.60)


def test_depth_data_attach_baf_summary_validates_required_columns_and_effective_pairs():
    df = pd.DataFrame([
        {"Chr": "chr1", "Start": 100, "End": 200, "sample1": 2.0}
    ])
    data = DepthData(df)

    with pytest.raises(ValueError, match="missing required columns"):
        data.attach_baf_summary(
            pd.DataFrame([
                {
                    "array_idx": 0,
                    "sample": "sample1",
                    "baf_median": 0.5,
                    "minor_baf_median": 0.5,
                    "baf_variance": 0.1,
                }
            ]),
            mappings=None,
        )

    with pytest.raises(ValueError, match="must provide both baf_effective_variance and baf_effective_n_sites"):
        data.attach_baf_summary(
            pd.DataFrame([
                {
                    "array_idx": 0,
                    "sample": "sample1",
                    "baf_median": 0.5,
                    "minor_baf_median": 0.5,
                    "baf_variance": 0.1,
                    "baf_n_sites": 2,
                    "baf_effective_variance": 0.2,
                }
            ]),
            mappings=None,
        )


def test_depth_data_attach_sample_ploidy_requires_complete_pairs_and_attaches_matrix():
    df = pd.DataFrame([
        {"Chr": "chr1", "Start": 100, "End": 200, "sample1": 2.0, "sample2": 2.5},
        {"Chr": "chrX", "Start": 200, "End": 260, "sample1": 1.5, "sample2": 1.0},
    ])
    data = DepthData(df)

    with pytest.raises(ValueError, match="sample2/chrX"):
        data.attach_sample_ploidy({
            ("sample1", "chr1"): 2,
            ("sample2", "chr1"): 2,
            ("sample1", "chrX"): 1,
        })

    data.attach_sample_ploidy({
        ("sample1", "chr1"): 2,
        ("sample2", "chr1"): 2,
        ("sample1", "chrX"): 1,
        ("sample2", "chrX"): 2,
    })

    assert data.sample_ploidy.detach().cpu().numpy().tolist() == [[2, 2], [1, 2]]


def test_depth_data_subsamples_bins_and_samples_deterministically(monkeypatch):
    df = pd.DataFrame([
        {"Chr": "chr1", "Start": 100, "End": 150, "sample1": 1.0, "sample2": 2.0, "sample3": 3.0},
        {"Chr": "chr1", "Start": 150, "End": 200, "sample1": 1.1, "sample2": 2.1, "sample3": 3.1},
        {"Chr": "chr2", "Start": 200, "End": 260, "sample1": 1.2, "sample2": 2.2, "sample3": 3.2},
        {"Chr": "chr2", "Start": 260, "End": 320, "sample1": 1.3, "sample2": 2.3, "sample3": 3.3},
    ])

    choices = iter([
        np.asarray([2, 0], dtype=np.int64),
        np.asarray([2, 0], dtype=np.int64),
    ])
    monkeypatch.setattr(depth_module.np.random, "choice", lambda *args, **kwargs: next(choices))

    data = DepthData(
        df,
        subsample_bins=2,
        subsample_samples=2,
        seed=123,
        clamp_threshold=None,
    )

    assert data.sample_ids == ["sample3", "sample1"]
    assert data.chr.tolist() == ["chr1", "chr2"]
    assert data.start.tolist() == [100, 200]
    assert np.allclose(
        data.depth.detach().cpu().numpy(),
        np.asarray([[3.0, 1.0], [3.2, 1.2]], dtype=np.float32),
    )
    assert data.interval_sizes.detach().cpu().numpy().reshape(-1).tolist() == pytest.approx([50.0, 60.0])


def test_depth_data_attach_baf_summary_without_effective_counts_leaves_effective_fields_unset():
    df = pd.DataFrame([
        {"Chr": "chr1", "Start": 100, "End": 200, "sample1": 2.0}
    ])
    data = DepthData(df)

    data.attach_baf_summary(
        pd.DataFrame([
            {
                "array_idx": 0,
                "sample": "sample1",
                "baf_median": 0.5,
                "minor_baf_median": 0.5,
                "baf_variance": 0.1,
                "baf_n_sites": 2,
            }
        ]),
        mappings=None,
    )

    assert data.has_baf is True
    assert data.has_baf_effective_count is False
    assert data.baf_effective_variance is None
    assert data.baf_effective_n_sites is None
    assert data.baf_n_sites.detach().cpu().numpy().tolist() == [[2]]


def test_depth_data_attach_sample_ploidy_ignores_empty_map():
    df = pd.DataFrame([
        {"Chr": "chr1", "Start": 100, "End": 200, "sample1": 2.0}
    ])
    data = DepthData(df)

    data.attach_sample_ploidy({})

    assert data.sample_ploidy is None


def test_torch_depth_helper_variants_match_expected_numeric_behavior():
    torch = depth_module.torch

    expected_depth = torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float32)
    scaled = depth_module._depth_variance_scale_torch(expected_depth)
    floored = depth_module._variance_expected_depth_torch(expected_depth, 0.5)
    unchanged = depth_module._variance_expected_depth_torch(expected_depth, 0.0)

    reference_variance = depth_module._count_anchored_reference_variance_torch(
        torch.tensor([100.0, 200.0], dtype=torch.float32),
        reference_bin_size=1000.0,
        bin_size_factor=500.0,
    )
    size_modifier = depth_module._size_modifier_torch(
        torch.tensor([50.0, 100.0], dtype=torch.float32),
        bin_size_factor=100.0,
    )
    spatial = depth_module._spatial_aggregate_variance_scale_torch(
        torch.tensor([1.0, 1_000.0, 20_000.0], dtype=torch.float32),
        torch.tensor(1_000.0, dtype=torch.float32),
    )

    assert np.allclose(
        scaled.detach().cpu().numpy(),
        _depth_variance_scale_numpy(np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float32)),
    )
    assert np.allclose(floored.detach().cpu().numpy(), np.asarray([0.5, 1.0, 2.0, 4.0], dtype=np.float32))
    assert np.allclose(unchanged.detach().cpu().numpy(), np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float32))
    assert np.allclose(reference_variance.detach().cpu().numpy(), np.asarray([[0.08, 0.04]], dtype=np.float32))
    assert np.allclose(size_modifier.detach().cpu().numpy(), np.asarray([2.0, 1.0], dtype=np.float32))
    assert np.allclose(
        spatial.detach().cpu().numpy(),
        _spatial_aggregate_variance_scale_numpy(
            np.asarray([1.0, 1_000.0, 20_000.0], dtype=np.float32),
            1_000.0,
        ),
        rtol=1e-5,
        atol=1e-6,
    )


def test_torch_center_and_baf_helpers_cover_reference_and_clipping_paths():
    torch = depth_module.torch

    log_lik = torch.tensor(
        [[[-2.0, -1.0]], [[-0.5, -3.0]]],
        dtype=torch.float32,
    )
    reference_probs = torch.tensor([0.75, 0.25], dtype=torch.float32)
    centered = depth_module._center_state_log_likelihood_table_torch(log_lik, reference_probs)

    centered_np = centered.detach().cpu().numpy()
    expected_np = _center_state_log_likelihood_table_numpy(
        log_lik.detach().cpu().numpy(),
        reference_probs.detach().cpu().numpy(),
    )
    assert np.allclose(centered_np, expected_np, rtol=1e-5, atol=1e-6)

    normalized = np.log(reference_probs.detach().cpu().numpy()).reshape(-1, 1, 1) + centered_np
    assert np.allclose(np.exp(normalized).sum(axis=0), np.ones((1, 2), dtype=np.float32), rtol=1e-5, atol=1e-6)

    clipped = depth_module._clip_baf_variance_torch(
        torch.tensor([float("nan"), float("inf"), float("-inf"), 0.0, 0.25], dtype=torch.float32)
    )
    robust_none = depth_module._robust_baf_log_likelihood_torch(log_lik, outlier_rate=0.0)
    robust_full = depth_module._robust_baf_log_likelihood_torch(log_lik, outlier_rate=1.0)
    robust_mixed = depth_module._robust_baf_log_likelihood_torch(log_lik, outlier_rate=0.25)

    assert np.allclose(clipped.detach().cpu().numpy(), np.asarray([1e6, 1e6, 1e-6, 1e-6, 0.25], dtype=np.float32))
    assert np.allclose(robust_none.detach().cpu().numpy(), log_lik.detach().cpu().numpy())
    assert np.allclose(
        robust_full.detach().cpu().numpy(),
        np.full_like(log_lik.detach().cpu().numpy(), math.log(2.0)),
    )
    assert np.allclose(
        robust_mixed.detach().cpu().numpy(),
        np.logaddexp(
            log_lik.detach().cpu().numpy() + math.log(0.75),
            math.log(2.0) + math.log(0.25),
        ),
        rtol=1e-5,
        atol=1e-6,
    )


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


def test_train_logs_periodic_elbo_progress(monkeypatch, caplog):
    model = object.__new__(CNVModel)
    model.model = object()
    model.guide = object()
    model.guide_type = "delta"
    model._build_guide = lambda *args, **kwargs: model.guide
    model.loss_history = {"epoch": [], "elbo": []}
    model.current_data = None

    losses = iter([100.0, 99.5, 99.25, 99.125])

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

    caplog.set_level(logging.INFO, logger="gatk_sv_gd.training")

    model.train(
        data,
        max_iter=4,
        log_freq=2,
        early_stopping=False,
    )

    messages = [record.getMessage() for record in caplog.records]
    assert "Training started: max_iter=4 early_stopping=False" in messages
    assert "Training progress: epoch=2 loss=99.5000" in messages
    assert "Training progress: epoch=4 loss=99.1250" in messages
    assert "Training completed: epochs=4 final_loss=99.1250" in messages


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

    assert model.latent_sites == ["sample_var", "length_scale_var", "baf_temperature", "pair_state_probs"]
    assert block_calls["expose"] == ["sample_var", "length_scale_var", "baf_temperature", "pair_state_probs"]


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

    assert model.latent_sites == ["bin_bias", "sample_var", "length_scale_var", "baf_temperature"]
    assert block_calls["expose"] == ["bin_bias", "sample_var", "length_scale_var", "baf_temperature"]


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

    assert "length_scale_var" in model.latent_sites
    assert "baf_temperature" not in model.latent_sites
    assert "baf_temperature" not in block_calls["expose"]


def test_model_requires_count_anchored_normalization_metadata():
    model = object.__new__(CNVModel)
    model.debug = False
    model._zero_t = 0.0
    model._count_anchored_reference_variance_t = None

    with pytest.raises(RuntimeError, match="sample_raw_count_medians and reference_bin_size are required"):
        model.model(depth=None, interval_sizes=None, n_bins=1, n_samples=1)


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


@pytest.mark.parametrize("baf_outlier_rate", [-0.1, 1.0])
def test_init_rejects_invalid_baf_outlier_rate(baf_outlier_rate):
    with pytest.raises(ValueError, match=r"baf_outlier_rate must be in \[0, 1\)\."):
        CNVModel(baf_outlier_rate=baf_outlier_rate)


@pytest.mark.parametrize("null_state_prior", [-0.1, 1.0])
def test_init_rejects_invalid_null_state_prior(null_state_prior):
    with pytest.raises(ValueError, match=r"null_state_prior must be in \[0, 1\)\."):
        CNVModel(null_state_prior=null_state_prior)


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


def test_select_state_log_likelihood_torch_matches_mask_loop():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "gather"):
        pytest.skip("real torch is not available")

    log_lik_table = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[1.1, 1.2], [1.3, 1.4]],
            [[2.1, 2.2], [2.3, 2.4]],
        ],
        dtype=torch.float32,
    )
    pair_state = torch.tensor([[0, 2], [1, 0]], dtype=torch.long)

    selected = _select_state_log_likelihood_torch(log_lik_table, pair_state)

    expected = torch.zeros_like(selected)
    for state_idx in range(log_lik_table.shape[0]):
        expected = expected + torch.where(
            pair_state == state_idx,
            log_lik_table[state_idx],
            torch.zeros_like(selected),
        )

    assert torch.allclose(selected, expected)


def test_select_state_log_likelihood_torch_matches_enumerated_mask_loop():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "gather"):
        pytest.skip("real torch is not available")

    log_lik_table = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[1.1, 1.2], [1.3, 1.4]],
            [[2.1, 2.2], [2.3, 2.4]],
        ],
        dtype=torch.float32,
    )
    pair_state = torch.tensor([[[0]], [[2]], [[1]]], dtype=torch.long)

    selected = _select_state_log_likelihood_torch(log_lik_table, pair_state)

    expected = torch.zeros((pair_state.shape[0],) + log_lik_table.shape[1:], dtype=log_lik_table.dtype)
    for state_idx in range(log_lik_table.shape[0]):
        expected = expected + torch.where(
            pair_state == state_idx,
            log_lik_table[state_idx],
            torch.zeros_like(log_lik_table[state_idx]),
        )

    assert torch.allclose(selected, expected)


def test_depth_variance_scale_numpy_tracks_expected_depth_and_stays_positive():
    scales = _depth_variance_scale_numpy(np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float32))

    assert scales[0] == pytest.approx(1e-6)
    assert scales[1] == pytest.approx(0.5)
    assert scales[2] == pytest.approx(1.0)
    assert scales[3] == pytest.approx(2.0)

    powered_scales = _depth_variance_scale_numpy(
        np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float32),
        power=2.0,
    )

    assert powered_scales[0] == pytest.approx(1e-6)
    assert powered_scales[1] == pytest.approx(0.25)
    assert powered_scales[2] == pytest.approx(1.0)
    assert powered_scales[3] == pytest.approx(4.0)


def test_lognormal_location_from_mean_matches_requested_natural_scale_mean():
    location = _lognormal_location_from_mean(1.5, 0.25)

    assert np.exp(location + 0.5 * 0.25 ** 2) == pytest.approx(1.5)


def test_count_anchored_reference_variance_numpy_matches_poisson_scale():
    variance = _count_anchored_reference_variance_numpy(
        np.asarray([2000.0, 4000.0], dtype=np.float32),
        reference_bin_size=10000.0,
        bin_size_factor=10000.0,
    )

    assert variance.shape == (1, 2)
    assert variance[0, 0] == pytest.approx(0.002)
    assert variance[0, 1] == pytest.approx(0.001)


def test_count_anchored_reference_variance_numpy_skips_rescaling_without_positive_bin_factor():
    variance = _count_anchored_reference_variance_numpy(
        np.asarray([2000.0, 4000.0], dtype=np.float32),
        reference_bin_size=10000.0,
        bin_size_factor=0.0,
    )

    assert variance.shape == (1, 2)
    assert variance[0].tolist() == pytest.approx([0.002, 0.001])


def test_variance_expected_depth_numpy_applies_floor_only_when_positive():
    expected_depth = np.asarray([0.01, 0.2, 2.0], dtype=np.float32)

    unchanged = _variance_expected_depth_numpy(expected_depth, min_expected_depth=0.0)
    floored = _variance_expected_depth_numpy(expected_depth, min_expected_depth=0.1)

    assert unchanged.tolist() == pytest.approx([0.01, 0.2, 2.0])
    assert floored.tolist() == pytest.approx([0.1, 0.2, 2.0])


def test_size_modifier_numpy_uses_active_bin_factor_or_unity():
    interval_sizes = np.asarray([100.0, 200.0, 400.0], dtype=np.float32)

    assert _size_modifier_numpy(interval_sizes, bin_size_factor=200.0).tolist() == pytest.approx([2.0, 1.0, 0.5])
    assert _size_modifier_numpy(interval_sizes, bin_size_factor=0.0).tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_init_adds_length_scale_var_site_for_count_anchored_models(monkeypatch):
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
        sample_raw_count_medians=[2000.0, 2500.0],
        reference_bin_size=10000.0,
    )

    assert "length_scale_var" in model.latent_sites
    assert "length_scale_var" in block_calls["expose"]
    assert "bin_var" not in model.latent_sites


def test_spatial_aggregate_variance_scale_numpy_matches_small_bin_limit():
    factor = _spatial_aggregate_variance_scale_numpy(
        np.asarray([1.0], dtype=np.float64),
        1_000_000.0,
    )

    assert factor[0] == pytest.approx(1.0, abs=1e-6)


def test_spatial_aggregate_variance_scale_numpy_scales_as_inverse_length_for_large_bins():
    interval_sizes = np.asarray([100_000.0, 200_000.0], dtype=np.float64)
    factor = _spatial_aggregate_variance_scale_numpy(interval_sizes, 1_000.0)

    assert factor[0] == pytest.approx((2.0 * 1_000.0) / 100_000.0, rel=1e-2)
    assert factor[1] / factor[0] == pytest.approx(0.5, rel=1e-2)


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


def test_run_discrete_inference_always_uses_full_pair_state_prior():
    model = object.__new__(CNVModel)
    model.pair_states = [(1, 1), (1, 2)]
    model.max_total_cn = 3
    model.bin_size_factor = 1.0
    model.baf_temperature = 0.0
    model.baf_outlier_rate = 0.0
    model.null_state_prior = 0.0
    model.var_length_scale = 1.0
    model._count_anchored_reference_variance_np = _count_anchored_reference_variance_numpy(
        np.asarray([4.0, 4.0], dtype=np.float32),
        reference_bin_size=1.0,
        bin_size_factor=1.0,
    )

    model.get_map_estimates = lambda data: {
        "bin_bias": np.asarray([1.0, 1.0], dtype=np.float32),
        "sample_var": np.asarray([0.0, 0.0], dtype=np.float32),
        "bin_var": np.asarray([0.0, 0.0], dtype=np.float32),
        "pair_state_probs": np.asarray([[0.99, 0.01], [0.99, 0.01]], dtype=np.float32),
        "length_scale_var": np.asarray(1.0, dtype=np.float32),
    }

    data = SimpleNamespace(
        depth=_FakeTensor([[2.9, 2.9], [2.9, 2.9]]),
        interval_sizes=_FakeTensor([[1.0], [1.0]]),
        n_bins=2,
        n_samples=2,
        has_baf=False,
    )

    posterior = CNVModel.run_discrete_inference(model, data)
    cn_posterior = posterior["cn_posterior"]

    var_cn2, var_cn3 = _depth_variance_scale_numpy(np.asarray([2.0, 3.0], dtype=np.float32))
    log_lik_cn2 = -0.5 * np.log(2.0 * np.pi * var_cn2) - ((2.9 - 2.0) ** 2) / (2.0 * var_cn2)
    log_lik_cn3 = -0.5 * np.log(2.0 * np.pi * var_cn3) - ((2.9 - 3.0) ** 2) / (2.0 * var_cn3)
    odds_cn3 = np.exp(log_lik_cn3 - log_lik_cn2) * (0.01 / 0.99)
    expected_p_cn3 = odds_cn3 / (1.0 + odds_cn3)

    assert cn_posterior[0, 0, 3] == pytest.approx(expected_p_cn3, rel=1e-5)
    assert cn_posterior[0, 0, 3] < 0.1


def test_run_discrete_inference_assigns_extreme_outliers_to_null_state():
    model = object.__new__(CNVModel)
    model.pair_states = [(1, 1), (1, 2)]
    model.max_total_cn = 3
    model.bin_size_factor = 1.0
    model.baf_temperature = 0.0
    model.baf_outlier_rate = 0.0
    model.null_state_prior = 0.2
    model.var_length_scale = 1.0
    model._count_anchored_reference_variance_np = _count_anchored_reference_variance_numpy(
        np.asarray([4.0, 4.0], dtype=np.float32),
        reference_bin_size=1.0,
        bin_size_factor=1.0,
    )

    model.get_map_estimates = lambda data: {
        "bin_bias": np.asarray([1.0, 1.0], dtype=np.float32),
        "sample_var": np.asarray([0.0, 0.0], dtype=np.float32),
        "bin_var": np.asarray([0.0, 0.0], dtype=np.float32),
        "pair_state_probs": np.asarray([[0.99, 0.01], [0.99, 0.01]], dtype=np.float32),
        "length_scale_var": np.asarray(1.0, dtype=np.float32),
    }

    data = SimpleNamespace(
        depth=_FakeTensor([[100.0, 100.0], [100.0, 100.0]]),
        interval_sizes=_FakeTensor([[1.0], [1.0]]),
        n_bins=2,
        n_samples=2,
        has_baf=False,
    )

    posterior = CNVModel.run_discrete_inference(model, data)

    assert posterior["null_posterior"][0, 0] > 0.999
    assert posterior["cn_posterior"][0, 0].sum() < 1e-6


def test_run_discrete_inference_uses_count_anchored_poisson_baseline_when_available():
    model = object.__new__(CNVModel)
    model.pair_states = [(1, 1), (1, 2)]
    model.max_total_cn = 3
    model.bin_size_factor = 1.0
    model.baf_temperature = 0.0
    model.baf_outlier_rate = 0.0
    model.null_state_prior = 0.0
    model.reference_bin_size = 1.0
    model._count_anchored_reference_variance_np = _count_anchored_reference_variance_numpy(
        np.asarray([2000.0, 2000.0], dtype=np.float32),
        reference_bin_size=1.0,
        bin_size_factor=1.0,
    )

    model.get_map_estimates = lambda data: {
        "bin_bias": np.asarray([1.0, 1.0], dtype=np.float32),
        "sample_var": np.asarray([0.0, 0.0], dtype=np.float32),
        "bin_var": np.asarray([0.0, 0.0], dtype=np.float32),
        "pair_state_probs": np.asarray([[0.99, 0.01], [0.99, 0.01]], dtype=np.float32),
    }

    data = SimpleNamespace(
        depth=_FakeTensor([[2.1, 2.1], [2.1, 2.1]]),
        interval_sizes=_FakeTensor([[1.0], [1.0]]),
        n_bins=2,
        n_samples=2,
        has_baf=False,
    )

    posterior = CNVModel.run_discrete_inference(model, data)
    cn_posterior = posterior["cn_posterior"]

    var_cn2, var_cn3 = np.asarray([0.002, 0.003], dtype=np.float64)
    log_lik_cn2 = -0.5 * np.log(2.0 * np.pi * var_cn2) - ((2.1 - 2.0) ** 2) / (2.0 * var_cn2)
    log_lik_cn3 = -0.5 * np.log(2.0 * np.pi * var_cn3) - ((2.1 - 3.0) ** 2) / (2.0 * var_cn3)
    odds_cn3 = np.exp(log_lik_cn3 - log_lik_cn2) * (0.01 / 0.99)
    expected_p_cn3 = odds_cn3 / (1.0 + odds_cn3)

    assert cn_posterior[0, 0, 3] == pytest.approx(expected_p_cn3, rel=1e-5)
    assert cn_posterior[0, 0, 3] < 1e-20


def test_run_discrete_inference_uses_count_anchored_length_scale_var_when_available():
    model = object.__new__(CNVModel)
    model.pair_states = [(1, 1), (1, 2)]
    model.max_total_cn = 3
    model.bin_size_factor = 1.0
    model.baf_temperature = 0.0
    model.baf_outlier_rate = 0.0
    model.null_state_prior = 0.0
    model.reference_bin_size = 1.0
    model.var_length_scale = 5_000.0
    model._count_anchored_reference_variance_np = _count_anchored_reference_variance_numpy(
        np.asarray([2000.0, 2000.0], dtype=np.float32),
        reference_bin_size=1.0,
        bin_size_factor=1.0,
    )

    sample_var_val = 5e-4
    length_scale_var_val = 1_000.0

    model.get_map_estimates = lambda data: {
        "bin_bias": np.asarray([1.0, 1.0], dtype=np.float32),
        "sample_var": np.asarray([sample_var_val, sample_var_val], dtype=np.float32),
        "bin_var": np.asarray([0.0, 0.0], dtype=np.float32),
        "pair_state_probs": np.asarray([[0.99, 0.01], [0.99, 0.01]], dtype=np.float32),
        "length_scale_var": np.asarray(length_scale_var_val, dtype=np.float32),
    }

    data = SimpleNamespace(
        depth=_FakeTensor([[2.1, 2.1], [2.1, 2.1]]),
        interval_sizes=_FakeTensor([[1.0], [1.0]]),
        n_bins=2,
        n_samples=2,
        has_baf=False,
    )

    posterior = CNVModel.run_discrete_inference(model, data)
    cn_posterior = posterior["cn_posterior"]

    # Poisson variance at L=1, L_ref=1, m=2000: 4/m*(L_ref/L)*(d/2)
    poisson_cn2 = 0.002
    poisson_cn3 = 0.003
    spatial_factor = _spatial_aggregate_variance_scale_numpy(
        np.asarray([1.0], dtype=np.float64),
        length_scale_var_val,
    )[0]
    excess_cn2 = (2.0 ** 2) * sample_var_val * spatial_factor
    excess_cn3 = (3.0 ** 2) * sample_var_val * spatial_factor
    var_cn2 = poisson_cn2 + excess_cn2
    var_cn3 = poisson_cn3 + excess_cn3

    log_lik_cn2 = -0.5 * np.log(2.0 * np.pi * var_cn2) - ((2.1 - 2.0) ** 2) / (2.0 * var_cn2)
    log_lik_cn3 = -0.5 * np.log(2.0 * np.pi * var_cn3) - ((2.1 - 3.0) ** 2) / (2.0 * var_cn3)
    odds_cn3 = np.exp(log_lik_cn3 - log_lik_cn2) * (0.01 / 0.99)
    expected_p_cn3 = odds_cn3 / (1.0 + odds_cn3)

    assert cn_posterior[0, 0, 3] == pytest.approx(expected_p_cn3, rel=1e-5)

    # Reference: a shorter correlation length should reduce the excess
    # variance for the same interval size and make CN=3 less likely here.
    shorter_factor = _spatial_aggregate_variance_scale_numpy(
        np.asarray([1.0], dtype=np.float64),
        0.1,
    )[0]
    shorter_var_cn2 = poisson_cn2 + (2.0 ** 2) * sample_var_val * shorter_factor
    shorter_var_cn3 = poisson_cn3 + (3.0 ** 2) * sample_var_val * shorter_factor
    log_lik_cn2_ref = -0.5 * np.log(2.0 * np.pi * shorter_var_cn2) - (
        (2.1 - 2.0) ** 2
    ) / (2.0 * shorter_var_cn2)
    log_lik_cn3_ref = -0.5 * np.log(2.0 * np.pi * shorter_var_cn3) - (
        (2.1 - 3.0) ** 2
    ) / (2.0 * shorter_var_cn3)
    odds_cn3_ref = np.exp(log_lik_cn3_ref - log_lik_cn2_ref) * (0.01 / 0.99)
    p_cn3_short_length_scale = odds_cn3_ref / (1.0 + odds_cn3_ref)
    assert cn_posterior[0, 0, 3] > p_cn3_short_length_scale


def test_run_discrete_inference_reweights_reference_prior_for_haploid_ploidy():
    model = object.__new__(CNVModel)
    model.pair_states = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    model.n_states = len(model.pair_states)
    model.max_total_cn = 4
    model.bin_size_factor = 1.0
    model.baf_temperature = 0.0
    model.baf_outlier_rate = 0.0
    model.null_state_prior = 0.0
    model.var_length_scale = 1_000.0
    model.alpha_ref = 50.0
    model.alpha_non_ref = 1.0
    model.ref_state_idx = model.pair_states.index((1, 1))
    model.min_variance_expected_depth = 0.1
    model._count_anchored_reference_variance_np = _count_anchored_reference_variance_numpy(
        np.asarray([200.0, 200.0], dtype=np.float32),
        reference_bin_size=1.0,
        bin_size_factor=1.0,
    )

    pair_priors = np.asarray(
        [[1.0 / 55.0, 1.0 / 55.0, 1.0 / 55.0, 50.0 / 55.0, 1.0 / 55.0, 1.0 / 55.0]] * 2,
        dtype=np.float32,
    )
    model.get_map_estimates = lambda data: {
        "bin_bias": np.asarray([1.0, 1.0], dtype=np.float32),
        "sample_var": np.asarray([0.2, 0.2], dtype=np.float32),
        "pair_state_probs": pair_priors,
        "length_scale_var": np.asarray(1_000.0, dtype=np.float32),
    }

    base_data = {
        "depth": _FakeTensor([[1.5, 1.5], [1.5, 1.5]]),
        "interval_sizes": _FakeTensor([[1.0], [1.0]]),
        "n_bins": 2,
        "n_samples": 2,
        "has_baf": False,
    }

    diploid_posterior = CNVModel.run_discrete_inference(
        model,
        SimpleNamespace(**base_data),
    )["pair_state_posterior"][0, 0]
    haploid_posterior = CNVModel.run_discrete_inference(
        model,
        SimpleNamespace(
            sample_ploidy=_FakeTensor([[1, 1], [1, 1]]),
            **base_data,
        ),
    )["pair_state_posterior"][0, 0]

    assert diploid_posterior[3] > diploid_posterior[1]
    assert haploid_posterior[1] > haploid_posterior[3]
    assert haploid_posterior[1] > 0.9


def test_run_discrete_inference_min_variance_expected_depth_prevents_copy0_null_collapse():
    def make_model(min_variance_expected_depth: float):
        model = object.__new__(CNVModel)
        model.pair_states = [(0, 0), (0, 1)]
        model.n_states = len(model.pair_states)
        model.max_total_cn = 1
        model.bin_size_factor = 1.0
        model.baf_temperature = 0.0
        model.baf_outlier_rate = 0.0
        model.null_state_prior = 0.2
        model.var_length_scale = 1_000.0
        model.min_variance_expected_depth = min_variance_expected_depth
        model._count_anchored_reference_variance_np = _count_anchored_reference_variance_numpy(
            np.asarray([200.0, 200.0], dtype=np.float32),
            reference_bin_size=1.0,
            bin_size_factor=1.0,
        )
        model.get_map_estimates = lambda data: {
            "bin_bias": np.asarray([1.0, 1.0], dtype=np.float32),
            "sample_var": np.asarray([0.2, 0.2], dtype=np.float32),
            "pair_state_probs": np.asarray([[0.4, 0.4], [0.4, 0.4]], dtype=np.float32),
            "length_scale_var": np.asarray(1_000.0, dtype=np.float32),
        }
        return model

    data = SimpleNamespace(
        depth=_FakeTensor([[0.1, 0.1], [0.1, 0.1]]),
        interval_sizes=_FakeTensor([[1.0], [1.0]]),
        n_bins=2,
        n_samples=2,
        has_baf=False,
    )

    baseline = CNVModel.run_discrete_inference(make_model(0.0), data)
    floored = CNVModel.run_discrete_inference(make_model(0.1), data)

    baseline_pair = baseline["pair_state_posterior"][0, 0]
    floored_pair = floored["pair_state_posterior"][0, 0]

    assert baseline["null_posterior"][0, 0] > 0.75
    assert baseline_pair[0] == pytest.approx(0.0)
    assert floored_pair[0] > 0.65
    assert floored["null_posterior"][0, 0] < 0.3


def test_run_discrete_inference_baf_outlier_rate_caps_contradictory_baf_penalty():
    maps = {
        "bin_bias": np.asarray([1.0, 1.0], dtype=np.float32),
        "sample_var": np.asarray([0.05, 0.05], dtype=np.float32),
        "bin_var": np.asarray([0.0, 0.0], dtype=np.float32),
        "pair_state_probs": np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
    }

    def make_model(baf_outlier_rate: float):
        model = object.__new__(CNVModel)
        model.pair_states = [(1, 1), (1, 2)]
        model.max_total_cn = 3
        model.bin_size_factor = 1.0
        model.baf_temperature = 1.0
        model.baf_outlier_rate = baf_outlier_rate
        model.var_length_scale = 1.0
        model._count_anchored_reference_variance_np = _count_anchored_reference_variance_numpy(
            np.asarray([4.0, 4.0], dtype=np.float32),
            reference_bin_size=1.0,
            bin_size_factor=1.0,
        )
        model._pair_state_prior_mean_np = np.asarray([0.5, 0.5], dtype=np.float64)
        model.get_map_estimates = lambda data: {
            **maps,
            "length_scale_var": np.asarray(1.0, dtype=np.float32),
        }
        return model

    data = SimpleNamespace(
        depth=_FakeTensor([[2.9, 2.9], [2.9, 2.9]]),
        interval_sizes=_FakeTensor([[1.0], [1.0]]),
        minor_baf_median=_FakeTensor([[0.49, 0.49], [0.49, 0.49]]),
        baf_variance=_FakeTensor([[1e-4, 1e-4], [1e-4, 1e-4]]),
        baf_n_sites=_FakeTensor([[5, 5], [5, 5]]),
        n_bins=2,
        n_samples=2,
        has_baf=True,
    )

    plain_posterior = CNVModel.run_discrete_inference(make_model(0.0), data)["cn_posterior"][0, 0, 3]
    robust_posterior = CNVModel.run_discrete_inference(make_model(0.05), data)["cn_posterior"][0, 0, 3]

    assert plain_posterior < 1e-10
    assert robust_posterior > 1e-3
    assert robust_posterior > plain_posterior * 1e6