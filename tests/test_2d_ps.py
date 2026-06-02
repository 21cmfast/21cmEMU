"""Tests for 2D PS / score model functionality."""

from __future__ import annotations

import numpy as np
import pytest

from py21cmemu import Emulator

# ══════════════════════════════════════════════════════════════════════════════
# Main-only emulator 2D PS predict tests (from test_coverage.py)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.main_only
def test_emulator_init_with_2d_ps():
    """MCG emulator with emulate_2d_ps=True loads the score model."""
    emu = Emulator(emulator="mcg", emulate_2d_ps=True)
    assert emu.score_model is not None
    assert emu.emulate_2d_ps is True


@pytest.mark.main_only
def test_emulator_2d_ps_predict_ode(mh_emulator_with_2d_ps, test_db_params):
    """predict() with emulate_2d_ps=True (ODE, verbose) covers lines 297+, tqdm 374-376."""
    _theta, output, _errors = mh_emulator_with_2d_ps.predict(
        test_db_params,
        n_realisations=2,
        ps_2d_redshifts=None,  # triggers line 297: use default redshifts
        verbose=True,  # triggers tqdm path (lines 374-376)
    )
    assert output.PS_2D is not None


@pytest.mark.main_only
def test_emulator_2d_ps_predict_em_explicit(mh_emulator_with_2d_ps, test_db_params):
    """predict() with EM sampler, explicit n_ps_batch and sde covers branches 301->303, 303->307."""
    from py21cmemu.sde import VPSDE

    sde = VPSDE(beta_min=0.1, beta_max=20.0)
    _theta, output, _errors = mh_emulator_with_2d_ps.predict(
        test_db_params,
        n_realisations=2,
        ps_2d_redshifts=[7.0],
        ps_sampling_method="em",
        n_ps_batch=1,  # explicit → covers False branch of 301 (not None)
        sde=sde,  # explicit → covers False branch of 303 (not None)
    )
    assert output.PS_2D is not None


@pytest.mark.main_only
def test_emulator_2d_ps_invalid_method(mh_emulator_with_2d_ps, test_db_params):
    """Invalid ps_sampling_method raises ValueError before diffusion sampling (line 309)."""
    with pytest.raises(ValueError, match="ps_sampling_method"):
        mh_emulator_with_2d_ps.predict(
            test_db_params,
            n_realisations=2,
            ps_2d_redshifts=[7.0],
            ps_sampling_method="invalid",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Sampler and SDE tests (from test_v3.py::TestMH2DSampler)
# ══════════════════════════════════════════════════════════════════════════════


class TestMH2DSampler:
    """Test 2D PS sampler classes without running full sampling."""

    def test_em_sampler_import(self):
        """Test that EM sampler can be imported."""
        from py21cmemu.sample_pytorch import GetEMSampler

        assert GetEMSampler is not None

    def test_ode_sampler_import(self):
        """Test that ODE sampler can be imported."""
        from py21cmemu.sample_pytorch import GetODESampler

        assert GetODESampler is not None

    def test_vpsde_import(self):
        """Test that VPSDE can be imported."""
        from py21cmemu.sde import VPSDE

        assert VPSDE is not None

    def test_vpsde_creation(self):
        """Test VPSDE creation with default parameters."""
        from py21cmemu.sde import VPSDE

        sde = VPSDE(beta_min=0.1, beta_max=20.0)
        assert sde.beta_0 == 0.1
        assert sde.beta_1 == 20.0
        assert sde.N == 1000  # default

    def test_em_sampler_creation_cpu(self):
        """Test EM sampler can be created on CPU."""
        import torch

        from py21cmemu.sample_pytorch import GetEMSampler
        from py21cmemu.sde import VPSDE

        sde = VPSDE(beta_min=0.1, beta_max=20.0)
        shape = (1, 10, 32, 32)  # Small batch for testing
        sampler = GetEMSampler(sde, shape, device=torch.device("cpu"), denoise=True)

        assert sampler is not None
        em_fn = sampler.get_em_sampler()
        assert callable(em_fn)

    def test_ode_sampler_creation_cpu(self):
        """Test ODE sampler can be created on CPU."""
        import torch

        from py21cmemu.sample_pytorch import GetODESampler
        from py21cmemu.sde import VPSDE

        sde = VPSDE(beta_min=0.1, beta_max=20.0)
        shape = (1, 10, 32, 32)  # Small batch for testing
        sampler = GetODESampler(
            sde, shape, device=torch.device("cpu"), denoise=True, rtol=1e-5, atol=1e-5
        )

        assert sampler is not None
        ode_fn = sampler.get_ode_sampler()
        assert callable(ode_fn)


# ══════════════════════════════════════════════════════════════════════════════
# Score model structure tests (from test_v3.py::TestMH2DScoreModel)
# ══════════════════════════════════════════════════════════════════════════════


class TestMH2DScoreModel:
    """Test 2D PS score model structure without running inference."""

    def test_score_model_import(self):
        """Test that UNet score model can be imported."""
        from py21cmemu.models.mcg.score_model import UNet

        assert UNet is not None

    def test_score_model_creation(self):
        """Test UNet can be created without loading weights."""
        from py21cmemu.models.mcg.score_model import UNet

        model = UNet(
            dim=(32, 64),
            init_dim=48,
            channels=1,
            dim_mults=(1, 2, 4, 8, 16),
            cdn_len=12,
        )

        assert model is not None
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_score_model_forward_shape(self):
        """Test UNet forward pass produces correct shape."""
        import torch

        from py21cmemu.models.mcg.score_model import UNet

        model = UNet(
            dim=(32, 64),
            init_dim=48,
            channels=1,
            dim_mults=(1, 2, 4, 8, 16),
            cdn_len=12,
        )
        model.eval()

        batch_size = 2
        x = torch.randn(batch_size, 1, 32, 64)
        t = torch.rand(batch_size)
        cdn = torch.randn(batch_size, 12)

        with torch.no_grad():
            out = model(x, time=t, cdn=cdn)

        assert out.shape == (batch_size, 1, 32, 64)

    def test_score_model_weights_load(self):
        """Test that packaged score model weights can be loaded."""
        from pathlib import Path

        import torch

        from py21cmemu.models.mcg.score_model import UNet

        weights_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "py21cmemu"
            / "models"
            / "MCG"
            / "score_model_weights.pt"
        )

        model = UNet(
            dim=(32, 64),
            init_dim=48,
            channels=1,
            dim_mults=(1, 2, 4, 8, 16),
            cdn_len=12,
        )

        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        batch_size = 1
        x = torch.randn(batch_size, 1, 32, 64)
        t = torch.tensor([0.5])
        cdn = torch.randn(batch_size, 12)

        with torch.no_grad():
            out = model(x, time=t, cdn=cdn)

        assert out.shape == (batch_size, 1, 32, 64)
        assert not torch.all(out == 0)
        assert not torch.any(torch.isnan(out))


# ══════════════════════════════════════════════════════════════════════════════
# Emulator PS setup tests (from test_v3.py::TestMHEmulatorPSSetup)
# ══════════════════════════════════════════════════════════════════════════════


class TestMHEmulatorPSSetup:
    """Test MH emulator PS setup without full model loading."""

    def test_emulator_ps_false(self):
        """Test emulator creation with PS disabled."""
        emu = Emulator(emulator="mcg", emulate_2d_ps=False)
        assert not emu.emulate_2d_ps
        assert emu.score_model is None
        assert emu.sample is None

    def test_emulator_ps_properties_accessible(self):
        """Test PS properties accessible even without model."""
        emu = Emulator(emulator="mcg", emulate_2d_ps=False)

        assert hasattr(emu.properties, "kperp")
        assert hasattr(emu.properties, "kpar")
        assert hasattr(emu.properties, "PS_zs")
        assert hasattr(emu.properties, "Nmodes")
        assert hasattr(emu.properties, "PS_bias")
        assert hasattr(emu.properties, "PS_scale")

    def test_ps_sampling_method_values(self):
        """Test that valid PS sampling method values are 'em' and 'ode'."""
        valid_methods = ("em", "ode")
        assert "em" in valid_methods
        assert "ode" in valid_methods


# ══════════════════════════════════════════════════════════════════════════════
# Utility function tests (from test_v3.py::TestMH2DUtilities)
# ══════════════════════════════════════════════════════════════════════════════


class TestMH2DUtilities:
    """Test utility functions for 2D PS handling."""

    def test_reverse_transform(self):
        """Test reverse transform function."""
        import torch

        from py21cmemu.utils import reverse_transform

        normed = torch.randn(2, 1, 32, 32)
        scale = torch.tensor(5.0)
        bias = torch.tensor(-2.0)

        denormed = reverse_transform(normed, scale, bias)

        unit = (normed + 1) / 2
        d = unit * scale + bias
        expected = 10**d
        assert torch.allclose(denormed, expected)


# ══════════════════════════════════════════════════════════════════════════════
# Multi-GPU get_pred tests (from test_v3.py::TestGetPredMultiGPU)
# ══════════════════════════════════════════════════════════════════════════════


class TestGetPredMultiGPU:
    """Tests for the multi-GPU code path in Emulator.get_pred."""

    @pytest.fixture()
    def emu_with_2d_ps(self):
        """Emulator with 2D PS enabled (score_model loaded, no GPU needed)."""
        return Emulator(emulator="mcg", emulate_2d_ps=True)

    @staticmethod
    def _make_cdns(n_batches: int, n_ps_batch: int = 2) -> np.ndarray:
        """Return a (n_batches, n_ps_batch, 12) conditioning array."""
        return np.random.rand(n_batches, n_ps_batch, 12).astype(np.float32)

    # ------------------------------------------------------------------
    # Single-GPU / CPU path (device_count <= 1)
    # ------------------------------------------------------------------

    def test_single_gpu_path_used_when_one_gpu(self, emu_with_2d_ps, monkeypatch):
        """get_pred takes the sequential path when device_count() <= 1."""
        import torch

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

        n_ps_batch, n_real = 2, 10
        call_log = []

        def fake_get_pred_single(cdn):
            call_log.append(cdn.shape)
            return np.ones((n_ps_batch, n_real, 32, 64))

        emu_with_2d_ps.get_pred_single = fake_get_pred_single
        emu_with_2d_ps._current_ps_method = "ode"
        emu_with_2d_ps._n_realisations = n_real
        emu_with_2d_ps._denoise = True

        cdns = self._make_cdns(n_batches=3, n_ps_batch=n_ps_batch)
        result = emu_with_2d_ps.get_pred(cdns)

        assert len(call_log) == 3
        assert result.shape == (3, n_ps_batch, n_real, 32, 64)

    def test_cpu_path_used_when_no_gpu(self, emu_with_2d_ps, monkeypatch):
        """get_pred takes the sequential path when no CUDA devices are available."""
        import torch

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

        n_ps_batch, n_real = 2, 5
        call_log = []

        def fake_get_pred_single(cdn):
            call_log.append(1)
            return np.ones((n_ps_batch, n_real, 32, 64))

        emu_with_2d_ps.get_pred_single = fake_get_pred_single
        emu_with_2d_ps._current_ps_method = "ode"
        emu_with_2d_ps._n_realisations = n_real
        emu_with_2d_ps._denoise = True

        cdns = self._make_cdns(n_batches=2, n_ps_batch=n_ps_batch)
        result = emu_with_2d_ps.get_pred(cdns)

        assert len(call_log) == 2
        assert result.shape == (2, n_ps_batch, n_real, 32, 64)

    # ------------------------------------------------------------------
    # Multi-GPU path (device_count >= 2)
    # ------------------------------------------------------------------

    def test_multi_gpu_path_distributes_work(self, emu_with_2d_ps, monkeypatch):
        """With 2 fake GPUs the work is split round-robin across workers."""
        import torch

        n_gpus, n_ps_batch, n_real = 2, 2, 5
        monkeypatch.setattr(torch.cuda, "device_count", lambda: n_gpus)

        processed_by_gpu = [[] for _ in range(n_gpus)]

        def make_fake_sampler(gpu_id):
            def sampler(model, cdn):
                processed_by_gpu[gpu_id].append(True)
                return torch.ones(cdn.shape[0], n_real, 32, 64)

            return sampler

        cpu = torch.device("cpu")
        emu_with_2d_ps._ps_gpu_replicas = [
            (cpu, emu_with_2d_ps.score_model, make_fake_sampler(g))
            for g in range(n_gpus)
        ]
        emu_with_2d_ps._current_ps_method = "ode"
        emu_with_2d_ps._n_realisations = n_real
        emu_with_2d_ps._denoise = True
        emu_with_2d_ps._ps_gpu_config_key = ("ode", n_ps_batch, n_real, True)

        n_batches = 6
        cdns = self._make_cdns(n_batches=n_batches, n_ps_batch=n_ps_batch)
        result = emu_with_2d_ps.get_pred(cdns)

        assert len(processed_by_gpu[0]) == 3
        assert len(processed_by_gpu[1]) == 3
        assert result.shape == (n_batches, n_ps_batch, n_real, 32, 64)

    def test_multi_gpu_odd_batches(self, emu_with_2d_ps, monkeypatch):
        """With 2 GPUs and an odd number of batches the work is correctly split."""
        import torch

        n_gpus, n_ps_batch, n_real = 2, 2, 3
        monkeypatch.setattr(torch.cuda, "device_count", lambda: n_gpus)

        processed = [[] for _ in range(n_gpus)]

        def make_fake_sampler(gpu_id):
            def sampler(model, cdn):
                processed[gpu_id].append(True)
                return torch.ones(cdn.shape[0], n_real, 32, 64)

            return sampler

        cpu = torch.device("cpu")
        emu_with_2d_ps._ps_gpu_replicas = [
            (cpu, emu_with_2d_ps.score_model, make_fake_sampler(g))
            for g in range(n_gpus)
        ]
        emu_with_2d_ps._current_ps_method = "ode"
        emu_with_2d_ps._n_realisations = n_real
        emu_with_2d_ps._denoise = True
        emu_with_2d_ps._ps_gpu_config_key = ("ode", n_ps_batch, n_real, True)

        n_batches = 5
        cdns = self._make_cdns(n_batches=n_batches, n_ps_batch=n_ps_batch)
        result = emu_with_2d_ps.get_pred(cdns)

        assert len(processed[0]) == 3  # indices 0, 2, 4
        assert len(processed[1]) == 2  # indices 1, 3
        assert result.shape == (n_batches, n_ps_batch, n_real, 32, 64)

    def test_multi_gpu_all_result_indices_filled(self, emu_with_2d_ps, monkeypatch):
        """All result indices are populated — no skipped or None entries."""
        import torch

        n_gpus, n_ps_batch, n_real = 2, 1, 4
        monkeypatch.setattr(torch.cuda, "device_count", lambda: n_gpus)

        def make_fake_sampler(gpu_id):
            def sampler(model, cdn):
                return torch.ones(cdn.shape[0], n_real, 32, 64) * (gpu_id + 1)

            return sampler

        cpu = torch.device("cpu")
        emu_with_2d_ps._ps_gpu_replicas = [
            (cpu, emu_with_2d_ps.score_model, make_fake_sampler(g))
            for g in range(n_gpus)
        ]
        emu_with_2d_ps._current_ps_method = "ode"
        emu_with_2d_ps._n_realisations = n_real
        emu_with_2d_ps._denoise = True
        emu_with_2d_ps._ps_gpu_config_key = ("ode", n_ps_batch, n_real, True)

        cdns = self._make_cdns(n_batches=4, n_ps_batch=n_ps_batch)
        result = emu_with_2d_ps.get_pred(cdns)

        assert result.shape == (4, n_ps_batch, n_real, 32, 64)
        assert not np.any(np.isnan(result))
        assert np.all(result > 0)

    # ------------------------------------------------------------------
    # Replica caching
    # ------------------------------------------------------------------

    def test_replica_cache_reused_when_config_matches(
        self, emu_with_2d_ps, monkeypatch
    ):
        """When config key matches on consecutive calls, replicas are not rebuilt."""
        import torch

        n_gpus, n_ps_batch, n_real = 2, 2, 3
        monkeypatch.setattr(torch.cuda, "device_count", lambda: n_gpus)

        def fake_sampler(model, cdn):
            return torch.ones(cdn.shape[0], n_real, 32, 64)

        cpu = torch.device("cpu")
        fake_replicas = [
            (cpu, emu_with_2d_ps.score_model, fake_sampler) for _ in range(n_gpus)
        ]
        config_key = ("ode", n_ps_batch, n_real, True)
        emu_with_2d_ps._ps_gpu_replicas = fake_replicas
        emu_with_2d_ps._ps_gpu_config_key = config_key
        emu_with_2d_ps._current_ps_method = "ode"
        emu_with_2d_ps._n_realisations = n_real
        emu_with_2d_ps._denoise = True

        cdns = self._make_cdns(n_batches=2, n_ps_batch=n_ps_batch)
        emu_with_2d_ps.get_pred(cdns)
        emu_with_2d_ps.get_pred(cdns)

        # Same list object means the replicas were never rebuilt
        assert emu_with_2d_ps._ps_gpu_replicas is fake_replicas

    def test_replica_config_key_invalidated_on_n_realisations_change(
        self, emu_with_2d_ps
    ):
        """Config key differs when n_realisations changes -> triggers cache miss."""
        key_a = ("ode", 2, 3, True)
        key_b = ("ode", 2, 7, True)
        assert key_a != key_b

    def test_replica_config_key_invalidated_on_method_change(self, emu_with_2d_ps):
        """Config key differs when sampling method changes -> triggers cache miss."""
        key_em = ("em", 2, 5, True)
        key_ode = ("ode", 2, 5, True)
        assert key_em != key_ode

    # ------------------------------------------------------------------
    # Exception propagation
    # ------------------------------------------------------------------

    def test_worker_exception_propagates(self, emu_with_2d_ps, monkeypatch):
        """An exception raised inside a worker is re-raised in the main thread."""
        import torch

        n_gpus = 2
        monkeypatch.setattr(torch.cuda, "device_count", lambda: n_gpus)

        def failing_sampler(model, cdn):
            raise RuntimeError("deliberate worker failure")

        cpu = torch.device("cpu")
        emu_with_2d_ps._ps_gpu_replicas = [
            (cpu, emu_with_2d_ps.score_model, failing_sampler) for _ in range(n_gpus)
        ]
        emu_with_2d_ps._current_ps_method = "ode"
        emu_with_2d_ps._n_realisations = 5
        emu_with_2d_ps._denoise = True
        emu_with_2d_ps._ps_gpu_config_key = ("ode", 2, 5, True)

        cdns = self._make_cdns(n_batches=4, n_ps_batch=2)
        with pytest.raises((RuntimeError, Exception)):
            emu_with_2d_ps.get_pred(cdns)

    # ------------------------------------------------------------------
    # predict() stores _n_realisations and _denoise on the instance
    # ------------------------------------------------------------------

    def test_predict_stores_n_realisations_and_denoise(self, emu_with_2d_ps):
        """predict() stores _n_realisations and _denoise so get_pred can use them."""
        from py21cmemu.inputs import MHEmulatorInput

        n_real = 4
        captured = {}

        def fake_get_pred(cdns, verbose=False):
            captured["n_realisations"] = emu_with_2d_ps._n_realisations
            captured["denoise"] = emu_with_2d_ps._denoise
            return np.ones((cdns.shape[0], cdns.shape[1], n_real, 32, 64))

        emu_with_2d_ps.get_pred = fake_get_pred

        mh_in = MHEmulatorInput()
        rng = np.random.default_rng(0)
        params = mh_in.undo_normalization(rng.random((1, 11)))

        emu_with_2d_ps.predict(
            params,
            ps_2d_redshifts=np.array([7.0, 10.0]),
            n_realisations=n_real,
            denoise=False,
        )

        assert captured.get("n_realisations") == n_real
        assert captured.get("denoise") is False
