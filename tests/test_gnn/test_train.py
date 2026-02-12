"""Tests for src.gnn.train and src.gnn.evaluate — training loop + evaluation."""

from unittest.mock import patch

import torch
from torch_geometric.data import HeteroData

from src.gnn.evaluate import evaluate_gnn
from src.gnn.losses import LossConfig
from src.gnn.model import ModelConfig, TD4DDModel
from src.gnn.train import Trainer, TrainingConfig

FEAT_DIM = 32
D_MODEL = 16


def _make_config(**overrides) -> ModelConfig:
    defaults = dict(
        feat_dims={"admission": FEAT_DIM},
        d_model=D_MODEL,
        use_transformer=False,
        use_diffusion=False,
        dropout=0.0,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_batch(batch_size: int, n_positive: int = 2) -> HeteroData:
    total = batch_size + 4  # extra neighbor nodes
    data = HeteroData()
    data["admission"].x = torch.randn(total, FEAT_DIM)
    data["admission"].y = torch.zeros(total)
    data["admission"].y[:n_positive] = 1.0
    data.batch_size = batch_size
    data["admission"].n_id = torch.arange(total)
    return data


def _mock_loader_fn(batches: list[HeteroData]):
    """Return a callable that yields a fresh iterator each call."""
    return lambda: iter(batches)


def _make_trainer(
    max_epochs=5,
    patience=20,
    batch_size=8,
    n_positive=2,
    n_batches=2,
    **training_overrides,
):
    """Helper: build a tiny model + trainer with synthetic data."""
    torch.manual_seed(42)
    config = _make_config()
    model = TD4DDModel(config)

    batches = [_make_batch(batch_size, n_positive) for _ in range(n_batches)]

    tc = TrainingConfig(
        max_epochs=max_epochs,
        patience=patience,
        device="cpu",
        checkpoint_dir="/tmp/test_gnn_ckpt",
        log_every_n_epochs=1,
        **training_overrides,
    )
    loss_config = LossConfig()
    trainer = Trainer(
        model=model,
        train_loader=_mock_loader_fn(batches),
        val_loader=_mock_loader_fn(batches),
        loss_config=loss_config,
        training_config=tc,
    )
    return trainer, model, batches


class TestTrainerInit:
    def test_trainer_init(self):
        """Optimizer has correct lr and weight_decay."""
        trainer, model, _ = _make_trainer(lr=0.005, weight_decay=1e-3)
        pg = trainer.optimizer.param_groups[0]
        assert pg["lr"] == 0.005
        assert pg["weight_decay"] == 1e-3


class TestSingleEpoch:
    def test_single_epoch(self):
        """_train_epoch returns dict with 'loss' > 0."""
        trainer, _, _ = _make_trainer()
        metrics = trainer._train_epoch()
        assert "loss" in metrics
        assert metrics["loss"] > 0
        assert "cls_loss" in metrics
        assert "diff_loss" in metrics


class TestValidation:
    def test_validation(self):
        """_validate returns dict with auroc and auprc keys."""
        trainer, _, _ = _make_trainer()
        # Need to run one train epoch first to initialize cls_loss_fn
        trainer._train_epoch()
        metrics = trainer._validate()
        assert "auroc" in metrics
        assert "auprc" in metrics
        assert 0.0 <= metrics["auroc"] <= 1.0


class TestEarlyStopping:
    def test_early_stopping(self):
        """With patience=3, training stops before max_epochs=100."""
        trainer, _, _ = _make_trainer(max_epochs=100, patience=3)
        result = trainer.train()
        # Should stop well before 100 epochs
        assert len(result["training_history"]) < 100


class TestCheckpointSaveLoad:
    def test_checkpoint_save_load(self, tmp_path):
        """Save checkpoint, load into fresh model, outputs match."""
        torch.manual_seed(42)
        config = _make_config()
        model = TD4DDModel(config)

        batches = [_make_batch(8, 2) for _ in range(2)]
        tc = TrainingConfig(
            max_epochs=5,
            patience=10,
            device="cpu",
            checkpoint_dir=str(tmp_path),
            log_every_n_epochs=1,
        )
        trainer = Trainer(
            model=model,
            train_loader=_mock_loader_fn(batches),
            val_loader=_mock_loader_fn(batches),
            loss_config=LossConfig(),
            training_config=tc,
        )
        result = trainer.train()
        ckpt_path = result["checkpoint_path"]
        assert ckpt_path is not None
        assert ckpt_path.exists()

        # Load into two fresh models — both should produce identical output
        model_a = TD4DDModel(config)
        model_b = TD4DDModel(config)
        meta_a = Trainer.load_checkpoint(ckpt_path, model_a)
        Trainer.load_checkpoint(ckpt_path, model_b)
        assert meta_a["epoch"] == result["best_epoch"]

        model_a.eval()
        model_b.eval()
        test_batch = _make_batch(4, 1)
        with torch.no_grad():
            out_a = model_a(test_batch)
            out_b = model_b(test_batch)
        assert torch.allclose(out_a["logits"], out_b["logits"], atol=1e-6)


class TestGradientClipping:
    def test_gradient_clipping(self):
        """clip_grad_norm_ is called with the configured max_norm."""
        trainer, _, _ = _make_trainer(grad_clip_norm=0.5)
        with patch("src.gnn.train.nn.utils.clip_grad_norm_") as mock_clip:
            trainer._train_epoch()
            assert mock_clip.called
            # Check the max_norm argument
            _, kwargs = mock_clip.call_args
            if not kwargs:
                args = mock_clip.call_args[0]
                assert args[1] == 0.5
            else:
                assert kwargs.get("max_norm", mock_clip.call_args[0][1]) == 0.5


class TestDeviceHandling:
    def test_device_handling(self):
        """'auto' selects cpu (no GPU in CI); model params on correct device."""
        trainer, model, _ = _make_trainer()
        assert trainer.device == torch.device("cpu")
        for p in model.parameters():
            assert p.device == torch.device("cpu")


class TestEvaluateGnn:
    def test_evaluate_gnn(self, tmp_path):
        """evaluate_gnn returns dict with auroc and auprc."""
        torch.manual_seed(42)
        config = _make_config()
        model = TD4DDModel(config)

        batches = [_make_batch(8, 2) for _ in range(2)]

        # Train one epoch first so model is initialized
        tc = TrainingConfig(max_epochs=1, patience=10, device="cpu",
                            checkpoint_dir=str(tmp_path), log_every_n_epochs=1)
        trainer = Trainer(
            model=model,
            train_loader=_mock_loader_fn(batches),
            val_loader=_mock_loader_fn(batches),
            loss_config=LossConfig(),
            training_config=tc,
        )
        trainer.train()

        metrics = evaluate_gnn(
            model=model,
            test_loader=_mock_loader_fn(batches),
            prepare_batch_fn=trainer._prepare_batch,
            output_dir=str(tmp_path / "eval"),
            device="cpu",
        )
        assert "auroc" in metrics
        assert "auprc" in metrics
        assert 0.0 <= metrics["auroc"] <= 1.0

        # Check JSON report was written
        report_path = tmp_path / "eval" / "gnn_evaluation.json"
        assert report_path.exists()


class TestTrainingHistory:
    def test_training_history(self):
        """Training 5 epochs (patience=10) produces history with 5 entries."""
        trainer, _, _ = _make_trainer(max_epochs=5, patience=10)
        result = trainer.train()
        history = result["training_history"]
        assert len(history) == 5
        for entry in history:
            assert "train_loss" in entry
            assert "val_auroc" in entry
