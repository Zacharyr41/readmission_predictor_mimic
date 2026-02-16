"""Training loop with early stopping, gradient clipping, and checkpointing."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch_geometric.data import HeteroData

from src.gnn.hop_extraction import HopExtractor, HopIndices
from src.gnn.losses import LossConfig, build_classification_loss, compute_total_loss

logger = logging.getLogger(__name__)


def _get_batch_size(batch: HeteroData) -> int:
    """Extract batch_size from a HeteroData batch.

    NeighborLoader stores batch_size in the input node store, not the global
    store.  Fall back through several access patterns.
    """
    # Global store (manually constructed batches)
    try:
        return batch.batch_size
    except AttributeError:
        pass
    # Node-type store (NeighborLoader output)
    for ntype in batch.node_types:
        store = batch[ntype]
        if hasattr(store, "batch_size"):
            return store.batch_size
    raise AttributeError("batch has no batch_size attribute")


def _move_hop_indices(hop: HopIndices, device: str | torch.device) -> HopIndices:
    """Move all tensors in a HopIndices to the target device."""
    hop.contextual_indices = [
        [t.to(device) for t in path] for path in hop.contextual_indices
    ]
    hop.contextual_masks = [
        [t.to(device) for t in path] for path in hop.contextual_masks
    ]
    if hop.temporal_indices is not None:
        hop.temporal_indices = [t.to(device) for t in hop.temporal_indices]
    if hop.temporal_masks is not None:
        hop.temporal_masks = [t.to(device) for t in hop.temporal_masks]
    if hop.temporal_deltas is not None:
        hop.temporal_deltas = [t.to(device) for t in hop.temporal_deltas]
    return hop


@dataclass
class TrainingConfig:
    """Configuration for the GNN training loop."""

    lr: float = 0.001
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 20
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    device: str = "auto"
    checkpoint_dir: Path = Path("outputs/gnn_models/")
    log_every_n_epochs: int = 5


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class Trainer:
    """Trains a TD4DD model with early stopping on validation AUROC.

    Parameters
    ----------
    model : nn.Module
        The TD4DD model instance.
    train_loader : Callable[[], Iterator[HeteroData]]
        Factory that returns a fresh training batch iterator each call.
    val_loader : Callable[[], Iterator[HeteroData]]
        Factory that returns a fresh validation batch iterator each call.
    loss_config : LossConfig
        Loss function configuration.
    training_config : TrainingConfig
        Training hyperparameters.
    aux_edge_indices : list[torch.Tensor] or None
        Global auxiliary edge indices for the diffusion branch.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: Callable[[], Iterator[HeteroData]],
        val_loader: Callable[[], Iterator[HeteroData]],
        loss_config: LossConfig,
        training_config: TrainingConfig | None = None,
        aux_edge_indices: list[torch.Tensor] | None = None,
        hop_extractor: HopExtractor | None = None,
    ) -> None:
        self.config = training_config or TrainingConfig()
        self.device = _resolve_device(self.config.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_config = loss_config
        self.aux_edge_indices = aux_edge_indices
        self.hop_extractor = hop_extractor

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self._cls_loss_fn: Callable | None = None
        self._best_val_auroc = 0.0
        self._best_epoch = 0
        self._history: list[dict] = []

    def train(self) -> dict:
        """Run the full training loop with early stopping.

        Returns
        -------
        dict
            Keys: best_val_auroc, best_epoch, training_history, checkpoint_path
        """
        patience_counter = 0
        checkpoint_path = None

        for epoch in range(1, self.config.max_epochs + 1):
            train_metrics = self._train_epoch()
            val_metrics = self._validate()

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_cls_loss": train_metrics["cls_loss"],
                "train_diff_loss": train_metrics["diff_loss"],
                "val_auroc": val_metrics["auroc"],
                "val_auprc": val_metrics["auprc"],
                "val_loss": val_metrics["loss"],
            }
            self._history.append(epoch_record)

            if epoch % self.config.log_every_n_epochs == 0 or epoch == 1:
                logger.info(
                    "Epoch %d/%d — train_loss=%.4f  val_auroc=%.4f  val_loss=%.4f",
                    epoch,
                    self.config.max_epochs,
                    train_metrics["loss"],
                    val_metrics["auroc"],
                    val_metrics["loss"],
                )

            if val_metrics["auroc"] > self._best_val_auroc:
                self._best_val_auroc = val_metrics["auroc"]
                self._best_epoch = epoch
                patience_counter = 0
                checkpoint_path = self._save_checkpoint(epoch, val_metrics)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d). "
                        "Best AUROC=%.4f at epoch %d.",
                        epoch,
                        self.config.patience,
                        self._best_val_auroc,
                        self._best_epoch,
                    )
                    break

        return {
            "best_val_auroc": self._best_val_auroc,
            "best_epoch": self._best_epoch,
            "training_history": self._history,
            "checkpoint_path": checkpoint_path,
        }

    def _train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_diff = 0.0
        n_batches = 0

        for batch in self.train_loader():
            kwargs = self._prepare_batch(batch)
            out = self.model(**kwargs)

            labels = batch["admission"].y[:_get_batch_size(batch)].to(self.device)

            # Lazy-init classification loss on first batch
            if self._cls_loss_fn is None:
                self._cls_loss_fn = build_classification_loss(
                    self.loss_config, labels
                )

            losses = compute_total_loss(
                self._cls_loss_fn,
                out["logits"].squeeze(-1),
                labels,
                out["L_diff"],
                self.loss_config,
            )

            self.optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )
            self.optimizer.step()

            total_loss += losses["total"].item()
            total_cls += losses["cls"].item()
            total_diff += losses["diff"].item()
            n_batches += 1

        n = max(n_batches, 1)
        return {
            "loss": total_loss / n,
            "cls_loss": total_cls / n,
            "diff_loss": total_diff / n,
        }

    @torch.no_grad()
    def _validate(self) -> dict:
        self.model.eval()
        all_probs: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader():
            kwargs = self._prepare_batch(batch)
            out = self.model(**kwargs)

            labels = batch["admission"].y[:_get_batch_size(batch)].to(self.device)
            probs = out["probabilities"].squeeze(-1)

            if self._cls_loss_fn is not None:
                losses = compute_total_loss(
                    self._cls_loss_fn,
                    out["logits"].squeeze(-1),
                    labels,
                    out["L_diff"],
                    self.loss_config,
                )
                total_loss += losses["total"].item()

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            n_batches += 1

        y_proba = torch.cat(all_probs).numpy()
        y_true = torch.cat(all_labels).numpy()

        auroc = roc_auc_score(y_true, y_proba)
        from sklearn.metrics import average_precision_score

        auprc = average_precision_score(y_true, y_proba)

        return {
            "auroc": float(auroc),
            "auprc": float(auprc),
            "loss": total_loss / max(n_batches, 1),
        }

    def _prepare_batch(self, batch: HeteroData) -> dict:
        """Convert a HeteroData batch into model.forward() keyword arguments."""
        batch = batch.to(self.device)
        batch_size = _get_batch_size(batch)

        # Remap global aux edges to batch-local indices
        local_aux = None
        if self.aux_edge_indices is not None and hasattr(batch["admission"], "n_id"):
            n_id = batch["admission"].n_id
            global_to_local = {}
            for i in range(batch_size):
                global_to_local[n_id[i].item()] = i

            local_aux = []
            for edge_index in self.aux_edge_indices:
                src = edge_index[0].tolist()
                dst = edge_index[1].tolist()
                new_src, new_dst = [], []
                for s, d in zip(src, dst):
                    if s in global_to_local and d in global_to_local:
                        new_src.append(global_to_local[s])
                        new_dst.append(global_to_local[d])
                if new_src:
                    local_edge = torch.tensor(
                        [new_src, new_dst], dtype=torch.long, device=self.device
                    )
                else:
                    local_edge = torch.zeros(
                        (2, 0), dtype=torch.long, device=self.device
                    )
                local_aux.append(local_edge)

        hop_indices = None
        if self.hop_extractor is not None:
            hop_indices = self.hop_extractor.extract(batch, batch_size)
            hop_indices = _move_hop_indices(hop_indices, self.device)

        return {
            "batch_data": batch,
            "aux_edge_indices": local_aux,
            "hop_indices": hop_indices,
        }

    def _save_checkpoint(self, epoch: int, metrics: dict) -> Path:
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"best_model_epoch{epoch}.pt"

        model_config = None
        if hasattr(self.model, "config"):
            model_config = asdict(self.model.config)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_auroc": self._best_val_auroc,
                "history": self._history,
                "model_config": model_config,
            },
            path,
        )
        logger.info("Checkpoint saved → %s (AUROC=%.4f)", path, metrics["auroc"])
        return path

    @staticmethod
    def load_checkpoint(path: Path | str, model: nn.Module) -> dict:
        """Load model weights from a checkpoint file.

        Parameters
        ----------
        path : Path or str
            Path to the checkpoint ``.pt`` file.
        model : nn.Module
            Model instance to load weights into.

        Returns
        -------
        dict
            Checkpoint metadata (epoch, best_val_auroc, history, model_config).
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        return {
            "epoch": ckpt["epoch"],
            "best_val_auroc": ckpt["best_val_auroc"],
            "history": ckpt.get("history", []),
            "model_config": ckpt.get("model_config"),
        }
