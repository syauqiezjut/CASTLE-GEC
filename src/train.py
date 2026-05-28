"""
train.py
CASTLE Training Script
========================
Training loop implementing:

  Loss (Eq. 9):  L = L_CE + λ_kg·L_KG + λ_reg·L_reg
  L_KG  (Eq. 10): -1/|S| Σ_{(v_i,v_j)∈S} w_ij · log p(v_j | v_i, x)
  L_reg (Eq. 11): ||λ_cat||² + ||γ||² + ||α||² + ||β||²

Hyperparameters (Table 8):
  Optimizer:     Adam (β1=0.9, β2=0.98)
  LR:            5e-4 with inverse square root scheduler
  Warmup:        4000 steps
  Label smooth:  0.1
  Batch size:    128, update_freq=2 (effective 256)
  Max epochs:    10, patience=5
  FP16:          True
  Hardware:      NVIDIA RTX 3090 (paper); any CUDA GPU works
"""

import os
import sys
import math
import logging
import argparse
import random
import time
from pathlib import Path
from typing import Dict, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from castle_model import CASTLE, build_castle_model
from dataset import load_iged, get_dataloaders, PAD_TOKEN
from knowledge_graph import CASTLEKnowledgeGraph

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Loss functions (Eq. 9-11)
# ──────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing (L_CE).
    Smoothing coefficient = 0.1 (Table 8).
    """
    def __init__(self, vocab_size: int, smoothing: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, T, V]
        targets: [B, T]
        """
        B, T, V = logits.shape
        logits = logits.reshape(-1, V)      # [B*T, V]
        targets = targets.reshape(-1)        # [B*T]

        # Ignore padding
        mask = targets != self.pad_idx
        logits = logits[mask]
        targets = targets[mask]

        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth distribution
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (V - 2))
            smooth_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
            smooth_dist[:, self.pad_idx] = 0.0

        loss = -(smooth_dist * log_probs).sum(dim=-1).mean()
        return loss


def compute_kg_loss(
    logits: torch.Tensor,      # [B, T, V]
    kg_edges: list,            # list of (src, dst, relation, weight) for batch
    tokenizer,
    pad_idx: int,
    subset_size: int = 32,
) -> torch.Tensor:
    """
    Knowledge consistency loss (Equation 10):
      L_KG = -1/|S| Σ_{(v_i,v_j)∈S} w_ij · log p(v_j | v_i, x)

    Operates on a randomly sampled subset S ⊂ E_rel with |S|=32
    (paper: "randomly sampled subset S ⊂ E_rel(x) with |S|=32").
    """
    if not kg_edges or len(kg_edges) == 0:
        return torch.tensor(0.0, device=logits.device)

    # Sample subset of edges
    if len(kg_edges) > subset_size:
        kg_edges = random.sample(kg_edges, subset_size)

    B, T, V = logits.shape

    # Eq. (10): L_KG gradients flow back to model (do NOT detach).
    # Vectorized gather (below) keeps backward fast — no need to detach.
    log_probs = F.log_softmax(logits.float(), dim=-1)  # [B, T, V], with grad

    # Vectorized: build token_id list and weight list for all sampled edges at once
    token_ids_list = []
    weights_list   = []
    for (src_word, dst_word, relation, weight) in kg_edges:
        try:
            enc = tokenizer.encode(dst_word)
            ids = [tid for tid in enc.ids if tid != pad_idx and tid < V]
        except Exception:
            ids = []
        for tid in ids:
            token_ids_list.append(tid)
            weights_list.append(weight)

    if not token_ids_list:
        return torch.tensor(0.0, device=logits.device)

    # Batch gather: log_probs[:, :, token_ids] → [B, T, N_edges]
    idx = torch.tensor(token_ids_list, dtype=torch.long, device=logits.device)
    w   = torch.tensor(weights_list,   dtype=torch.float32, device=logits.device)

    # [B, T, N] → mean over B and T → [N], then dot with weights
    gathered = log_probs[:, :, idx]              # [B, T, N]
    mean_lp  = gathered.mean(dim=(0, 1))         # [N]
    l_kg     = -(w * mean_lp).mean()

    return l_kg.to(logits.device)


def compute_reg_loss(model: CASTLE) -> torch.Tensor:
    """
    Regularization loss (Equation 11):
      L_reg = ||λ_cat||² + ||γ||² + ||α||² + ||β||²

    where α and β are vectors of linked attention gates for encoder/decoder.
    """
    reg = torch.tensor(0.0)
    device = next(model.parameters()).device

    # Collect linked attention gate params (α for encoder, β for decoder)
    for layer in model.encoder_layers:
        a = layer.castle_attn.linked_self_attn.gate
        reg = reg + a.to("cpu") ** 2

    for layer in model.decoder_layers:
        b_self = layer.self_attn.gate
        b_cross = layer.cross_attn.gate
        reg = reg + b_self.to("cpu") ** 2 + b_cross.to("cpu") ** 2

    # γ (residual weights)
    for layer in model.encoder_layers:
        g = layer.castle_attn.gamma
        reg = reg + g.to("cpu") ** 2
    for layer in model.decoder_layers:
        g = layer.gamma
        reg = reg + g.to("cpu") ** 2

    # λ_cat (category weights in KG gate)
    for layer in model.encoder_layers:
        w = layer.castle_attn.kg_gate.category_weights
        reg = reg + (w.to("cpu") ** 2).sum()

    return reg.to(device)


class CASTLELoss(nn.Module):
    """
    CASTLE training loss: standard cross-entropy + label smoothing.

    Note: Paper describes L_KG (Eq. 10) and L_reg (Eq. 11) as auxiliary losses,
    but the original training code used only L_CE (label_smoothed_cross_entropy).
    lambda_kg and lambda_reg are kept in the interface for reference but default to 0.
    """
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int = 0,
        label_smoothing: float = 0.1,
        lambda_kg: float = 0.0,   # NOT used in original training
        lambda_reg: float = 0.0,  # NOT used in original training
    ):
        super().__init__()
        self.ce_loss = LabelSmoothingCrossEntropy(vocab_size, label_smoothing, pad_idx)
        self.pad_idx = pad_idx

    def forward(
        self,
        outputs: Dict,
        targets: torch.Tensor,
        model: CASTLE,
        kg_edges: list = None,
        tokenizer=None,
    ) -> Dict[str, torch.Tensor]:
        # Use KG-biased logits (Eq. 8 applied during training too)
        logits = outputs.get("kg_logits", outputs["logits"])

        # Teacher forcing: predict tgt[1:] from tgt[:-1]
        tgt_label = targets[:, 1:]    # [B, T-1]
        logits_for_ce = logits[:, :tgt_label.size(1), :]
        l_ce = self.ce_loss(logits_for_ce, tgt_label)

        return {
            "loss": l_ce,
            "l_ce": l_ce.detach(),
            "l_kg": torch.tensor(0.0),
            "l_reg": torch.tensor(0.0),
        }


# ──────────────────────────────────────────────
# Learning rate scheduler (inverse square root)
# ──────────────────────────────────────────────

class InverseSqrtScheduler:
    """
    Inverse square root LR scheduler (Eq. from Fairseq):
      lr = lr_scale * min(step^{-0.5}, step * warmup_steps^{-1.5})
    """
    def __init__(self, optimizer, d_model: int, warmup_steps: int, base_lr: float = 5e-4):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.base_lr = base_lr
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _get_lr(self) -> float:
        step = max(1, self._step)
        # Linear warmup to base_lr, then inverse sqrt decay.
        # Peak at step == warmup_steps → exactly base_lr (5e-4 per Table 8).
        if step < self.warmup_steps:
            return self.base_lr * step / self.warmup_steps
        else:
            return self.base_lr * (self.warmup_steps / step) ** 0.5

    def state_dict(self):
        return {"_step": self._step}

    def load_state_dict(self, state):
        self._step = state["_step"]


# ──────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────

def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> float:
    """Token-level accuracy (ignoring pad tokens)."""
    preds = logits.argmax(dim=-1)   # [B, T]
    mask = targets != pad_idx
    correct = (preds == targets) & mask
    return correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0


def save_checkpoint(
    model: CASTLE,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    val_loss: float,
    val_f1: float,
    save_dir: str,
    is_best: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
        "val_f1": val_f1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    path = os.path.join(save_dir, f"checkpoint_epoch{epoch:02d}.pt")
    torch.save(ckpt, path)
    if is_best:
        best_path = os.path.join(save_dir, "checkpoint_best.pt")
        torch.save(ckpt, best_path)
        logger.info(f"  ✓ New best model saved: val_loss={val_loss:.4f}, F1={val_f1:.4f}")
    return path


def load_checkpoint(path: str, model: CASTLE, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    logger.info(f"Checkpoint loaded from {path} (epoch {ckpt['epoch']})")
    return ckpt


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: CASTLE,
    val_loader,
    loss_fn: CASTLELoss,
    tokenizer,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = total_ce = total_acc = 0.0
    n_batches = 0

    for i, batch in enumerate(val_loader):
        if max_batches and i >= max_batches:
            break

        src = batch["src_tokens"].to(device)
        tgt = batch["tgt_tokens"].to(device)
        mask = batch["src_mask"].to(device)

        outputs = model(src, tgt[:, :-1], mask)
        losses = loss_fn(outputs, tgt, model)

        acc = compute_accuracy(
            outputs["logits"][:, :tgt.size(1)-1],
            tgt[:, 1:],
            loss_fn.ce_loss.pad_idx,
        )

        total_loss += losses["loss"].item()
        total_ce += losses["l_ce"].item()
        total_acc += acc
        n_batches += 1

    model.train()
    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_ce": total_ce / max(n_batches, 1),
        "val_acc": total_acc / max(n_batches, 1),
    }


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────

def train(cfg: Dict):
    # ── Setup ──────────────────────────────────────────────────
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    save_dir = cfg.get("save_dir", "checkpoints/castle")
    log_dir = cfg.get("log_dir", "runs/castle")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # ── Dataset ────────────────────────────────────────────────
    logger.info("Loading IGED dataset...")
    train_ds, val_ds, test_ds, tokenizer = load_iged(
        hf_dataset_name=cfg["dataset"].get("name", "syauqie/IGED"),
        tokenizer_dir=cfg.get("tokenizer_dir", "data/tokenizer"),
        local_csv=cfg.get("local_csv", None),
        max_length=cfg["model"].get("max_len", 128),
    )
    pad_idx = tokenizer.token_to_id(PAD_TOKEN)
    vocab_size = tokenizer.get_vocab_size()

    t_cfg = cfg["training"]
    train_loader, val_loader, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=t_cfg.get("batch_size", 128),
        num_workers=4,
        pad_id=pad_idx,
    )

    # ── Knowledge Graph ────────────────────────────────────────
    kg = None
    kg_path = cfg.get("knowledge_graph", {}).get("output_path", "data/castle_kg.pkl")
    if os.path.exists(kg_path):
        logger.info(f"Loading pre-built KG from {kg_path}")
        kg = CASTLEKnowledgeGraph.load(kg_path)
    else:
        logger.warning(
            f"KG not found at {kg_path}. "
            "Run `python scripts/build_kg.py` first for full CASTLE performance. "
            "Continuing without KG (KG integration disabled)."
        )

    # ── Model ─────────────────────────────────────────────────
    model = build_castle_model(cfg, vocab_size=vocab_size, pad_idx=pad_idx)
    if kg:
        model.set_knowledge_graph(kg, kg.word_to_idx())
    model = model.to(device)

    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Precompute token ID cache untuk KG nodes — eliminasi encode() dari inner loop
    if kg:
        model.build_token_id_cache(tokenizer, vocab_size=vocab_size, pad_idx=pad_idx)

    # ── Optimizer and Scheduler ────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=t_cfg.get("lr", 5e-4),
        betas=(t_cfg.get("beta1", 0.9), t_cfg.get("beta2", 0.98)),
        eps=1e-8,
    )
    scheduler = InverseSqrtScheduler(
        optimizer,
        d_model=cfg["model"].get("encoder_embed_dim", 256),
        warmup_steps=t_cfg.get("warmup_updates", 4000),
        base_lr=t_cfg.get("lr", 5e-4),
    )

    # ── Resume checkpoint ──────────────────────────────────────
    resume = cfg.get("resume", None)
    start_epoch = 0
    global_step = 0
    if resume and os.path.exists(resume):
        ckpt = load_checkpoint(resume, model, optimizer, scheduler)
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("step", 0)

    # ── Loss ───────────────────────────────────────────────────
    loss_fn = CASTLELoss(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        label_smoothing=t_cfg.get("label_smoothing", 0.1),
        lambda_kg=0.0,   # not used in original training
        lambda_reg=0.0,  # not used in original training
    )

    # ── FP16 ───────────────────────────────────────────────────
    use_fp16 = t_cfg.get("fp16", True) and torch.cuda.is_available()
    scaler = GradScaler("cuda") if use_fp16 else None
    update_freq = t_cfg.get("update_freq", 2)
    max_epochs = t_cfg.get("max_epochs", 10)
    patience = t_cfg.get("patience", 5)

    logger.info("=" * 60)
    logger.info("Starting CASTLE training")
    logger.info(f"  Epochs: {max_epochs}, Patience: {patience}")
    logger.info(f"  FP16: {use_fp16}, Update freq: {update_freq}")
    logger.info(f"  KG enabled: {kg is not None}")
    logger.info("=" * 60)

    best_val_loss = float("inf")
    patience_count = 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_loss = epoch_ce = epoch_kg = 0.0
        n_batches = 0
        t0 = time.time()

        optimizer.zero_grad()

        n_train_batches = len(train_loader)
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{max_epochs}",
            unit="batch",
            dynamic_ncols=True,
        )

        for batch_idx, batch in enumerate(pbar):
            _profile = (batch_idx < 3)  # print timing for first 3 batches only
            _t0 = time.time()

            src = batch["src_tokens"].to(device)
            tgt = batch["tgt_tokens"].to(device)
            mask = batch["src_mask"].to(device)
            _t1 = time.time()

            # KG edges — fast O(1) lookup via precomputed cache
            # Compute per-sample edge lists once; reuse for both model bias and L_KG loss
            src_texts = batch["src_texts"]
            kg_edges_per_sample = []  # List[List[edge]] — one list per batch item
            kg_edges = []             # flat list for L_KG loss computation
            if kg:
                for txt in src_texts:
                    sample_edges = kg.get_edge_set_for_sequence(txt.lower().split())
                    kg_edges_per_sample.append(sample_edges)
                    kg_edges.extend(sample_edges)
            else:
                kg_edges_per_sample = None
            _t2 = time.time()

            # Forward pass dengan KG bias (Eq. 8) sesuai paper
            # kg_edges_per_sample avoids recomputing edges inside model._apply_kg_bias
            with autocast("cuda", enabled=use_fp16):
                outputs = model(src, tgt[:, :-1], mask,
                                kg_edges_per_sample=kg_edges_per_sample)
                losses = loss_fn(outputs, tgt, model, kg_edges, tokenizer)
                loss = losses["loss"] / update_freq
            _t3 = time.time()

            # Backward
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % update_freq == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                # Fairseq default clip_norm=0.1
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                lr = scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging to TensorBoard
                if global_step % 100 == 0:
                    writer.add_scalar("train/loss", losses["loss"].item(), global_step)
                    writer.add_scalar("train/l_ce", losses["l_ce"].item(), global_step)
                    writer.add_scalar("train/l_kg", losses["l_kg"].item(), global_step)
                    writer.add_scalar("train/lr", lr, global_step)

            _t4 = time.time()
            if _profile:
                logger.info(
                    f"  [TIMING batch {batch_idx}] "
                    f"data={_t1-_t0:.3f}s  "
                    f"kg_lookup={_t2-_t1:.3f}s  "
                    f"fwd+loss={_t3-_t2:.3f}s  "
                    f"bwd={_t4-_t3:.3f}s  "
                    f"total={_t4-_t0:.3f}s  "
                    f"n_kg_edges={len(kg_edges)}"
                )

            epoch_loss += losses["loss"].item()
            epoch_ce += losses["l_ce"].item()
            epoch_kg += losses["l_kg"].item()
            n_batches += 1

            # Update tqdm postfix setiap batch
            pbar.set_postfix({
                "loss": f"{losses['loss'].item():.4f}",
                "ce": f"{losses['l_ce'].item():.4f}",
                "kg": f"{losses['l_kg'].item():.4f}",
                "step": global_step,
            })

            # Log ke file setiap 50 batch
            if n_batches % 50 == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"  Epoch {epoch+1} | batch {n_batches}/{n_train_batches} | "
                    f"step {global_step} | "
                    f"loss={epoch_loss/n_batches:.4f} | "
                    f"ce={epoch_ce/n_batches:.4f} | "
                    f"kg={epoch_kg/n_batches:.4f} | "
                    f"elapsed={elapsed:.0f}s"
                )

        pbar.close()

        # ── Validation ──────────────────────────────────────────
        val_metrics = validate(model, val_loader, loss_fn, tokenizer, device, max_batches=200)
        val_loss = val_metrics["val_loss"]

        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_metrics["val_acc"], epoch)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_count = 0
        else:
            patience_count += 1

        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler,
            epoch=epoch, step=global_step,
            val_loss=val_loss, val_f1=val_metrics["val_acc"],
            save_dir=save_dir, is_best=is_best,
        )

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch+1}/{max_epochs} | "
            f"train_loss={epoch_loss/n_batches:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['val_acc']:.4f} | "
            f"patience={patience_count}/{patience} | "
            f"time={elapsed:.0f}s"
        )

        # Early stopping
        if patience_count >= patience:
            logger.info(f"Early stopping after {patience} epochs without improvement.")
            break

    writer.close()
    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")
    logger.info(f"Best model saved to {save_dir}/checkpoint_best.pt")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("train.log"),
        ],
    )

    parser = argparse.ArgumentParser(description="Train CASTLE model")
    parser.add_argument("--config", type=str, default="configs/castle_base.yaml")
    parser.add_argument("--local_csv", type=str, default=None,
                        help="Path to IGED_stratified_dataset.csv (overrides HF dataset)")
    parser.add_argument("--save_dir", type=str, default="checkpoints/castle")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_kg", action="store_true", help="Disable KG integration")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.local_csv:
        cfg["local_csv"] = args.local_csv
    if args.save_dir:
        cfg["save_dir"] = args.save_dir
    if args.resume:
        cfg["resume"] = args.resume
    if args.no_kg:
        cfg["model"]["kg_enabled"] = False
    if args.epochs:
        cfg["training"]["max_epochs"] = args.epochs

    train(cfg)
