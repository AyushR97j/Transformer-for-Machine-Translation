"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional

import wandb
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing with ε = smoothing.
    y_smooth = (1-ε)*one_hot(y) + ε/(vocab_size-1)
    PAD positions receive 0 probability.
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits : [batch * tgt_len, vocab_size]
        target : [batch * tgt_len]
        """
        log_probs = F.log_softmax(logits, dim=-1)   # [N, V]

        # Build smoothed target distribution
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_dist[:, self.pad_idx] = 0.0           # zero out PAD
            # Mask PAD positions entirely
            pad_mask = target.eq(self.pad_idx)
            smooth_dist[pad_mask] = 0.0

        loss = -(smooth_dist * log_probs).sum(dim=-1)    # [N]
        # Average only over non-PAD tokens
        non_pad = (~pad_mask).sum().clamp(min=1)
        return loss.sum() / non_pad


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    log_wandb: bool = True,
    grad_norm_log: bool = False,      # for ablation experiment (Section 2.2)
) -> float:
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_tokens = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    phase = "train" if is_train else "val"

    with context:
        for batch_idx, (src, tgt) in enumerate(tqdm(data_iter, desc=f"Epoch {epoch_num} [{phase}]")):
            src = src.to(device)
            tgt = tgt.to(device)

            # Decoder input: all tokens except last (<eos>)
            tgt_in  = tgt[:, :-1]
            # Decoder target: all tokens except first (<sos>)
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx=model.pad_idx).to(device)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx=model.pad_idx).to(device)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            # logits: [batch, tgt_len, vocab_size]

            # Flatten for loss
            logits_flat = logits.contiguous().view(-1, logits.size(-1))
            tgt_flat    = tgt_out.contiguous().view(-1)

            loss = loss_fn(logits_flat, tgt_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (helps training stability)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Log gradient norms for ablation (Section 2.2)
                if grad_norm_log and log_wandb:
                    for name, param in model.named_parameters():
                        if param.grad is not None and ('W_q' in name or 'W_k' in name):
                            wandb.log({f"grad_norm/{name}": param.grad.norm().item()})

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                if log_wandb and batch_idx % 50 == 0:
                    wandb.log({
                        f"{phase}/loss": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch_num,
                    })

            non_pad = tgt_flat.ne(model.pad_idx).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

    avg_loss = total_loss / max(total_tokens, 1)
    if log_wandb:
        wandb.log({f"{phase}/epoch_loss": avg_loss, "epoch": epoch_num})
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Auto-regressive greedy decoding.
    Returns [1, out_len] token-index tensor (includes start_symbol).
    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)                     # [1, src_len, d_model]
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)  # [1, 1]

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=model.pad_idx).to(device)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)  # [1, cur_len, V]
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
            ys = torch.cat([ys, next_tok], dim=1)

            if next_tok.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab: dict,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """Corpus-level BLEU (0–100)."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    itos = {v: k for k, v in tgt_vocab.items()} if isinstance(tgt_vocab, dict) else None
    def idx_to_token(idx):
        if itos:
            return itos.get(idx, '<unk>')
        return tgt_vocab.lookup_token(idx)

    sos_idx = tgt_vocab['<sos>'] if isinstance(tgt_vocab, dict) else tgt_vocab['<sos>']
    eos_idx = tgt_vocab['<eos>'] if isinstance(tgt_vocab, dict) else tgt_vocab['<eos>']
    pad_idx = model.pad_idx

    hypotheses = []
    references = []

    model.eval()
    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU evaluation"):
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                src_i = src[i].unsqueeze(0)                         # [1, src_len]
                src_mask = make_src_mask(src_i, pad_idx=pad_idx).to(device)
                out = greedy_decode(model, src_i, src_mask, max_len,
                                    start_symbol=sos_idx,
                                    end_symbol=eos_idx,
                                    device=device)
                pred_tokens = [
                    idx_to_token(t.item()) for t in out[0]
                    if idx_to_token(t.item()) not in ('<sos>', '<eos>', '<pad>')
                ]
                ref_tokens = [
                    idx_to_token(t.item()) for t in tgt[i]
                    if idx_to_token(t.item()) not in ('<sos>', '<eos>', '<pad>')
                ]
                hypotheses.append(pred_tokens)
                references.append([ref_tokens])

    sf = SmoothingFunction().method1
    score = corpus_bleu(references, hypotheses, smoothing_function=sf) * 100.0
    return score


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config': {
            'src_vocab_size': model.src_vocab_size,
            'tgt_vocab_size': model.tgt_vocab_size,
            'd_model':        model.d_model,
            'N':              model.N,
            'num_heads':      model.num_heads,
            'd_ff':           model.d_ff,
            'dropout':        model.dropout_p,
            'pad_idx':        model.pad_idx,
        },
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt.get('epoch', 0)


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment(config=None, run_name="baseline") -> None:
    """Full training experiment with W&B logging."""
    from dataset import Multi30kDataset
    from lr_scheduler import NoamScheduler

    # ── Default hyper-parameters ──────────────────────────────────────
    default_cfg = dict(
        d_model      = 256,
        N            = 3,
        num_heads    = 8,
        d_ff         = 512,
        dropout      = 0.1,
        batch_size   = 128,
        num_epochs   = 15,
        warmup_steps = 4000,
        smoothing    = 0.1,
        min_freq     = 2,
    )
    if config is not None:
        default_cfg.update(config)
    cfg = default_cfg

    # ── W&B init ──────────────────────────────────────────────────────
    wandb.init(
        project="da6401-a3",
        name=run_name,
        config=cfg,
    )
    cfg = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────
    ds = Multi30kDataset()
    ds.build_vocab(min_freq=cfg.min_freq)
    ds.process_data()

    train_loader = ds.get_dataloader('train',      batch_size=cfg.batch_size, shuffle=True)
    val_loader   = ds.get_dataloader('validation', batch_size=cfg.batch_size, shuffle=False)
    test_loader  = ds.get_dataloader('test',       batch_size=cfg.batch_size, shuffle=False)

    src_vocab_size = len(ds.src_vocab)
    tgt_vocab_size = len(ds.tgt_vocab)

    # ── Model ─────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size = src_vocab_size,
        tgt_vocab_size = tgt_vocab_size,
        d_model        = cfg.d_model,
        N              = cfg.N,
        num_heads      = cfg.num_heads,
        d_ff           = cfg.d_ff,
        dropout        = cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    wandb.log({"model/n_params": n_params})

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)

    # ── Loss ──────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size = tgt_vocab_size,
        pad_idx    = 1,
        smoothing  = cfg.smoothing,
    )

    best_val_loss = float('inf')
    best_ckpt     = "best_checkpoint.pt"

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device
        )
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        save_checkpoint(model, optimizer, scheduler, epoch,
                        path=f"checkpoint_epoch{epoch}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path=best_ckpt)
            print(f"  ✓ New best model saved (val_loss={val_loss:.4f})")

    # ── Final BLEU ────────────────────────────────────────────────────
    print("Computing test BLEU …")
    epoch = load_checkpoint(best_ckpt, model)
    model.to(device)
    bleu = evaluate_bleu(model, test_loader, ds.tgt_vocab, device=device)
    print(f"Test BLEU: {bleu:.2f}")
    wandb.log({"test_bleu": bleu})
    wandb.finish()
    return bleu, ds, model


# ══════════════════════════════════════════════════════════════════════
#  ABLATION HELPERS  (called from wandb_experiments.py)
# ══════════════════════════════════════════════════════════════════════

def run_fixed_lr_experiment(lr=1e-4):
    """Section 2.1: fixed LR baseline (no Noam scheduler)."""
    from dataset import Multi30kDataset
    from lr_scheduler import NoamScheduler

    cfg = dict(
        d_model=256, N=3, num_heads=8, d_ff=512,
        dropout=0.1, batch_size=128, num_epochs=15,
        warmup_steps=4000, smoothing=0.1, min_freq=2,
    )
    wandb.init(project="da6401-a3", name="fixed_lr_1e-4", config=cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Multi30kDataset()
    ds.build_vocab(min_freq=2)
    ds.process_data()

    train_loader = ds.get_dataloader('train',      batch_size=128, shuffle=True)
    val_loader   = ds.get_dataloader('validation', batch_size=128, shuffle=False)

    model = Transformer(len(ds.src_vocab), len(ds.tgt_vocab),
                        d_model=256, N=3, num_heads=8, d_ff=512, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = LabelSmoothingLoss(len(ds.tgt_vocab), pad_idx=1, smoothing=0.1)

    for epoch in range(15):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, None,
                               epoch_num=epoch, is_train=True, device=device)
        val_loss   = run_epoch(val_loader,   model, loss_fn, None, None,
                               epoch_num=epoch, is_train=False, device=device)
        print(f"[fixed-lr] Epoch {epoch} | train={train_loss:.4f} | val={val_loss:.4f}")
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()