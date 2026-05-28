"""
Trainer — lives in rankbench root.

所有正则化均解耦，不加在 loss 里：
  l2          : W *= (1 - lr * lam)
  lrd_decoupled: W -= lr * lam * polar(W)
  lrd_elastic  : W -= lr * lam_lrd * polar(W)  +  W *= (1 - lr * lam_l2)
"""

import os
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import DEVICE
from models.newton_schulz_lrd import (apply_lrd_decoupled, apply_l2_decoupled,
                                       lrd_grad_stats)
from models.rank_monitor import mean_qk_stable_rank, mean_qk_weight_norm

# ── 优化器开关 ──────────────────────────────────────────────────────────────
USE_MUON = False   # True = Muon + AdamW，False = 纯 AdamW
# ───────────────────────────────────────────────────────────────────────────


_DEFAULTS = {
    "l2":            dict(lr=2e-2, lam=1e-3),
    "lrd_decoupled": dict(lr=2e-2, lam=1e-1),
    "lrd_elastic":   dict(lr=2e-2, lam_lrd=1e-2, lam_l2=1e-3),
}


def _make_optimizers(model, lr):
    if USE_MUON:
        muon_params = [p for name, p in model.named_parameters()
                       if p.ndim >= 2
                       and all(k not in name for k in ('embed', 'head', 'pos_emb', 'proj'))]
        adam_params = [p for name, p in model.named_parameters()
                       if p.ndim < 2
                       or any(k in name for k in ('embed', 'head', 'pos_emb', 'proj'))]
        muon = torch.optim.Muon(muon_params, lr=lr, momentum=0.95, nesterov=True)
        adam = torch.optim.AdamW(adam_params, lr=lr * 0.1, weight_decay=0.0)
        return muon, adam
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        return None, opt


def train(
    task,
    reg:           str          = "l2",
    lr:            float | None = None,
    lam:           float | None = None,
    lam_lrd:       float | None = None,
    lam_l2:        float | None = None,
    steps:         int          = 20_000,
    batch_size:    int | None   = None,
    record:        int          = 1,
    report:        int          = 200,
    verbose_grads: bool         = False,
) -> dict:

    defaults = _DEFAULTS.get(reg, dict(lr=2e-2, lam=1e-3))
    lr = lr if lr is not None else defaults["lr"]

    if reg == "lrd_elastic":
        lam_lrd = lam_lrd if lam_lrd is not None else defaults["lam_lrd"]
        lam_l2  = lam_l2  if lam_l2  is not None else defaults["lam_l2"]
        lam     = None
    else:
        lam = lam if lam is not None else defaults.get("lam", 1e-3)

    model        = task.make_model().to(DEVICE)
    muon, adam   = _make_optimizers(model, lr)
    warmup_steps = min(200, steps // 20)

    bs_str = "full" if batch_size is None else str(batch_size)
    if reg == "lrd_elastic":
        print(f"  lr={lr:.2e}  lam_lrd={lam_lrd:.2e}  lam_l2={lam_l2:.2e}"
              f"  batch={bs_str}  warmup={warmup_steps}  record={record}  device={DEVICE}")
    else:
        print(f"  lr={lr:.2e}  lam={lam:.2e}  batch={bs_str}"
              f"  warmup={warmup_steps}  record={record}  device={DEVICE}")

    log_steps, log_loss, log_train, log_test, log_rank, log_grad_norm, log_weight_norm = [], [], [], [], [], [], []

    for step in range(1, steps + 1):

        # warmup（只调 muon/opt 的 lr，adam 的 lr 保持不变）
        if USE_MUON and step <= warmup_steps:
            for g in muon.param_groups:
                g["lr"] = lr * step / warmup_steps
        elif USE_MUON:
            for g in muon.param_groups:
                g["lr"] = lr
        if not USE_MUON:
            for g in adam.param_groups:
                g["lr"] = (lr * step / warmup_steps) if step <= warmup_steps else lr

        if batch_size is None:
            x, y = task.full_batch()
        else:
            x, y = task.sample_batch(batch_size)
        x, y = x.to(DEVICE), y.to(DEVICE)

        # loss 只有 task loss，不加任何正则项
        loss = task.loss(model, (x, y))

        if USE_MUON:
            muon.zero_grad()
        adam.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if USE_MUON:
            muon.step()
        adam.step()

        # 解耦正则化，在 optimizer 之后单独施加
        current_lr = muon.param_groups[0]["lr"] if USE_MUON else adam.param_groups[0]["lr"]
        with torch.no_grad():
            if reg == "l2":
                apply_l2_decoupled(model, lr=current_lr, lam=lam)
            elif reg == "lrd_decoupled":
                apply_lrd_decoupled(model, lr=current_lr, lam=lam)
            elif reg == "lrd_elastic":
                apply_lrd_decoupled(model, lr=current_lr, lam=lam_lrd)
                apply_l2_decoupled(model,  lr=current_lr, lam=lam_l2)

        if step % record == 0:
            train_acc, test_acc = task.evaluate(model)
            rank        = mean_qk_stable_rank(model)
            weight_norm = mean_qk_weight_norm(model)
            # 计算 test loss
            with torch.no_grad():
                test_loss = task.loss(model, (task.x_test, task.y_test)).item()
            log_steps.append(step)
            log_loss.append(test_loss)   # 改为记录 test loss
            log_train.append(train_acc)
            log_test.append(test_acc)
            log_rank.append(rank)
            log_grad_norm.append(grad_norm.item())
            log_weight_norm.append(weight_norm)

            if step % report == 0:
                rank_warn = " ⚠" if rank < 0.5 else ""
                print(f"[{reg}] step {step:>6} | test_loss {test_loss:.4f} | "
                      f"train {train_acc:.3f} | test {test_acc:.3f} | "
                      f"rank {rank:.2f} | gnorm {grad_norm:.3f} | wnorm {weight_norm:.3f}{rank_warn}")
                if verbose_grads and reg in ("lrd_decoupled", "lrd_elastic"):
                    for name, mag in lrd_grad_stats(model).items():
                        print(f"    {name:40s}  polar={mag:.4f}")

            if test_acc >= 0.99:
                print(f"[{reg}] ★ 早停：test acc={test_acc:.4f} @ step {step}")
                break

    return dict(reg=reg, lr=lr, lam=lam, lam_lrd=lam_lrd, lam_l2=lam_l2,
                steps=log_steps, loss=log_loss,
                train_acc=log_train, test_acc=log_test,
                rank=log_rank, grad_norm=log_grad_norm,
                weight_norm=log_weight_norm)


def plot_comparison(results: list, task_name: str, save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"{task_name}_{timestamp}.png")

    fig  = plt.figure(figsize=(25, 4))
    gs   = gridspec.GridSpec(1, 5, figure=fig, wspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, r in enumerate(results):
        c = colors[i % len(colors)]
        if r["reg"] == "lrd_elastic":
            label = (f"lrd_elastic  lr={r['lr']:.2e}"
                     f" lrd={r['lam_lrd']:.2e} l2={r['lam_l2']:.2e}")
        else:
            label = f"{r['reg']}  lr={r['lr']:.2e} λ={r['lam']:.2e}"
        for key, ax in zip(["loss", "train_acc", "test_acc", "rank", "grad_norm"], axes):
            ax.plot(r["steps"], r[key], color=c, label=label, linewidth=1.0)

    for ax, title, ylabel in zip(
        axes,
        ["Loss", "Train Accuracy", "Test Accuracy", "QK Stable Rank", "Grad Norm"],
        ["loss", "accuracy", "accuracy", "stable rank", "grad norm"],
    ):
        ax.set_xlabel("step"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    axes[2].set_ylim(0, 1.05)
    fig.suptitle(f"Grokking — {task_name}", fontsize=13, y=1.02)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {save_path}")
