"""
quick_rank_check.py — 单元测试，循环跑所有任务，三种正则化对比
正则化全部解耦，不加在 loss 里。
"""

import os
import csv
import json
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from tasks.modular_addition       import ModularAddition
from tasks.modular_multiplication import ModularMultiplication
from tasks.modular_log            import ModularLog
from tasks.group_composition      import GroupComposition
from tasks.parity                 import Parity
from tasks.modular_polynomial     import ModularPolynomial

from models.transformer       import TinyTransformer, LinearProjectTransformer
from models.rank_monitor      import mean_qk_stable_rank, mean_qk_weight_norm
from models.newton_schulz_lrd import apply_lrd_decoupled, apply_l2_decoupled
from models.muon              import get_muon_cls
from utils import DEVICE

# ── 优化器开关 ──────────────────────────────────────────────────────────────
USE_MUON = False   # True = Muon + AdamW，False = 纯 AdamW
# ───────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TASK       = ["ModularMultiplication","ModularLog","GroupComposition","Parity","ModularPolynomial"]
DIMS       = [128]
TRAIN_SIZE = None
TRAIN_FRAC = 0.4
TEST_SIZE  = 1000
STEPS      = 6000
BATCH_SIZE = 1024
LABEL_SMOOTHING = 0.0
RECORD     = 10
REPORT     = 500

REGS = ["l2", "lrd_decoupled", "lrd_elastic"]

DEFAULTS = {
    "l2":            dict(lr=2e-2, lam=1e-3),
    "lrd_decoupled": dict(lr=2e-2, lam=1e-1),
    "lrd_elastic":   dict(lr=2e-2, lam_lrd=1e-2, lam_l2=1e-3),
}

SAVE_DIR = "results/quick"
SEED     = 0

ADAPTIVE_LRD      = True
LRD_REF_TEST_ACC  = 0.7
RANK_EMA_BETA     = 0.95
LRD_ALPHA_MIN     = 0.0
LRD_ALPHA_MAX     = 1.0
LRD_ALPHA_POWER   = 1.0

# ══════════════════════════════════════════════════════════════════════════════

def adaptive_lrd_alpha(rank_ema, rank_ref):
    if rank_ref is None:
        return 1.0
    ratio = rank_ema / (rank_ref + 1e-8)
    alpha = ratio ** LRD_ALPHA_POWER
    return max(LRD_ALPHA_MIN, min(LRD_ALPHA_MAX, alpha))


def _safe_float_label(value):
    return str(value).replace(".", "p").replace("-", "m")


def make_run_id(timestamp, task_name):
    batch = "full" if BATCH_SIZE is None else str(BATCH_SIZE)
    frac = _safe_float_label(TRAIN_FRAC)
    ls = _safe_float_label(LABEL_SMOOTHING)
    return f"{timestamp}_{task_name}_seed{SEED}_bs{batch}_ls{ls}_frac{frac}"


def write_run_config(run_dir, run_id, task_name):
    config = dict(
        run_id=run_id,
        task=task_name,
        dims=DIMS,
        train_size=TRAIN_SIZE,
        train_frac=TRAIN_FRAC,
        test_size=TEST_SIZE,
        steps=STEPS,
        batch_size=BATCH_SIZE,
        label_smoothing=LABEL_SMOOTHING,
        record=RECORD,
        report=REPORT,
        regs=REGS,
        defaults=DEFAULTS,
        seed=SEED,
        use_muon=USE_MUON,
        adaptive_lrd=ADAPTIVE_LRD,
        lrd_ref_test_acc=LRD_REF_TEST_ACC,
        rank_ema_beta=RANK_EMA_BETA,
        lrd_alpha_min=LRD_ALPHA_MIN,
        lrd_alpha_max=LRD_ALPHA_MAX,
        lrd_alpha_power=LRD_ALPHA_POWER,
        device=str(DEVICE),
    )
    path = os.path.join(run_dir, "run.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return path


def write_metrics_csv(log, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(log.keys())
    rows = zip(*(log[k] for k in fields))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)

def make_task(name, train_size, train_frac, test_size):
    kw = dict(train_size=train_size, train_frac=train_frac, test_size=test_size)
    if name == "ModularAddition":         return ModularAddition(p=97, **kw)
    elif name == "ModularMultiplication": return ModularMultiplication(p=97, **kw)
    elif name == "ModularLog":            return ModularLog(p=97, **kw)
    elif name == "ModularPolynomial":     return ModularPolynomial(p=97, **kw)
    elif name == "GroupComposition":      return GroupComposition(n=5, **kw)
    elif name == "Parity":                return Parity(dim=16, **kw)
    else: raise ValueError(f"Unknown task: {name}")


def resolve_tasks(task_config, all_task_names):
    if isinstance(task_config, str):
        return all_task_names if task_config.lower() == "all" else [task_config]
    return list(task_config)


def make_model(task_name, task, dim):
    return task.make_model(dim=dim, heads=4).to(DEVICE)

def run_one(task_name, task, reg, dim, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    lr      = DEFAULTS[reg]["lr"]
    lam     = DEFAULTS[reg].get("lam")
    lam_lrd = DEFAULTS[reg].get("lam_lrd")
    lam_l2  = DEFAULTS[reg].get("lam_l2")

    model = make_model(task_name, task, dim)

    warmup_steps = min(200, STEPS // 20)
    rank = mean_qk_stable_rank(model)
    rank_ema = rank
    rank_ref = None
    lrd_alpha = 1.0
    lam_lrd_eff = 0.0

    # ── 根据 USE_MUON 开关正确初始化优化器 ─────────────────────────────────
    if USE_MUON:
        Muon = get_muon_cls()
        muon_params =[p for name, p in model.named_parameters()
                       if p.ndim >= 2
                       and all(k not in name for k in ('embed', 'head', 'pos_emb', 'proj'))]
        adam_params =[p for name, p in model.named_parameters()
                       if p.ndim < 2
                       or any(k in name for k in ('embed', 'head', 'pos_emb', 'proj'))]

        muon = Muon(muon_params, lr=lr, momentum=0.95, nesterov=True)
        adam = torch.optim.AdamW(adam_params, lr=lr * 0.1, weight_decay=0.0)
    else:
        # 纯 AdamW 模式，接管所有参数
        adam = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        muon = None
    # ───────────────────────────────────────────────────────────────────────────

    if reg == "lrd_elastic":
        print(f"[{reg} dim={dim}] lr={lr:.0e} lam_lrd={lam_lrd:.0e} lam_l2={lam_l2:.0e}  "
              f"batch={BATCH_SIZE or 'full'} warmup={warmup_steps}")
    else:
        print(f"  [{reg} dim={dim}] lr={lr:.0e} lam={lam:.0e}  "
              f"batch={BATCH_SIZE or 'full'} warmup={warmup_steps}")
    if reg in ("lrd_decoupled", "lrd_elastic") and ADAPTIVE_LRD:
        print(f"  adaptive_lrd: ref_test_acc={LRD_REF_TEST_ACC:.2f} beta={RANK_EMA_BETA:.2f} "
              f"alpha=[{LRD_ALPHA_MIN:.2f}, {LRD_ALPHA_MAX:.2f}] power={LRD_ALPHA_POWER:.1f}")

    log = dict(steps=[], loss=[], train_acc=[], test_acc=[], rank=[], rank_ema=[], rank_ref=[],
               grad_norm=[], weight_norm=[], lrd_alpha=[], lam_lrd_eff=[])

    for step in range(1, STEPS + 1):
        
        # ── 调整学习率 ──
        scale = step / warmup_steps if step <= warmup_steps else 1.0
        if USE_MUON:
            for g in muon.param_groups:
                g["lr"] = lr * scale
            # 如果原逻辑是使用 Muon 时 Adam 的 LR 固定不预热，则保留原意
        else:
            for g in adam.param_groups:
                g["lr"] = lr * scale

        if BATCH_SIZE is None:
            x, y = task.full_batch()
        else:
            x, y = task.sample_batch(BATCH_SIZE)
        x, y = x.to(DEVICE), y.to(DEVICE)

        # loss 只有 task loss，正则化全部解耦
        if LABEL_SMOOTHING > 0:
            loss = F.cross_entropy(model(x), y, label_smoothing=LABEL_SMOOTHING)
        else:
            loss = task.loss(model, (x, y))

        # ── 梯度清零与反向传播 ──
        if USE_MUON:
            muon.zero_grad()
        adam.zero_grad()
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        
        # ── 步进更新 ──
        if USE_MUON:
            muon.step()
        adam.step()

        train_acc, test_acc = task.evaluate(model)
        rank = mean_qk_stable_rank(model)
        weight_norm = mean_qk_weight_norm(model)
        rank_ema = RANK_EMA_BETA * rank_ema + (1.0 - RANK_EMA_BETA) * rank
        if (reg in ("lrd_decoupled", "lrd_elastic") and ADAPTIVE_LRD
                and rank_ref is None and test_acc >= LRD_REF_TEST_ACC):
            rank_ref = rank_ema
            print(f"  [{reg} dim={dim}] adaptive_lrd rank_ref set @ step {step}: "
                  f"test={test_acc:.3f} rank_ref={rank_ref:.2f}")

        # ── 解耦正则化 ──
        current_lr = muon.param_groups[0]["lr"] if USE_MUON else adam.param_groups[0]["lr"]
        if reg in ("lrd_decoupled", "lrd_elastic") and ADAPTIVE_LRD:
            lrd_alpha = adaptive_lrd_alpha(rank_ema, rank_ref)
        else:
            lrd_alpha = 1.0
        
        if reg == "l2":
            apply_l2_decoupled(model, lr=current_lr, lam=lam)
        elif reg == "lrd_decoupled":
            lam_lrd_eff = lam * lrd_alpha
            apply_lrd_decoupled(model, lr=current_lr, lam=lam_lrd_eff)
        elif reg == "lrd_elastic":
            lam_lrd_eff = lam_lrd * lrd_alpha
            apply_lrd_decoupled(model, lr=current_lr, lam=lam_lrd_eff)
            apply_l2_decoupled(model, lr=current_lr, lam=lam_l2)

        # ── 日志记录与早停 ──
        if step % RECORD == 0:
            log["steps"].append(step)
            log["loss"].append(loss.item())
            log["train_acc"].append(train_acc)
            log["test_acc"].append(test_acc)
            log["rank"].append(rank)
            log["rank_ema"].append(rank_ema)
            log["rank_ref"].append(float("nan") if rank_ref is None else rank_ref)
            log["grad_norm"].append(grad_norm.item())
            log["weight_norm"].append(weight_norm)
            log["lrd_alpha"].append(lrd_alpha)
            log["lam_lrd_eff"].append(lam_lrd_eff)

            if step % REPORT == 0:
                msg = (f"  [{reg} dim={dim}] step {step:>5} "
                       f"| loss {loss.item():.4f} | train {train_acc:.3f} "
                       f"| test {test_acc:.3f} | rank {rank:.2f}"
                       f"| gnorm {grad_norm:.3f} | wnorm {weight_norm:.3f}")
                if reg in ("lrd_decoupled", "lrd_elastic") and ADAPTIVE_LRD:
                    msg += f"| alpha {lrd_alpha:.3f} | lam_lrd_eff {lam_lrd_eff:.2e}"
                print(msg)

            if test_acc >= 0.99:
                print(f"  [{reg} dim={dim}] ★ 早停 @ step {step}")
                break

    return log, model


def qk_spectrum(model):
    spectra = []
    for module in model.modules():
        if hasattr(module, "q") and hasattr(module, "k"):
            for linear in (module.q, module.k):
                W = linear.weight.detach().float().view(linear.weight.shape[0], -1)
                sv = torch.linalg.svdvals(W)
                sv = sv / (sv[0] + 1e-12)
                spectra.append(sv.cpu().numpy())
    if not spectra:
        return None
    min_len = min(len(sv) for sv in spectra)
    return np.stack([sv[:min_len] for sv in spectra]).mean(axis=0)


def qk_pattern(model):
    patterns = []
    for module in model.modules():
        if hasattr(module, "q") and hasattr(module, "k"):
            q = module.q.weight.detach().float().view(module.q.weight.shape[0], -1)
            k = module.k.weight.detach().float().view(module.k.weight.shape[0], -1)
            pattern = (q @ k.T).abs()
            pattern = pattern / (pattern.max() + 1e-12)
            patterns.append(pattern.cpu().numpy())
    if not patterns:
        return None
    return np.stack(patterns).mean(axis=0)

def plot(results, task_name, dims, train_frac, save_dir, run_id):
    os.makedirs(save_dir, exist_ok=True)
    colors = {"l2": "#e05c5c", "lrd_decoupled": "#4a90d9", "lrd_elastic": "#2ca02c"}

    n_rows = len(dims)
    fig, axes = plt.subplots(n_rows, 6, figsize=(30, 4 * n_rows), squeeze=False)
    fig.suptitle(f"{run_id} | Quick Check | {task_name}  train_frac={train_frac}", fontsize=14, y=1.01)

    for row, dim in enumerate(dims):
        axes[row, 0].set_ylabel(f"dim={dim}", fontsize=11, labelpad=10)
        for reg in REGS:
            key = (reg, dim)
            if key not in results or results[key] is None:
                continue
            log = results[key]
            c   = colors.get(reg, "gray")
            if reg == "lrd_elastic":
                label = (f"lrd_elastic  lr={DEFAULTS[reg]['lr']:.0e}"
                         f" lrd={DEFAULTS[reg]['lam_lrd']:.0e}"
                         f" l2={DEFAULTS[reg]['lam_l2']:.0e}")
            else:
                label = f"{reg}  lr={DEFAULTS[reg]['lr']:.0e} λ={DEFAULTS[reg]['lam']:.0e}"
            for col, metric in enumerate(["loss", "train_acc", "test_acc", "rank", "grad_norm", "weight_norm"]):
                axes[row, col].plot(log["steps"], log[metric],
                                    color=c, linewidth=1.0, label=label)

        titles  = ["Test Loss", "Train Accuracy", "Test Accuracy", "QK Stable Rank", "Grad Norm", "QK Weight Norm"]
        ylabels = ["loss", "accuracy", "accuracy", "stable rank", "grad norm", "weight norm"]
        for col in range(6):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(titles[col])
            ax.set_xlabel("step")
            ax.set_ylabel(ylabels[col])
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            if 1 <= col <= 2:
                ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{run_id}_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  图已保存 → {path}")


def plot_spectral_report(results, models, task_name, dim, save_dir, run_id):
    keys = [(reg, dim) for reg in REGS if (reg, dim) in results and (reg, dim) in models]
    if not keys:
        return

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.28)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax_acc = fig.add_subplot(gs[0, 0])
    ax_rank = fig.add_subplot(gs[0, 1])
    ax_spec = fig.add_subplot(gs[0, 2])

    for i, key in enumerate(keys):
        reg, _ = key
        log = results[key]
        color = colors[i % len(colors)]
        ax_acc.plot(log["steps"], log["test_acc"], label=reg, color=color)
        ax_rank.plot(log["steps"], log["rank"], label=reg, color=color)

        spectrum = qk_spectrum(models[key])
        if spectrum is not None:
            ax_spec.plot(np.arange(len(spectrum)), spectrum, marker="o", markersize=2,
                         linewidth=1.0, label=reg, color=color)

    ax_acc.set_title("Grokking Speed")
    ax_acc.set_xlabel("step")
    ax_acc.set_ylabel("test acc")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(fontsize=8)

    ax_rank.set_title("Effective Rank of QK")
    ax_rank.set_xlabel("step")
    ax_rank.set_ylabel("stable rank")
    ax_rank.grid(alpha=0.3)
    ax_rank.legend(fontsize=8)

    ax_spec.set_title("Singular Value Spectrum (Normalized)")
    ax_spec.set_xlabel("singular value index")
    ax_spec.set_ylabel("sigma_i / sigma_1")
    ax_spec.set_yscale("log")
    ax_spec.grid(alpha=0.3)
    ax_spec.legend(fontsize=8)

    for i, key in enumerate(keys[:3]):
        reg, _ = key
        ax = fig.add_subplot(gs[1, i])
        pattern = qk_pattern(models[key])
        if pattern is not None:
            ax.imshow(pattern, aspect="auto", cmap="viridis")
        ax.set_title(f"{reg} QK Pattern")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{run_id} | Spectral Report | {task_name} dim={dim}", fontsize=13)
    path = os.path.join(save_dir, f"{run_id}_spectral_dim{dim}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  谱结构图已保存 → {path}")


if __name__ == "__main__":
    ALL_TASK_NAMES = ["ModularAddition", "ModularMultiplication", "ModularLog",
                      "ModularPolynomial", "GroupComposition", "Parity"]
    tasks_to_run = resolve_tasks(TASK, ALL_TASK_NAMES)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Tasks: {tasks_to_run}  |  dims: {DIMS}  |  steps: {STEPS}"
          f"  |  batch: {BATCH_SIZE or 'full'}  |  train_frac: {TRAIN_FRAC}"
          f"  |  device: {DEVICE}\n")

    for task_name in tasks_to_run:
        run_id = make_run_id(timestamp, task_name)
        run_dir = os.path.join(SAVE_DIR, run_id)
        metrics_dir = os.path.join(run_dir, "metrics")
        plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        config_path = write_run_config(run_dir, run_id, task_name)

        print(f"\n{'█'*60}\n  Task: {task_name}\n{'█'*60}")
        print(f"  Run ID : {run_id}")
        print(f"  Config : {config_path}")
        results = {}
        models = {}
        for dim in DIMS:
            for reg in REGS:
                print(f"\n{'═'*55}\n  {reg}  dim={dim}\n{'═'*55}")
                task = make_task(task_name, TRAIN_SIZE, TRAIN_FRAC, TEST_SIZE)
                log, model = run_one(task_name, task, reg, dim, SEED)
                results[(reg, dim)] = log
                models[(reg, dim)] = model
                metrics_path = os.path.join(metrics_dir, f"{run_id}_{reg}_dim{dim}.csv")
                write_metrics_csv(log, metrics_path)
                print(f"  metrics → {metrics_path}")
        plot(results, task_name, DIMS, TRAIN_FRAC, plots_dir, run_id)
        for dim in DIMS:
            plot_spectral_report(results, models, task_name, dim, plots_dir, run_id)
        print(f"  Run outputs → {run_dir}")
