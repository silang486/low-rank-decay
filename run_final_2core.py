"""
run_final_2core.py — 双 GPU 版本

用法：
  # GPU 0
  CUDA_VISIBLE_DEVICES=0 python run_final_2core.py --tasks ModularAddition ModularMultiplication ModularLog

  # GPU 1（同一个 RUN_ID）
  CUDA_VISIBLE_DEVICES=1 python run_final_2core.py --tasks GroupComposition Parity ModularPolynomial

两个进程各自写独立的 best_params_{suffix}.json，不会冲突。
阶段3结束后自动尝试合并两份结果出完整图，如果另一边还没跑完就只画自己的。
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

from tasks.modular_addition       import ModularAddition
from tasks.modular_multiplication import ModularMultiplication
from tasks.modular_polynomial     import ModularPolynomial
from tasks.modular_log            import ModularLog
from tasks.group_composition      import GroupComposition
from tasks.parity                 import Parity

from trainer       import train
from hparam_search import search_for_task

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TEST_SIZE = 1000

ALL_TASKS = [
    ModularAddition(p=97,       test_size=TEST_SIZE, train_frac=0.4),
    ModularMultiplication(p=97, test_size=TEST_SIZE, train_frac=0.4),
    ModularPolynomial(p=97,     test_size=TEST_SIZE, train_frac=0.4),
    ModularLog(p=97,            test_size=TEST_SIZE, train_frac=0.4),
    GroupComposition(n=5,       test_size=TEST_SIZE, train_frac=0.4),
    Parity(dim=16,              test_size=TEST_SIZE, train_frac=0.4),
]

REGS  = ["l2", "lrd_decoupled", "lrd_elastic"]
SEEDS = [0, 1, 2]

SEARCH_N_CALLS = {
    "l2":            40,
    "lrd_decoupled": 40,
    "lrd_elastic":   60,
}

TASK_CONFIGS = {
    "ModularAddition":       dict(search_steps=10_000,  train_steps=20_000),
    "ModularMultiplication": dict(search_steps=10_000,  train_steps=20_000),
    "ModularPolynomial":     dict(search_steps=50_000,  train_steps=100_000),
    "ModularLog":            dict(search_steps=30_000,  train_steps=60_000),
    "GroupComposition":      dict(search_steps=10_000,  train_steps=20_000),
    "Parity":                dict(search_steps=15_000,  train_steps=40_000),
}

BATCH_SIZE            = None
EARLY_STOP_MIN_TRIALS = 8

# 断点续跑时把 RUN_ID 改成上次的文件夹名
RUN_ID      = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join("results", RUN_ID)

# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def _task_kwargs(task):
    cls = task.__class__.__name__
    if cls in ("ModularAddition", "ModularMultiplication",
               "ModularPolynomial", "ModularLog"):
        return dict(p=task.p, test_size=TEST_SIZE, train_frac=0.4)
    if cls == "GroupComposition":
        return dict(n=task.n, test_size=TEST_SIZE, train_frac=0.4)
    if cls == "Parity":
        return dict(dim=task.dim, test_size=TEST_SIZE, train_frac=0.4)
    return {}

# ── Step 1 ────────────────────────────────────────────────────────────────────

def run_search(tasks, regs):
    best_params = {}
    for task in tasks:
        task_name = task.__class__.__name__
        cfg = TASK_CONFIGS[task_name]
        best_params[task_name] = search_for_task(
            task,
            regs                  = regs,
            n_calls               = SEARCH_N_CALLS,
            max_steps             = cfg["search_steps"],
            batch_size            = BATCH_SIZE,
            results_dir           = RESULTS_DIR,
            early_stop_steps      = cfg["search_steps"] * 0.1,
            early_stop_min_trials = EARLY_STOP_MIN_TRIALS,
        )
    return best_params

# ── Step 2 ────────────────────────────────────────────────────────────────────

def run_training(tasks, regs, best_params):
    results = {}
    for task in tasks:
        task_name = task.__class__.__name__
        print(f"\n{'═'*60}\n  Task: {task_name}\n{'═'*60}")
        results[task_name] = {}

        for reg in regs:
            p       = best_params[task_name][reg]
            # 兼容 best_lr/lr 两种键名（search 返回 best_lr，summary JSON 存 lr）
            lr      = p.get("best_lr") or p.get("lr")
            lam     = p.get("best_lam") or p.get("lam")
            lam_lrd = p.get("best_lam_lrd") or p.get("lam_lrd")
            lam_l2  = p.get("best_lam_l2")  or p.get("lam_l2")

            if reg == "lrd_elastic":
                print(f"\n  [{reg}]  lr={lr:.2e}  lam_lrd={lam_lrd:.2e}  lam_l2={lam_l2:.2e}")
            else:
                print(f"\n  [{reg}]  lr={lr:.2e}  lam={lam:.2e}")
            seed_results = []

            for seed in SEEDS:
                print(f"    seed={seed}")
                set_seed(seed)
                task_fresh = task.__class__(**_task_kwargs(task))
                metrics = train(
                    task_fresh, reg=reg, lr=lr,
                    lam=lam, lam_lrd=lam_lrd, lam_l2=lam_l2,
                    steps      = TASK_CONFIGS[task_name]["train_steps"],
                    batch_size = BATCH_SIZE,
                    record     = 10,
                    report     = 500,
                )
                seed_results.append(metrics)
            results[task_name][reg] = seed_results
    return results

# ── Step 3 ────────────────────────────────────────────────────────────────────

def plot_task(task_name, task_results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{task_name}.png")

    fig  = plt.figure(figsize=(30, 4))
    gs   = gridspec.GridSpec(1, 6, figure=fig, wspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(6)]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (reg, seed_list) in enumerate(task_results.items()):
        c     = colors[i % len(colors)]
        steps = np.array(seed_list[0]["steps"])
        for key, ax in zip(["loss", "train_acc", "test_acc", "rank", "grad_norm", "weight_norm"], axes):
            min_len = min(len(s[key]) for s in seed_list)
            vals = np.array([s[key][:min_len] for s in seed_list])
            mean = vals.mean(axis=0)
            std  = vals.std(axis=0)
            ax.plot(steps[:min_len], mean, color=c, linewidth=1.5,
                    label=f"{reg} (n={len(seed_list)})")
            ax.fill_between(steps[:min_len], mean-std, mean+std,
                            color=c, alpha=0.15)

    for ax, title, ylabel in zip(
        axes,
        ["Test Loss", "Train Accuracy", "Test Accuracy", "QK Stable Rank", "Grad Norm", "QK Weight Norm"],
        ["loss", "accuracy", "accuracy", "stable rank", "grad norm", "weight norm"],
    ):
        ax.set_xlabel("step"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    axes[2].set_ylim(0, 1.05)

    fig.suptitle(f"Grokking — {task_name}", fontsize=13, y=1.02)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

def plot_all(results, save_dir):
    for task_name, task_results in results.items():
        plot_task(task_name, task_results, save_dir)

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ALL_TASK_NAMES = [t.__class__.__name__ for t in ALL_TASKS]
    ALL_TASK_MAP   = {t.__class__.__name__: t for t in ALL_TASKS}

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", nargs="+", default=ALL_TASK_NAMES,
        choices=ALL_TASK_NAMES,
        help="要跑的任务。例：--tasks ModularAddition ModularMultiplication"
    )
    args = parser.parse_args()
    tasks = [ALL_TASK_MAP[name] for name in args.tasks]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    task_suffix = "_".join(sorted([t.__class__.__name__ for t in tasks]))

    print(f"Run ID  : {RUN_ID}")
    print(f"Tasks   : {[t.__class__.__name__ for t in tasks]}")
    print(f"Results → {RESULTS_DIR}/")

    # ── 阶段 1：搜索 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 1 / 3 — Per-task hyperparameter search")
    print("=" * 60)
    best_params = run_search(tasks, REGS)

    # 各自写独立的 json，不会和另一个进程冲突
    json_path = os.path.join(RESULTS_DIR, f"best_params_{task_suffix}.json")
    summary = {
        task_name: {
            reg: dict(
                lr      = res["best_lr"],
                lam     = res.get("best_lam"),
                lam_lrd = res.get("best_lam_lrd"),
                lam_l2  = res.get("best_lam_l2"),
                score   = res["best_score"],
            )
            for reg, res in per_reg.items()
        }
        for task_name, per_reg in best_params.items()
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Best params → {json_path}")
    for task_name, per_reg in summary.items():
        for reg, v in per_reg.items():
            if reg == "lrd_elastic":
                print(f"    {task_name:30s} {reg:20s} "
                      f"lr={v['lr']:.2e}  lam_lrd={v['lam_lrd']:.2e}"
                      f"  lam_l2={v['lam_l2']:.2e}  score={v['score']:.0f}")
            else:
                print(f"    {task_name:30s} {reg:20s} "
                      f"lr={v['lr']:.2e}  lam={v['lam']:.2e}  "
                      f"score={v['score']:.0f}")

    # ── 阶段 2：训练 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 2 / 3 — Full training (3 seeds)")
    print("=" * 60)
    results = run_training(tasks, REGS, best_params)

    # ── 阶段 3：画图 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 3 / 3 — Plotting")
    print("=" * 60)

    # 先画自己的任务
    plot_all(results, save_dir=RESULTS_DIR)

    # 尝试合并另一个进程的结果出完整图
    all_json = [
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith("best_params_") and f.endswith(".json")
    ]
    if len(all_json) >= 2:
        print("\n  检测到两份 best_params，尝试合并出完整图...")
        merged = {}
        for fn in all_json:
            with open(os.path.join(RESULTS_DIR, fn)) as f:
                merged.update(json.load(f))
        merged_path = os.path.join(RESULTS_DIR, "best_params_merged.json")
        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"  Merged → {merged_path}")
    else:
        print("\n  另一个进程尚未完成，跳过合并。")
        print("  全部完成后手动运行 merge_and_plot.py 生成完整图。")

    print(f"\nAll done. Results in '{RESULTS_DIR}/'")
