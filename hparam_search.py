"""
Bayesian hyperparameter search — lives in rankbench root.

所有正则化均解耦，不加在 loss 里。
"""

import os
import json
import torch
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from utils import DEVICE
from models.newton_schulz_lrd import apply_lrd_decoupled, apply_l2_decoupled

# ── 优化器开关（和 trainer.py 保持一致）────────────────────────────────────
USE_MUON = False   # True = Muon + AdamW，False = 纯 AdamW
# ───────────────────────────────────────────────────────────────────────────


class _EarlyStop(Exception):
    pass


SPACES = {
    "l2":            [Real(1e-3, 1e-1, prior="log-uniform", name="lr"),
                      Real(1e-6, 1e-1, prior="log-uniform", name="lam")],
    "lrd_decoupled": [Real(1e-3, 1e-1, prior="log-uniform", name="lr"),
                      Real(1e-4, 1e0,  prior="log-uniform", name="lam")],
    "lrd_elastic":   [Real(1e-3, 1e-1, prior="log-uniform", name="lr"),
                      Real(1e-4, 1e0,  prior="log-uniform", name="lam_lrd"),
                      Real(1e-6, 1e-1, prior="log-uniform", name="lam_l2")],
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


def _run_trial(task, reg, lr, max_steps, batch_size, target_acc,
               eval_interval=50, lam=None, lam_lrd=None, lam_l2=None):
    model = task.make_model().to(DEVICE)
    muon, adam = _make_optimizers(model, lr)
    best_test_acc = 0.0
    warmup_steps  = min(200, max_steps // 20)

    for step in range(1, max_steps + 1):
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

        # loss 只有 task loss
        loss = task.loss(model, (x, y))

        if USE_MUON:
            muon.zero_grad()
        adam.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if USE_MUON:
            muon.step()
        adam.step()

        # 解耦正则化
        current_lr = muon.param_groups[0]["lr"] if USE_MUON else adam.param_groups[0]["lr"]
        with torch.no_grad():
            if reg == "l2":
                apply_l2_decoupled(model, lr=current_lr, lam=lam)
            elif reg == "lrd_decoupled":
                apply_lrd_decoupled(model, lr=current_lr, lam=lam)
            elif reg == "lrd_elastic":
                apply_lrd_decoupled(model, lr=current_lr, lam=lam_lrd)
                apply_l2_decoupled(model,  lr=current_lr, lam=lam_l2)

        if step % eval_interval == 0:
            _, test_acc = task.evaluate(model)
            best_test_acc = max(best_test_acc, test_acc)
            if test_acc >= target_acc:
                return float(step)

    return float(max_steps) + float(max_steps) * (1.0 - best_test_acc)


def _gp_search(tasks, reg, n_calls, max_steps, batch_size,
               target_acc, random_state, eval_interval=50,
               save_path=None, early_stop_steps=300,
               early_stop_min_trials=15):
    space     = SPACES[reg]
    trial_log = []

    @use_named_args(space)
    def objective(**kwargs):
        lr      = kwargs["lr"]
        lam     = kwargs.get("lam")
        lam_lrd = kwargs.get("lam_lrd")
        lam_l2  = kwargs.get("lam_l2")

        scores = [
            _run_trial(t, reg, lr, max_steps, batch_size, target_acc,
                       eval_interval, lam=lam, lam_lrd=lam_lrd, lam_l2=lam_l2)
            for t in tasks
        ]
        mean_score = float(np.mean(scores))

        if reg == "lrd_elastic":
            trial_log.append((mean_score, lr, lam_lrd, lam_l2))
        else:
            trial_log.append((mean_score, lr, lam))

        n_done = len(trial_log)
        best   = min(trial_log, key=lambda x: x[0])

        if n_done == 1 or n_done % 5 == 0 or n_done == n_calls:
            if reg == "lrd_elastic":
                param_str = f"lr={lr:.2e} lam_lrd={lam_lrd:.2e} lam_l2={lam_l2:.2e}"
                best_str  = f"lr={best[1]:.2e} lam_lrd={best[2]:.2e} lam_l2={best[3]:.2e}"
            else:
                param_str = f"lr={lr:.2e} lam={lam:.2e}"
                best_str  = f"lr={best[1]:.2e} lam={best[2]:.2e}"
            task_str = "  ".join(f"{t.__class__.__name__}={s:.0f}"
                                 for t, s in zip(tasks, scores))
            print(f"  [{reg}] {n_done:>3}/{n_calls} | {param_str}"
                  f" | {task_str} | this={mean_score:.0f}"
                  f" | best={best[0]:.0f} ({best_str})")

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(dict(
                    reg          = reg,
                    n_done       = n_done,
                    best_score   = best[0],
                    best_lr      = best[1],
                    best_lam     = best[2] if reg != "lrd_elastic" else None,
                    best_lam_lrd = best[2] if reg == "lrd_elastic" else None,
                    best_lam_l2  = best[3] if reg == "lrd_elastic" else None,
                    all_trials   = trial_log,
                ), f, indent=2)

        if best[0] <= early_stop_steps and n_done >= early_stop_min_trials:
            print(f"  [{reg}] ★ 早退 trial {n_done} (score={best[0]:.0f})")
            raise _EarlyStop()

        return mean_score

    try:
        result = gp_minimize(
            func=objective, dimensions=space, n_calls=n_calls,
            n_initial_points=max(10, n_calls // 5),
            acq_func="EI", random_state=random_state, noise="gaussian",
        )
        best_x     = result.x
        best_score = result.fun
    except _EarlyStop:
        best       = min(trial_log, key=lambda x: x[0])
        best_x     = list(best[1:])
        best_score = best[0]

    if reg == "lrd_elastic":
        print(f"\n{'─'*60}")
        print(f"  [{reg}]  BEST  lr={best_x[0]:.2e}"
              f"  lam_lrd={best_x[1]:.2e}  lam_l2={best_x[2]:.2e}"
              f"  score={best_score:.0f}  (trials={len(trial_log)})")
        print(f"{'─'*60}\n")
        return dict(reg=reg, best_lr=best_x[0],
                    best_lam=None, best_lam_lrd=best_x[1], best_lam_l2=best_x[2],
                    best_score=best_score)
    else:
        print(f"\n{'─'*60}")
        print(f"  [{reg}]  BEST  lr={best_x[0]:.2e}  lam={best_x[1]:.2e}"
              f"  score={best_score:.0f}  (trials={len(trial_log)})")
        print(f"{'─'*60}\n")
        return dict(reg=reg, best_lr=best_x[0], best_lam=best_x[1],
                    best_lam_lrd=None, best_lam_l2=None, best_score=best_score)


def search_for_task(task, regs=("l2", "lrd_elastic"),
                    n_calls=60, max_steps=10_000,
                    batch_size=None, target_acc=0.99,
                    random_state=42, results_dir="results",
                    eval_interval=50,
                    early_stop_steps=300,
                    early_stop_min_trials=15):
    task_name = task.__class__.__name__
    print(f"\n{'═'*60}\n  Per-task search — {task_name}\n{'═'*60}")

    out = {}
    for reg in regs:
        calls     = n_calls[reg] if isinstance(n_calls, dict) else n_calls
        save_path = os.path.join(results_dir, f"search_{task_name}_{reg}.json")

        if os.path.exists(save_path):
            with open(save_path) as f:
                ckpt = json.load(f)
            if ckpt.get("n_done", 0) >= calls:
                print(f"  [{reg}] 已有完整结果，跳过 (score={ckpt['best_score']:.0f})")
                out[reg] = dict(
                    reg          = reg,
                    best_lr      = ckpt["best_lr"],
                    best_lam     = ckpt.get("best_lam"),
                    best_lam_lrd = ckpt.get("best_lam_lrd"),
                    best_lam_l2  = ckpt.get("best_lam_l2"),
                    best_score   = ckpt["best_score"],
                )
                continue

        out[reg] = _gp_search(
            tasks=[task], reg=reg, n_calls=calls,
            max_steps=max_steps, batch_size=batch_size,
            target_acc=target_acc, random_state=random_state,
            eval_interval=eval_interval,
            save_path=save_path,
            early_stop_steps=early_stop_steps,
            early_stop_min_trials=early_stop_min_trials,
        )
    return out
