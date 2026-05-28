"""
run_explore.py
==============
单次运行探索脚本，追踪 grad update vs decay 比值及奇异值谱演化。

在你的 rankbench 根目录直接运行：
    python run_explore.py

不修改任何原有文件，只依赖：
    trainer.py          ← train() 接口
    explore_update_ratio.py  ← UpdateRatioTracker（放在同目录）
    tasks/*             ← 和 run_final.py 一致
    models/*            ← 和 run_final.py 一致
    utils.py            ← DEVICE

CONFIG 区域是唯一需要改的地方。
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from datetime import datetime
from typing import Optional

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ── 你的项目模块 ──────────────────────────────────────────────────────────────
from utils import DEVICE
from models.newton_schulz_lrd import apply_lrd_decoupled, apply_l2_decoupled
from models.rank_monitor import mean_qk_stable_rank

# ── 选一个任务来探索，换成你想观察的 ──────────────────────────────────────────
from tasks.group_composition import GroupComposition
EXPLORE_TASK = GroupComposition(n=8, train_frac=0.4)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — 只改这里
# ══════════════════════════════════════════════════════════════════════════════

# 要对比的正则化方式，每个会跑一次完整训练并生成独立图表
CONFIGS = [
    dict(reg="l2",            lr=2e-2, lam=1e-3),
    dict(reg="lrd_decoupled", lr=2e-2, lam=1e-1),
]

STEPS         = 10_000   # 训练步数
BATCH_SIZE    = 256
REPORT_EVERY  = 100      # 基础指标（train/test acc, rank）记录间隔
TRACK_EVERY   = 200      # UpdateRatioTracker 记录间隔（比 REPORT 稍稀，节省 SVD 开销）
LRD_ITERS     = 30       # NS 迭代次数，和主代码保持一致
TOP_K_SV      = 6        # 追踪前几个奇异值
SEED          = 42

SAVE_DIR = os.path.join("results", "explore_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

# ══════════════════════════════════════════════════════════════════════════════
#  UpdateRatioTracker（内联，不需要单独 import explore_update_ratio.py）
# ══════════════════════════════════════════════════════════════════════════════

def _polar_ns(W: torch.Tensor, iters: int = 30, eps: float = 1e-6) -> torch.Tensor:
    transposed = W.shape[0] < W.shape[1]
    if transposed:
        W = W.T
    X = W / (W.norm() + eps)
    I = torch.eye(X.shape[1], device=W.device, dtype=W.dtype)
    for _ in range(iters):
        A = X.T @ X
        X = 0.5 * (X @ (3.0 * I - A))
    if transposed:
        X = X.T
    return X


def _adamw_effective_step(param: torch.Tensor,
                           optimizer: torch.optim.Optimizer,
                           lr: float) -> Optional[torch.Tensor]:
    """从 AdamW state 里还原实际自适应步长向量（含 bias correction）。"""
    state = optimizer.state.get(param, {})
    if 'exp_avg' not in state:
        return None
    m    = state['exp_avg']
    v    = state['exp_avg_sq']
    step = state.get('step', 1)
    step_val = step.item() if isinstance(step, torch.Tensor) else int(step)

    betas, adam_eps = (0.9, 0.999), 1e-8
    for group in optimizer.param_groups:
        if any(p is param for p in group['params']):
            betas    = group.get('betas', betas)
            adam_eps = group.get('eps', adam_eps)
            break

    beta1, beta2 = betas
    m_hat = m / (1 - beta1 ** step_val)
    v_hat = v / (1 - beta2 ** step_val)
    return lr * m_hat / (v_hat.sqrt() + adam_eps)


def _svd_stats(W: torch.Tensor, top_k: int = 6, eps_ratio: float = 0.01) -> dict:
    W2d  = W.detach().float().view(W.shape[0], -1)
    sv   = torch.linalg.svdvals(W2d)
    s1   = sv[0].item() + 1e-12
    fsq  = sv.pow(2).sum().item()
    return dict(
        sv_top        = sv[:top_k].tolist(),
        sv_normalized = (sv[:top_k] / s1).tolist(),
        stable_rank   = fsq / s1**2,
        numerical_rank= int((sv > eps_ratio * s1).sum().item()),
        spectral_norm = s1,
        frob_norm     = fsq**0.5,
    )


class UpdateRatioTracker:
    """
    记录每个 2D 权重矩阵上：
      grad_norm  : Adam 实际步长的 Frobenius 范数
      l2_norm    : L2 decay 步长范数   (lr * lam * W)
      lrd_norm   : LRD decay 步长范数  (lr * lam * polar(W))
      ratio_l2   : l2_norm  / grad_norm
      ratio_lrd  : lrd_norm / grad_norm
    以及奇异值谱的完整统计。
    """
    SKIP = ('embed', 'head', 'pos_emb', 'norm', 'bias')

    def __init__(self, model, optimizer, lr, lam_l2, lam_lrd,
                 lrd_iters=30, top_k=6):
        self.model     = model
        self.opt       = optimizer
        self.lr        = lr
        self.lam_l2    = lam_l2
        self.lam_lrd   = lam_lrd
        self.iters     = lrd_iters
        self.top_k     = top_k
        self.history   = defaultdict(lambda: defaultdict(list))
        self.steps     = []

    def _track(self, name, p):
        if not p.requires_grad or p.ndim < 2:
            return False
        return not any(k in name for k in self.SKIP)

    @torch.no_grad()
    def record(self, step: int):
        self.steps.append(step)
        for name, p in self.model.named_parameters():
            if not self._track(name, p):
                continue
            W   = p.detach().float()
            W2d = W.view(W.shape[0], -1)

            # Adam 步长
            adam = _adamw_effective_step(p, self.opt, self.lr)
            grad_norm = adam.view(W.shape[0], -1).norm('fro').item() \
                        if adam is not None else float('nan')

            # L2 步长
            l2_norm  = (self.lr * self.lam_l2  * W2d).norm('fro').item()

            # LRD 步长
            polar    = _polar_ns(W2d, iters=self.iters)
            lrd_norm = (self.lr * self.lam_lrd * polar).norm('fro').item()

            safe = grad_norm + 1e-12
            sv   = _svd_stats(p, top_k=self.top_k)

            h = self.history[name]
            h['grad_norm'].append(grad_norm)
            h['l2_norm'].append(l2_norm)
            h['lrd_norm'].append(lrd_norm)
            h['ratio_l2'].append(l2_norm  / safe)
            h['ratio_lrd'].append(lrd_norm / safe)
            h['stable_rank'].append(sv['stable_rank'])
            h['numerical_rank'].append(sv['numerical_rank'])
            h['spectral_norm'].append(sv['spectral_norm'])
            h['frob_norm'].append(sv['frob_norm'])
            for i, v in enumerate(sv['sv_normalized']):
                h[f'sv_norm_{i}'].append(v)

    def print_summary(self, last_n=50):
        print(f"\n{'='*72}")
        print(f"  Update Ratio Summary  (mean of last {last_n} steps)")
        print(f"{'='*72}")
        print(f"  {'param':<38} {'ratio_l2':>9} {'ratio_lrd':>10}"
              f" {'stbl_rank':>10} {'num_rank':>9}")
        print(f"  {'-'*70}")
        for name, h in self.history.items():
            def tail_mean(k):
                v = h.get(k, [])
                return float(np.mean(v[-last_n:])) if v else float('nan')
            rl2  = tail_mean('ratio_l2')
            rlrd = tail_mean('ratio_lrd')
            sr   = tail_mean('stable_rank')
            nr   = tail_mean('numerical_rank')
            flag = '  <-- decay > grad' if (rl2 > 1 or rlrd > 1) else ''
            print(f"  {name:<38} {rl2:>9.3f} {rlrd:>10.3f}"
                  f" {sr:>10.2f} {nr:>9.0f}{flag}")
        print(f"{'='*72}\n")



def train_with_tracker(task, reg: str, lr: float, lam: float,
                        steps: int, batch_size: int,
                        report_every: int, track_every: int,
                        lrd_iters: int, top_k_sv: int) -> dict:
    """
    和 trainer.train() 接口兼容，额外返回 tracker 对象。
    返回 (metrics_dict, tracker)。
    """
    model = task.make_model().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    # tracker 的 lam_l2 / lam_lrd 都用同一个 lam，
    # 这样可以直接对比两种正则化在同等 lam 下的步长大小
    tracker = UpdateRatioTracker(
        model    = model,
        optimizer= opt,
        lr       = lr,
        lam_l2   = lam,
        lam_lrd  = lam,
        lrd_iters= lrd_iters,
        top_k    = top_k_sv,
    )

    log = defaultdict(list)

    for step in range(1, steps + 1):
        x, y = task.sample_batch(batch_size)
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss  = task.loss(model, (x, y))

        # coupled regularisation
        if reg == "l2":
            loss = loss + lam * sum((p**2).sum() for p in model.parameters())

        opt.zero_grad()
        loss.backward()
        opt.step()

        # decoupled regularisation（在 optimizer.step 之后）
        if reg == "lrd_decoupled":
            apply_lrd_decoupled(model, lr=lr, lam=lam, iters=lrd_iters)
        elif reg == "l2":
            # l2 decoupled 版本可选；coupled 已在 loss 里处理，这里不重复
            pass

        # ── 基础指标 ─────────────────────────────────────────────────────────
        if step % report_every == 0:
            train_acc, test_acc = task.evaluate(model)
            rank = mean_qk_stable_rank(model)
            log['steps'].append(step)
            log['loss'].append(loss.item())
            log['train_acc'].append(train_acc)
            log['test_acc'].append(test_acc)
            log['rank'].append(rank)
            print(f"  [{reg}] step {step:>6} | loss {loss.item():.4f}"
                  f" | train {train_acc:.3f} | test {test_acc:.3f}"
                  f" | rank {rank:.2f}")

        # ── UpdateRatioTracker ───────────────────────────────────────────────
        if step % track_every == 0:
            tracker.record(step)

    return dict(reg=reg, lr=lr, lam=lam, **{k: v for k, v in log.items()}), tracker


# ══════════════════════════════════════════════════════════════════════════════
#  绘图
# ══════════════════════════════════════════════════════════════════════════════

def plot_basic(all_results: list, task_name: str, save_dir: str):
    """train acc / test acc / stable rank 三联图，和 run_final.py 风格一致。"""
    fig = plt.figure(figsize=(15, 4))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, r in enumerate(all_results):
        c     = colors[i % len(colors)]
        label = f"{r['reg']}  lr={r['lr']:.1e} lambda={r['lam']:.1e}"
        steps = r['steps']
        for ax, key in zip(axes, ['train_acc', 'test_acc', 'rank']):
            ax.plot(steps, r[key], color=c, label=label, linewidth=1.8)

    titles  = ['Train Accuracy', 'Test Accuracy', 'QK Stable Rank']
    ylabels = ['accuracy', 'accuracy', 'stable rank']
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_xlabel('step'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 1.05)

    fig.suptitle(f'Grokking - {task_name}', fontsize=13, y=1.02)
    path = os.path.join(save_dir, f'basic_{task_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved -> {path}')


def plot_tracker(tracker: UpdateRatioTracker,
                 reg: str, lr: float, lam: float,
                 save_dir: str, max_params: int = 8):
    """
    为每个被追踪的参数生成一张五行图：
      行1  更新步长绝对大小（log scale）
      行2  ratio = decay / grad（核心，ratio>1 说明正则化压制了学习）
      行3  权重范数：σ₁ 和 ‖W‖_F
      行4  归一化奇异值 σᵢ/σ₁（消除范数影响，看真实秩结构）
      行5  stable rank 与 numerical rank
    """
    import matplotlib.gridspec as gridspec

    steps  = np.array(tracker.steps)
    params = list(tracker.history.keys())[:max_params]

    for name in params:
        h    = tracker.history[name]
        safe = lambda k: np.array(h.get(k, [np.nan] * len(steps)))

        fig = plt.figure(figsize=(13, 17))
        fig.suptitle(f'[{reg}  lr={lr:.1e} lambda={lam:.1e}]\n{name}',
                     fontsize=11, y=0.99)
        gs = gridspec.GridSpec(5, 1, hspace=0.5)

        # ── 行1：步长绝对大小 ─────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(steps, safe('grad_norm'), label='Adam ||delta||', color='steelblue')
        ax1.plot(steps, safe('l2_norm'),   label='L2 ||delta||', color='coral',    ls='--')
        ax1.plot(steps, safe('lrd_norm'),  label='LRD ||delta||', color='seagreen', ls='-.')
        ax1.set_yscale('log')
        ax1.set_ylabel('update norm F (log)')
        ax1.set_title('Update step magnitude')
        ax1.legend(fontsize=8); ax1.grid(alpha=0.25)

        # ── 行2：比值（最重要）───────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(steps, safe('ratio_l2'),  label='L2  / Adam', color='coral')
        ax2.plot(steps, safe('ratio_lrd'), label='LRD / Adam', color='seagreen')
        ax2.axhline(1.0, color='gray', ls=':', lw=0.9, label='ratio = 1')
        ax2.axhline(0.1, color='gray', ls=':', lw=0.5, alpha=0.5)
        ax2.set_yscale('log')
        ax2.set_ylabel('decay / grad  (log)')
        ax2.set_title('Regularization vs gradient ratio: >1 means decay dominates learning')
        ax2.legend(fontsize=8); ax2.grid(alpha=0.25)

        # ── 行3：权重范数 ────────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(steps, safe('spectral_norm'), label='sigma_1', color='mediumpurple')
        ax3.plot(steps, safe('frob_norm'),     label='||W||_F', color='goldenrod', ls='--')
        ax3.set_ylabel('norm')
        ax3.set_title('Weight norms')
        ax3.legend(fontsize=8); ax3.grid(alpha=0.25)

        # ── 行4：归一化奇异值 ─────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[3])
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, tracker.top_k))
        for i in range(tracker.top_k):
            key = f'sv_norm_{i}'
            if key in h:
                ax4.plot(steps, safe(key),
                         label=f'sigma_{i+1}/sigma_1', color=cmap[i], alpha=0.85)
        ax4.set_ylabel('sigma_i / sigma_1')
        ax4.set_title('Normalized singular values')
        ax4.legend(fontsize=7, ncol=3); ax4.grid(alpha=0.25)
        ax4.set_ylim(-0.05, 1.05)

        # ── 行5：rank 指标 ─────────────────────────────────────────────
        ax5  = fig.add_subplot(gs[4])
        ax5r = ax5.twinx()
        ax5.plot(steps,  safe('stable_rank'),    color='steelblue', label='stable rank')
        ax5r.plot(steps, safe('numerical_rank'), color='tomato',    label='numerical rank', ls='--')
        ax5.set_ylabel('stable rank',    color='steelblue')
        ax5r.set_ylabel('numerical rank', color='tomato')
        ax5.set_xlabel('training step')
        ax5.set_title('Rank metrics')
        lines  = ax5.get_legend_handles_labels()
        lines2 = ax5r.get_legend_handles_labels()
        ax5.legend(lines[0]+lines2[0], lines[1]+lines2[1], fontsize=8)
        ax5.grid(alpha=0.25)

        safe_name = name.replace('.', '_').replace('/', '_')
        path = os.path.join(save_dir, f'tracker_{reg}_{safe_name}.png')
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  saved -> {path}')


def plot_ratio_overlay(all_trackers: list[tuple[str, UpdateRatioTracker]],
                       save_dir: str):
    """
    把所有 config 的 ratio_lrd 画在同一张图上，便于直接对比。
    每个子图对应一个参数矩阵。
    """
    # 收集所有被追踪的参数名（取并集）
    all_names = []
    for _, t in all_trackers:
        for n in t.history:
            if n not in all_names:
                all_names.append(n)
    all_names = all_names[:8]

    n_params = len(all_names)
    if n_params == 0:
        return

    fig, axes = plt.subplots(n_params, 1, figsize=(12, 3.5 * n_params),
                              squeeze=False)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for row, name in enumerate(all_names):
        ax = axes[row][0]
        for ci, (label, tracker) in enumerate(all_trackers):
            if name not in tracker.history:
                continue
            h     = tracker.history[name]
            steps = np.array(tracker.steps)
            ratio = np.array(h.get('ratio_lrd', [np.nan]*len(steps)))
            ax.plot(steps, ratio, color=colors[ci % len(colors)],
                    label=label, linewidth=1.5)
        ax.axhline(1.0, color='gray', ls=':', lw=0.8)
        ax.set_yscale('log')
        ax.set_title(name, fontsize=9)
        ax.set_ylabel('LRD / Adam')
        ax.legend(fontsize=7); ax.grid(alpha=0.25)
        if row == n_params - 1:
            ax.set_xlabel('training step')

    fig.suptitle('LRD decay / Adam grad ratio - all configs', fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, 'ratio_overlay.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved -> {path}')


# ══════════════════════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    task_name = EXPLORE_TASK.__class__.__name__
    print(f'\n{"="*64}')
    print(f'  Explore: {task_name}')
    print(f'  Configs: {[c["reg"] for c in CONFIGS]}')
    print(f'  Steps  : {STEPS}   batch={BATCH_SIZE}')
    print(f'  Output : {SAVE_DIR}/')
    print(f'{"="*64}\n')

    all_results  = []
    all_trackers = []

    for cfg in CONFIGS:
        reg, lr, lam = cfg['reg'], cfg['lr'], cfg['lam']
        print(f'\n{"-"*60}')
        print(f'  Training  reg={reg}  lr={lr:.1e}  lambda={lam:.1e}')
        print(f'{"-"*60}')

        # 每次用新鲜的 task（确保 train/test split 一致）
        torch.manual_seed(SEED)
        task_fresh = EXPLORE_TASK.__class__(
            **{k: getattr(EXPLORE_TASK, k)
               for k in ('n', 'train_frac')   # GroupComposition 的构造参数
               if hasattr(EXPLORE_TASK, k)}
        )

        metrics, tracker = train_with_tracker(
            task        = task_fresh,
            reg         = reg,
            lr          = lr,
            lam         = lam,
            steps       = STEPS,
            batch_size  = BATCH_SIZE,
            report_every= REPORT_EVERY,
            track_every = TRACK_EVERY,
            lrd_iters   = LRD_ITERS,
            top_k_sv    = TOP_K_SV,
        )
        all_results.append(metrics)
        all_trackers.append((f'{reg} lr={lr:.1e} lambda={lam:.1e}', tracker))

        # 文字摘要
        tracker.print_summary(last_n=10)

        # 每个 config 的详细图（每个参数一张）
        plot_tracker(tracker, reg=reg, lr=lr, lam=lam, save_dir=SAVE_DIR)

    # 基础对比图（所有 config 在同一张图）
    print(f'\n{"-"*60}')
    print('  Plotting basic comparison...')
    plot_basic(all_results, task_name=task_name, save_dir=SAVE_DIR)

    # ratio overlay（所有 config 叠在一起对比）
    print('  Plotting ratio overlay...')
    plot_ratio_overlay(all_trackers, save_dir=SAVE_DIR)

    # 保存 metrics 为 JSON，方便后续分析
    json_path = os.path.join(SAVE_DIR, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump([{k: v for k, v in r.items()} for r in all_results],
                  f, indent=2, default=lambda x: x if isinstance(x, (int, float, str)) else str(x))
    print(f'  metrics -> {json_path}')

    print(f'\n{"="*64}')
    print(f'  Done. All outputs in: {SAVE_DIR}/')
    print(f'{"="*64}\n')
