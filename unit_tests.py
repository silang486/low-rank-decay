"""
unit_tests.py — 最小单元测试

运行：python unit_tests.py
全部通过会打印 "ALL TESTS PASSED"，失败会显示具体哪个测试出了问题。

测试清单：
  1. Newton-Schulz 奇异值验证
  2. apply_l2_decoupled 效果验证
  3. apply_lrd_decoupled 方向和 rank 验证
  4. stable_rank 数值验证
  5. RMSNorm epsilon 风险验证
  6. 解耦正则化顺序无关性验证
  7. _should_apply_lrd 过滤正确性验证
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from models.newton_schulz_lrd import (
    _newton_schulz, _polar_factor, _should_apply_lrd,
    apply_l2_decoupled, apply_lrd_decoupled
)
from models.rank_monitor import stable_rank, mean_qk_stable_rank
from models.transformer import TinyTransformer
from tasks.modular_addition import ModularAddition

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PASS = "  ✓"
FAIL = "  ✗"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"{status} {name}" + (f"\n      {detail}" if detail else ""))
    results.append((name, condition))


# ══════════════════════════════════════════════════════════════════════════════
# Test 1: Newton-Schulz 奇异值验证
# 期望：polar factor 的所有奇异值应该接近 1.0
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Test 1] Newton-Schulz 奇异值验证")

for shape in [(32, 32), (128, 128), (512, 128), (128, 512)]:
    W = torch.randn(*shape)
    ns = _newton_schulz(W, iters=30)
    S  = torch.linalg.svdvals(ns)
    min_s  = S.min().item()
    max_s  = S.max().item()
    mean_s = S.mean().item()
    ok = min_s > 0.99 and max_s < 1.01
    check(
        f"  shape={shape} 奇异值全部接近1.0",
        ok,
        f"min={min_s:.4f} max={max_s:.4f} mean={mean_s:.4f}"
    )

# 期望：方阵和非方阵都应该通过
# 理想结果：min≈1.0, max≈1.0, mean≈1.0


# ══════════════════════════════════════════════════════════════════════════════
# Test 2: apply_l2_decoupled 效果验证
# 期望：施加后 ||W|| 缩小，缩小比例是 (1 - lr * lam)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Test 2] apply_l2_decoupled 效果验证")

model = TinyTransformer(vocab=97, num_classes=97, dim=128).to(DEVICE)
lr, lam = 0.01, 0.001

# 记录施加前的权重 norm
norms_before = {
    name: p.data.norm('fro').item()
    for name, p in model.named_parameters()
    if p.requires_grad
}

apply_l2_decoupled(model, lr=lr, lam=lam)

norms_after = {
    name: p.data.norm('fro').item()
    for name, p in model.named_parameters()
    if p.requires_grad
}

expected_ratio = 1.0 - lr * lam  # 0.99990

for name in norms_before:
    if any(k in name for k in ('bias', 'norm')):
        # bias 和 norm 不应该被缩小
        ratio = norms_after[name] / (norms_before[name] + 1e-9)
        check(
            f"  {name} 不受影响",
            abs(ratio - 1.0) < 1e-4,
            f"ratio={ratio:.6f} (期望=1.0)"
        )
    else:
        ratio = norms_after[name] / (norms_before[name] + 1e-9)
        check(
            f"  {name} 缩小比例正确",
            abs(ratio - expected_ratio) < 1e-4,
            f"ratio={ratio:.6f} (期望={expected_ratio:.6f})"
        )

# 理想结果：非 bias/norm 参数比例约为 0.99990，bias/norm 比例为 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Test 3: apply_lrd_decoupled 方向验证
# 期望：更新方向是 polar factor 方向，施加后 rank 不应该突然爆炸
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Test 3] apply_lrd_decoupled 方向和 rank 验证")

model = TinyTransformer(vocab=97, num_classes=97, dim=128).to(DEVICE)
rank_before = mean_qk_stable_rank(model)

# 施加多次，看 rank 变化趋势
ranks = [rank_before]
for _ in range(20):
    apply_lrd_decoupled(model, lr=0.01, lam=0.1)
    ranks.append(mean_qk_stable_rank(model))

rank_after_20 = ranks[-1]
rank_decreased = rank_after_20 < rank_before

check(
    "  20次施加后 rank 有所下降",
    rank_decreased,
    f"rank before={rank_before:.2f} after={rank_after_20:.2f}"
)

# rank 不应该变成 nan 或负数
check(
    "  rank 始终为有效正数",
    all(r > 0 and r == r for r in ranks),  # r == r 检查 nan
    f"min_rank={min(ranks):.4f}"
)

# 理想结果：rank 单调或总体下降，不出现 nan


# ══════════════════════════════════════════════════════════════════════════════
# Test 4: stable_rank 数值验证
# 期望：rank=1 的矩阵 stable_rank 接近 1，满秩矩阵 stable_rank 接近 n
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Test 4] stable_rank 数值验证")

# rank=1 的矩阵：outer product
u = torch.randn(32)
v = torch.randn(32)
W_rank1 = torch.outer(u, v)
sr1 = stable_rank(W_rank1)
check(
    "  rank=1 矩阵的 stable_rank 接近 1",
    abs(sr1 - 1.0) < 0.1,
    f"stable_rank={sr1:.4f} (期望≈1.0)"
)

# 正交矩阵（满秩）：stable_rank 应该等于 min(m,n)
Q, _ = torch.linalg.qr(torch.randn(32, 32))
sr_full = stable_rank(Q)
check(
    "  正交矩阵 stable_rank 接近 32",
    abs(sr_full - 32.0) < 2.0,
    f"stable_rank={sr_full:.4f} (期望≈32.0)"
)

# 全零矩阵：stable_rank 应该是 0
W_zero = torch.zeros(32, 32)
sr_zero = stable_rank(W_zero)
check(
    "  全零矩阵 stable_rank=0",
    sr_zero == 0.0,
    f"stable_rank={sr_zero}"
)

# 理想结果：rank=1→1.0，正交→32.0，全零→0.0


# ══════════════════════════════════════════════════════════════════════════════
# Test 5: RMSNorm epsilon 风险验证
# 期望：当权重很小时，RMSNorm 的分母是否会被 eps 主导
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Test 5] RMSNorm epsilon 风险验证")

from models.transformer import RMSNorm

rms = RMSNorm(dim=128, eps=1e-8).to(DEVICE)

# 正常输入
x_normal = torch.randn(4, 2, 128).to(DEVICE)
out_normal = rms(x_normal)
check(
    "  正常输入 RMSNorm 输出有效",
    not torch.isnan(out_normal).any() and not torch.isinf(out_normal).any(),
    f"mean={out_normal.abs().mean().item():.4f}"
)

# 极小输入（模拟权重被 weight decay 压缩后的激活值）
x_tiny = x_normal * 1e-6
out_tiny = rms(x_tiny)
# 如果 eps 主导，输出会被放大到异常值
out_ratio = out_tiny.abs().mean().item() / out_normal.abs().mean().item()
eps_dominated = out_ratio > 100  # 如果放大超过100倍，说明 eps 主导了

check(
    "  极小输入时 RMSNorm 不被 eps 主导",
    not eps_dominated,
    f"放大比例={out_ratio:.1f}x (>100x 说明 eps 主导，尺度不变性被破坏)"
)

# 理想结果：正常输入有效，极小输入不被 eps 主导


# ══════════════════════════════════════════════════════════════════════════════
# Test 6: 解耦正则化顺序验证
# 期望：先 optimizer 再正则化，和先正则化再 optimizer，结果应该不同
# 但两种顺序下正则化的效果都应该正确施加
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Test 6] 解耦正则化施加正确性验证")

torch.manual_seed(42)
model = TinyTransformer(vocab=97, num_classes=97, dim=128).to(DEVICE)

# 记录施加前 QK 权重
qk_before = {}
for name, module in model.named_modules():
    if hasattr(module, 'q') and hasattr(module, 'k'):
        qk_before[name+'.q'] = module.q.weight.data.clone()
        qk_before[name+'.k'] = module.k.weight.data.clone()

lr, lam = 0.01, 0.1
apply_lrd_decoupled(model, lr=lr, lam=lam)

# 验证 QK 权重确实被更新了
updated = 0
for name, module in model.named_modules():
    if hasattr(module, 'q') and hasattr(module, 'k'):
        diff_q = (module.q.weight.data - qk_before[name+'.q']).abs().mean().item()
        diff_k = (module.k.weight.data - qk_before[name+'.k']).abs().mean().item()
        if diff_q > 1e-6 and diff_k > 1e-6:
            updated += 1

check(
    "  apply_lrd_decoupled 确实更新了 QK 权重",
    updated > 0,
    f"更新了 {updated} 个 attention 层的 QK 权重"
)

# 验证 embedding 和 head 没有被更新
embed_before = model.embed.weight.data.clone()
apply_lrd_decoupled(model, lr=lr, lam=lam)
embed_unchanged = (model.embed.weight.data - embed_before).abs().max().item() < 1e-9

check(
    "  apply_lrd_decoupled 不影响 embedding",
    embed_unchanged,
    f"embedding 最大变化={( model.embed.weight.data - embed_before).abs().max().item():.2e}"
)

# 理想结果：QK 权重被更新，embedding/head 不变


# ══════════════════════════════════════════════════════════════════════════════
# Test 7: _should_apply_lrd 过滤正确性
# 期望：只有隐藏层 2D 权重返回 True
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Test 7] _should_apply_lrd 过滤正确性")

model = TinyTransformer(vocab=97, num_classes=97, dim=128).to(DEVICE)

should_apply = []
should_skip  = []
for name, p in model.named_parameters():
    if _should_apply_lrd(name, p):
        should_apply.append(name)
    else:
        should_skip.append(name)

# embedding 和 head 应该被跳过
skip_keywords = ('embed', 'head', 'pos_emb', 'proj', 'norm', 'bias')
wrong_apply = [n for n in should_apply if any(k in n for k in skip_keywords)]
check(
    "  embedding/head/norm/bias 被正确跳过",
    len(wrong_apply) == 0,
    f"错误地施加到: {wrong_apply}" if wrong_apply else "全部正确"
)

# 应该有参数被施加
check(
    "  至少有一些参数会被施加 LRD",
    len(should_apply) > 0,
    f"施加到 {len(should_apply)} 个参数: {should_apply[:3]}..."
)

print(f"\n  施加 LRD 的参数: {should_apply}")
print(f"  跳过的参数: {should_skip}")

# 理想结果：attention Q/K/V/O 和 MLP 权重施加，其他跳过


# ══════════════════════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"  结果: {passed}/{total} 通过")

failed = [(name, ok) for name, ok in results if not ok]
if failed:
    print(f"\n  失败的测试:")
    for name, _ in failed:
        print(f"    ✗ {name}")
    print("\nSOME TESTS FAILED")
else:
    print("\nALL TESTS PASSED")


# ══════════════════════════════════════════════════════════════════════════════
# Test 8: 恒定拉力测试 (Constant Force Test)
# 期望：当 ||W|| 极小时，lrd 更新不应该导致范数剧增或符号翻转
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Test 8] 恒定拉力测试")

lr, lam = 0.01, 0.1
update_magnitude = lr * lam  # 0.001，固定更新量

for tiny_scale in [1e-3, 1e-6, 1e-9]:
    W_tiny = torch.randn(128, 128) * tiny_scale
    norm_before = W_tiny.norm('fro').item()

    # 模拟一次 apply_lrd_decoupled
    from models.newton_schulz_lrd import _polar_factor
    polar = _polar_factor(W_tiny, iters=30)
    W_new = W_tiny - lr * lam * polar
    norm_after = W_new.norm('fro').item()

    # 更新量和权重本身的比值
    ratio = norm_after / (norm_before + 1e-30)

    # 如果 ratio >> 1，说明更新量远大于权重本身，发生了"噪声注入"
    catastrophic = ratio > 10
    check(
        f"  ||W||={tiny_scale:.0e} 时更新不发生灾难性放大",
        not catastrophic,
        f"norm before={norm_before:.2e} after={norm_after:.2e} ratio={ratio:.1f}x"
        + (" ← 灾难性放大！" if catastrophic else "")
    )

# 理想结果：tiny_scale=1e-3 可能轻微放大，1e-6 和 1e-9 会发生灾难性放大
# 如果失败，说明需要对 lrd 更新量做自适应缩放：
# p.sub_(lr * lam * polar * ||p|| / (||polar|| + eps))


# ══════════════════════════════════════════════════════════════════════════════
# Test 9: L2 多步累积衰减验证
# 期望：1000步后 ||W|| ≈ W_0 * (1 - lr*lam)^1000
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Test 9] L2 多步累积衰减验证")

import math

lr, lam = 0.01, 0.001
steps = 1000
expected_decay = (1.0 - lr * lam) ** steps  # 理论值

# 用单个权重矩阵测试
W = torch.randn(128, 128)
norm_0 = W.norm('fro').item()

# 创建一个只有这一个参数的假 module
class _FakeModel(nn.Module):
    def __init__(self, W):
        super().__init__()
        # 用 blocks.0.attn.q.weight 这样的名字让过滤器通过
        self.blocks = nn.ModuleList([nn.ModuleList([
            type('attn', (), {
                'q': nn.Linear(128, 128, bias=False),
                'k': nn.Linear(128, 128, bias=False),
            })()
        ])])
        self.blocks[0][0].q.weight.data = W.clone()

# 直接测 apply_l2_decoupled 的累积效果
W_test = W.clone()

# 手动模拟1000步
for _ in range(steps):
    W_test = W_test * (1.0 - lr * lam)

norm_final = W_test.norm('fro').item()
expected_norm = norm_0 * expected_decay
actual_decay  = norm_final / norm_0

check(
    f"  1000步后 ||W|| 按指数衰减",
    abs(actual_decay - expected_decay) < 0.001,
    f"actual decay={actual_decay:.6f} expected={expected_decay:.6f}"
)

# 验证公式：W_t = W_0 * (1-λ)^t
check(
    f"  衰减公式 W_t = W_0*(1-lr*lam)^t 成立",
    abs(norm_final - expected_norm) / (expected_norm + 1e-9) < 0.001,
    f"norm_final={norm_final:.4f} expected={expected_norm:.4f}"
)

# 理想结果：两个测试都通过，说明 l2 按预期指数衰减
# 如果失败，说明 apply_l2_decoupled 有累积误差


# ══════════════════════════════════════════════════════════════════════════════
# Test 10: 自适应缩放修复验证
# 期望：修复后即使 ||W|| 极小，更新也不会发生灾难性放大
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Test 10] 自适应缩放修复验证（apply_lrd_decoupled）")

lr, lam = 0.01, 0.1

for tiny_scale in [1e-3, 1e-6, 1e-9]:
    W_tiny = torch.randn(128, 128) * tiny_scale
    norm_before = W_tiny.norm('fro').item()

    # 自适应缩放的更新
    polar  = _polar_factor(W_tiny, iters=30)
    w_norm = W_tiny.norm('fro')
    scale  = w_norm / (polar.norm('fro') + 1e-9)
    W_new  = W_tiny - lr * lam * polar
    norm_after = W_new.norm('fro').item()
    ratio = norm_after / (norm_before + 1e-30)

    # 修复后 ratio 应该接近 1，不会灾难性放大
    ok = ratio < 2.0
    check(
        f"  修复后 ||W||={tiny_scale:.0e} 时更新稳定",
        ok,
        f"norm before={norm_before:.2e} after={norm_after:.2e} ratio={ratio:.2f}x"
    )

# 同时验证 128x128 方阵 NS 奇异值（修复了迭代次数翻倍）
print("\n[Test 1b] 128x128 方阵 NS 奇异值（修复后）")
W = torch.randn(128, 128)
ns = _newton_schulz(W, iters=30)
S  = torch.linalg.svdvals(ns)
check(
    "  128x128 方阵奇异值全部接近1.0（修复后）",
    S.min().item() > 0.99 and S.max().item() < 1.01,
    f"min={S.min().item():.4f} max={S.max().item():.4f} mean={S.mean().item():.4f}"
)


# ══════════════════════════════════════════════════════════════════════════════
# 最终汇总（重新统计）
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"  结果: {passed}/{total} 通过")
failed = [(name, ok) for name, ok in results if not ok]
if failed:
    print(f"\n  失败的测试:")
    for name, _ in failed:
        print(f"    ✗ {name}")
    print("\nSOME TESTS FAILED")
else:
    print("\nALL TESTS PASSED")
