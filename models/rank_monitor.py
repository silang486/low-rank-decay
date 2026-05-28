"""
Stable rank monitor for transformer QK matrices.

Stable rank:  r(W) = ||W||_F^2 / ||W||_2^2

This is a smooth, noise-robust proxy for the true matrix rank.
We track Q and K projection matrices across all attention layers,
matching the analysis in Figure 2 of the paper.
"""

import torch
import torch.nn as nn


def stable_rank(W: torch.Tensor) -> float:
    """Compute stable rank of a 2-D weight matrix."""
    W2d = W.view(W.shape[0], -1).detach().float()
    frob_sq = W2d.pow(2).sum().item()
    spec_sq = torch.linalg.norm(W2d, ord=2).pow(2).item()
    if frob_sq < 1e-16:
        return 0.0
    return frob_sq / (spec_sq + 1e-9)


def mean_qk_stable_rank(model: nn.Module) -> float:
    """
    Return the mean stable rank of all Q and K weight matrices
    across every attention layer in the model.
    """
    ranks = []
    for module in model.modules():
        if hasattr(module, 'q') and hasattr(module, 'k'):
            ranks.append(stable_rank(module.q.weight))
            ranks.append(stable_rank(module.k.weight))
    if not ranks:
        return float('nan')
    return sum(ranks) / len(ranks)


def mean_qk_weight_norm(model: nn.Module) -> float:
    """
    Return the mean Frobenius norm of all Q and K weight matrices.
    用于监控 ||W|| 是否趋近于 0（弹弓效应的前兆）。
    """
    norms = []
    for module in model.modules():
        if hasattr(module, 'q') and hasattr(module, 'k'):
            norms.append(module.q.weight.detach().norm('fro').item())
            norms.append(module.k.weight.detach().norm('fro').item())
    if not norms:
        return float('nan')
    return sum(norms) / len(norms)
