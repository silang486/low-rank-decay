import torch
import torch.nn as nn


_SKIP_KEYWORDS = ("embed", "head", "pos_emb", "proj", "norm", "bias")
_L2_SKIP_KEYWORDS = ("bias", "norm")


def _should_apply_lrd(name: str, p: torch.Tensor) -> bool:
    if not p.requires_grad:
        return False
    if p.ndim < 2:
        return False
    if any(k in name for k in _SKIP_KEYWORDS):
        return False
    return True


def _newton_schulz(W: torch.Tensor, iters: int = 30, eps: float = 1e-6) -> torch.Tensor:
    """
    Approximate the polar factor with Newton-Schulz iterations.
    K=30 is the default because smaller values were not reliable enough here.
    """
    X = W / (W.norm() + eps)
    I = torch.eye(X.shape[1], device=W.device, dtype=W.dtype)
    for _ in range(iters):
        A = X.T @ X
        X = 0.5 * (X @ (3.0 * I - A))
    return X


def _polar_factor(p: torch.Tensor, iters: int = 30) -> torch.Tensor:
    W = p.view(p.shape[0], -1)
    return _newton_schulz(W, iters=iters).view(p.shape)


def lrd_penalty(model: nn.Module, iters: int = 30) -> torch.Tensor:
    device = next(model.parameters()).device
    total = torch.zeros(1, device=device)
    for name, p in model.named_parameters():
        if not _should_apply_lrd(name, p):
            continue
        polar = _polar_factor(p, iters=iters)
        total = total + (p * polar).sum()
    return total


@torch.no_grad()
def apply_lrd_decoupled(model: nn.Module, lr: float, lam: float, iters: int = 30):
    """Decoupled Low-Rank Decay: W -= lr * lam * polar(W)."""
    for name, p in model.named_parameters():
        if not _should_apply_lrd(name, p):
            continue
        polar = _polar_factor(p, iters=iters)
        p.sub_(lr * lam * polar)


@torch.no_grad()
def apply_l2_decoupled(model: nn.Module, lr: float, lam: float):
    """Decoupled L2 weight decay: W *= (1 - lr * lam), skipping bias/norm."""
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in _L2_SKIP_KEYWORDS):
            continue
        p.mul_(1.0 - lr * lam)


def lrd_grad_stats(model: nn.Module, iters: int = 30) -> dict:
    stats = {}
    for name, p in model.named_parameters():
        if not _should_apply_lrd(name, p):
            continue
        polar = _polar_factor(p.detach(), iters=iters)
        stats[name] = polar.abs().mean().item()
    return stats
