import torch


def _orthogonalize_newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    original_shape = G.shape
    X = G.reshape(G.shape[0], -1).float()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T

    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        X = 1.5 * X - 0.5 * A @ X

    if transposed:
        X = X.T
    return X.reshape(original_shape).to(dtype=G.dtype)


class Muon(torch.optim.Optimizer):
    """Small local Muon optimizer fallback for 2D+ tensor parameters."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                if g.ndim < 2:
                    p.add_(g, alpha=-lr)
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = _orthogonalize_newton_schulz(update, steps=ns_steps)
                p.add_(update, alpha=-lr)

        return loss


def get_muon_cls():
    return getattr(torch.optim, "Muon", Muon)
