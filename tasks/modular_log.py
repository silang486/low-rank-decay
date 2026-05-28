import torch
import torch.nn.functional as F
from .base import Task
from utils import DEVICE


def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def _find_primitive_root(p):
    phi = p - 1
    factors = set()
    n = phi
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    for g in range(2, p):
        if all(pow(g, phi // q, p) != 1 for q in factors):
            return g
    raise ValueError(f"No primitive root for p={p}")


class ModularLog(Task):
    """
    二元输入版本：y = log_g(x1 * x2) mod p
    输入是 (x1, x2) 对，x1,x2 ∈ [1, p-1]，数据量 (p-1)^2。
    """

    def __init__(self, p: int = 97, train_size: int = None,
                 test_size: int = 1000, train_frac: float = 0.4):
        assert _is_prime(p)
        self.p           = p
        self.num_classes = p - 1
        self.g           = _find_primitive_root(p)
        total            = (p - 1) ** 2

        dlog = torch.zeros(p, dtype=torch.long)
        val  = 1
        for y in range(p - 1):
            dlog[val] = y
            val = (val * self.g) % p

        xs    = torch.arange(1, p)
        x1    = xs.repeat_interleave(p - 1)
        x2    = xs.repeat(p - 1)
        prod  = (x1 * x2) % p
        y_all = dlog[prod]
        x_all = torch.stack([x1, x2], dim=1)

        perm = torch.randperm(total)
        if train_size is None:
            train_size = int(total * train_frac)
        test_size = min(test_size, total - train_size)

        self.x_train = x_all[perm[:train_size]].to(DEVICE)
        self.y_train = y_all[perm[:train_size]].to(DEVICE)
        self.x_test  = x_all[perm[total - test_size:]].to(DEVICE)
        self.y_test  = y_all[perm[total - test_size:]].to(DEVICE)

    def make_model(self, **kwargs):
        from models.transformer import TinyTransformer
        return TinyTransformer(vocab=self.p, num_classes=self.num_classes, **kwargs)

    def sample_batch(self, batch_size: int = 256):
        idx = torch.randint(0, len(self.x_train), (batch_size,), device=DEVICE)
        return self.x_train[idx], self.y_train[idx]

    def loss(self, model, batch):
        x, y = batch
        return F.cross_entropy(model(x), y)

    def evaluate(self, model, batches=None):
        model.eval()
        with torch.no_grad():
            train_acc = (model(self.x_train).argmax(-1) == self.y_train).float().mean().item()
            test_acc  = (model(self.x_test ).argmax(-1) == self.y_test ).float().mean().item()
        model.train()
        return train_acc, test_acc
