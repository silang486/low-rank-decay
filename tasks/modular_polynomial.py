import torch
import torch.nn.functional as F
from .base import Task
from utils import DEVICE


class ModularPolynomial(Task):
    """
    二元输入版本：y = (a*x1^2 + b*x2 + c) mod p
    输入是 (x1, x2) 对，数据量 p^2，和加法乘法任务一致。
    """

    def __init__(self, p: int = 97, train_size: int = None,
                 test_size: int = 1000, train_frac: float = 0.4):
        self.p = p
        self.a = torch.randint(1, p, (1,)).item()
        self.b = torch.randint(1, p, (1,)).item()
        self.c = torch.randint(1, p, (1,)).item()
        total  = p * p

        x1    = torch.arange(p).repeat_interleave(p)
        x2    = torch.arange(p).repeat(p)
        y     = (self.a * x1 * x1 + self.b * x2 + self.c) % p
        x_all = torch.stack([x1, x2], dim=1)

        perm = torch.randperm(total)
        if train_size is None:
            train_size = int(total * train_frac)
        test_size = min(test_size, total - train_size)

        self.x_train = x_all[perm[:train_size]].to(DEVICE)
        self.y_train = y    [perm[:train_size]].to(DEVICE)
        self.x_test  = x_all[perm[total - test_size:]].to(DEVICE)
        self.y_test  = y    [perm[total - test_size:]].to(DEVICE)

    def make_model(self, **kwargs):
        from models.transformer import TinyTransformer
        return TinyTransformer(vocab=self.p, num_classes=self.p, **kwargs)

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
