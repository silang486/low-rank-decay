import torch
import torch.nn.functional as F
from .base import Task
from utils import DEVICE


class Parity(Task):

    def __init__(self, dim: int = 32, train_size: int = None,
                 test_size: int = 1000, train_frac: float = 0.4,
                 pool_size: int = 50_000):
        self.dim         = dim
        self.num_classes = 2

        oversample = 3
        x_raw = torch.randint(0, 2, (pool_size * oversample, dim))

        if dim <= 64:
            powers = (2 ** torch.arange(dim)).long()
            keys   = (x_raw.long() * powers).sum(dim=1)
            seen, keep = {}, []
            for i, k in enumerate(keys.tolist()):
                if k not in seen:
                    seen[k] = True
                    keep.append(i)
                    if len(keep) >= pool_size:
                        break
            x_dedup = x_raw[torch.tensor(keep)]
        else:
            x_dedup = x_raw[:pool_size]

        x_dedup = x_dedup.float()
        y_dedup = (x_dedup.sum(dim=1) % 2).long()
        total   = len(x_dedup)

        perm = torch.randperm(total)
        if train_size is None:
            train_size = int(total * train_frac)
        test_size = min(test_size, total - train_size)

        self.x_train = x_dedup[perm[:train_size]].to(DEVICE)
        self.y_train = y_dedup[perm[:train_size]].to(DEVICE)
        self.x_test  = x_dedup[perm[total - test_size:]].to(DEVICE)
        self.y_test  = y_dedup[perm[total - test_size:]].to(DEVICE)

    def make_model(self, **kwargs):
        from models.transformer import LinearProjectTransformer
        return LinearProjectTransformer(
            input_dim=self.dim, num_classes=self.num_classes, **kwargs)

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
