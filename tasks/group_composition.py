import torch
import torch.nn.functional as F
from itertools import permutations
from .base import Task
from utils import DEVICE


def _build_s5_table():
    perms = list(permutations(range(5)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    n = len(perms)
    table = torch.zeros(n, n, dtype=torch.long)
    for i, p1 in enumerate(perms):
        for j, p2 in enumerate(perms):
            composed = tuple(p1[p2[k]] for k in range(5))
            table[i, j] = perm_to_idx[composed]
    return perms, table


class GroupComposition(Task):
    """
    S5 置换群的群运算任务。
    输入：(i, j)，两个群元素的索引
    输出：i ∘ j 的索引，数据量 120^2 = 14400
    """

    def __init__(self, n: int = 5, train_size: int = None,
                 test_size: int = 1000, train_frac: float = 0.4):
        self.n = n
        _, table = _build_s5_table()
        num_elements  = table.shape[0]
        self.num_elements = num_elements
        total         = num_elements ** 2

        i_all = torch.arange(num_elements).repeat_interleave(num_elements)
        j_all = torch.arange(num_elements).repeat(num_elements)
        y_all = table[i_all, j_all]
        x_all = torch.stack([i_all, j_all], dim=1)

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
        return TinyTransformer(
            vocab=self.num_elements,
            num_classes=self.num_elements,
            **kwargs
        )

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
