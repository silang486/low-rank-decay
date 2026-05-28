import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class SwiGLU(nn.Module):

    def __init__(self, dim):
        super().__init__()
        hidden = 4 * dim
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):

    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn  = Attention(dim, heads)
        self.mlp   = SwiGLU(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyTransformer(nn.Module):
    """
    Token-based transformer.
    输入 x: (B, T) long tensor of token indices.

    结构：
    - token embedding
    - 可学习位置编码
    - N 个 transformer block
    - RMSNorm
    - 取最后一个位置的输出接分类头（无 pooling）
    """

    def __init__(self, vocab, dim=128, heads=4, layers=2, num_classes=None,
                 max_seq_len=16):
        """
        vocab       : embedding table size
        num_classes : 输出类别数，默认等于 vocab
        max_seq_len : 位置编码支持的最大序列长度，默认 16 够用
        注意：head 输入维度是 max_seq_len * dim（flatten 所有位置）
        """
        super().__init__()

        self.dim         = dim
        self.max_seq_len = max_seq_len

        self.embed   = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.blocks = nn.ModuleList([
            Block(dim, heads) for _ in range(layers)
        ])

        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, num_classes if num_classes is not None else vocab)

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)

        x = self.embed(x) + self.pos_emb(pos)   # (B, T, D)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)                         # (B, T, D)

        return self.head(x[:, -1, :])            # (B, num_classes)


class LinearProjectTransformer(nn.Module):
    """
    Continuous-input transformer.
    输入 x: (B, T) float tensor（例如 Parity 任务的 bit 序列）。

    结构：
    - 线性投影替代 embedding
    - 可学习位置编码
    - N 个 transformer block
    - RMSNorm
    - 取最后一个位置的输出接分类头（无 pooling）
    """

    def __init__(self, input_dim, num_classes, dim=128, heads=4, layers=4):
        super().__init__()

        self.dim       = dim
        self.input_dim = input_dim

        self.proj    = nn.Linear(1, dim, bias=False)
        self.pos_emb = nn.Embedding(input_dim, dim)

        self.blocks = nn.ModuleList([
            Block(dim, heads) for _ in range(layers)
        ])

        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)

        x = self.proj(x.unsqueeze(-1)) + self.pos_emb(pos)   # (B, T, D)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, -1, :])
