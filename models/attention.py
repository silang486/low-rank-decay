import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, dim, heads):

        super().__init__()

        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x):

        B,T,D = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(B,T,self.heads,self.head_dim)
        k = k.view(B,T,self.heads,self.head_dim)
        v = v.view(B,T,self.heads,self.head_dim)

        # QK norm
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        att = torch.einsum("bthd,bshd->bhts", q, k) * self.scale

        att = torch.softmax(att, dim=-1)

        out = torch.einsum("bhts,bshd->bthd", att, v)

        out = out.reshape(B,T,D)

        return self.o(out)
