"""Small decoder-only Transformer. Each token is one UTF-8 byte (0..255)."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = [t.view(batch, length, self.heads, width // self.heads)
                   .transpose(1, 2) for t in (q, k, v)]
        # Causal attention prevents the model from looking at future answers.
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attention = attention.transpose(1, 2).contiguous().view(batch, length, width)
        x = x + self.projection(attention)
        return x + self.mlp(self.norm2(x))


class TinyLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context_length = config['context_length']
        width = config['width']
        if width % config['heads']:
            raise ValueError('width must be divisible by heads')
        self.tokens = nn.Embedding(256, width)
        self.positions = nn.Embedding(self.context_length, width)
        self.blocks = nn.Sequential(*[Block(width, config['heads'])
                                      for _ in range(config['layers'])])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 256, bias=False)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.tokens(tokens) + self.positions(positions)
        return self.output(self.norm(self.blocks(x)))


def choose_device(requested='auto'):
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
