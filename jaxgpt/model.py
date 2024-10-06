import math
from typing import Type

import jax
import jax.lax as lax
import jax.numpy as jnp
from flax import linen as nn

from jaxgpt.utils import CfgNode as CN

def masked_fill(a: jnp.Array, mask: jnp.Array, fill: jax.typing.DTypeLike) -> jnp.Array:
    """
    fill tensor a with value fill where mask is True
    """
    return lax.select(mask, a , lax.broadcast(fill, a.shape))

class CasualSelfAttention(nn.Module):
    config: CN

    def setup(self) -> None:
        assert self.config.n_embd % self.config.n_head == 0
        self.c_attn = nn.Dense(3 * self.config.n_embd) 
        self.c_proj = nn.Dense(self.config.n_embd)
        self.attn_dropout = nn.Dropout(self.config.attn_pdrop)
        self.resid_dropout = nn.Dropout(self.config.resid_pdrop)

        self.bias = jnp.tril(jnp.ones((self.config.block_size, self.config.block_size)))
        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd

    def __call__(self, x: jnp.Array) -> jnp.Array:
        assert x.ndim == 3
        B, T, C = x.shape

        #qkv = qkv.reshape(B, T, self.n_head, 3 * (C // self.n_head))
        #q, k, v = jnp.split(qkv, 3, axis=-1)

        q, k, v = jnp.split(self.c_attn(x), self.n_embd, axis=2)

        # (B, nh, T, hs)
        q = k.reshape(B, T, self.n_head, C // self.n_head).transpose(1,2)
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = k.reshape(B, T, self.n_head, C // self.n_head).transpose(1,2)
        print(q.shape, k.shape, v.shape)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.shape[-1]))
        att = masked_fill(att, self.bias[:, :, :T, :T], -jnp.inf)
        att = nn.softmax(att, axis=-1)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        print(att.shape, v.shape)
        y = att @ v
        print(y.shape)
        y = y.transpose(1, 2).reshape(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    config: CN

    def setup(self) -> None:
        self.ln1 = nn.LayerNorm()
        self.attn = CasualSelfAttention(self.config)
        self.ln2 = nn.LayerNorm()

        # MLP
        self.c_fc = nn.Dense(self.config.n_embd * 4)
        self.c_proj = nn.Dense(self.config.n_embd)
        self.dropout = nn.Dropout(self.config.resid_pdrop)

    def mlp(self, x: jnp.Array) -> jnp.Array:
        x = self.c_fc(x)
        x = nn.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)

    def __call__(self, x: jnp.Array) -> jnp.Array:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x