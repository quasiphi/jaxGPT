from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn


# set default config
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1

class CasualSelfAttention(nn.Module):
    config: GPTConfig

    def setup(self) -> None:
        config = self.config

        assert config.n_embd % config.n_head == 0

        #head_size = config.n_embd // config.n_head

        self.c_attn = nn.Dense(config.n_embd * 3)
        self.c_proj = nn.Dense(config.n_embd) # output projection
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    # `train` is keyword-only
    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        B, T, C = x.shape # batch size, seq len, n_embd

        q, k, v = self.c_attn(x)._split(3, axis=-1)
        q = q.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2)
        k = k.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2)
        v = v.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2)

        mask = jnp.tril(jnp.ones((T, T))).reshape((1, 1, T, T))

        att = (q @ k.swapaxes(-2, -1)) * (1.0 / jnp.sqrt(k.shape[-1]))
        att = jnp.where(mask == 0, float('-inf'), att)
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic = not train)

        y = att @ v
        y = y.swapaxes(1, 2).reshape(B, T, C)
        y = self.resid_dropout(self.c_proj(y), deterministic = not train)
        return y


class MLP(nn.Module):
    config: GPTConfig

    def setup(self) -> None:
        config = self.config
        self.c_fc = nn.Dense(config.n_embd * 4)
        self.c_proj = nn.Dense(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        x = self.c_fc(x)
        x = nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        x = self.dropout(x, deterministic = not train)
        return x


class Block(nn.Module):
    config: GPTConfig

    def setup(self) -> None:
        config = self.config
        self.ln1 = nn.LayerNorm(epsilon=1e-5)
        self.attn = CasualSelfAttention(config)
        self.ln2 = nn.LayerNorm(epsilon=1e-5)
        self.mlp = MLP(config)

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        x = x + self.attn(self.ln1(x), train=train)
        x = x + self.mlp(self.ln2(x), train=train)
        return x
