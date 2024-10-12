import math
import jax.numpy as jnp
from flax import linen as nn

# (batch, seq, head, dims)
BSHD = (0, 2, 1, 3)

def expand_mask(mask: jnp.ndarray) -> jnp.ndarray:
    assert mask.ndim >= 2
    if mask.ndim == 3:
        mask = mask.expand_dims(axis=1)
    while mask.ndim < 4:
        mask = mask.expand_dims(axis=0)
    return mask

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention


class MultiHeadAttention(nn.Module):
    num_heads: int
    d_model: int

    def setup(self):
        self.qkv_proj = nn.Dense(
            3 * self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros
        )
        self.out_proj = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros
        )
        #self.head_dim = self.d_model // self.num_heads

    def __call__(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        if mask is not None:
            mask = expand_mask(mask)

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, -1)
        qkv = qkv.transpose(BSHD)
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.transpose(BSHD)
        values = values.reshape(batch_size, seq_len, embed_dim)
        output = self.out_proj(values)
        return output, attention


class EncoderBlock(nn.Module):
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float

    def setup(self):
        # deterministic parameters on dropouts are set in the __call__ method
        self.self_attn = MultiHeadAttention(
            num_heads=self.num_heads,
            d_model=self.input_dim
        )

        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(rate=self.dropout_prob),
            nn.relu,
            nn.Dense(self.input_dim)
        ]

        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(rate=self.dropout_prob)

    def __call__(self, x, mask=None, train=True):
        attn_out, _ = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm1(x)
        lin_out = x
        for l in self.linear:
            if not isinstance(l, nn.Dropout): lin_out = l(lin_out)
            else: lin_out = l(lin_out, deterministic=not train)
        x = x + self.dropout(lin_out, deterministic=not train)
        x = self.norm2(x)
        return x