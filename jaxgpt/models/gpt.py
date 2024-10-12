import jax.numpy as jnp
from flax import linen as nn

from jaxgpt.models.layers import EncoderBlock

class GPT(nn.Module):
    num_layers: int
    num_heads: int
    d_model: int
    d_ff: int
    vocab_size: int
    max_seq_len: int
    dropout_rate: float = 0.1

    def setup(self):
        self.token_embedding = nn.Embed(self.vocab_size, self.d_model)
        self.position_embedding = nn.Embed(self.max_seq_len, self.d_model)
        self.transformer = [EncoderBlock(
            input_dim=self.d_model,
            num_heads=self.num_heads,
            dim_feedforward=self.d_ff,
            dropout_prob=self.dropout_rate
        ) for _ in range(self.num_layers)]
        self.layernorm = nn.LayerNorm()
        self.lm_head = nn.Dense(self.vocab_size, use_bias=False)

    def __call__(self, input_ids, mask=None, train=True):
        token_embeddings = self.token_embedding(input_ids)
        position_ids = jnp.arange(input_ids.shape[1])[None, :]
        position_embeddings = self.position_embedding(position_ids)
        x = token_embeddings + position_embeddings
        for block in self.transformer:
            x = block(x, mask=mask, train=train)
        x = self.layernorm(x)
        logits = self.lm_head(x)
        return logits
