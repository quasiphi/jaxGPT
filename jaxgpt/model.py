import math
from typing import Type

import jax
import jax.lax as lax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
import optax

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

class GPT(nn.Module):
    config: CN

    @staticmethod
    def get_default_config() -> CN:
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def setup(self) -> None:
        assert self.config.vocab_size is not None
        assert self.config.block_size is not None
        self.block_size = self.config.block_size
        type_given = self.config.model_type is not None

        params_given = all(
            [
                self.config.n_layer is not None, self.config.n_head is not None, 
                self.config.n_embd is not None
            ]
        )

        assert type_given ^ params_given # xor

        if type_given:
            self.config.merge_from_dict({
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-macro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-mini':     dict(n_layer=3, n_head=3, n_embd=48),
            }[self.config.model_type])

        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # but used Dense instead of Linear 
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            kernel_init = nn.initializers.normal(stdev=0.02),
            use_bias=False
        )

        # transformer components
        self.wte = nn.Embed(
            self.config.vocab_size,
            self.config.n_embd,
            embedding_init = nn.initializers.normal(stddev=0.02)
        )
        self.wpe = nn.Embed(
            self.config.block_size,
            self.config.n_embd,
            embedding_init = nn.initializers.normal(stddev=0.02)
        )
        self.drop = nn.Dropout(self.config.embd_pdrop)
        self.h = [Block(self.config) for _ in range(self.config.n_layer)]
        self.ln_f = nn.LayerNorm(
            scale_init = nn.initializers.ones,
            bias_init = nn.initializers.zeros
        )

        # TODO: what about weight initialization?

    def transformer(self, x: jnp.Array) -> jnp.Array:
        x = self.wte(x)
        x = self.wpe(x)
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return x

    def _init_weights(self, module: nn.Module):
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, model_type: str) -> Type["GPT"]: 
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        from transformers import FlaxGPT2LMHeadModel # use flax directly :)

        config = cls.get_default_config()
        config.model_type = model_type
        # openai defined

        config.block_size = 1024
        model = GPT(config)
        model_params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 1), dtype=jnp.int32))["params"]

        hf_model = FlaxGPT2LMHeadModel.from_pretrained(model_type)
        hf_params = hf_model.params

        model_params = cls._transfer_weights(hf_params, model_params)
        return model, model_params

    @staticmethod
    def _transfer_weights(hf_params, model_params):
        """
        Helper function to map and transfer Hugging Face model parameters
        to our custom GPT model's parameter structure. This includes handling
        any necessary reshaping (like transposing Conv1D layers).
        """
        new_params = {}

        # Recursively copy over parameters, making adjustments as necessary
        for key, value in model_params.items():
            if key in hf_params:
                if 'Dense' in key and 'kernel' in hf_params[key]:
                    # Transpose the weights for Dense layers (same as Conv1D in PyTorch)
                    new_params[key] = hf_params[key].T # TODO: debug this transpose
                else:
                    new_params[key] = hf_params[key]
            else:
                # Handle any keys that might not match exactly
                raise ValueError(f"Missing key in Hugging Face params: {key}")

        return new_params


    def configure_optimizers(self, params):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Dense, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embed)

        param_dict = freeze(params)

        for path, param in param_dict.items():
            if path.endswith('bias') or isinstance(param, blacklist_weight_modules):
                no_decay.add(path)
            elif path.endswith('wieght') and isinstance(param, whitelist_weight_modules):
                decay.add(path)

        decay_params = {p: param_dict[p] for p in decay}
        no_decay_params = {p: param_dict[p] for p in no_decay}

        decay_schedule = optax.add_decayed_weights(
            decay_params, weight_decay=self.config.weight_decay
        )

        no_decay_schedule = optax.add_decayed_weights(
            no_decay_params, weight_decay=0.0
        )

        optim = optax.chain(
            decay_schedule,
            no_decay_schedule,
            optax.scale_by_adam(
                b1=self.config.betas[0],
                b2=self.config.betas[1],
            )
        )

        return optim



