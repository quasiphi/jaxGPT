import math
from typing import Optional, Tuple

import optax
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import path_aware_map
from flax.core import freeze
from flax.training import train_state
from flax import traverse_util
import numpy as np

from jaxgpt.models.layers import Block, GPTConfig
from jaxgpt.utils.common import colored, colored_bool

SEED = 80085

class GPT(nn.Module):
    config: GPTConfig

    def setup(self) -> None:
        config = self.config

        assert config.vocab_size is not None
        assert config.block_size is not None

        self.wte = nn.Embed(config.vocab_size, config.n_embd)
        self.wpe = nn.Embed(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm()

    def __call__(
            self,
            idx: jax.Array,
            *,
            train: bool,
            targets: Optional[jax.Array] = None
        ) -> jax.Array:
    
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of lenght {T} when block size is only {self.config.block_size}"
        pos = jnp.arange(0, T, dtype=jnp.int32)[None]

        # embeddings
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb, deterministic = not train)
        for block in self.h:
            x = block(x, train = train)
        x = self.ln_f(x)

        logits = self.wte.attend(x)
        logits = jax.lax.clamp(-100.0, logits, 100.0)

        if targets is not None:
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        else:
            loss = None

        return logits, loss

    def crop_block_size(self, params, block_size: int):
        # in order to load pretrained gpt2 (block size 1024) 
        # and cut it's block size to make the model smaller

        assert block_size <= self.config.block_size
        self.config.block_size = block_size

        def crop_weights(path: Tuple[str, ...], x):
            if path[-2:] == ('wpe', 'embedding'): return x[:block_size]
            return x
        
        return freeze(path_aware_map(crop_weights, params))

    @classmethod
    def from_pretrained(cls, model_type, override_args: Optional[dict] = None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large'}

        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)

        from transformers import GPT2LMHeadModel
        print(f'loading weights from pretrained gpt: {model_type}')

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280)
        }[model_type]

        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']

        config = GPTConfig(block_size=1024, **config_args)
        model = GPT(config)
        variables = jax.eval_shape(
            lambda: model.init(
                jax.random.PRNGKey(SEED),
                jnp.ones((1, 1), dtype=jnp.int32),
                train = False,
            )
        )
        params = variables['params']
        flat_params = traverse_util.flatten_dict(params, sep='.')

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        def copy_from(flax_name, pt_name, transpose=False, add_head_dim=False, debug=True):
            if debug:
                print(f"copying {colored(pt_name, 'red')} -> {colored(flax_name, 'green')} :: {colored_bool(transpose, 'T')} :: {colored_bool(add_head_dim, 'H')}")
            pt_tensor = sd_hf[pt_name]
            jax_array = flat_params[flax_name]
            if transpose:
                pt_tensor = pt_tensor.T()
            pt_array = pt_tensor.detach().cpu().numpy()

            if add_head_dim:
                pass
            assert pt_array.shape == jax_array.shape

            flat_params[flax_name] = pt_array

        copy_from('wte.embedding', 'transformer.wte.weight')
        copy_from('wpe.embedding', 'transformer.wpe.weight')

        for i in range(config.n_layer):
            copy_from(f'h_{i}.ln_1.scale', f'transformer.h.{i}.ln_1.weight')
            copy_from(f'h_{i}.ln_1.bias', f'transformer.h.{i}.ln_1.bias')
            copy_from(f'h_{i}.attn.c_attn.kernel', f'transformer.h.{i}.attn.c_attn.weight', add_head_dim=True)
            copy_from(f'h_{i}.attn.c_attn.bias', f'transformer.h.{i}.attn.c_attn.bias', add_head_dim=True)
            copy_from(f'h_{i}.attn.c_proj.kernel', f'transformer.h.{i}.attn.c_proj.weight')
            copy_from(f'h_{i}.attn.c_proj.bias', f'transformer.h.{i}.attn.c_proj.bias')
            copy_from(f'h_{i}.ln_2.scale', f'transformer.h.{i}.ln_2.weight')
            copy_from(f'h_{i}.ln_2.bias', f'transformer.h.{i}.ln_2.bias')
            copy_from(f'h_{i}.mlp.c_fc.kernel', f'transformer.h.{i}.mlp.c_fc.weight')
            copy_from(f'h_{i}.mlp.c_fc.bias', f'transformer.h.{i}.mlp.c_fc.bias')
            copy_from(f'h_{i}.mlp.c_proj.kernel', f'transformer.h.{i}.mlp.c_proj.weight')
            copy_from(f'h_{i}.mlp.c_proj.bias', f'transformer.h.{i}.mlp.c_proj.bias')

        copy_from('ln_f.scale', 'transformer.ln_f.weight')
        copy_from('ln_f.bias', 'transformer.ln_f.bias')

        params = freeze(traverse_util.unflatten_dict(flat_params, sep='.'))
        return model, params

    def configure_optimizers(self, params, weight_decay, learning_rate, betas):
        def get_optim(decay):
            return optax.adamw(learning_rate=learning_rate, b1=betas[0], b2=betas[1], weight_decay=decay)
        
        def partition_fn(path: Tuple[str, ...], x) -> str:
            if path[-1] in ('bias', 'scale', 'embedding'):
                return 'no_decay'
            elif path[-1] == 'kernel':
                return 'decay'
            else: raise ValueError(f'Unrecognized parameter: {path}')

        partition_optims = {
            'decay': get_optim(weight_decay),
            'no_decay': get_optim(0.0)
        }
        param_partitions = freeze(path_aware_map(partition_fn, params))
        tx = optax.multi_transform(partition_optims, param_partitions)
        return tx

    def generate(self, key, params, input_tokens, max_new_tokens, temperature=1.0, top_k=None):
        B, T = input_tokens.shape
        padding = jnp.zeros((B, max_new_tokens), dtype=jnp.int32)
        tokens = jnp.concatenate([input_tokens, padding], axis=-1)
        indexes = jnp.arange(T, T+max_new_tokens)

        def scan_f(tokens, i):
            step_key = jax.random.fold_in(key, i)
            logits, _ = self.apply({'params': params}, tokens, train=False)
            logits = logits[:, i - 1, :] / temperature
            if top_k is not None:
                top_logits, top_tokens = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
                token_idx = jax.random.categorical(step_key, top_logits, axis=-1)
                next_token = jnp.take_along_axis(top_tokens, token_idx[:, None], axis=-1).squeeze(-1)
            else:
                next_token = jax.random.categorical(step_key, logits, axis=-1)
            tokens = tokens.at[:, i].set(next_token)
            return tokens, None

        tokens, _ = jax.lax.scan(scan_f, tokens, indexes)
        return tokens

    def create_train_state(
            self,
            learning_rate,
            weight_decay,
            beta1,
            beta2,
            decay_lr = None,
            warmup_iters = None,
            lr_decay_iters = None,
            min_lr = None,
            params = None,
            **kwargs
    ):
        if params is None:
            variables = self.init(jax.random.PRNGKey(SEED), jnp.ones((1,1), dtype=jnp.int32), train=False)
            params = variables['params']

        params = freeze(params)

        if decay_lr:
            assert warmup_iters is not None and lr_decay_iters is not None and min_lr is not None
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_iters,
                decay_steps=lr_decay_iters,
                end_value=min_lr
            )
        else:
            lr_schedule = learning_rate
        
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            self.configure_optimizers(
                params,
                weight_decay,
                lr_schedule,
                betas=(beta1, beta2)
            )
        )
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)
