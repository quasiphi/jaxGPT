#!/usr/bin/env python3

import jax
import jax.numpy as jnp

from jaxgpt.models.gpt import GPT 
from jaxgpt.models.layers import GPTConfig

model_config = GPTConfig()
#model_config.vocab_size = 50257
#model_config.block_size = 1024

model = GPT(model_config)
idx = jnp.ones((2, 10), dtype=jnp.int32)
key = jax.random.PRNGKey(80085)
print(model.tabulate(key, idx, train = False, depth=2))