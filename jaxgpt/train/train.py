#!/usr/bin/env python3

import sys
from pathlib import Path
from ast import literal_eval
from pprint import pprint

import jax
import jax.numpy as jnp
import jax.random as random
from flax.training import train_state
from optax import adamw, softmax_cross_entropy

from jaxgpt.models.gpt import GPT, GPTConfig

out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
always_save_checkpoint = True
init_from = 'scratch' # or 'resume' or 'gpt2' for pretrained

# data
dataset = 'shakespeare'
batch_size = 12
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0

# optim
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95

decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000 # set it ~ max_iters
min_lr = 6e-5

# i think these are unused
device = 'cpu'
dtype = 'bfloat16'

max_new_tokens = 100
temperature = 0.8
top_k = 200

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
print(config_keys)

# check if the config has been supplied
for arg in sys.argv[1:]:
    if '=' not in arg:
        continue
    kv = arg.split('=')
    assert len(kv) == 2, f"expecting each override arg to be of form --arg=value, got {arg}"
    k, v = kv
    assert k[:2] == '--'
    k = k[2:]
    if k == 'config':
        config_file = Path(v)
        if not config_file.exists():
            print(f'Config file {v} not found')
            continue
        config = {} 
        with open(config_file, 'r') as f:
            for line in f:
                key, value = line.strip().split('=')
                key = key.strip()
                value = value.strip()
                print(key, value)
                if key in config_keys:
                    config[key] = eval(value)
        globals().update(config)

config = {k: globals()[k] for k in config_keys}
pprint(config)