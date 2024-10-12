#!/usr/bin/env python3
"""
Trains a shakespeare-level langauge model.
"""

import os
import sys

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np
from collections import defaultdict

from jaxgpt.model import GPT
from jaxgpt.trainer import Trainer
from jaxgpt.utils import set_seed, setup_logging, CfgNode as CN
from jaxgpt.tokenizer import Tokenizer  

def get_config() -> CN:
    C = CN()

    C.system = CN()
    C.system.seed = 80085
    C.system.work_dir = "./out/shakespeare"

    C.data = ShakespeareDataset.get_default_config()

    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-nano'
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # make it bigger to run training faster
    C.trainer.seed = C.system.seed

    return C

class ShakespeareDataset():
    @staticmethod
    def get_default_config() -> CN:
        C = CN()
        C.block_size = 128
        C.vocab_size = 8000  # Adjust as needed
        return C

    def __init__(self, config, data):
        self.config = config
        self.data = data
        
        # Initialize and train the custom tokenizer
        self.tokenizer = Tokenizer()

        self.vocab_size = self.config.vocab_size
        self.block_size = self.config.block_size
        self.data_size = sum(len(self.tokenizer.encode(line)) for line in data)

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return self.data_size - self.block_size

    def __getitem__(self, idx):
        # Find the appropriate line and position
        for line in self.data:
            encoded = self.tokenizer.encode(line)
            if idx < len(encoded) - self.block_size:
                break
            idx -= len(encoded) - self.block_size

        chunk = encoded[idx:idx + self.block_size + 1]
        x = jnp.array(chunk[:-1], dtype=jnp.int32)
        y = jnp.array(chunk[1:], dtype=jnp.int32)
        return x, y

def train():
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    fn = 'tinyshakespeare.txt'
    lines = open(fn, 'r').read().strip('\n').split('\n')
    dataset = ShakespeareDataset(config.data, lines)

    config.model.vocab_size = dataset.get_vocab_size()
    config.model.block_size = dataset.get_block_size()
    model = GPT(config.model)

    trainer = Trainer(config.trainer, model, dataset)

    # Define a callback function for training and generation
    def training_callback(trainer, step, loss, lr):
        if step % 10 == 0:
            print(f"Step {step}: train loss {loss:.5f}, learning rate {lr:.6f}")

        if step % 500 == 0:
            # Evaluate and generate text
            context = "O God, O God!"
            context_tokens = jnp.array(dataset.tokenizer.encode(context), dtype=jnp.int32)
            
            @jax.jit
            def generate_sample(params, context, max_new_tokens, temperature=1.0, top_k=10):
                def sample_top_k(logits, top_k):
                    top_logits, top_indices = jax.lax.top_k(logits, k=top_k)
                    return jax.random.categorical(jax.random.PRNGKey(int(jnp.sum(top_logits))), top_logits) + top_indices[0]

                def generate_token(params, x):
                    logits, _ = model.apply({'params': params}, x, train=True)
                    logits = logits[:, -1, :] / temperature
                    return sample_top_k(logits, top_k)

                generated = context
                for _ in range(max_new_tokens):
                    next_token = generate_token(params, generated)
                    generated = jnp.concatenate([generated, next_token[None]], axis=0)
                return generated

            generated_tokens = generate_sample(trainer.state.params, context_tokens[None, :], 500)
            completion = dataset.tokenizer.decode(generated_tokens[0].tolist())
            print(f"Generated text:\n{completion}\n")

            # Save the latest model
            print("Saving model...")
            ckpt_path = os.path.join(config.system.work_dir, "model.params")
            with open(ckpt_path, "wb") as f:
                f.write(jax.device_get(jax.tree_util.tree_map(lambda x: x.tobytes(), trainer.state.params)))

    trainer.set_callback('on_batch_end',training_callback)

    trainer.run()

if __name__ == "__main__":
    train()