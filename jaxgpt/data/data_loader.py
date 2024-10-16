import jax.numpy as jnp
from typing import Generator
import tiktoken

class DataLoader:
    def __init__(self, dataset, batch_size, seq_len):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.tokenizer = tiktoken.get_encoding('gpt2')

    def tokenize(self, text: str) -> jnp.ndarray:
        return jnp.array(self.tokenizer.encode_ordinary(text))

    def create_batches(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        n = len(self.dataset)
        train_data = self.dataset[:int(n*0.9)]
        val_data = self.dataset[int(n*0.9):]

        train_ids = self.tokenize(train_data)
        val_ids = self.tokenize(val_data)

        return train_ids, val_ids
