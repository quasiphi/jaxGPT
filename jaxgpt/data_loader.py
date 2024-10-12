import jax.numpy as jnp
from typing import Generator
from transformers import GPT2Tokenizer


class DataLoader:
    def __init__(self, dataset, batch_size, seq_len):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def tokenize(self, text: str):
        return self.tokenizer(text, return_tensor='jax', padding=True, truncation=True)['input_ids']

    def create_batches(self) -> Generator[jnp.ndarray]:
        n = len(self.dataset) // self.batch_size
        for i in range(n):
            batch = self.dataset[i*self.batch_size:(i+1)*self.batch_size]
            tokenized_batch = jnp.array([self.tokenize(text) for text in batch])
            yield tokenized_batch
