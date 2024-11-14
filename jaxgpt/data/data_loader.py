import os
import zipfile
import platform
from pathlib import Path
from typing import Generator, Optional

import tiktoken
import jax.numpy as jnp
import jax.random as jrandom

from jaxgpt.utils.common import fetch

OSX = platform.system() == 'Darwin'
CACHE_DIR = (
    os.path.expanduser('~/Library/Caches') if OSX else os.path.expanduser('~/.cache')
)

DATASETS = {
    "enron-corpus": "https://tile.loc.gov/storage-services/master/gdc/gdcdatasets/2018487913/2018487913.zip",
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}

class DataLoader:
    """
    Custom data loader for this llm usage.
    """

    def __init__(self, source_dict: dict[str, str], batch_size: int, seq_len: int, split: str, seed: int = 0):
        self.source_dict = source_dict 
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.split = split
        self.rng_key = jrandom.PRNGKey(seed)

        self.tokenizer = tiktoken.get_encoding('gpt2')

        self.train_data, self.test_data = self.prepare()


    # TODO handle zip files
    def prepare(self) -> tuple[Path, Path]:
        out_dir = Path(CACHE_DIR) / 'jaxgpt' 
        out_dir.mkdir(parents=True, exist_ok=True)

        test_data = []
        train_data = []

        for name, url in self.source_dict.items():
            if url.startswith(('http', 'https')):
                file = fetch(url)
            else:
                file = Path(url)
                if not file.exists():
                    raise FileNotFoundError(f'Source at {file} does not exist')

            if zipfile.is_zipfile(file):
                extract_dir = out_dir / name
                extract_dir.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(file, 'r') as zref:
                    zref.extractall(extract_dir)

                for extracted_file in extract_dir:
                    with open(extracted_file, 'r') as f:
                        file_data = f.read()
                    n = len(file_data)

                    train_data.append(file_data[:int(.9*n)])
                    test_data.append(file_data[int(.9*n):])

            else:
                with open(file, 'r') as f:
                    file_data = f.read()
                n = len(file_data)

                train_data.append(file_data[:int(.9*n)])
                test_data.append(file_data[int(.9*n):])

        # not sure if thats a best way to join datasets
        train_data = ' '.join(train_data)
        test_data = ' '.join(test_data)

        train_ids = self.tokenizer.encode_ordinary(train_data)
        test_ids = self.tokenizer.encode_ordinary(test_data)

        train_ids = jnp.array(train_ids, dtype=jnp.uint16)
        test_ids = jnp.array(test_ids, dtype=jnp.uint16)

        train_out = out_dir / 'train.bin'
        test_out = out_dir / 'test.bin'

        train_ids.tofile(train_out)
        test_ids.tofile(test_out)

        print(f"saved train and test data to {out_dir}")
        print(f"train @ {train_out}")
        print(f"test @ {test_out}")

        return train_ids, test_ids

    def __iter__(self) -> Generator[tuple[jnp.array, jnp.array], None, None]:
        data = self.train_data if self.split == 'train' else self.test_data
        while True:
            self.rng_key, subkey = jrandom.split(self.rng_key)
            indicies = jrandom.randint(subkey, (self.batch_size,), 0, len(data) - self.seq_len - 1)
            x = jnp.stack([data[i : i + self.seq_len].astype(jnp.int32) for i in indicies])
            y = jnp.stack([data[i + 1 : i + self.seq_len + 1].astype(jnp.int32) for i in indicies])

            yield x, y

