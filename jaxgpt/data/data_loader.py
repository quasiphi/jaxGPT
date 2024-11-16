import os
import zipfile
import platform
from pathlib import Path
from typing import Generator, Optional

import tiktoken
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from jaxgpt.utils.common import fetch, colored

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

    TRAIN_SPLIT = 0.75
    VAL_SPLIT = 0.20
    TEST_SPLIT = 0.05

    def __init__(self, source_dict: dict[str, str], batch_size: int, seq_len: int, split: str, seed: int = 0):
        self.source_dict = source_dict 
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.split = split
        self.rng_key = jrandom.PRNGKey(seed)

        self.tokenizer = tiktoken.get_encoding('gpt2')

        self.train_data, self.val_data, self.test_data = self.prepare()


    # TODO handle zip files
    def prepare(self) -> tuple[Path, Path]:
        out_dir = Path(CACHE_DIR) / 'jaxgpt' 
        out_dir.mkdir(parents=True, exist_ok=True)

        test_data = []
        val_data = []
        train_data = []

        for name, url in self.source_dict.items():
            print(f"preparing {name} from {url}")
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
                    train_end = int(self.TRAIN_SPLIT * n)
                    val_end = int((self.TRAIN_SPLIT + self.VAL_SPLIT) * n)

                    train_data.append(file_data[:train_end])
                    val_data.append(file_data[train_end:val_end])
                    test_data.append(file_data[val_end:])

            else:
                with open(file, 'r') as f:
                    file_data = f.read()

                n = len(file_data)
                train_end = int(self.TRAIN_SPLIT * n)
                val_end = int((self.TRAIN_SPLIT + self.VAL_SPLIT) * n)

                train_data.append(file_data[:train_end])
                val_data.append(file_data[train_end:val_end])
                test_data.append(file_data[val_end:])

        # not sure if thats a best way to join datasets
        train_data = ' '.join(train_data)
        val_data = ' '.join(val_data)
        test_data = ' '.join(test_data)

        train_ids = self.tokenizer.encode_ordinary(train_data)
        val_ids = self.tokenizer.encode_ordinary(val_data)
        test_ids = self.tokenizer.encode_ordinary(test_data)

        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtpye=np.uint16)
        test_ids = np.array(test_ids, dtype=np.uint16)

        train_out = out_dir / 'train.bin'
        val_out = out_dir / 'val.bin'
        test_out = out_dir / 'test.bin'

        train_ids.tofile(train_out)
        val_ids.tofile(val_out)
        test_ids.tofile(test_out)

        print(f"saved train and test data to {out_dir}")
        print(f"train @ {train_out}")
        print(f"val @ {val_out}")
        print(f"test @ {test_out}")

        return (
            jnp.array(train_ids), 
            jnp.array(val_ids),
            jnp.array(test_ids)
        )

    def __iter__(self) -> Generator[tuple[jnp.array, jnp.array], None, None]:
        data = {
            'train': self.train_data,
            'val': self.val_data,
            'test': self.test_data,
        }.get(self.split)

        if data is None:
            raise ValueError(f"invalid split '{self.split}'. Must be one of train, val, test")

        while True:
            self.rng_key, subkey = jrandom.split(self.rng_key)
            indicies = jrandom.randint(subkey, (self.batch_size,), 0, len(data) - self.seq_len - 1)
            x = jnp.stack([data[i : i + self.seq_len].astype(jnp.int32) for i in indicies])
            y = jnp.stack([data[i + 1 : i + self.seq_len + 1].astype(jnp.int32) for i in indicies])

            yield x, y

