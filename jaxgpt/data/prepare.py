#!/usr/bin/env python3

import os
import sys
import argparse
import tiktoken
import numpy as np
from pathlib import Path
from typing import Optional

from jaxgpt.utils.common import colored, fetch, valid_dir

'''
Usage:

./prepare.py --file data/shakespeare.txt --out_dir data/shakespeare 
./prepare.py --file https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt --out_dir data/shakespeare

you can supply --file flag with either a local file or a URL.
'''

parser = argparse.ArgumentParser(description='Prepare data for training GPT')
parser.add_argument('--file', type=str, required=True, help='Path to the file or URL to download the data')
parser.add_argument('--out_dir', type=str, required=True, help='Directory to save the processed data')

def handle_args(args: list):
    return parser.parse_args(args) 

def main():
    args = handle_args(sys.argv[1:])
    if args.file.startswith(('http', 'https')):
        file = fetch(args.file)
    else:
        file = Path(args.file)
        if not file.exists():
            raise FileNotFoundError(f'File {file} does not exist')

    out_dir = Path(args.out_dir)
    valid_dir(out_dir)

    with open(file, 'r') as f:
        data = f.read()
    n = len(data)

    train_data = data[:int(.9*n)]
    test_data = data[int(.9*n):]

    enc = tiktoken.get_encoding('gpt2')

    train_ids = enc.encode_ordinary(train_data)
    test_ids = enc.encode_ordinary(test_data)

    print(f"train has {len(train_ids)} tokens, test has {len(test_ids)} tokens")

    train_ids = np.array(train_ids, dtype=np.uint16)
    test_ids = np.array(test_ids, dtype=np.uint16)

    train_ids.tofile(out_dir / 'train.bin')
    test_ids.tofile(out_dir / 'test.bin')

    print(f"saved train and test data to {out_dir}")

if __name__ == "__main__":
    main()