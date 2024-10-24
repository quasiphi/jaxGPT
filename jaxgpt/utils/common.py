import os
import sys
import time
import json
import random
import pathlib
import hashlib
import tempfile
import platform
import functools
import contextlib
import urllib.request
from tqdm import tqdm
from ast import literal_eval
from typing import Optional, Type

import numpy as np 
import jax
import jax.numpy as jnp

OSX = platform.system() == 'Darwin'
CACHE_DIR = (
    os.path.expanduser('~/Library/Caches') if OSX else os.path.expanduser('~/.cache')
)

KeyArray = Type[jax.Array]


def colored(st: str, color: Optional[str] = None, background: int | bool = False) -> str:
    if color is not None:
        return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m"
    else: return st

def colored_bool(b: bool, st: Optional[str] = None) -> str:
    if st: return colored(st, 'green' if b else 'red')
    return colored(str(b), 'green' if b else 'red')

class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix: str = "", on_exit: Optional[callable] = None, enabled: bool = True) -> None:
        self.prefix = prefix
        self.on_exit = on_exit
        self.enabled = enabled

    def __enter__(self) -> None:
        self.st = time.perf_counter_ns()

    def __exit__(self, *exc):
        et = time.perf_counter_ns()
        self.t = et - self.st
        if self.enabled:
            print(f'{self.prefix}{self.t*1e-6:6.2} ms'+(self.on_exit(self.t) if self.on_exit else ''))

def set_seed(seed: int) -> KeyArray:
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.key(seed)
    return key

def setup_logging(config):
    work_dir = config.system.work_dir
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))

class CfgNode:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        return self._str_helper(0)

    # can't this be replaced with pprint?
    def _str_helper(self, indent: int) -> str:
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append(f"{k}:\n")
                parts.append(v._str_helper(indent+1))
            else:
                parts.append(f"{k}: {v}\n")
        parts = [" " * (indent*4) + p for p in parts]
        return "".join(parts)

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, CfgNode) else v for k,v in self.__dict__.items()}

    def merge_from_dict(self, d: dict):
        self.__dict__.update(d)

    def merge_from_args(self, args: list[str]):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:
            keyval = arg.split("=")
            assert len(keyval) == 2, f"expecting each override arg to be of form --arg=value, got {arg}"
            key, val = keyval

            try: val = literal_eval(val)
            except ValueError: pass

            assert key[:2] == "--"
            key = key[2:]
            keys = key.split(".")
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"
            print(f"command line overwriting config attribute {key} with {val}")
            setattr(obj, leaf_key, val)


IS_RUNNING = False
def print_compiling(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        global IS_RUNNING
        revert = False
        try:
            if not IS_RUNNING:
                print(f'compiling {colored(f.__name__, "yellow")}')
                IS_RUNNING = True
                revert = True
            return f(*args, **kwargs)
        finally:
            if revert:
                IS_RUNNING = False
    return wrapper

def valid_dir(fn: str | pathlib.Path):
    if not isinstance(fn, pathlib.Path):
        fn = pathlib.Path(fn)

    if not fn.exists():
        fn.mkdir(parents=True)

def getenv(key: str, default=0): return type(default)(os.getenv(key, default))

# TODO: add signature matching 
def fetch(url: str, name: Optional[str]=None, allow_cache=(not getenv('DISABLE_HTTP_CACHE'))):
    if url.startswith(('/', '.')): return pathlib.Path(url)
    fp = None
    if name is not None and (isinstance(name, pathlib.Path) or '/' in name):
        fp = pathlib.Path(name)
    else:
        if name: fn = name
        else: fn = hashlib.md5(url.encode('utf-8')).hexdigest()
        fp = pathlib.Path(CACHE_DIR) / 'jaxgpt' / 'downloads' / fn
    if not fp.is_file() or not allow_cache:
        with urllib.request.urlopen(url, timeout=10) as r:
            assert r.status == 200
            print(f'saving from {r} to tmp file at {fp}')
            total_bytes = int(r.headers.get('Content-Length', 0))  
            progress_bar = tqdm(total=total_bytes, unit='B', unit_scale=True, desc=url)
            (path := fp.parent).mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path, delete=False) as f:
                while chunk := r.read(16384):
                    progress_bar.update(f.write(chunk))
                f.close()
                if (file_size := os.stat(f.name).st_size) != total_bytes:
                    raise RuntimeError(f"fetch incomplete, file size mismatch: {file_size} < {total_bytes}")
                pathlib.Path(f.name).rename(fp)
    else:
        print(f'fetching from a cached file at {fp}')
    return fp