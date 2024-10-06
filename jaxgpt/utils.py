import os
import sys
import time
import json
import random
import contextlib
from typing import Optional, Type
from ast import literal_eval

import numpy as np 
import jax
import jax.numpy as jnp

KeyArray = Type[jnp.Array]

def colored(st: str, color: Optional[str] = None, background: int | bool = False) -> str:
    if color is not None:
        return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m"
    else: return st

class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix: str = "", on_exit: Optional[callable] = None, enabled: bool = True) -> None:
        self.prefix = prefix
        self.on_exit = on_exit
        self.enabled = enabled

    def __enter__(self) -> None:
        self.st = time.perf_counter_ns()

    def __exit__(self, *exc):
        et = time.perf_counter_ns()
        t = et - self.st
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

