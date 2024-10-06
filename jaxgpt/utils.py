import time
import contextlib
from typing import Optional

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
