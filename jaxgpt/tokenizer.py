import os
import json
import regex as re
import requests

import jax

def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word: str) -> set[tuple[str, str]]:
    """
    Return all bigrams as a set of tuples, of consecutive elements in the iterable word. Set has no order.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

if __name__ == "__main__":
    print(bytes_to_unicode())
    word = "kocham Pati"
    print(get_pairs(word))
