import os
import json
import regex as re
import requests

import jax

"""
    ok so what is this regex looking for, exactly?
    python re reference: https://docs.python.org/3/library/re.html
    - the vertical bars | is OR, so re.findall will chunkate text as the pieces match, from left to right
    - '\'s' would split up things like Andrej's -> (Andrej, 's)
    - ' ?\p{L}': optional space followed by 1+ unicode code points in the category "letter"
    - ' ?\p{N}': optional space followed by 1+ unicode code points in the category "number"
    - ' ?[^\s\p{L}\p{N}]+': optional space, then 1+ things that are NOT a whitespace, letter or number
    - '\s+(?!\S)': 1+ whitespace characters (e.g. space or tab or etc) UNLESS they are followed by non-whitespace
                    so this will consume whitespace characters in a sequence but exclude the last whitespace in
                    that sequence. that last whitespace has the opportunity to then match the optional ' ?' in
                    earlier patterns.
    - '\s+': 1+ whitespace characters, intended probably to catch a full trailing sequence of whitespaces at end of string
    So TLDR:
    - we are special casing a few common apostrophe constructs ('s, 't, 're, ...) and making those into separate tokens
    - we then separate out strings into consecutive chunks of 1) letters, 2) numbers, 3) non-letter-numbers, 4) whitespaces
"""
PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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


class Encoder:
    # TODO: check type for encoder
    def __init__(self, encoder, bpe_merges: list[tuple[list[str]]]) -> None:
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip( bpe_merges, range(len(bpe_merges))))

        # splitting pattern for pre-tokenization
        self.pat = re.compile(PATTERN)
        self.cache = {}

if __name__ == "__main__":
    print(bytes_to_unicode())
    word = "kocham Pati"
    print(get_pairs(word))
