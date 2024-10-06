import os
import json
import regex as re
import requests
import pprint

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
    def __init__(self, encoder: dict[str, int], bpe_merges: list[tuple[list[str]]]) -> None:
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip( bpe_merges, range(len(bpe_merges))))

        # splitting pattern for pre-tokenization
        self.pat = re.compile(PATTERN)
        self.cache = {}

    def bpe(self, token: str) -> str:
        """
        this function uses self.bpe_ranks to iteratively merge all the possible bpe tokens
        up the tree. token is a string of one individual 'word' (after regex tokenization)
        and after byte encoding, e.g. 'Ġthere'.
        """

        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            # lowest rank bigram
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str, debug: bool = False) -> list[int]:
        bpe_idx = []
        parts = []
        # pre-tokenization
        tokens = re.findall(self.pat, text)
        for token in tokens:
            token_bytes = token.encode('utf-8')
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(' ')
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)
            print(token, token_bytes, token_translated, token_merged)
            if debug:
                parts.append({
                    'token': token,
                    'token_bytes': token_bytes,
                    'token_translated': token_translated,
                    'token_merged': token_merged,
                    'token_ix': token_ix
                })

        if debug:
            pprint.pp({
                'bpe_idx': bpe_idx,
                'tokens': tokens,
                'parts': parts
            })

        return bpe_idx
    
    def decode(self, bpe_idx: list[int], debug: bool = False) -> str:
        tokens_merged = [self.decoder[token] for token in bpe_idx]
        tokens_flat = "".join(tokens_merged)
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        text = tokens_bytes.decode("utf-8", errors="replace")
        if debug:
            pprint.pp({
                'bpe_idx': bpe_idx,
                'tokens_merged': tokens_merged,
                'tokens_flat': tokens_flat,
                'tokens_bytes': tokens_bytes,
                'text': text
            })
        return text


def get_file(local_file: str, url: str) -> None:
    if not os.path.isfile(local_file):
        print(f"Downloading {url} to {local_file}")
        response = requests.get(url)
        open(local_file, "wb").write(response.content)

def get_encoder() -> Encoder:
    """
    Returns an instance of the GPT/BPE Encoder/Decoder
    and handles caching of "database" files.
    """

    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".jaxgpt")
    os.makedirs(cache_dir, exist_ok=True)

    encoder_local_file = os.path.join(cache_dir, "encoder.json")
    encoder_remote_url = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json"
    get_file(encoder_local_file, encoder_remote_url)
    with open(encoder_local_file, "r") as f:
        encoder = json.load(f)
    # 256 individual byte tokens, 50,000 merged tokens, and 1 special <|endoftext|> token
    assert len(encoder) == 50257

    vocab_local_file = os.path.join(cache_dir, "vocab.bpe")
    vocab_remote_url = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe"
    get_file(vocab_local_file, vocab_remote_url)
    with open(vocab_local_file, "r", encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
    assert len(bpe_merges) == 50000

    return Encoder(encoder, bpe_merges)


if __name__ == "__main__":
    word = "kocham Pati"
    E = get_encoder()
    enc = E.encode(word, debug = True)
    dec = E.decode(enc, debug = True)
