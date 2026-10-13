# Vendored (trimmed) from mlx-examples/clip/tokenizer.py — Copyright © 2023-2024
# Apple Inc., MIT License.

import json
from pathlib import Path

import mlx.core as mx
import regex


class CLIPTokenizer:
    """A simple port of HF's CLIPTokenizer."""

    def __init__(self, bpe_ranks, vocab):
        self.bpe_ranks = bpe_ranks
        self.vocab = vocab
        self.pat = regex.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            regex.IGNORECASE,
        )
        self._cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}

    @property
    def bos_token(self) -> int:
        return self.vocab["<|startoftext|>"]

    @property
    def eos_token(self) -> int:
        return self.vocab["<|endoftext|>"]

    def bpe(self, text):
        if text in self._cache:
            return self._cache[text]

        unigrams = list(text[:-1]) + [text[-1] + "</w>"]
        unique_bigrams = set(zip(unigrams, unigrams[1:]))
        while unique_bigrams:
            bigram = min(
                unique_bigrams, key=lambda pair: self.bpe_ranks.get(pair, float("inf"))
            )
            if bigram not in self.bpe_ranks:
                break

            new_unigrams = []
            skip = False
            for a, b in zip(unigrams, unigrams[1:]):
                if skip:
                    skip = False
                    continue
                if (a, b) == bigram:
                    new_unigrams.append(a + b)
                    skip = True
                else:
                    new_unigrams.append(a)
            if not skip:
                new_unigrams.append(b)

            unigrams = new_unigrams
            unique_bigrams = set(zip(unigrams, unigrams[1:]))

        self._cache[text] = unigrams
        return unigrams

    def tokenize(self, text, prepend_bos=True, append_eos=True) -> mx.array:
        if isinstance(text, list):
            return [self.tokenize(t, prepend_bos, append_eos) for t in text]

        clean_text = regex.sub(r"\s+", " ", text.lower())
        tokens = regex.findall(self.pat, clean_text)
        bpe_tokens = [ti for t in tokens for ti in self.bpe(t)]

        out = []
        if prepend_bos:
            out.append(self.bos_token)
        out.extend(self.vocab[t] for t in bpe_tokens)
        if append_eos:
            out.append(self.eos_token)
        return mx.array(out)

    def pad(self, sequences: list[mx.array], pad_id: int = 0) -> mx.array:
        """Right-pad token id sequences to a common length."""
        max_len = max(len(s) for s in sequences)
        rows = []
        for seq in sequences:
            ids = seq.tolist()
            rows.append(ids + [pad_id] * (max_len - len(ids)))
        return mx.array(rows)

    @staticmethod
    def from_pretrained(path: str) -> "CLIPTokenizer":
        path = Path(path)
        with open(path / "vocab.json", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(path / "merges.txt", encoding="utf-8") as f:
            bpe_merges = f.read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]
        bpe_merges = [tuple(m.split()) for m in bpe_merges]
        bpe_ranks = dict(map(reversed, enumerate(bpe_merges)))
        return CLIPTokenizer(bpe_ranks, vocab)
