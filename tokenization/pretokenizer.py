from collections import Counter
import regex as re

class Pretokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, special_tokens: list[str]):
        self.string_counter = Counter()
        self.special_tokens = special_tokens

    def count(self, corpus):
        print("counting")
        special_pat = "|".join(re.escape(s) for s in self.special_tokens)
        segments = re.split(special_pat, corpus)
        for segment in segments:
            for match in re.finditer(self.PAT, segment):
                self.string_counter[match.group()] += 1
        print("count finished")
        return self.string_counter



