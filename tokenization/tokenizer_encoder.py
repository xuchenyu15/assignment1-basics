import json
import regex as re
from collections.abc import Iterable, Iterator
from functools import lru_cache


class TokenizerEncoder():
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.indexes: dict[bytes, int] = {
            token: id
            for id, token in vocab.items()
        }
        self.merges = merges
        self.merge_ranks = {
            pair: rank 
            for rank, pair in enumerate(self.merges)
        }
        if special_tokens == None:
            self.special_tokens = list()
        else:
            self.special_tokens = special_tokens
            self.special_tokens.sort(
                key = lambda x: (len(x), x),
                reverse=True
            )

    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None
    ):
        with open(vocab_filepath, "r", encoding = "utf-8") as f:
            raw_vocab = json.load(f)

        vocab: dict[int, bytes] = {
            int(token_id): bytes.fromhex(token_hex)
            for token_id, token_hex in raw_vocab.items()
        }

        with open(merges_filepath, "r", encoding = "utf-8") as f:
            raw_merges = json.load(f)

        merges: list[tuple[bytes, bytes]] = [
            (bytes.fromhex(left), bytes.fromhex(right)) 
            for left, right in raw_merges
        ]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        encoded_list = list()
        chunks = self.preprocess_corpus(text)
        for chunk in chunks:
            # print(self.encode_pretoken.cache_info())
            chunk_encoded = self.encode_chunk(chunk)
            encoded_list.extend(chunk_encoded)
            # self.encode_pretoken.cache_clear()
        return encoded_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            iterable_encoded: list[int] = self.encode(text)
            yield from iterable_encoded

    def preprocess_corpus(self, corpus: str) -> list[str]:
        # divide whole corpus into 
        if not corpus or len(corpus) == 0:
            return []
        if not self.special_tokens:
            special_pat = corpus
        else: 
            special_pat = "|".join(
                re.escape(s) for s in self.special_tokens
            )
        chunks = re.split(f"({special_pat})", corpus)
        return chunks
    
    def encode_chunk(self, chunk: str) -> list[str]:
        if chunk in self.special_tokens:
            return [self.indexes[chunk.encode("utf-8")]]
        
        pretokens_found = re.findall(self.PAT, chunk)
        if not pretokens_found:
            return []
        encoded_chunk = list()

        for pretoken in pretokens_found:
            encoded_pretoken = self.encode_pretoken(pretoken)
            encoded_chunk.extend(encoded_pretoken)
        return encoded_chunk

    def decode(self, ids: list[int]) -> str:
        combined_bytes = b"".join([self.vocab[id] for id in ids])
        return combined_bytes.decode("utf-8", errors = "replace")

    @lru_cache(maxsize=50_000)
    def encode_pretoken(self, pretoken: str) -> list[int]:
        bytes_list = [bytes([b]) for b in pretoken.encode("utf-8")]
        while True: 
            pairs_positions = self.get_pairs_positions(bytes_list)
            pair_merge = self.find_next_pair(pairs_positions)
            if pair_merge == None:
                break
            self.apply_pair_merge(bytes_list, pairs_positions, pair_merge)

        encoded_list = [self.indexes[encoded_bytes] for encoded_bytes in bytes_list]
        return tuple(encoded_list)

    def find_next_pair(self, pairs_positions: dict) -> tuple[bytes, bytes] | None:
        pair = None
        for candidate in pairs_positions.keys():
            if candidate not in self.merge_ranks:
                continue
            if pair == None:
                pair = candidate
            else:
                if self.merge_ranks[candidate] < self.merge_ranks[pair]:
                    pair = candidate
        return pair

    def apply_pair_merge(self, bytes_list, pairs_positions, pair):
        if pair not in pairs_positions:
            return
        positions = pairs_positions[pair]
        applied = 0
        last = -1
        for i in positions:
            updated_i = i - applied
            if updated_i == last:
                continue
            else:
                bytes_list[updated_i : updated_i + 2] = [pair[0] + pair[1]]
                applied += 1
                last = updated_i

    def get_pairs_positions(
        self, 
        bytes_list: list[bytes]
    ) -> dict[tuple[bytes, bytes], list[int]]:
        pairs_positions = dict()
        for i in range(len(bytes_list) - 1):
            pair = (bytes_list[i], bytes_list[i + 1])
            if pair not in pairs_positions:
                pairs_positions[pair] = []
            pairs_positions[pair].append(i)

        return pairs_positions


        
