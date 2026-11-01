import json
import regex as re
from collections.abc import Iterable, Iterator


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
            chunk_encoded = self.encode_chunk(chunk)
            encoded_list.extend(chunk_encoded)
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

    def encode_pretoken(self, pretoken: str) -> list[int]:
        # print(f"current text have {len(text)} as length")
        bytes_list = [bytes([b]) for b in pretoken.encode("utf-8")]
        # print(f"current encoded list have {len(encoded_list)} tokens")
        for bytes1, bytes2 in self.merges:
            new_bytes_list = list()
            j = 0
            while j < len(bytes_list):
                if j == len(bytes_list) - 1:
                    new_bytes_list.append(bytes_list[j])
                    j += 1
                elif bytes_list[j] == bytes1 and bytes_list[j + 1] == bytes2:
                    new_bytes_list.append(bytes1 + bytes2)
                    j += 2
                else:
                    new_bytes_list.append(bytes_list[j])
                    j += 1
            bytes_list = new_bytes_list
        encoded_list = [self.indexes[encoded_bytes] for encoded_bytes in bytes_list]
        return encoded_list
        
