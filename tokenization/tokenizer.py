import regex as re
from .pretoken import Pretoken

class Tokenizer: 
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def __init__(
        self, 
        special_tokens: list[str]
    ):
        self.special_tokens = special_tokens
        self.vocab = dict()
        for i in range(256):
            self.vocab[i] = bytes([i])
        for j in range(len(special_tokens)):
            self.vocab[256 + j] = special_tokens[j].encode("utf-8")
        self.merges = list() # list[tuple(int, int, int)]

    def vocab_size(self):
        return len(self.vocab)

    def train_tokenizer(
        self, 
        target_vocab_size: int, 
        corpus: str
    ):  
        if len(self.vocab) >= target_vocab_size:
            print(f"train finished, current vocabulary size: {len(self.vocab)}")
            return

        pretokens_map, pairs_map, merged_counts = self.aggregate_counts(corpus)

        while len(self.vocab) < target_vocab_size:
            next_merge_pair = self.find_next_merge(merged_counts)
            if next_merge_pair == None:
                print("No merges to be performed")
                break
            next_merge = self.merge(next_merge_pair)
            self.apply_merge_batch(next_merge, merged_counts, pretokens_map, pairs_map)
        print(f"train finished, current vocabulary size: {len(self.vocab)}")
        return

    def train_tokenizer_from_string_counter(
        self, 
        target_vocab_size: int, 
        string_counter: dict[str, int]
    ):  
        if len(self.vocab) >= target_vocab_size:
            print(f"train finished, current vocabulary size: {len(self.vocab)}")
            return

        pretokens_map, pairs_map, merged_counts = self.aggregate_counts_from_string_counter(string_counter)

        print("Start training")
        while len(self.vocab) < target_vocab_size:
            next_merge_pair = self.find_next_merge(merged_counts)
            if next_merge_pair == None:
                print("No merges to be performed")
                break
            next_merge = self.merge(next_merge_pair)
            self.apply_merge_batch(next_merge, merged_counts, pretokens_map, pairs_map)
        print(f"train finished, current vocabulary size: {len(self.vocab)}")
        return

    def train_tokenizer_from_aggregated_counts(
        self, 
        target_vocab_size: int, 
        pretokens_map: dict[tuple, Pretoken], 
        pairs_map: dict[tuple[int, int], set[tuple]], 
        merged_counts: dict[tuple[int, int], int]
    ):  
        if len(self.vocab) >= target_vocab_size:
            print(f"train finished, current vocabulary size: {len(self.vocab)}")
            return

        while len(self.vocab) < target_vocab_size:
            next_merge_pair = self.find_next_merge(merged_counts)
            if next_merge_pair == None:
                print("No merges to be performed")
                break
            next_merge = self.merge(next_merge_pair)
            self.apply_merge_batch(next_merge, merged_counts, pretokens_map, pairs_map)
        print(f"train finished, current vocabulary size: {len(self.vocab)}")
        return

    def preprocess(self, corpus: str) -> list[str]:
        contents = list()
        special_pat = "|".join(re.escape(s) for s in self.special_tokens)
        chunks = re.split(special_pat, corpus)
        print(f"chunks size: {len(chunks)}")
        for chunk in chunks:
            contents_found = re.findall(self.PAT, chunk)
            for content_found in contents_found:
                contents.append(content_found)
        return contents

    def aggregate_counts(
        self, 
        corpus: str
    ) -> tuple[dict[tuple, Pretoken], dict[tuple[int, int], set[tuple]], dict[tuple[int, int], int]]:
        contents = self.preprocess(corpus)
        encoded_lists = [self.tokenize(content) for content in contents]
        print(f"encoded_lists size: {len(encoded_lists)}")
        
        pretokens_map = dict()
        pairs_map = dict()
        merged_counts = dict()
        for encoded_list in encoded_lists:
            encoded_tuple = tuple(encoded_list)
            if encoded_tuple not in pretokens_map:
                pretokens_map[encoded_tuple] = Pretoken(0, encoded_list)
            pretoken = pretokens_map[encoded_tuple]
            pretoken.increase_occurrence(1)

        for encoded_tuple, pretoken in pretokens_map.items():
            pair_counts = pretoken.get_pair_counts()
            self.ingest_counts(merged_counts, pair_counts)
            self.ingest_pair_map(pairs_map, pair_counts.keys(), encoded_tuple)

        return (pretokens_map, pairs_map, merged_counts)

    def aggregate_counts_from_string_counter(
        self, 
        string_counter: dict[str, int]
    ) -> tuple[dict[tuple, Pretoken], dict[tuple[int, int], set[tuple]], dict[tuple[int, int], int]]:
        pretokens_map = dict()
        pairs_map = dict()
        merged_counts = dict()

        print("Aggregating from string counter inside tokenizer.")

        for content, occurrence in string_counter.items():
            encoded_list = self.tokenize(content)
            encoded_tuple = tuple(encoded_list)
            if encoded_tuple not in pretokens_map:
                pretokens_map[encoded_tuple] = Pretoken(0, encoded_list)
            pretoken = pretokens_map[encoded_tuple]
            pretoken.increase_occurrence(occurrence)

        for encoded_tuple, pretoken in pretokens_map.items():
            pair_counts = pretoken.get_pair_counts()
            self.ingest_counts(merged_counts, pair_counts)
            self.ingest_pair_map(pairs_map, pair_counts.keys(), encoded_tuple)

        print("Aggregated counts and maps.")

        return (pretokens_map, pairs_map, merged_counts)

    def find_next_merge(
        self,
        merged_counts: dict[tuple[int, int], int]
    ) -> None | tuple[int, int]:
        return self.get_max_pair(merged_counts)

    def tokenize(self, text: str) -> list[int]:
        # print(f"current text have {len(text)} as length")
        encoded_list = list(text.encode("utf-8"))
        # print(f"current encoded list have {len(encoded_list)} tokens")
        for index, i1, i2 in self.merges:
            new_encoded_list = list()
            j = 0
            while j < len(encoded_list):
                if j == len(encoded_list) - 1:
                    new_encoded_list.append(encoded_list[j])
                    j += 1
                elif encoded_list[j] == i1 and encoded_list[j + 1] == i2:
                    new_encoded_list.append(index)
                    j += 2
                else:
                    new_encoded_list.append(encoded_list[j])
                    j += 1
            encoded_list = list(new_encoded_list)
        return encoded_list

    def merge_counts(self, counts_list: list[dict[tuple[int, int], list[int]]]) -> dict[tuple[int, int], int]:
        merged_counts = dict()
        for counts in counts_list:
            # print(f"single counts have a size of {len(counts)}")
            for pair in counts:
                if pair not in merged_counts:
                    merged_counts[pair] = len(counts[pair])
                else:
                    merged_counts[pair] += len(counts[pair])
        return merged_counts

    def get_max_pair(self, merged_counts: dict[tuple[int, int], int]) -> None | tuple[int, int]:
        # print(f"merge counts have a size of {len(merged_counts)}")
        counts_list = [(count, pair) for pair, count in merged_counts.items()]
        if len(counts_list) == 0:
            print("No counts, thus no max pair")
            return None
        counts_list.sort(
            key = lambda entry: (
                entry[0],
                (self.vocab[entry[1][0]], self.vocab[entry[1][1]])
            ),
            reverse = True
        )
        # print(counts_list[0])
        return counts_list[0][1]

    def merge(self, pair: tuple[int, int]) -> tuple[int, tuple[int, int]]:
        index = len(self.vocab)
        self.merges.append((index, pair[0], pair[1]))
        # print("merging: ")
        # print(self.vocab[pair[0]] + self.vocab[pair[1]])
        self.vocab[index] = self.vocab[pair[0]] + self.vocab[pair[1]]
        return (index, pair)

    def apply_merge_batch(
        self, 
        merge, 
        merged_counts,
        pretokens_map, 
        pairs_map
    ) -> dict[tuple[int, int], int]:
        merge_pair = merge[1]
        if merge_pair not in pairs_map or len(pairs_map[merge_pair]) == 0:
            return
        for encoded_tuple in list(pairs_map[merge_pair]):
            pretoken = pretokens_map[encoded_tuple]
            self.remove_pair_map(pairs_map, pretoken.pair_positions.keys(), encoded_tuple)
            diff_counts = pretoken.apply_merge(merge)
            new_encoded_tuple = pretoken.get_encoded_tuple()
            if new_encoded_tuple not in pretokens_map:
                pretokens_map[new_encoded_tuple] = pretoken
            del pretokens_map[encoded_tuple]
            self.ingest_counts(merged_counts, diff_counts)
            self.ingest_pair_map(pairs_map, pretoken.pair_positions.keys(), new_encoded_tuple)

        return

    def ingest_pair_map(self, pairs_map, pairs, encoded_tuple):
        for pair in pairs: 
            if pair not in pairs_map:
                pairs_map[pair] = set()
            pairs_map[pair].add(encoded_tuple)
        return

    def remove_pair_map(self, pairs_map, pairs, encoded_tuple):
        for pair in pairs: 
            if pair not in pairs_map:
                continue
            pairs_map[pair].remove(encoded_tuple)
            if len(pairs_map[pair]) == 0:
                del pairs_map[pair]
        return

    def ingest_counts(self, merged_counts, pair_counts):
        for pair, count in pair_counts.items():
            merged_counts[pair] = merged_counts.get(pair, 0) + count
            if merged_counts[pair] == 0:
                del merged_counts[pair]
        return

    def get_merges_bytes(self):
        merges_bytes = list()
        for _, i1, i2 in self.merges:
            merges_bytes.append((self.vocab[i1], self.vocab[i2]))
        return merges_bytes

    