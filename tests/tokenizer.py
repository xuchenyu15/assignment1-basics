import regex as re

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
        special_pat = "|".join(re.escape(s) for s in self.special_tokens)
        chunks = re.split(special_pat, corpus)
        print(f"chunks size: {len(chunks)}")
        # print(chunks)
        contents = list()
        for chunk in chunks:
            contents = contents + re.findall(self.PAT, chunk)
        encoded_lists = [self.tokenize(content) for content in contents]
        print(f"encoded_lists size: {len(encoded_lists)}")
        
        if len(self.vocab) >= target_vocab_size:
            print(f"train finished, current vocabulary size: {len(self.vocab)}")
            return

        first_merge_pair, aggregated_counts, merged_counts = self.find_first_merge(encoded_lists)
        if first_merge_pair == None:
            print("No merges to be performed")
            return
        first_merge = self.merge(first_merge_pair)
        self.apply_merge_batch(first_merge, encoded_lists, aggregated_counts, merged_counts)

        while len(self.vocab) < target_vocab_size:
            next_merge_pair = self.find_next_merge(merged_counts)
            if next_merge_pair == None:
                print("No merges to be performed")
                break
            next_merge = self.merge(next_merge_pair)
            self.apply_merge_batch(next_merge, encoded_lists, aggregated_counts, merged_counts)
        print(f"train finished, current vocabulary size: {len(self.vocab)}")
        return

    def find_first_merge(
            self, 
            encoded_lists: list[list[int]]
        ) -> tuple[None | tuple[int, int], list[dict[tuple[int, int], list[int]]], dict[tuple[int, int], int]]:
        aggregated_counts = list()
        for encoded_list in encoded_lists:
            counts = self.count_pairs(encoded_list)
            aggregated_counts.append(counts)
        # print(f"Aggregated counts has a size of {len(aggregated_counts)}")
        merged_counts = self.merge_counts(aggregated_counts)
        return (self.get_max_pair(merged_counts), aggregated_counts, merged_counts)

    def find_next_merge(
            self,
            merged_counts: dict[tuple[int, int], int]
        ) -> None | tuple[int, int]:
        # for encoded_list in encoded_lists:
        #     counts = self.count_pairs(encoded_list)
        #     aggregated_counts.append(counts)
        # print(f"Aggregated counts has a size of {len(aggregated_counts)}")
        # merged_counts = self.merge_counts(aggregated_counts)
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

    def count_pairs(self, encoded_list: list[int]) -> dict[tuple[int, int], list[int]]:
        counts = dict()
        # print(f"current encoded list have {len(encoded_list)} tokens")
        for i in range(len(encoded_list) - 1):
            pair = (encoded_list[i], encoded_list[i + 1])
            if pair not in counts:
                counts[pair] = []
            counts[pair].append(i)
        # print(f"current counts have {len(counts)} pairs")
        return counts

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
        encoded_lists, 
        aggregated_counts,
        merged_counts
    ) -> dict[tuple[int, int], int]:
        pair = merge[1]
        for i in range(len(encoded_lists)):
            if pair not in aggregated_counts[i]:
                # print(aggregated_counts[i])
                # print("skipping")
                continue
            diff = self.apply_merge(
                merge,
                i,
                encoded_lists, 
                aggregated_counts,
            )
            self.update_merged_counts(merged_counts, diff)
        return

    def apply_merge(
        self, 
        merge: tuple[int, tuple[int, int]],
        list_index: int,
        encoded_lists: list[list[int]],
        aggregated_counts: list[dict[tuple[int, int], list[int]]]
    ) -> dict[tuple[int, int], int]:
        # print(encoded_list)
        original_counts = aggregated_counts[list_index]
        diff_counts = dict()
        index, pair = merge[0], merge[1]
        last = None
        applied = 0
        new_encoded_list = list(encoded_lists[list_index])
        positions = original_counts[pair]
        for position in positions:
            updated_position = position - applied
            if updated_position == last:
                # two pairs share one position, the second pair is invalid, 
                # eg: [3, 3, 3] with merge (3, 3) -> 4 should become: [4, 3]
                continue
            new_encoded_list[updated_position : updated_position + 2] = [index]
            last = updated_position 
            applied += 1
        # print(encoded_list)
        new_counts = self.count_pairs(new_encoded_list)
        encoded_lists[list_index] = new_encoded_list
        aggregated_counts[list_index] = new_counts
        # calculate counts diff to be applied to merged counts
        for pair_reduce in original_counts:
            diff_counts[pair_reduce] = diff_counts.get(pair_reduce, 0) - len(original_counts[pair_reduce]) 
        for pair_increase in new_counts:
            diff_counts[pair_increase] = diff_counts.get(pair_increase, 0) + len(new_counts[pair_increase]) 
        return diff_counts

    def update_merged_counts(
        self, 
        origin: dict[tuple[int, int], int],
        diff: dict[tuple[int, int], int]
    ):
        for pair in diff:
            if pair not in origin:
                origin[pair] = diff[pair]
            else:
                origin[pair] += diff[pair]
        return

    def get_merges_bytes(self):
        merges_bytes = list()
        for _, i1, i2 in self.merges:
            merges_bytes.append((self.vocab[i1], self.vocab[i2]))
        return merges_bytes

    