from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import Counter
import json

from tokenization.chunker import Chunker
from tokenization.tokenizer import Tokenizer
from tokenization.pretoken import Pretoken
from tokenization.pretokenizer import Pretokenizer

def process_chunk(task):
    path, start, end = task 

    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")
    
    pretokenizer = Pretokenizer(["<|endoftext|>"])
    worker_counts = pretokenizer.count(chunk)
    return worker_counts


# def aggregate_pretokens_map(
#     aggregated_pretokens_map: dict[tuple, Pretoken], 
#     pretokens_map: dict[tuple, Pretoken]
# ):
#     for encoded_tuple in pretokens_map.keys():
#         if encoded_tuple not in aggregated_pretokens_map:
#             aggregated_pretokens_map[encoded_tuple] = pretokens_map[encoded_tuple]
#         else:
#             aggregated_pretokens_map[encoded_tuple].increase_occurrence(pretokens_map[encoded_tuple].occurrence)
#     return 

# def aggregate_pairs_map(
#     aggregated_pairs_map: dict[tuple[int, int], set[tuple]], 
#     pairs_map: dict[tuple[int, int], set[tuple]]
# ):
#     for pair in pairs_map:
#         if pair not in aggregated_pairs_map:
#             aggregated_pairs_map[pair] = pairs_map[pair]
#         else:
#             aggregated_pairs_map[pair].update(pairs_map[pair])
#     return 

# def aggregate_merged_counts(
#     aggregated_merged_counts: dict[tuple[int, int], int], 
#     merged_counts: dict[tuple[int, int], int]
# ):
#     for pair in merged_counts:
#         if pair not in aggregated_merged_counts:
#             aggregated_merged_counts[pair] = merged_counts[pair]
#         else:
#             aggregated_merged_counts[pair] += merged_counts[pair]
#     return 


def main():
    # path = "mypath"
    path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "TinyStoriesV2-GPT4-train.txt"
    )
    # path = (
    #     Path(__file__).resolve().parent.parent
    #     / "tests"
    #     / "fixtures"
    #     / "tinystories_sample_5M.txt"
    # )
    num_processes = 4
    chunker = Chunker()
    combined_counts = Counter()
    # aggregated_pretokens_map = dict()
    # aggregated_pairs_map = dict()
    # aggregated_merged_counts = dict()

    with open(path, "rb") as f:
        boundaries = chunker.find_chunk_boundaries(
            f, num_processes, b"<|endoftext|>"
        )

    tasks = [
        (path, start, end) 
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with ProcessPoolExecutor(max_workers=num_processes) as pool:
        for worker_counts in pool.map(process_chunk, tasks):
            combined_counts.update(worker_counts)

    print("Count combined")        

    tokenizer = Tokenizer(["<|endoftext|>"])
    tokenizer.train_tokenizer_from_string_counter(10000, combined_counts)
    print("printing merges: ")
    print(tokenizer.get_merges_bytes())
    token_id, token = max(tokenizer.vocab.items(), key=lambda item: len(item[1]))
    print("printing longest token: ")
    print(token_id, token)
    print("Length in bytes:", len(token))

    # Hex preserves arbitrary token bytes, including incomplete UTF-8 sequences.
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump({token_id: token.hex() for token_id, token in tokenizer.vocab.items()}, f, indent=2)
        f.write("\n")

    with open("merges.json", "w", encoding="utf-8") as f:
        json.dump([[left.hex(), right.hex()] for left, right in tokenizer.get_merges_bytes()], f, indent=2)
        f.write("\n")
            
            

if __name__ == "__main__":
    main()
