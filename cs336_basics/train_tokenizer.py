from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import Counter
import json
import time

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


def main():
    # path = "mypath"
    # path = (
    #     Path(__file__).resolve().parent.parent
    #     / "data"
    #     / "owt_train.txt"
    # )
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
    num_chunks = 48
    calculated = 48
    chunker = Chunker()
    combined_counts = Counter()
    # aggregated_pretokens_map = dict()
    # aggregated_pairs_map = dict()
    # aggregated_merged_counts = dict()

    with open(path, "rb") as f:
        boundaries = chunker.find_chunk_boundaries(
            f, num_chunks, b"<|endoftext|>"
        )

    tasks = [
        (path, start, end) 
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    # tasks = [
    #     (path, start, end) 
    #     for start, end in zip(boundaries[:calculated], boundaries[1:calculated + 1])
    # ]

    with ProcessPoolExecutor(max_workers=num_processes) as pool:
        for worker_counts in pool.map(process_chunk, tasks):
            combined_counts.update(worker_counts)

    print("Count combined")        

    tokenizer = Tokenizer(["<|endoftext|>"])
    tokenizer.train_tokenizer_from_string_counter_heap(32000, combined_counts)
    print("printing merges: ")
    print(tokenizer.get_byte_merges())
    token_id, token = max(tokenizer.vocab.items(), key=lambda item: len(item[1]))
    print("printing longest token: ")
    print(token_id, token)
    print("Length in bytes:", len(token))

    # Hex preserves arbitrary token bytes, including incomplete UTF-8 sequences.
    with open(f"vocab-{str(calculated)}.json", "w", encoding="utf-8") as f:
        json.dump({token_id: token.hex() for token_id, token in tokenizer.vocab.items()}, f, indent=2)
        f.write("\n")

    with open(f"merges-{str(calculated)}.json", "w", encoding="utf-8") as f:
        json.dump([[left.hex(), right.hex()] for left, right in tokenizer.get_byte_merges()], f, indent=2)
        f.write("\n")
            
            

if __name__ == "__main__":
    start = time.perf_counter()
    try: 
        main()
    finally:
        elapsed = time.perf_counter() - start 
        print(
            f"Total elapsed: {elapsed:.4f}s ({elapsed / 60:.4f} min)",
            flush=True,
        )
