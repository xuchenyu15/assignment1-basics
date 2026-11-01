import time
from tokenization.tokenizer_encoder import TokenizerEncoder
from pathlib import Path
from itertools import islice
import numpy as np

total_input_bytes = 0

def iter_documents(path: str):
    delimiter = "<|endoftext|>"
    buffer = ""

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            buffer += line
            while delimiter in buffer:
                document, buffer = buffer.split(delimiter, 1)
                yield document + delimiter

        if buffer:
            yield buffer  # Preserve trailing text without a delimiter.

def counted_documents(documents):
    global total_input_bytes

    for document in documents:
        total_input_bytes += len(document.encode("utf-8"))
        yield document

def main():
    target = "owt"
    path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "owt_train.txt"
    )
    # path = (
    #     Path(__file__).resolve().parent.parent
    #     / "data"
    #     / "TinyStoriesV2-GPT4-train.txt"
    # )
    vocab_filepath = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "bpe"
        / f"{target}-vocab.json"
    )
    merges_filepath = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "bpe"
        / f"{target}-merges.json"
    )
    
    encoder = TokenizerEncoder.from_files(vocab_filepath, merges_filepath, ["<|endoftext|>"])
    documents = counted_documents(iter_documents(path))
    token_ids = encoder.encode_iterable(documents)

    start = time.perf_counter()
    total_tokens = 0
    last_report = start

    with open(f"results/bpe/{target}_train.bin", "wb") as f:
        while True:
            batch = list(islice(token_ids, 100_000))
            if not batch:
                break

            if min(batch) < 0 or max(batch) > 65535:
                raise ValueError("Token ID is outside the uint16 range")

            np.asarray(batch, dtype="<u2").tofile(f)
            total_tokens += len(batch)

            now = time.perf_counter()
            if now - last_report >= 5:
                elapsed = now - start
                print(
                    f"{total_tokens:,} tokens | "
                    f"{total_input_bytes:,} input bytes | "
                    f"{total_input_bytes / elapsed:,.0f} bytes/s",
                    flush=True,
                )
                last_report = now

    elapsed = time.perf_counter() - start
    print(
        f"Finished: {total_tokens:,} tokens in {elapsed:.2f}s "
        f"({total_tokens / elapsed:,.0f} tokens/s)"
    )

    return 


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