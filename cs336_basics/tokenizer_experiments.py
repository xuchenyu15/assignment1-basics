import time
from tokenization.chunker import Chunker
from tokenization.tokenizer_encoder import TokenizerEncoder
from pathlib import Path
import json


def get_samples(
        path: str, 
        sample_number: int
) -> list[bytes]:
    chunker = Chunker()
    with open(path, "rb") as f:
        samples = chunker.find_samples(
            f, b"<|endoftext|>", sample_number
        )
    print(f"Found {str(len(samples))} samples.")
    return samples

def main():
    target = "owt"
    output_prefix = "owt"
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
    samples = get_samples(path, 1000)
    encoder = TokenizerEncoder.from_files(vocab_filepath, merges_filepath, ["<|endoftext|>"])
    total_bytes = 0
    total_token = 0
    final_tokens = []
    total_time = 0
    
    for sample in samples: 
        start = time.perf_counter()
        encoded_list = encoder.encode(sample.decode("utf-8"))
        encoding_seconds = time.perf_counter() - start
        total_time += encoding_seconds

        total_bytes += len(sample)
        total_token += len(encoded_list)
        final_tokens.append(encoded_list)

    throughput = total_bytes / total_time
    estimated_seconds = 825 * 10 ** 9 / throughput

    print(f"Total bytes: {total_bytes}")
    print(f"Total tokens: {total_token}")
    print(f"Total time: {total_time:.4f} s")
    print(f"The ratio of bytes/token is {total_bytes/total_token:.4f}.")
    print(f"The throughput of tokenizer is {throughput:.4f} bytes/s.")
    print(f"Estimated Pile time: {estimated_seconds / 3600:.2f} hours")

    output_dir = Path(__file__).resolve().parent.parent / "results" / "bpe"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{output_prefix}-samples.json").open("w", encoding="utf-8") as f:
        json.dump([sample.hex() for sample in samples], f, indent=2)

    with (output_dir / f"{output_prefix}-final_tokens.json").open("w", encoding="utf-8") as f:
        json.dump(final_tokens, f, indent=2)

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