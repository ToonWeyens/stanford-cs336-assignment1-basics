import os
import regex as re

from collections import Counter, defaultdict
from typing import BinaryIO
from tqdm import tqdm

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()

def split_on_specials(data: bytes, special_tokens: list[bytes]) -> list[bytes]:
    pattern = b"|".join(re.escape(tok) for tok in special_tokens)
    return re.split(pattern, data)

def build_pretokens(data: bytes, special_tokens: list[bytes]) -> Counter[bytes]:
    # GPT-2 byte-level tokenizer
    PAT = rb"'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"

    # tokenize segments but never cross specials
    segments = split_on_specials(data, special_tokens)
    c = Counter()
    for seg in tqdm(segments):
        # for m in PAT.finditer(seg):
        for m in re.finditer(PAT, seg):
            c[m.group()] += 1
    return c

class SymTab:
    # id -> tuple[int,...] of base bytes; seq -> id
    def __init__(self):
        # We start with the first 256 bytes
        self.id2seq = {i: (i,) for i in range(256)}
        self.seq2id = {(i,): i for i in range(256)}
        self.next_id = 256

    # Add another to it
    def add_symbol(self, seq: tuple[int, ...]) -> int:
        if seq in self.seq2id:
            return self.seq2id[seq]
        i = self.next_id
        self.next_id += 1
        self.seq2id[seq] = i
        self.id2seq[i] = seq
        return i

def make_dataset(
    pretoks: Counter[bytes],
    *,
    debug_output: bool = False,
) -> list[tuple[tuple[int, ...], int]]:
    dataset = []
    for tok, freq in pretoks.items():
        pretok_with_freq = (tuple(tok), freq)
        if debug_output:
            printable = tok.decode("utf-8", errors="backslashreplace")
            print(f"tok {printable}  - freq {freq}: {pretok_with_freq}")
        dataset.append(pretok_with_freq)
    return dataset

def get_stats(dataset):
    pairs = defaultdict(int)
    for word, freq in dataset:
        if len(word) < 2:
            continue
        for a, b in zip(word, word[1:]):
            pairs[(a,b)] += freq
    return pairs

def apply_merge(dataset,
                a:int,
                b: int,
                new_id: int
    ) -> list[tuple[tuple[int, ...], int]]: 
    out = []
    for seq, f in dataset:
        i = 0
        n = len(seq)
        buf = []
        while i < n:
            if i + 1 < n and seq[i] == a and seq[i+1] == b:
                buf.append(new_id)
                i += 2
            else:
                buf.append(seq[i])
                i += 1
        out.append((tuple(buf), f))
    return out


def my_run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    debug: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # make sure they're byte objects
    special_tokens_b = [s.encode("utf-8") if isinstance(s, str) else s for s in special_tokens]


    data = read_bytes(input_path)
    
    if debug:
        print("computing pretokens")
    pretoks = build_pretokens(data, special_tokens_b)

    sym = SymTab()
    for s in special_tokens_b:
        sym.add_symbol(tuple(s))

    dataset = make_dataset(pretoks, debug_output=debug)

    merges: list[tuple[bytes, bytes]] = []

    if debug:
        print("merging tokens")
    pbar = tqdm(
        total=vocab_size,
        initial=len(sym.id2seq),
        unit="items",
        disable=not debug,
    )
    while len(sym.id2seq) < vocab_size:
        pairs = get_stats(dataset)
        if not pairs:
            break

        # Match GPT-2 training tie-break behaviour: prefer the highest
        # frequency, and when multiple pairs share the frequency choose the
        # pair whose underlying byte sequence is lexicographically largest.
        # This mirrors the reference implementation's deterministic ordering.
        a, b = max(
            pairs.items(),
            key=lambda kv: (
                kv[1],
                sym.id2seq[kv[0][0]],
                sym.id2seq[kv[0][1]],
            ),
        )[0]

        # make/ensure merged symbol representing underlying bytes
        merged_seq = sym.id2seq[a] + sym.id2seq[b] # adding tupples appends them
        new_id = sym.add_symbol(merged_seq)

        # record merge for output
        merges.append((bytes(sym.id2seq[a]), bytes(sym.id2seq[b])))

        # update dataset
        ba = bytes(sym.id2seq[a])
        bb = bytes(sym.id2seq[b])
        bab = ba+bb
        if debug:
            a_repr = ba.decode("utf-8", errors="backslashreplace")
            b_repr = bb.decode("utf-8", errors="backslashreplace")
            merged_repr = bab.decode("utf-8", errors="backslashreplace")
            print(
                f"merging byte pair {a} ({a_repr}), {b} ({b_repr}) to ({merged_repr})"
            )
        dataset = apply_merge(dataset, a, b, new_id)

        pbar.update(1)

    pbar.close()

    id2bytes = {i: bytes(seq) for i, seq in sym.id2seq.items()}

    return id2bytes, merges


if __name__ == "__main__":
    from pathlib import Path

    # filename = Path("data") / "owt_valid.txt"
    filename = Path("data") / "TinyStoriesV2-GPT4-valid.txt"
    print(f"Using {filename} for tokenization")

    special_tokens = [b"<|endoftext|>",
                      b"<|fim_prefix|>",
                      b"<|fim_suffix|>",
                      b"<|fim_middle|>",
                      b"<|file_separator|>"]

    vocab, merges = my_run_train_bpe(filename, 500, special_tokens)
