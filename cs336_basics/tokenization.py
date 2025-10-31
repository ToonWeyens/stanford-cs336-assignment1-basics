import os
import pickle
import regex as re
import heapq

from collections import Counter, defaultdict
from pathlib import Path
from typing import BinaryIO
from tqdm import tqdm

# Note: Currently unused as pretokenization doesn't seem to be the biggest job
def _find_chunk_boundaries(
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

def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()
    
def _split_on_specials(data: bytes, special_tokens: list[bytes]) -> list[bytes]:
    pattern = b"|".join(re.escape(tok) for tok in special_tokens)
    return re.split(pattern, data)

def _build_indexes(dataset):
    # words[i]  -> mutable list of ids for word i
    # freqs[i]  -> frequency of word i
    # counts[(x,y)] -> total weighted count across all word pairs
    # occ[(x,y)]    -> list of occurrence sites (word_index, position) where position goes from 0 to word length - 1
    words = [list(seq) for seq, f in dataset]
    freqs = [f for _, f in dataset]
    counts = Counter()
    occ = defaultdict(set)

    for wi, s in enumerate(words):
        if len(s) < 2:
            continue
        f = freqs[wi]
        for i in range(len(s) - 1):
            p = (s[i], s[i+1])
            counts[p] += f
            occ[p].add((wi, i))

    # Max-heap with lazy invalidation, meaning that the version will be incremented and the last version will be used
    heap = []
    ver = {p: 0 for p in counts}       # version per pair
    for p, c in counts.items():
        # take -counts because python's heapq is min-heap, not max, so smallest index is in position 0
        heapq.heappush(heap, (-c, p, ver[p]))
    return words, freqs, counts, occ, heap, ver

def _heap_push(pair, counts, ver, heap):
    ver[pair] = ver.get(pair, 0) + 1
    c = counts.get(pair)
    if c is None:
        raise RuntimeError(f"Trying push to heap pair {pair} with version {ver[pair]} but counts has no entry")
    if c < 0:
        raise RuntimeError(f"Trying to push to heap pair {pair} with version {ver[pair]} with negative count {c}")
    # print(f'Pushed new entry {(-c, pair, ver[pair])} to heap')

    heapq.heappush(heap, (-c, pair, ver[pair]))

def _dec(pair, f, counts, ver, heap):
    # print(f"decreasing pair {pair} by {f}")
    if f == 0: 
        raise RuntimeError(f"Tried decreasing counts for pair {pair} by 0 which is not possible")
    if pair not in counts:
        raise RuntimeError(f"Pair {pair} was not found in counts")
    counts[pair] -= f
    # Todo: After validating everything works, delete counts 0. This would mean we have to modify
    #   some values to counts.get(pair, 0) in other places.
    #   This would free up some memory in the heap, but it's probably less relevant?
    # if counts[pair] == 0:
    #     del counts[pair]
    # elif counts[pair] < 0:
    if counts[pair] < 0:
        raise RuntimeError(f"Tried decreasing counts for pair {pair} by {f} to {counts[pair]}")
    _heap_push(pair, counts, ver, heap)

def _inc(pair, f, counts, ver, heap):
    # print(f"increasing pair {pair} by {f}")
    if f == 0: 
        raise RuntimeError(f"Tried increasing counts for pair {pair} by 0 which is not possible")
    counts[pair] = counts.get(pair, 0) + f # lazily create
    _heap_push(pair, counts, ver, heap)

def _occ_del(occ, pair, site):
    s = occ.get(pair)
    s.discard(site)
    if not s: # this pair stopped occuring completely
        occ.pop(pair, None)

def _occ_add(occ, pair, site):
    occ[pair].add(site)


def _build_pretokens(data: bytes, special_tokens: list[bytes]) -> Counter[bytes]:
    # GPT-2 byte-level tokenizer
    PAT = rb"'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"

    # tokenize segments but never cross specials
    segments = _split_on_specials(data, special_tokens)
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

def _make_dataset(
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

def _get_stats(dataset):
    pairs = defaultdict(int)
    for word, freq in dataset:
        if len(word) < 2:
            continue
        for a, b in zip(word, word[1:]):
            pairs[(a,b)] += freq
    return pairs

def _apply_merge(dataset,
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



def train_bpe_incremental(dataset, sym, target_vocab_size, merges_out, debug=False):
    words, freqs, counts, occ, heap, ver = _build_indexes(dataset)

    # Explanation:
    #   words[i] is word converted to its byte integer, with i=0..(tot nr. of words)
    #   freqs[i] is the corresponding frequency of each word
    #   counts[(i,j)] is the total numer of times a pair (i,j) appears with i=0..(tot nr. of pairs)
    #   occ[(i,j)] is the occurences where these pairs occur, indexed the same as counts
    #   heap is a minheap containing tripples (-count, pair, version), one for each pair
    #   ver is version, one for each pair
    # where
    #   tot nr. of pairs >> 255 but probably < 256^2
    #   version is
    #       initialized at 0
    #       will be incremented later, so we don't have to overwrite (which is slow) and can compact periodically to release memory
    #   We need version because heaps just add new values, which will almost always come after the old ones (count typically decreases)
    #   I'm not entirely sure if counts could go up, but this is a safe implementation

    # ---------------------  merge loop  ---------------------
    while len(sym.id2seq) < target_vocab_size:
        # get best pair from heap; skip stale entries
        while heap:
            negc, pair, stamp = heapq.heappop(heap)
            # heap entry version is old
            if ver[pair] != stamp:
                continue
            # this should never happen, so let's error if it does
            # it catches inconsisistencies
            c = -negc
            if counts.get(pair, 0) != c or c == 0:
                # continue
                raise RuntimeError(f"Heap invariant broken for pair {pair}: stored {c}, real {counts.get(pair, 0)}")
            break
        else:
            if debug: print("no more pairs")
            return

        a, b = pair

        # materialize merged token id
        merged_seq = sym.id2seq[a] + sym.id2seq[b]
        new_id = sym.add_symbol(merged_seq)
        merges_out.append((bytes(sym.id2seq[a]), bytes(sym.id2seq[b])))

        # --------- key picture for every site (w,i) where (a,b) occurs ----------
        # Before:      [ ...  L |  a  b  |  R  ... ]
        # Touch old:        (L,a)   (a,b)   (b,R)
        # Merge:            [ ...  L | new |  R  ... ]
        # Touch new:        (L,new)         (new,R)
        # Only these 5 pairs can change at this site; nothing else moves.
        # ------------------------------------------------------------------------

        # pop the entry for this pair in occ
        # Later we will also modify the occ for neighbour pairs of each site
        sites = occ.pop(pair, set())    # <<< set default
        for site in sites:
            print(f">> word 11660: {words[11660]}")
            if len(words[11660])<6:
                print(f">> word 11660 decreased in length")
            if site is None:
                raise RuntimeError(f"sites should never be None")
            wi, i = site
            s = words[wi]
            f = freqs[wi]

            # Validate still (a,b) at this location; if edits made it stale, skip.
            if not (0 <= i < len(s)-1 and s[i] == a and s[i+1] == b):
                # continue
                raise RuntimeError(f"location {(i,i+1)} in word {s} did not contain pair {[a,b]}")

            L = s[i-1] if i-1 >= 0 else None
            R = s[i+2] if i+2 < len(s) else None

            # old neighbors disappear at this site
            print(f"removing stale neighbors of pair {pair}")
            if L is not None:
                _dec((L, a), f, counts, ver, heap)
                _occ_del(occ, (L, a), (wi, i-1))   # move out
            _dec((a, b), f, counts, ver, heap)     # whole pair already popped from occ
            if R is not None:
                _dec((b, R), f, counts, ver, heap)
                _occ_del(occ, (b, R), (wi, i+1))   # move out

            # in-place merge inside the word
            s[i:i+2] = [new_id]

            # new neighbors appear at this site
            print(f"adding new neighbors of pair {pair}")
            if L is not None:
                _inc((L, new_id), f, counts, ver, heap)
                _occ_add(occ, (L, new_id), (wi, i-1))  # move in
            if R is not None:
                _inc((new_id, R), f, counts, ver, heap)
                _occ_add(occ, (new_id, R), (wi, i))    # move in; right pair starts at i

        if debug:
            def b2s(tup):
                return bytes(tup).decode("utf-8", errors="backslashreplace")
            print(f"merge ({a},{b}) -> new_id={new_id} "
                  f"[{b2s(sym.id2seq[a])}+{b2s(sym.id2seq[b])}={b2s(merged_seq)}] "
                  f"distinct_pairs={len(counts)}")


def my_run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    debug: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # make sure they're byte objects
    special_tokens_b = [s.encode("utf-8") if isinstance(s, str) else s for s in special_tokens]

    # (1) pretokenize
    cache_path = Path(input_path)
    cache_path = cache_path.with_name(cache_path.name + ".pretoks.pkl")

    if cache_path.exists():
        if debug:
            print(f"loading pretokens from {cache_path}")
        with cache_path.open("rb") as f:
            pretoks = pickle.load(f)
    else:
        data = _read_bytes(input_path)

        if debug:
            print("computing pretokens")
        pretoks = _build_pretokens(data, special_tokens_b)

        with cache_path.open("wb") as f:
            pickle.dump(pretoks, f)

        # Re-load from disk so downstream code always works with the cached artifact.
        with cache_path.open("rb") as f:
            pretoks = pickle.load(f)

    # (2) set up dataset
    sym = SymTab()
    for s in special_tokens_b:
        sym.add_symbol(tuple(s))

    dataset = _make_dataset(pretoks, debug_output=debug)

    # (3) run merge algorithm
    merges: list[tuple[bytes, bytes]] = []

    if debug:
        print("merging tokens (indexed)")
    train_bpe_incremental(dataset, sym, vocab_size, merges_out=merges, debug=debug)
    id2bytes = {i: bytes(seq) for i, seq in sym.id2seq.items()}
    return id2bytes, merges


    if debug:
        print("merging tokens")
    pbar = tqdm(
        total=vocab_size,
        initial=len(sym.id2seq),
        unit="items",
        disable=not debug,
    )
    while len(sym.id2seq) < vocab_size:
        pairs = _get_stats(dataset)
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
        dataset = _apply_merge(dataset, a, b, new_id)

        pbar.update(1)

    pbar.close()

    id2bytes = {i: bytes(seq) for i, seq in sym.id2seq.items()}

    return id2bytes, merges


if __name__ == "__main__":
    from pathlib import Path

    # filename = Path("data") / "owt_valid.txt"
    # filename = Path("data") / "TinyStoriesV2-GPT4-train.txt"
    filename = Path("data") / "TinyStoriesV2-GPT4-valid.txt"
    print(f"Using {filename} for tokenization")

    special_tokens = [b"<|endoftext|>",
                      b"<|fim_prefix|>",
                      b"<|fim_suffix|>",
                      b"<|fim_middle|>",
                      b"<|file_separator|>"]

    vocab, merges = my_run_train_bpe(filename, 10000, special_tokens, debug=True)
