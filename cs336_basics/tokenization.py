import os
import pickle
import regex as re
import heapq

from collections import Counter, defaultdict
from pathlib import Path
from typing import BinaryIO
from tqdm import tqdm

from datetime import datetime

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

# we use a linked list so we can automatically reference them in occ (see _build_indexes)
class Node:
    __slots__ = ("id", "prev", "next", "wi", "pos")
    def __init__(self, sym_id: int, wi: int):
        self.id = sym_id
        self.prev = None     # type: Node | None
        self.next = None     # type: Node | None
        self.wi = wi         # word index (to access freqs)
        self.pos = None       # position within word (to be able to sort left to right)

    def __hash__(self):
        # allow putting Node into sets; identity semantics
        return id(self)
    
    def __repr__(self):
        left = self.prev.id if self.prev else None
        right = self.next.id if self.next else None
        return f"Node(id={self.id}, wi={self.wi} (pos {self.pos}), prev={left}, next={right})"

    __str__ = __repr__



class _DescendingSeq:
    """Wrap sequence tuples so the heap prefers lexicographically larger ones."""

    __slots__ = ("seq",)

    def __init__(self, seq: tuple[int, ...]):
        self.seq = seq

    def __lt__(self, other: "_DescendingSeq") -> bool:
        if not isinstance(other, _DescendingSeq):
            return NotImplemented
        return self.seq > other.seq

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _DescendingSeq):
            return NotImplemented
        return self.seq == other.seq


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()
    
def _split_on_specials(data: bytes, special_tokens: list[bytes]) -> list[bytes]:
    pattern = b"|".join(re.escape(tok) for tok in special_tokens)
    return re.split(pattern, data)

def _build_indexes(dataset, sym):
    """"
    Build:
        - words: list[Node | None]: head node of each word's linked list
        - freqs: list[int]: frequency of word i
        - counts: Counter[(int,int)]: weighted pair counts
        - occ: dict[(int,int)]: set[Node] where Node is the LEFT node of the pair
        - heap, ver: heap over (-count, pair, ver[pair]) with versions for lazy invalidation
    """
    words = [None] * len(dataset)
    freqs = [f for _, f in dataset]
    counts = Counter()
    occ = defaultdict(set)

    # build linked lists per word and index pairs by left node
    for wi, (seq, f) in enumerate(dataset):
        if not seq:
            continue
        # create nodes
        nodes = [Node(x, wi) for x in seq]
        # link
        for i in range(len(nodes)):
            nodes[i].pos = i
        for i in range(1, len(nodes)):
            nodes[i-1].next = nodes[i]
            nodes[i].prev = nodes[i-1]
        words[wi] = nodes[0]

        # index pairs
        if len(nodes) >= 2:
            n = nodes[0]
            while n and n.next:
                p = (n.id, n.next.id)
                counts[p] += f
                occ[p].add(n)   # left node is the site
                n = n.next

    # heap + versions
    heap = []
    ver = {}
    for p, c in counts.items():
        ver[p] = 0
        seq_left = sym.id2seq[p[0]]
        seq_right = sym.id2seq[p[1]]
        heapq.heappush(
            heap,
            (-c, _DescendingSeq(seq_left), _DescendingSeq(seq_right), ver[p], p),
        )
    return words, freqs, counts, occ, heap, ver

def _heap_push(pair, counts, ver, heap, sym):
    ver[pair] = ver.get(pair, 0) + 1
    c = counts.get(pair)
    if c is None:
        raise RuntimeError(f"Trying push to heap pair {pair} with version {ver[pair]} but counts has no entry")
    if c < 0:
        raise RuntimeError(f"Trying to push to heap pair {pair} with version {ver[pair]} with negative count {c}")
    # print(f'Pushed new entry {(-c, pair, ver[pair])} to heap')

    seq_left = sym.id2seq[pair[0]]
    seq_right = sym.id2seq[pair[1]]
    heapq.heappush(
        heap,
        (-c, _DescendingSeq(seq_left), _DescendingSeq(seq_right), ver[pair], pair),
    )

def _dec(pair, f, counts, ver, heap, sym):
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
    _heap_push(pair, counts, ver, heap, sym)

def _inc(pair, f, counts, ver, heap, sym):
    # print(f"increasing pair {pair} by {f}")
    if f == 0: 
        raise RuntimeError(f"Tried increasing counts for pair {pair} by 0 which is not possible")
    counts[pair] = counts.get(pair, 0) + f # lazily create
    _heap_push(pair, counts, ver, heap, sym)

def _occ_del(occ, pair, left_node):
    s = occ.get(pair)
    s.discard(left_node)
    if not s: # this pair stopped occuring completely
        occ.pop(pair, None)

def _occ_add(occ, pair, left_node):
    occ[pair].add(left_node)


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



def train_bpe_incremental(dataset, sym, target_vocab_size, merges_out, debug=False):
    words, freqs, counts, occ, heap, ver = _build_indexes(dataset, sym)

    # ---------------------  merge loop  ---------------------
    loopid=0
    while len(sym.id2seq) < target_vocab_size:
        # get best pair from heap; skip stale entries
        while heap:
            negc, _seq_left, _seq_right, stamp, pair = heapq.heappop(heap)
            # heap entry version is old
            if ver.get(pair) != stamp:
                continue
            # this should never happen, so let's error if it does as it catches inconsistencies
            c = -negc
            if counts.get(pair, 0) != c or c == 0:
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

        # ---------------------  affected occurence loop  ---------------------

        # --------- key picture for every site (w,i) where (a,b) occurs ----------
        # Before:      [ ...  L |  a  b  |  R  ... ]
        # Touch old:        (L,a)   (a,b)   (b,R)
        # Merge:            [ ...  L | new |  R  ... ]
        # Touch new:        (L,new)         (new,R)
        # Only these 5 pairs can change at this site; nothing else moves.
        # ------------------------------------------------------------------------

        # pop the entry for this pair in occ
        # Later we will also modify the occ for neighbour pairs of each site
        sites_list = list(occ.pop(pair, set()))
        sites = sorted(
            sites_list, 
            key=lambda node: (node.wi, node.pos)
        )

        sites_to_skip = set() # will contain corner cases, such as repeated characters, that need to be deleted
        
        for left in sites:  # snapshot; we'll mutate structures
            if left in sites_to_skip:
                # This list will contain cases such as o,o,o where the second pair needs to be removed
                # because it doesn't exist any more after the first pair was done
                continue

            if left is None or left.next is None:
                raise ValueError('left {left} or left.next {left.next} cannot be None!')
            # validate still (a,b) at this site
            if left.id != a or left.next.id != b:
                raise ValueError(f'pair {(a,b)} does not agree with {(left.id, left.next.id)}')

            right = left.next
            L = left.prev
            R = right.next
            wi = left.wi
            f = freqs[wi]

            # decrement counts and remove old neighbor sites
            if L is not None:
                _dec((L.id, a), f, counts, ver, heap, sym)
                if L.id==a and a==b: # remove from current sites, not occ
                    sites_to_skip.add(left.prev)
                else:
                    _occ_del(occ, (L.id, a), L)

            _dec((a, b), f, counts, ver, heap, sym)
            if R is not None:
                _dec((b, R.id), f, counts, ver, heap, sym)
                if b==a and R.id==b: # remove from current sites, not occ
                    sites_to_skip.add(right)
                else:
                    _occ_del(occ, (b, R.id), right)

            # splice: replace [left(a), right(b)] with [new_id]
            new_node = Node(new_id, wi)
            new_node.prev, new_node.next = L, R
            new_node.pos = left.pos
            if L is not None:
                L.next = new_node
            else:
                # left was head of this word
                words[wi] = new_node
            if R is not None:
                R.prev = new_node

            # increment counts and add new neighbor sites
            if L is not None:
                _inc((L.id, new_id), f, counts, ver, heap, sym)
                _occ_add(occ, (L.id, new_id), L)         # pair starts at L
            if R is not None:
                _inc((new_id, R.id), f, counts, ver, heap, sym)
                _occ_add(occ, (new_id, R.id), new_node)  # pair starts at new_node


        if debug:
            def b2s(tup):
                return bytes(tup).decode("utf-8", errors="backslashreplace")
            print(f"LOOP {loopid} - merge ({a},{b}) -> new_id={new_id} "
                  f"[{b2s(sym.id2seq[a])}+{b2s(sym.id2seq[b])}={b2s(merged_seq)}] "
                  f"distinct_pairs={len(counts)}")
        
        loopid+=1

    if debug:
        # check if for all places with count 0 we also have no occurences left
        for pair in counts.keys():
            if counts[pair] == 0:
                if occ[pair]:
                    raise RuntimeError(f"for pair {pair} the count was {counts[pair]} but occurences was nonzero: {occ[pair]}")

        # check if for all places with no occurences we also have no count zero
        for pair in occ.keys():
            if not occ[pair]:
                if counts[pair] > 0:
                    raise RuntimeError(f"for pair {pair} the count was {counts[pair]} but occurences was nonzero: {occ[pair]}")
        
        # print the tokens
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        id2seq_output_file = f"output_id2seq_{timestamp}.txt"

        with open(id2seq_output_file, "wb") as f:
            for id, seq in sym.id2seq.items():
                f.write(f"id {id} - seq {bytes(seq)} ({seq})\n".encode("utf-8"))


    return


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



if __name__ == "__main__":
    from pathlib import Path

    # filename = Path("data") / "owt_valid.txt"
    # filename = Path("data") / "TinyStoriesV2-GPT4-train.txt"
    # filename = Path("data") / "TinyStoriesV2-GPT4-valid.txt"
    # filename = Path("data") / "owt_valid.txt"
    filename = Path("data") / "owt_train.txt"
    print(f"Using {filename} for tokenization")

    special_tokens = [b"<|endoftext|>",
                      b"<|fim_prefix|>",
                      b"<|fim_suffix|>",
                      b"<|fim_middle|>",
                      b"<|file_separator|>"]

    vocab, merges = my_run_train_bpe(filename, 32000, special_tokens, debug=True)

    # print the vocab and the merges
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vocab_output_file = f"output_vocab_{timestamp}.txt"
    merges_output_file = f"output_merges_{timestamp}.txt"

    with open(vocab_output_file, "wb") as f:
        for id, v in vocab.items():
            f.write(f"{id} - {v}\n".encode("utf-8"))

    with open(merges_output_file, "wb") as f:
        for id, merge in enumerate(merges):
            f.write(f"{id} - {merge}\n".encode("utf-8"))

    print('done')
