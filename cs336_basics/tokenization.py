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

# we use a linked list so we can automatically reference them in occ (see _build_indexes)
class Node:
    __slots__ = ("id", "prev", "next", "wi")
    def __init__(self, sym_id: int, wi: int):
        self.id = sym_id
        self.prev = None     # type: Node | None
        self.next = None     # type: Node | None
        self.wi = wi         # word index (to access freqs)

    def __hash__(self):
        # allow putting Node into sets; identity semantics
        return id(self)
    
    def __repr__(self):
        left = self.prev.id if self.prev else None
        right = self.next.id if self.next else None
        return f"Node(id={self.id}, wi={self.wi}, prev={left}, next={right})"

    __str__ = __repr__



def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()
    
def _split_on_specials(data: bytes, special_tokens: list[bytes]) -> list[bytes]:
    pattern = b"|".join(re.escape(tok) for tok in special_tokens)
    return re.split(pattern, data)

def _build_indexes(dataset):
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
    ver = {p: 0 for p in counts}
    for p, c in counts.items():
        heapq.heappush(heap, (-c, p, ver[p]))
    return words, freqs, counts, occ, heap, ver

def _get_occ_for_wi_from_list(occ_list, wi: int):
    return [
        node for node in occ_list if node.wi == wi
    ]

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

    # ---------------------  merge loop  ---------------------
    loopid=0
    while len(sym.id2seq) < target_vocab_size:
        loopid+=1
        print(f'LOOP {loopid}')
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
        if a==b:
            print(f'weird corner case!!!!!!!!!!! {a,b}')


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
        sites = occ.pop(pair, set())
        for left in list(sites):  # snapshot; we'll mutate structures

            ########### TEST
            # print('check if this is already corrupted')
            # ssseq = dataset[left.wi][0]
            # if any(x == y == 111 for x, y in zip(ssseq, ssseq[1:])):
            #     print('first double id!!')
            sites_ones = occ[(111,111)]
            for l in list(sites_ones):
                if l.id != 111 and l.next.id != 111:
                    print(f">> {l.id}, {l.next.id}")
            # print('done')
            ########### TEST


            if left.wi == 8067:
                occ_for_this_node = _get_occ_for_wi_from_list(sites,8067)
                print("left.wi was 8067, occ is calculated")
            if left is None or left.next is None:
                raise ValueError('left or left.next is None!')
            # validate still (a,b) at this site
            if left.id != a or left.next.id != b:
                # THIS NEEDS TO BE DEBUGGED: IT LOOKS LIKE THE ORIGINAL DATASET (dataset[left.wi]) ALWAYS HAD 111,111.
                # BUT 111, LEFT.NEXT.ID should NOT BE INSIDE OCC
                # FOR WORD 371 we started with (32, 100, 111, 111, 114), as is in dataset still
                # Now we have [281, 111, 308] because we merged 32+100 into 281 and 111+114 into 308
                # This means that the end result should not appear in occ for [111,111] any more
                raise ValueError(f'pair {(a,b)} does not agree with {(left.id, left.next.id)}')

            right = left.next
            L = left.prev
            R = right.next
            wi = left.wi
            f = freqs[wi]

            # decrement counts and remove old neighbor sites
            if L is not None:
                _dec((L.id, a), f, counts, ver, heap)
                if not (L.id==a and a==b): # Only delete from occ if L,a is not the same as a,b because that's already popped
                    _occ_del(occ, (L.id, a), L)
            _dec((a, b), f, counts, ver, heap)
            if R is not None:
                _dec((b, R.id), f, counts, ver, heap)
                if not (b==a and R.id==b): # Only delete from occ if b,R is not the same as a,b because that's already popped
                    _occ_del(occ, (b, R.id), right)

            # splice: replace [left(a), right(b)] with [new_id]
            new_node = Node(new_id, wi)
            new_node.prev, new_node.next = L, R
            if L is not None:
                L.next = new_node
            else:
                # left was head of this word
                words[wi] = new_node
            if R is not None:
                R.prev = new_node

            # increment counts and add new neighbor sites
            if L is not None:
                _inc((L.id, new_id), f, counts, ver, heap)
                _occ_add(occ, (L.id, new_id), L)         # pair starts at L
            if R is not None:
                _inc((new_id, R.id), f, counts, ver, heap)
                _occ_add(occ, (new_id, R.id), new_node)  # pair starts at new_node


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
