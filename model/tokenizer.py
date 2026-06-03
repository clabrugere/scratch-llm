import heapq
import re
from collections import Counter, defaultdict


def string_to_byte(input: str, encoding: str) -> list[int]:
    """Encode a string into a list of byte values using the given encoding."""
    return [int(b) for b in input.encode(encoding)]


class TokenSequence:
    """Doubly-linked list over a fixed array of token IDs, supporting O(1) node removal."""

    def __init__(self, values: list[int]):
        n = len(values)
        self.values = values
        self.left = list(range(-1, n - 1))
        self.right = list(range(1, n + 1))
        self.n = n

    def delete(self, i: int) -> None:
        """Remove node i from the sequence, linking its neighbors directly."""
        if self.left[i] >= 0:
            self.right[self.left[i]] = self.right[i]
        if self.right[i] < self.n:
            self.left[self.right[i]] = self.left[i]

    def has_left(self, i: int) -> bool:
        return self.left[i] >= 0

    def has_right(self, i: int) -> bool:
        return self.right[i] < self.n

    def __iter__(self):
        i = 0
        while i < self.n:
            yield i, self.values[i]
            i = self.right[i]

    def __len__(self):
        return self.n


class PairIndex:
    """Tracks pair frequencies and their positions for incremental BPE training."""

    def __init__(self):
        self.counts = Counter()
        self.positions = defaultdict(set)
        self.heap = []

    def add(self, pair: tuple[int, int], pos: int) -> None:
        """Record that pair starts at pos."""
        self.counts[pair] += 1
        self.positions[pair].add(pos)
        heapq.heappush(self.heap, (-self.counts[pair], pair))

    def remove(self, pair: tuple[int, int], pos: int) -> None:
        """Remove the occurrence of pair at pos."""
        self.counts[pair] -= 1
        self.positions[pair].discard(pos)
        if self.counts[pair] > 0:
            heapq.heappush(self.heap, (-self.counts[pair], pair))
        else:
            del self.counts[pair]
            del self.positions[pair]

    def most_frequent(self) -> tuple[tuple[int, int], int]:
        """Return the pair with the highest count, skipping stale heap entries."""
        while self.heap:
            neg_count, pair = self.heap[0]
            count = -neg_count
            # stale if count no longer matches (pair was updated or removed since push)
            if self.counts.get(pair, 0) == count:
                return pair, count
            heapq.heappop(self.heap)

    def pop(self, pair: tuple[int, int]) -> set[int]:
        """Return all positions for pair and remove it from the index."""
        del self.counts[pair]
        return self.positions.pop(pair)

    def __bool__(self) -> bool:
        return bool(self.counts)


class BPETokenizer:
    def __init__(self, max_vocab_size: int):
        """UTF-8 encoding represents characters with 1, 2, 3 or 4 consecutive bytes, which means that the base vocabulary
        size is 256. It also guarantee that we can't have out of vocabulary tokens as long as the input string can be
        encoded in UTF-8."""

        if max_vocab_size <= 256:
            raise ValueError(f"max_vocab_size must be at least 256, got '{max_vocab_size}'.")

        self.max_vocab_size = max_vocab_size
        self.reset()

    def reset(self) -> None:
        """Reset the tokenizer to the initial 256-byte vocabulary, discarding all learned merges."""
        self.pairs = {}
        self.id_to_token = {i: bytes([i]) for i in range(256)}
        self.next_id = 256
        self.special_to_id = {}
        self.id_to_special = {}
        self._special_pattern = None

    def register_special_token(self, token: str) -> int:
        """Assign a unique ID to a special token (e.g. BOS, EOS) if not already registered."""
        if token not in self.special_to_id:
            self.special_to_id[token] = self.next_id
            self.id_to_special[self.next_id] = token
            self.next_id += 1
            self._special_pattern = None  # Invalidate cached regex pattern

        return self.special_to_id[token]

    def train(self, input: str, stop_early: bool = False) -> None:
        """Learn BPE merges from input text until max_vocab_size is reached."""
        # index all adjacent pairs in the initial byte sequence
        seq = TokenSequence(string_to_byte(input, "utf-8"))
        index = PairIndex()
        for pos, token in seq:
            if seq.has_right(pos):
                index.add((token, seq.values[seq.right[pos]]), pos)

        while self.vocab_size < self.max_vocab_size:
            if not index:
                break

            pair, count = index.most_frequent()

            if stop_early and count == 1:
                break

            # register the new merged token
            new_id = self.next_id
            self.pairs[pair] = new_id
            self.id_to_token[new_id] = self.id_to_token[pair[0]] + self.id_to_token[pair[1]]
            self.next_id += 1

            # apply the merge at every occurrence and update only the two affected boundary pairs
            for pos in index.pop(pair):
                right_pos = seq.right[pos]

                # skip stale positions: right neighbor was already consumed by an earlier merge in this loop
                if right_pos >= seq.n or seq.left[right_pos] != pos:
                    continue

                # left boundary: (left_token, pair[0]) -> (left_token, new_id)
                if seq.has_left(pos):
                    left_pos = seq.left[pos]
                    index.remove((seq.values[left_pos], pair[0]), left_pos)
                    index.add((seq.values[left_pos], new_id), left_pos)

                # right boundary: (pair[1], right_right_token) -> (new_id, right_right_token)
                if seq.has_right(right_pos):
                    right_right_pos = seq.right[right_pos]
                    index.remove((pair[1], seq.values[right_right_pos]), right_pos)
                    index.add((new_id, seq.values[right_right_pos]), pos)

                seq.values[pos] = new_id
                seq.delete(right_pos)

    def _encode_non_special(self, input: str) -> list[int]:
        """Encode a string by applying learned merges in priority order, ignoring special tokens."""
        seq = TokenSequence(string_to_byte(input, "utf-8"))

        if len(seq) < 2:
            return seq.values

        # seed the heap with all mergeable adjacent pairs; merge ID doubles as priority (lower = earlier learned)
        heap = []
        for pos, token in seq:
            if seq.has_right(pos):
                pair = (token, seq.values[seq.right[pos]])
                if pair in self.pairs:
                    heapq.heappush(heap, (self.pairs[pair], pos, token, seq.values[seq.right[pos]]))

        while heap:
            # heap entry stores expected token values at push time for stale-entry detection
            merged_id, pos, expected_left, expected_right = heapq.heappop(heap)

            # skip stale entries: pos was deleted, values changed, or pos is no longer left-adjacent to its right neighbor
            right_pos = seq.right[pos]
            if (
                right_pos >= seq.n
                or seq.values[pos] != expected_left
                or seq.values[right_pos] != expected_right
                or seq.left[right_pos] != pos
            ):
                continue

            # apply merge: overwrite left token with merged ID, remove right token from the sequence
            seq.values[pos] = merged_id
            seq.delete(right_pos)

            # only the two boundary pairs are affected; push them if they have a merge
            if seq.has_left(pos):
                pair = (seq.values[seq.left[pos]], merged_id)
                if pair in self.pairs:
                    heapq.heappush(heap, (self.pairs[pair], seq.left[pos], seq.values[seq.left[pos]], merged_id))
            if seq.has_right(pos):
                pair = (merged_id, seq.values[seq.right[pos]])
                if pair in self.pairs:
                    heapq.heappush(heap, (self.pairs[pair], pos, merged_id, seq.values[seq.right[pos]]))

        return [token for _, token in seq]

    def encode(self, input: str) -> list[int]:
        """Encode a string into token IDs, handling special tokens before applying BPE merges."""
        if len(self.special_to_id) > 0:
            if self._special_pattern is None:
                # sort longest-first so a token that is a prefix of another doesn't shadow it
                self._special_pattern = re.compile(
                    f"({'|'.join(re.escape(t) for t in sorted(self.special_to_id.keys(), key=len, reverse=True))})"
                )
            splits = self._special_pattern.split(input)
        else:
            splits = [input]

        indices = []

        for split in splits:
            if split in self.special_to_id:
                indices.append(self.special_to_id[split])
            else:
                indices.extend(self._encode_non_special(split))

        return indices

    def decode(self, indices: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back into a string."""
        decoded = []

        for id in indices:
            if id in self.id_to_special:
                if not skip_special_tokens:
                    decoded.append(self.id_to_special[id].encode("utf-8"))
            else:
                decoded.append(self.id_to_token[id])

        decoded = b"".join(decoded).decode("utf-8")

        return decoded

    @property
    def vocab_size(self) -> int:
        return self.next_id
