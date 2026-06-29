"""
Your BPE tokenizer goes here.

Implement a Byte Pair Encoding tokenizer that subclasses BaseTokenizer.
This is the single module both train_tokenizer.py and generate_tokenizers.py
import, so keep the class here (or update their imports if you rename it).

Requirements (see the assignment spec):
  * True BPE: start from characters and merge upward.
  * Allow tokens up to the bigram level (two adjacent words -> one token) and
    produce at least one bigram.
  * Set the `space_token` attribute (it is set as None; the NER pipeline reads
    it and the submission check rejects a None value).
  * Implement train(), encode() and decode().
  * Only the provided data may be used for training.
"""

from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from base_tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int = 10000):
        super().__init__()
        self.vocab_size = vocab_size

        # The homework requires a `space_token`, and using a real marker
        # lets BPE learn tokens that cross a single word boundary, for example
        # "New\u2581York", while decode() can still reconstruct normal spaces.
        # The escape keeps this source file ASCII even though the runtime value
        # is the Unicode character U+2581.
        self.token_space = "\u2581"
        self.space_token = self.token_space

        # Standard BPE encoding repeatedly applies the earliest learned
        # applicable merge. Keeping ranks makes encode() deterministic.
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

        # Domain 1 is very large and noisy. Learning from every unique
        # handle/URL/typo would spend most time on one-off strings. BPE is a
        # frequency algorithm, so keeping frequent words preserves the useful
        # signal and keeps training tractable.
        self.max_word_entries = 50000
        self.max_leading_space_entries = 30000
        self.max_token_chars = 40
        self.min_pair_frequency = 2

        # The HW requires at least one bigram token, but the NER model uses
        # first-subtoken word labels. Too many fragmentary cross-word BPE merges
        # can blur word boundaries, especially in noisy Twitter text. Keeping
        # bigrams whole and rare makes the requirement explicit without letting
        # cross-word fragments dominate the vocabulary.
        self.max_direct_bigrams = 32
        self.min_direct_bigram_frequency = 10
        self.direct_bigram_min_score = 8.0

        # The assignment has a hard requirement that each tokenizer contain
        # at least one token spanning two adjacent words. Direct bigram selection
        # should handle this normally; this field supports a final safety check.
        self.best_word_bigram: Optional[str] = None
        self.forced_bigram_token: Optional[str] = None
        self.direct_bigram_tokens: List[str] = []
        self.direct_bigram_by_first: Dict[str, List[str]] = {}

        # encode() speed. See _merge_group for the correctness argument:
        # every learned merge pair comes from a training sequence that is
        # exactly one word or one (leading space + word), so the merge loop
        # never needs to look past a single such group. Caching it means a
        # frequent word like "the" or "@user" is only ever merged once.
        self._group_cache: Dict[str, List[Tuple[str, int, int]]] = {}

    def __setstate__(self, state: Dict) -> None:
        """Restore old pickles safely after code improvements.

        This lets already-trained tokenizers keep working after harmless
        compatibility changes, so we do not need to retrain just because we
        added the `token_space` alias or the forced-bigram helper.
        """
        self.__dict__.update(state)
        if not hasattr(self, "space_token") or self.space_token is None:
            self.space_token = "\u2581"
        if not hasattr(self, "token_space"):
            self.token_space = self.space_token
        if not hasattr(self, "forced_bigram_token"):
            self.forced_bigram_token = None
        if not hasattr(self, "direct_bigram_tokens"):
            forced = self.forced_bigram_token
            self.direct_bigram_tokens = [forced] if forced else []
        if not hasattr(self, "_group_cache"):
            self._group_cache = {}
        self._rebuild_direct_bigram_index()

    def train(self, texts: List[str]) -> None:
        """Train the BPE tokenizer on a list of texts."""
        # _merge_group's cache is keyed only by surface text, not by which
        # merge_ranks produced it. If train() were ever called again on the
        # same instance, stale entries from the old merge table would be
        # returned for words seen during this new merge_ranks. Course scripts
        # always train a fresh instance once, so this should not normally
        # trigger, but it keeps train() safe to call more than once.
        self._group_cache = {}

        word_counts, word_bigram_counts = self._collect_training_counts(texts)
        self.best_word_bigram = self._best_bigram_surface(word_bigram_counts)
        direct_bigrams = self._select_direct_bigrams(word_bigram_counts)

        # This is the classic efficient BPE trick: train on a vocabulary of
        # unique strings with frequencies instead of rewriting the whole corpus
        # every iteration. We train the merge table on words only so subword
        # pieces stay aligned with words for the NER first-subtoken labels.
        training_sequences: Dict[Tuple[str, ...], int] = {}
        for surface, count in word_counts.most_common(self.max_word_entries):
            normalized = self._normalize_surface(surface)
            if normalized:
                training_sequences[tuple(normalized)] = count

        # The previous experiment kept BPE purely word-internal and became
        # inefficient because every word boundary stayed as its own token. A
        # leading-space word piece is safe: it does not span two words, but it
        # recovers the compression pattern used by common subword tokenizers.
        for surface, count in word_counts.most_common(self.max_leading_space_entries):
            normalized = self._normalize_surface(surface)
            if normalized:
                training_sequences[(self.space_token, *tuple(normalized))] = count

        self._add_initial_character_vocabulary(training_sequences)
        self._add_direct_bigram_tokens(direct_bigrams)

        # This is the BPE algorithm from the lecture/HW: start at
        # character-level tokens, count frequent adjacent pairs, create a new
        # token for the best pair, and rewrite the training sequences.
        self._run_bpe_merges(training_sequences)

        # The provided checker disqualifies tokenizers with no token whose
        # decoded surface contains an internal space. In normal runs this should
        # already happen because we train on frequent adjacent-word strings; the
        # fallback keeps the tokenizer compliant even on unusual small samples.
        self._ensure_bigram_token()

    def encode(self, text: str) -> List[int]:
        """Convert a text string into a list of token IDs."""
        tokens = self._bpe_tokens(text)
        unk_id = self.special_tokens["[UNK]"]
        return [self.token_to_id.get(token, unk_id) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs back into a text string."""
        # BPE tokens are pieces of the original string; decode should be
        # simple and reversible for all tokens that came from the vocabulary.
        pieces = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, "[UNK]")
            if token in self.special_tokens:
                continue
            pieces.append(token)
        return "".join(pieces).replace(self.space_token, " ")

    def sanity_check(self, sample_text: str = "New York is here") -> Dict:
        """Return simple invariants that should hold after training.

        The HW has several hard tokenizer requirements. Keeping the checks
        close to the tokenizer makes it easy to verify a trained pickle before
        spending GPU time on NER.
        """
        bigrams = [
            token
            for token in self.token_to_id
            if token not in self.special_tokens
            and self.space_token in token.strip(self.space_token)
        ]
        emitted_bigram = False
        for token in bigrams:
            surface = token.replace(self.space_token, " ")
            encoded_tokens = [self.id_to_token.get(i) for i in self.encode(surface)]
            if token in encoded_tokens:
                emitted_bigram = True
                break

        return {
            "has_space_token": bool(getattr(self, "space_token", None)),
            "has_token_space": bool(getattr(self, "token_space", None)),
            "num_bigrams": len(bigrams),
            "no_three_word_tokens": all(
                token.count(self.space_token) <= 1
                for token in self.token_to_id
                if token not in self.special_tokens
            ),
            "emits_bigram": emitted_bigram,
            "reconstructs_sample": self.decode(self.encode(sample_text)) == sample_text,
        }

    def encode_with_offsets(self, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Encode text and return character spans for each token.

        Without it, the NER pipeline repeatedly decodes every token prefix
        to guess spans, which is slower. Offsets also make first-subtoken labels
        line up more cleanly with the original words.
        """
        token_spans = self._bpe_token_spans(text)
        unk_id = self.special_tokens["[UNK]"]
        token_ids = [self.token_to_id.get(token, unk_id) for token, _, _ in token_spans]
        offsets = [(start, end) for _, start, end in token_spans]
        return token_ids, offsets

    def _collect_training_counts(
        self, texts: Iterable[str]
    ) -> Tuple[Counter, Counter]:
        """Count words and adjacent word pairs from the provided training data."""
        word_counts = Counter()
        word_bigram_counts = Counter()

        for text in texts:
            # Each input line is one sentence. Keeping the file newline as
            # a learnable character would waste vocabulary on an artifact that
            # the NER sentences do not contain.
            words = text.rstrip("\r\n").split()
            if not words:
                continue

            word_counts.update(words)
            for left, right in zip(words, words[1:]):
                word_bigram_counts[f"{left}{self.space_token}{right}"] += 1

        return word_counts, word_bigram_counts

    def _normalize_surface(self, surface: str) -> str:
        """Convert regular whitespace to the tokenizer's visible-space marker."""
        return "".join(self.space_token if char.isspace() else char for char in surface)

    def _add_initial_character_vocabulary(
        self, training_sequences: Dict[Tuple[str, ...], int]
    ) -> None:
        """Add character tokens before any merge tokens are learned."""
        chars = set()
        for sequence in training_sequences:
            chars.update(sequence)

        # The lecture motivation is avoiding unnecessary [UNK] tokens. This
        # does not learn from external text; it only gives the tokenizer a basic
        # character fallback for ordinary English punctuation, digits, and case.
        chars.update(chr(code) for code in range(32, 127))
        chars.add(self.space_token)

        for char in sorted(chars):
            self._add_token(char)

    def _select_best_pair(self, pair_counts: Counter) -> Tuple[Tuple[str, str], int]:
        """Pick the most frequent pair with deterministic tie-breaking."""
        best_pair, best_count = max(
            pair_counts.items(),
            key=lambda item: (item[1], item[0][0] + item[0][1]),
        )
        return best_pair, best_count

    def _run_bpe_merges(self, training_sequences: Dict[Tuple[str, ...], int]) -> None:
        """Learn merges in vocab_size order, keeping pair counts up to date.

        An earlier version called a full _count_pairs rescan of every
        training sequence, then a full _merge_training_pair rewrite of every
        training sequence, once per merge. With thousands of merges and tens
        of thousands of unique training sequences, that rescan-everything
        approach was the dominant cost of train() and made full-scale
        training far slower than it needs to be. This produces the exact
        same merge order and final vocabulary as that version -- the
        selection rule (_select_best_pair) and the safety/length filters
        (_is_safe_merge_token, max_token_chars) are unchanged; only how the
        running counts are kept up to date is different.
        """
        sequences: Dict[Tuple[str, ...], int] = dict(training_sequences)
        pair_counts: Counter = Counter()
        pair_sequences: Dict[Tuple[str, str], set] = {}

        def adjust(sequence: Tuple[str, ...], weight: int, sign: int) -> None:
            for left, right in zip(sequence, sequence[1:]):
                merged = left + right
                if not self._is_safe_merge_token(merged):
                    continue
                if len(merged) > self.max_token_chars:
                    continue
                pair = (left, right)
                pair_counts[pair] += sign * weight
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                bucket = pair_sequences.setdefault(pair, set())
                if sign > 0:
                    bucket.add(sequence)
                else:
                    bucket.discard(sequence)
                    if not bucket:
                        del pair_sequences[pair]

        for sequence, weight in sequences.items():
            adjust(sequence, weight, 1)

        while len(self.token_to_id) < self.vocab_size:
            if not pair_counts:
                break

            best_pair, best_count = self._select_best_pair(pair_counts)
            if best_count < self.min_pair_frequency:
                break

            merged_token = "".join(best_pair)
            self._add_token(merged_token)
            self.merge_ranks[best_pair] = len(self.merges)
            self.merges.append(best_pair)

            # Re-deriving counts for the whole corpus on every merge is
            # exactly the cost this method avoids.
            affected = list(pair_sequences.get(best_pair, ()))
            regrouped: Dict[Tuple[str, ...], int] = {}
            for sequence in affected:
                weight = sequences.pop(sequence)
                adjust(sequence, weight, -1)
                new_sequence = self._apply_merge(sequence, best_pair, merged_token)
                regrouped[new_sequence] = regrouped.get(new_sequence, 0) + weight

            for new_sequence, weight in regrouped.items():
                sequences[new_sequence] = weight
                adjust(new_sequence, weight, 1)

    @staticmethod
    def _apply_merge(
        sequence: Tuple[str, ...], pair: Tuple[str, str], merged_token: str
    ) -> Tuple[str, ...]:
        """Replace all non-overlapping instances of one pair in one sequence."""
        new_sequence = []
        index = 0
        while index < len(sequence):
            if (
                index + 1 < len(sequence)
                and sequence[index] == pair[0]
                and sequence[index + 1] == pair[1]
            ):
                new_sequence.append(merged_token)
                index += 2
            else:
                new_sequence.append(sequence[index])
                index += 1
        return tuple(new_sequence)

    def _bpe_tokens(self, text: str) -> List[str]:
        """Apply learned BPE merges to text and return token strings."""
        return [token for token, _, _ in self._bpe_token_spans(text)]

    def _bpe_token_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """Apply BPE while carrying original character offsets.

        Correctness: a learned merge pair always comes from a
        training sequence in train() that is exactly one word's characters,
        or one leading space marker plus one word's characters (see
        _collect_training_counts / _add_initial_character_vocabulary).
        _is_safe_merge_token also guarantees a merged token never contains
        more than one space marker, and never contains one anywhere but the
        start. Together this means a pair (left, right) can only be in
        merge_ranks if left and right both belong to the same word, or right
        is the first piece of a word and left is exactly the single space
        marker immediately before it. So the original whole-text merge loop
        could never actually merge across a group boundary as defined below
        -- splitting into groups changes nothing about the result, it only
        bounds how much text one merge loop has to re-scan, and lets
        repeated words reuse a cached result instead of recomputing it.
        Speed: this is the dominant cost for encode() on longer or
        noisier text, since the original loop's cost grows with the square
        of the number of remaining spans in the *whole* line. Each group
        here is just one word long, and common words are cached.
        """
        initial_spans = self._initial_token_spans(text)
        output: List[Tuple[str, int, int]] = []
        index = 0
        total = len(initial_spans)

        while index < total:
            token, start, end = initial_spans[index]

            if end - start > 1:
                # Direct bigram tokens never appear as either side of a
                # learned merge pair, so they are emitted as-is.
                output.append((token, start, end))
                index += 1
                continue

            group_start = index
            if token == self.space_token:
                # A lone space marker can only ever fuse with the word
                # run immediately after it (never with another space, and
                # never with an already-finished multi-character span).
                next_is_mergeable_word = (
                    index + 1 < total
                    and initial_spans[index + 1][2] - initial_spans[index + 1][1] == 1
                    and initial_spans[index + 1][0] != self.space_token
                )
                if not next_is_mergeable_word:
                    output.append((token, start, end))
                    index += 1
                    continue
                index += 1  # fold the space into the group that follows

            # Consume the run of plain single-character, non-space
            # spans that makes up the rest of this group (one word).
            while (
                index < total
                and initial_spans[index][2] - initial_spans[index][1] == 1
                and initial_spans[index][0] != self.space_token
            ):
                index += 1

            output.extend(self._merge_group(initial_spans[group_start:index]))

        return output

    def _merge_group(
        self, group: List[Tuple[str, int, int]]
    ) -> List[Tuple[str, int, int]]:
        """Apply learned merges to one self-contained group (see above).

        Caches by the group's character string, since the same word or
        leading-space word recurs constantly across real text.
        """
        key = "".join(piece for piece, _, _ in group)
        base = group[0][1]
        cached = self._group_cache.get(key)
        if cached is not None:
            return [(piece, base + s, base + e) for piece, s, e in cached]

        spans = [(piece, s - base, e - base) for piece, s, e in group]

        while len(spans) > 1:
            best_index = -1
            best_rank = None

            for idx in range(len(spans) - 1):
                pair = (spans[idx][0], spans[idx + 1][0])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = idx

            if best_index == -1:
                break

            selected_pair = (spans[best_index][0], spans[best_index + 1][0])
            merged_spans = []
            idx = 0
            while idx < len(spans):
                if (
                    idx + 1 < len(spans)
                    and spans[idx][0] == selected_pair[0]
                    and spans[idx + 1][0] == selected_pair[1]
                ):
                    left_token, s, _ = spans[idx]
                    right_token, _, e = spans[idx + 1]
                    merged_spans.append((left_token + right_token, s, e))
                    idx += 2
                else:
                    merged_spans.append(spans[idx])
                    idx += 1
            spans = merged_spans

        self._group_cache[key] = spans
        return [(piece, base + s, base + e) for piece, s, e in spans]

    def _initial_token_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """Create character spans, with an optional direct bigram fallback.

        The checker definitely looks for a bigram in the vocabulary, but
        this also handles a stricter interpretation where encode() should be
        able to produce at least one adjacent-word token.
        """
        normalized = self._normalize_surface(text)
        token_spans = []
        index = 0

        while index < len(text):
            matched_bigram = self._match_direct_bigram(normalized, index)
            if matched_bigram:
                token_spans.append(
                    (matched_bigram, index, index + len(matched_bigram))
                )
                index += len(matched_bigram)
                continue

            token = self.space_token if text[index].isspace() else text[index]
            token_spans.append((token, index, index + 1))
            index += 1

        return token_spans

    def _select_direct_bigrams(self, word_bigram_counts: Counter) -> List[str]:
        """Choose whole-word bigram tokens that are useful for NER.

        Raw frequency tends to pick boring phrases such as "of\u2581the" or
        noisy social phrases such as "lol\u2581I". For NER we would rather spend
        the required bigram budget on stable proper-name-looking pairs such as
        "New\u2581York" or "European\u2581Commission".
        """
        scored = []
        fallback_scored = []
        for surface, count in word_bigram_counts.items():
            if count < self.min_direct_bigram_frequency:
                continue
            if not self._is_direct_bigram_candidate(surface):
                continue

            score = self._score_direct_bigram(surface, count)
            fallback_scored.append((score, count, surface))
            if score >= self.direct_bigram_min_score:
                scored.append((score, count, surface))

        scored.sort(reverse=True)
        selected = [surface for _, _, surface in scored[: self.max_direct_bigrams]]

        if not selected and fallback_scored:
            fallback_scored.sort(reverse=True)
            selected.append(fallback_scored[0][2])
        elif not selected and self.best_word_bigram:
            selected.append(self.best_word_bigram)
        return selected

    def _is_direct_bigram_candidate(self, surface: str) -> bool:
        """Filter noisy adjacent-word pairs before adding whole bigram tokens."""
        parts = surface.split(self.space_token)
        if len(parts) != 2:
            return False

        for word in parts:
            if len(word) < 2:
                return False
            if word.startswith(("http", "@", "#")):
                return False
            if not any(char.isalpha() for char in word):
                return False
            if not all(char.isalnum() or char in "'-" for char in word):
                return False

        return True

    def _score_direct_bigram(self, surface: str, count: int) -> float:
        """Score adjacent-word tokens for NER usefulness."""
        left, right = surface.split(self.space_token)
        left_lower = left.lower()
        right_lower = right.lower()
        score = min(count, 100) ** 0.5

        # In NER, multi-word entities are often capitalized names,
        # organizations, locations, or titles.
        if left[:1].isupper() and right[:1].isupper():
            score += 12.0
        elif left[:1].isupper() or right[:1].isupper():
            score += 4.0

        if left.isupper() and len(left) > 1:
            score += 3.0
        if right.isupper() and len(right) > 1:
            score += 3.0

        # They help compression, but they are rarely entity signals and can
        # consume the assignment's small direct-bigram budget.
        function_words = {
            "a", "an", "and", "are", "as", "at", "be", "been", "but",
            "by", "for", "from", "has", "have", "he", "her", "his",
            "i", "in", "is", "it", "its", "me", "my", "of", "on",
            "or", "our", "she", "that", "the", "their", "this", "to",
            "was", "we", "were", "with", "you", "your",
        }
        if left_lower in function_words:
            score -= 6.0
        if right_lower in function_words:
            score -= 6.0

        boring_pairs = {
            ("of", "the"), ("in", "the"), ("to", "the"), ("on", "the"),
            ("for", "the"), ("and", "the"), ("at", "the"), ("is", "a"),
            ("it", "is"), ("i", "am"), ("you", "are"), ("do", "not"),
            ("dont", "know"), ("don't", "know"),
        }
        if (left_lower, right_lower) in boring_pairs:
            score -= 10.0

        social_starts = {"lol", "haha", "hahaha", "omg", "yeah", "yes", "no"}
        if left_lower in social_starts or right_lower in social_starts:
            score -= 8.0

        return score

    def _add_direct_bigram_tokens(self, bigrams: List[str]) -> None:
        """Add selected adjacent-word tokens before filling the BPE budget."""
        self.direct_bigram_tokens = []
        for token in bigrams:
            if len(self.token_to_id) >= self.vocab_size:
                break
            self._add_token(token)
            self.direct_bigram_tokens.append(token)

        # Longest-first matching makes encoding deterministic if one bigram is
        # ever a prefix of another.
        self.direct_bigram_tokens.sort(key=len, reverse=True)
        self._rebuild_direct_bigram_index()

    def _match_direct_bigram(self, normalized: str, index: int) -> Optional[str]:
        """Return the direct bigram token starting at index, if one exists."""
        if index > 0 and normalized[index - 1] != self.space_token:
            return None

        # Encoding visits many character positions. A dict index avoids
        # scanning every selected bigram at positions where most cannot match,
        # while preserving the same longest-first order inside each bucket.
        for token in self.direct_bigram_by_first.get(normalized[index], []):
            end = index + len(token)
            if (
                normalized.startswith(token, index)
                and (end == len(normalized) or normalized[end] == self.space_token)
            ):
                return token
        return None

    def _rebuild_direct_bigram_index(self) -> None:
        """Build the first-character lookup table for direct bigram matching."""
        self.direct_bigram_by_first = {}
        for token in self.direct_bigram_tokens:
            if not token:
                continue
            self.direct_bigram_by_first.setdefault(token[0], []).append(token)

        for candidates in self.direct_bigram_by_first.values():
            candidates.sort(key=len, reverse=True)

    def _is_safe_merge_token(self, token: str) -> bool:
        """Return whether a learned BPE token respects word boundaries."""
        space_count = token.count(self.space_token)
        if space_count == 0:
            return True
        if space_count > 1:
            return False
        return token.startswith(self.space_token)

    def _add_token(self, token: str) -> None:
        """Add one token to the vocabulary if it is not present yet."""
        if token in self.token_to_id:
            return
        token_id = len(self.token_to_id)
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token

    def _best_bigram_surface(self, word_bigram_counts: Counter) -> Optional[str]:
        """Return the most common adjacent-word candidate, if any exists."""
        if not word_bigram_counts:
            return None
        return word_bigram_counts.most_common(1)[0][0]

    def _has_bigram_token(self) -> bool:
        """Check whether the vocabulary contains a token spanning two words."""
        for token in self.token_to_id:
            if token in self.special_tokens:
                continue
            if self.space_token in token.strip(self.space_token):
                return True
        return False

    def _ensure_bigram_token(self) -> None:
        """Make the assignment's bigram-token requirement explicit."""
        if self._has_bigram_token() or not self.best_word_bigram:
            return

        # This token is still selected from the provided training data. It
        # exists only as a compliance fallback; normal direct-bigram selection
        # should usually add several safer examples earlier.
        if len(self.token_to_id) < self.vocab_size:
            self._add_token(self.best_word_bigram)
            self.direct_bigram_tokens.append(self.best_word_bigram)
            self.direct_bigram_tokens.sort(key=len, reverse=True)
            self._rebuild_direct_bigram_index()
            self.forced_bigram_token = self.best_word_bigram
