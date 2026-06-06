"""Stylometry Drift Engine — deterministic local text rewriter.

Transforms text to reduce identifiable stylistic patterns while preserving
meaning exactly. All transformations are seeded from the input text hash,
so the same input always produces the same output.

No external AI calls. Fully offline.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent / "data"
_SYNONYM_FILE = _DATA_DIR / "synonyms.json"

# ---------------------------------------------------------------------------
# Synonym bank — loaded once at import time
# ---------------------------------------------------------------------------

_SYNONYMS: Dict[str, List[str]] = {}


def _load_synonyms() -> Dict[str, List[str]]:
    """Load synonym dictionary from the data file."""
    if not _SYNONYM_FILE.exists():
        return {}
    try:
        with open(_SYNONYM_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _get_synonyms() -> Dict[str, List[str]]:
    """Lazy-load and cache synonyms."""
    global _SYNONYMS
    if not _SYNONYMS:
        _SYNONYMS = _load_synonyms()
    return _SYNONYMS


# ---------------------------------------------------------------------------
# Deterministic seed generation
# ---------------------------------------------------------------------------

def _make_seed(text: str, salt: str = "") -> int:
    """Generate a deterministic integer seed from *text* + *salt*."""
    raw = hashlib.sha256((text + salt).encode("utf-8")).hexdigest()
    return int(raw[:16], 16)


def _make_rng(text: str, salt: str = "") -> random.Random:
    """Create a seeded RNG for deterministic pseudo-random choices."""
    return random.Random(_make_seed(text, salt))


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

# Set of abbreviations whose trailing period should NOT be treated as
# a sentence boundary.
_ABBREVIATIONS: Set[str] = {
    "dr", "mr", "mrs", "ms", "st", "jr", "sr", "vs", "etc",
    "approx", "dept", "est", "govt", "capt", "col", "gen",
    "lt", "maj", "sgt", "cpl", "pvt", "sra",
    "e.g", "i.e", "al", "vol", "fig", "no", "p", "pp", "ch", "sec",
}

# Sentence boundary detection: split on ". " (period-space) while keeping
# abbreviations like "Dr." and "e.g." intact.
_SENTENCE_END = re.compile(r"\.\s+")

# Words that often signal non-restrictive clauses for em-dash insertion
_LIGHT_PUNCT_WORDS = {"however", "therefore", "moreover", "furthermore",
                       "nevertheless", "nonetheless", "consequently", "thus",
                       "hence", "meanwhile", "afterward"}

# Subordinating conjunctions for clause reordering
_SUBORDINATORS = {"if", "when", "whenever", "while", "although", "though",
                   "because", "since", "as", "unless", "until", "whereas",
                   "wherever", "whether", "even though", "even if"}

# Coordinating conjunctions for sentence splitting
_COORDINATORS = {"and", "but", "or", "yet", "so", "nor"}

# Words that can start hedge phrases (for removal/replacement)
_HEDGE_WORDS = {"actually", "basically", "essentially", "honestly",
                 "literally", "practically", "really", "simply",
                 "truly", "virtually", "quite", "rather", "somewhat"}

# Adverb placement variations for selected common adverbs
_ADVERBS_REORDER = {
    "always", "often", "sometimes", "usually", "rarely",
    "never", "certainly", "definitely", "probably",
}

# Temporal phrase start patterns for reordering
_TEMPORAL_RE = re.compile(
    r"^(After|Before|During|Following|Once|Since|Until|"
    r"Upon|As soon as|By the time)\s",
    re.IGNORECASE,
)

# "If X, Y" / "When X, Y" pattern
_CONDITIONAL_RE = re.compile(
    r"^(If|When|Whenever|While|Although|Though|Because|Since|As|"
    r"Unless|Until|Whereas)\s",
    re.IGNORECASE,
)

# Prepositional phrase start detection (for fronting variations)
_PREPOSITIONS = {"in", "on", "at", "by", "with", "from", "to",
                  "for", "of", "about", "during", "through",
                  "throughout", "across", "between", "among",
                  "beneath", "under", "over", "above", "below"}


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, handling common abbreviations.

    Uses a simple heuristic: split on ". " (period + space) but skip
    known abbreviations like "Dr.", "Mr.", "e.g.", etc. Also handles
    terminal punctuation: ! and ?.
    """
    if not text.strip():
        return []

    # Normalise whitespace first
    text = re.sub(r"\s+", " ", text.strip())

    # Split on . ! ? followed by a space and a capital letter or quote.
    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z\"'(])",
        text,
    )

    # Re-check each split boundary: if the preceding word is an
    # abbreviation, merge back.
    sentences: List[str] = []
    for part in parts:
        if not part:
            continue
        # Check if this part starts with an abbreviation boundary
        # (the previous sentence ended with an abbreviation).
        # Extract the last word of "part" if we merge or the first
        # word if it's standalone.
        last_word = part.strip().split()[-1].rstrip(".").lower() if part.strip() else ""
        if last_word in _ABBREVIATIONS and sentences:
            # Merge with previous sentence
            sentences[-1] += ". " + part
        else:
            sentences.append(part)

    return sentences


# ---------------------------------------------------------------------------
# Per-sentence transformation functions
# ---------------------------------------------------------------------------


def _replace_punctuation(sentence: str, rng: random.Random) -> str:
    """Replace some punctuation patterns with stylistic equivalents."""
    # Em-dash insertion around appositives flanked by commas
    # "The CEO, Jane Smith, announced" -> "The CEO — Jane Smith — announced"
    if rng.random() < 0.3:
        sentence = re.sub(
            r",\s+([^,]+?)\s*,",
            lambda m: rng.choice([" — " + m.group(1) + " — ", " (" + m.group(1) + ")", ", " + m.group(1) + ","]),
            sentence,
            count=1,
        )

    # Semicolons to join related independent clauses
    # (when there's a period followed by a light-punctuation word)
    for word in _LIGHT_PUNCT_WORDS:
        if rng.random() < 0.25:
            pattern = r"\.\s+" + word + r"\s*,?\s*"
            replacement = r"; " + word + r", "
            sentence = re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)

    # Replace some colons with em-dashes
    if rng.random() < 0.3:
        sentence = sentence.replace(": ", " \u2014 ", 1)

    # Replace some semicolons with periods (creating separate sentences)
    if rng.random() < 0.2:
        sentence = sentence.replace("; ", ". ", 1)

    return sentence


def _split_long_sentence(sentence: str, rng: random.Random) -> List[str]:
    """Split a long sentence at a coordinating conjunction if possible."""
    words = sentence.split()
    if len(words) < 12:
        return [sentence]

    # Find a coordinator that's not at the very start or end
    best_idx = -1
    for i, w in enumerate(words):
        if w.lower().rstrip(",") in _COORDINATORS and 3 < i < len(words) - 3:
            best_idx = i

    if best_idx < 0:
        return [sentence]

    first = " ".join(words[:best_idx]).rstrip(",")
    conjunction = words[best_idx].rstrip(",")
    rest = " ".join(words[best_idx + 1:])

    if not first.endswith((".", "!", "?")):
        first = first.rstrip(",") + "."
    rest = conjunction.capitalize() + " " + rest

    return [first, rest]


def _combine_short_sentences(sentences: List[str], rng: random.Random) -> List[str]:
    """Combine adjacent short sentences sharing a trivial link."""
    if len(sentences) < 2:
        return sentences

    result: List[str] = []
    skip_next = False

    for i in range(len(sentences) - 1):
        if skip_next:
            skip_next = False
            continue

        a_words = sentences[i].split()
        b_words = sentences[i + 1].split()

        # Both must be short
        if len(a_words) > 10 or len(b_words) > 10:
            result.append(sentences[i])
            continue

        a_clean = re.sub(r"[^a-zA-Z\s]", "", sentences[i]).strip()
        b_clean = re.sub(r"[^a-zA-Z\s]", "", sentences[i + 1]).strip()

        a_subject = a_clean.split()[:2] if a_clean.split() else []
        b_subject = b_clean.split()[:2] if b_clean.split() else []

        # If they share a common subject (first 1-2 words), combine
        if a_subject and b_subject and a_subject[0].lower() == b_subject[0].lower():
            b_text = sentences[i + 1]
            # Remove leading subject if it matches
            b_no_subject = re.sub(
                r"^\s*" + re.escape(a_subject[0]) + r"\s*",
                "",
                b_text,
                count=1,
                flags=re.IGNORECASE,
            )
            combined = sentences[i].rstrip(".!?") + ", and " + b_no_subject[0].lower() + b_no_subject[1:]
            result.append(combined)
            skip_next = True
        else:
            result.append(sentences[i])

    if not skip_next:
        result.append(sentences[-1])

    return result


def _reorder_clauses(sentence: str, rng: random.Random) -> str:
    """Reorder clauses: move subordinate clauses from start to end, or vice versa."""
    # Check if sentence starts with a conditional/ temporal clause
    m = _CONDITIONAL_RE.match(sentence)
    if m:
        subordinator = m.group(1).lower()
        rest = sentence[m.end():]
        # Find the comma that separates the two clauses
        comma_idx = rest.find(",")
        if comma_idx > 0 and comma_idx < len(rest) - 5:
            front_clause = rest[:comma_idx].strip()
            main_clause = rest[comma_idx + 1:].strip()
            if main_clause and front_clause:
                # Swap order
                return f"{main_clause} {subordinator} {front_clause}."

    # Check temporal fronting
    m = _TEMPORAL_RE.match(sentence)
    if m:
        front_word = m.group(1).lower()
        rest = sentence[m.end():]
        comma_idx = rest.find(",")
        if comma_idx > 0 and comma_idx < len(rest) - 5:
            front_clause = rest[:comma_idx].strip()
            main_clause = rest[comma_idx + 1:].strip()
            if main_clause and front_clause:
                return f"{main_clause} {front_word} {front_clause}."

    return sentence


def _substitute_synonyms(sentence: str, rng: random.Random, intensity: float) -> str:
    """Replace words with synonyms from the bank based on intensity."""
    synonyms = _get_synonyms()
    if not synonyms:
        return sentence

    words = sentence.split()
    new_words = list(words)
    max_subs = max(1, int(len(words) * intensity * 0.15))

    # Collect candidate positions
    candidates: List[int] = []
    for idx, w in enumerate(words):
        clean = w.strip(".,!?;:\"'()[]-").lower()
        if clean in synonyms and len(synonyms[clean]) > 0:
            candidates.append(idx)

    if not candidates:
        return sentence

    # Shuffle candidates deterministically
    rng.shuffle(candidates)

    subs_made = 0
    for idx in candidates[:max_subs]:
        clean = words[idx].strip(".,!?;:\"'()[]-").lower()
        punct = words[idx][len(clean):] if len(words[idx]) > len(clean) else ""
        options = [s for s in synonyms[clean] if s.lower() != clean]
        if not options:
            continue

        replacement = rng.choice(options)
        # Preserve case
        if words[idx][0].isupper():
            replacement = replacement[0].upper() + replacement[1:]

        new_words[idx] = replacement + punct
        subs_made += 1

        if subs_made >= max_subs:
            break

    return " ".join(new_words)


def _modify_hedge_phrases(sentence: str, rng: random.Random) -> str:
    """Remove or replace hedge words (I think, maybe, etc.)."""
    # Remove some hedge adverbs
    cleaned = sentence
    for word in _HEDGE_WORDS:
        if rng.random() < 0.5:
            cleaned = re.sub(
                r"\b" + re.escape(word) + r"\b\s*,?\s*",
                "",
                cleaned,
                count=1,
                flags=re.IGNORECASE,
            )

    return cleaned


def _vary_adverb_placement(sentence: str, rng: random.Random) -> str:
    """Move sentence-initial adverbs to mid-sentence or vice versa."""
    words = sentence.split()

    # Move a fronted adverb inward
    first_word = words[0].rstrip(",").lower() if words else ""
    if first_word in _ADVERBS_REORDER and len(words) > 4:
        adverb = words[0].rstrip(",")
        rest = words[1:]
        # Insert after the first verb or after the first 2-3 words
        for i, w in enumerate(rest):
            if w.lower() in {"is", "was", "are", "were", "has", "have",
                              "had", "will", "would", "can", "could",
                              "should", "may", "might", "does", "do",
                              "did", "shall", "must"}:
                adverb_set = adverb.lower() + ", " if rng.random() < 0.3 else " " + adverb.lower()
                rest[i] = rest[i] + adverb_set
                return " ".join(rest)

    return sentence


def _transform_sentence(
    sentence: str,
    rng: random.Random,
    intensity: float,
) -> str:
    """Apply a suite of deterministic transformations to a single sentence."""
    if not sentence.strip():
        return sentence

    result = sentence

    # 1. Clause reordering (when intensity > 0.2)
    if intensity > 0.2 and rng.random() < intensity * 0.5:
        result = _reorder_clauses(result, rng)

    # 2. Punctuation shifts (all intensity levels)
    if rng.random() < 0.3 + intensity * 0.4:
        result = _replace_punctuation(result, rng)

    # 3. Hedge phrase modification (intensity > 0.3)
    if intensity > 0.3 and rng.random() < intensity * 0.3:
        result = _modify_hedge_phrases(result, rng)

    # 4. Adverb placement variation (intensity > 0.4)
    if intensity > 0.4 and rng.random() < intensity * 0.3:
        result = _vary_adverb_placement(result, rng)

    # 5. Synonym substitution (all levels, scaled by intensity)
    result = _substitute_synonyms(result, rng, intensity)

    return result


# ---------------------------------------------------------------------------
# N-gram history avoidance
# ---------------------------------------------------------------------------


def _extract_ngrams(text: str, n: int = 4) -> Set[str]:
    """Extract character n-grams from text (lowercase, ignoring whitespace runs)."""
    cleaned = re.sub(r"\s+", " ", text.lower())
    return {cleaned[i:i + n] for i in range(len(cleaned) - n + 1)}


def _build_avoid_set(history: List[str]) -> Set[str]:
    """Build a set of character n-grams to avoid from history texts."""
    avoid: Set[str] = set()
    for h in history:
        if not h or not isinstance(h, str):
            continue
        for n in (3, 4, 5):
            avoid.update(_extract_ngrams(h, n))
    return avoid


def _contains_avoided(text: str, avoid: Set[str]) -> bool:
    """Check if *text* contains any n-grams from the avoid set."""
    if not avoid:
        return False
    cleaned = re.sub(r"\s+", " ", text.lower())
    for n in (3, 4, 5):
        for i in range(len(cleaned) - n + 1):
            if cleaned[i:i + n] in avoid:
                return True
    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def rewrite(
    text: str,
    intensity: float = 0.5,
    history: Optional[List[str]] = None,
) -> str:
    """Rewrite *text* to reduce identifiable stylistic patterns.

    Parameters
    ----------
    text : str
        The text to rewrite.
    intensity : float, optional
        How aggressively to transform (0.0–1.0). Default 0.5.
    history : list of str, optional
        Prior texts from the same author whose patterns to avoid.

    Returns
    -------
    str
        Rewritten text preserving the original meaning.
    """
    intensity = max(0.0, min(1.0, intensity))

    if not text or not text.strip():
        return text

    # Build the avoid set from history
    avoid = _build_avoid_set(history or [])

    # Deterministic RNG seeded from the input text
    rng = _make_rng(text)

    # Attempt up to N versions in case any hit an avoided n-gram pattern
    for attempt in range(5):
        result = _rewrite_once(text, rng, intensity, avoid, attempt)
        if not _contains_avoided(result, avoid):
            return result

    # If we exhausted attempts, return the last one (best effort)
    return result


def _rewrite_once(
    text: str,
    rng: random.Random,
    intensity: float,
    avoid: Set[str],
    attempt: int,
) -> str:
    """Single rewrite pass — may be called multiple times with different seeds."""
    # Apply per-attempt seed variation so retries produce different results
    per_attempt_rng = _make_rng(text, str(attempt))

    sentences = split_sentences(text)

    if len(sentences) <= 1:
        # Single sentence — just transform
        s = _transform_sentence(sentences[0], per_attempt_rng, intensity)
        return s

    # Sentence combining (light touch, before splitting)
    if intensity > 0.2 and per_attempt_rng.random() < intensity * 0.3:
        sentences = _combine_short_sentences(sentences, per_attempt_rng)

    # Split long sentences
    split_result: List[str] = []
    for s in sentences:
        if per_attempt_rng.random() < intensity * 0.4:
            split = _split_long_sentence(s, per_attempt_rng)
            split_result.extend(split)
        else:
            split_result.append(s)

    # Transform each sentence individually
    transformed = []
    for s in split_result:
        t = _transform_sentence(s, per_attempt_rng, intensity)
        transformed.append(t)

    return " ".join(transformed)
