"""
transcript_quality.py
─────────────────────
Reusable utilities for Bengali transcript quality assessment.

Responsibilities:
  - Repetition detection (token-level and n-gram-level)
  - Hallucination / degeneration detection
  - Per-call quality scoring
  - Segment deduplication / collapse

These helpers are called by the improved transcription pipeline
and can also be imported standalone for offline analysis.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Bengali Unicode range: \u0980–\u09FF
BENGALI_CHAR_RE = re.compile(r"[\u0980-\u09FF]")

# Known Whisper silence / hallucination patterns
_SILENCE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\[.*?\]"),           # [music], [silence], [noise]
    re.compile(r"\(.*?\)"),           # (inaudible), (crosstalk)
    re.compile(r"♪+"),                # music notes
    re.compile(r"\.{3,}"),            # ...
    re.compile(r"^\s*$"),             # empty
]

# Minimum chars for a segment to count as real speech
MIN_SEGMENT_CHARS = 3

# If a word repeats more than this fraction of total words → suspicious
WORD_REPEAT_RATIO_THRESHOLD = 0.40

# If the top n-gram covers more than this fraction → suspicious
NGRAM_REPEAT_RATIO_THRESHOLD = 0.35

# Quality score bands
SCORE_GOOD   = 0.75
SCORE_WARN   = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QualityMetrics:
    """Per-call quality metrics stored in the JSONL output."""

    repetition_ratio: float = 0.0
    """Fraction of words that are excessively repeated."""

    ngram_repeat_ratio: float = 0.0
    """Fraction of tokens covered by the most-repeated n-gram."""

    avg_log_prob: Optional[float] = None
    """Mean segment log-probability (from Whisper). Lower = less confident."""

    speech_density: float = 0.0
    """Ratio of speech tokens to total duration (tokens per second)."""

    hallucination_flags: List[str] = field(default_factory=list)
    """Human-readable list of detected issues."""

    quality_score: float = 1.0
    """Composite 0–1 score. Higher is better."""

    is_suspicious: bool = False
    """True if the transcript likely contains degeneration."""

    def to_dict(self) -> dict:
        return {
            "repetition_ratio":   round(self.repetition_ratio,   4),
            "ngram_repeat_ratio": round(self.ngram_repeat_ratio, 4),
            "avg_log_prob":       round(self.avg_log_prob, 4) if self.avg_log_prob is not None else None,
            "speech_density":     round(self.speech_density,     4),
            "hallucination_flags": self.hallucination_flags,
            "quality_score":      round(self.quality_score,      4),
            "is_suspicious":      self.is_suspicious,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenizer that works for Bengali.
    Strips punctuation but keeps Bengali script intact.
    """
    # Keep Bengali chars, ASCII alphanumeric, and spaces
    cleaned = re.sub(r"[^\w\s\u0980-\u09FF]", " ", text, flags=re.UNICODE)
    return [t for t in cleaned.split() if len(t) >= 1]


def is_silence_or_noise(text: str) -> bool:
    """Return True if the text looks like a Whisper hallucinated silence marker."""
    t = text.strip()
    if not t:
        return True
    for pat in _SILENCE_PATTERNS:
        if pat.fullmatch(t):
            return True
    # Mostly non-Bengali and non-ASCII — probably noise label
    bengali_chars = len(BENGALI_CHAR_RE.findall(t))
    total_chars   = len(t.replace(" ", ""))
    if total_chars > 0 and bengali_chars / total_chars < 0.10 and len(t) < 20:
        return True
    return False


def detect_word_repetition(words: List[str], threshold: float = WORD_REPEAT_RATIO_THRESHOLD) -> tuple[float, Optional[str]]:
    """
    Measure how much a single word dominates the transcript.

    Returns:
        (ratio, dominant_word)  — ratio is fraction of tokens that ARE the dominant word.
        If no word is dominant, ratio=0 and dominant_word=None.
    """
    if not words:
        return 0.0, None
    counts = Counter(words)
    most_common_word, most_common_count = counts.most_common(1)[0]
    ratio = most_common_count / len(words)
    if ratio >= threshold:
        return ratio, most_common_word
    return 0.0, None


def detect_ngram_repetition(words: List[str], n: int = 3, threshold: float = NGRAM_REPEAT_RATIO_THRESHOLD) -> tuple[float, Optional[tuple]]:
    """
    Detect repeated n-grams across the token sequence.

    Returns:
        (ratio, dominant_ngram)
    """
    if len(words) < n:
        return 0.0, None
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    if not counts:
        return 0.0, None
    most_common_ng, most_common_cnt = counts.most_common(1)[0]
    # Each n-gram occurrence covers n tokens, but they overlap — estimate coverage
    coverage = (most_common_cnt * n) / len(words)
    if coverage >= threshold:
        return min(coverage, 1.0), most_common_ng
    return 0.0, None


def collapse_repetitions(text: str, max_repeat: int = 2) -> str:
    """
    Collapse runs of identical words.

    Example:
        "এখানে এখানে এখানে এখানে" → "এখানে এখানে"  (if max_repeat=2)
    """
    words = text.split()
    if not words:
        return text

    out: List[str] = []
    run_word  = words[0]
    run_count = 1

    for word in words[1:]:
        if word == run_word:
            run_count += 1
        else:
            out.extend([run_word] * min(run_count, max_repeat))
            run_word  = word
            run_count = 1

    out.extend([run_word] * min(run_count, max_repeat))
    return " ".join(out)


def dedup_consecutive_segments(segments: list, similarity_threshold: float = 0.85) -> list:
    """
    Remove consecutive segments whose text is nearly identical.

    Two segments are considered duplicates when their texts overlap
    heavily (one is a substring of the other, or they are very similar
    after stripping whitespace).
    """
    if not segments:
        return segments

    deduped = [segments[0]]
    for seg in segments[1:]:
        prev_text = deduped[-1].get("text_bn", "").strip()
        curr_text = seg.get("text_bn", "").strip()
        if _text_similarity(prev_text, curr_text) >= similarity_threshold:
            continue  # skip near-duplicate
        deduped.append(seg)

    return deduped


def _text_similarity(a: str, b: str) -> float:
    """Simple token-overlap Jaccard similarity."""
    set_a = set(tokenize(a))
    set_b = set(tokenize(b))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union        = set_a | set_b
    return len(intersection) / len(union)


# ─────────────────────────────────────────────────────────────────────────────
# High-level quality scorer
# ─────────────────────────────────────────────────────────────────────────────

def compute_quality_metrics(
    segments: list,
    duration: float,
    avg_log_prob: Optional[float] = None,
) -> QualityMetrics:
    """
    Compute a QualityMetrics object from a list of segment dicts.

    Segments are expected to have at least:
        {"start": float, "end": float, "text_bn": str}

    Args:
        segments:     List of segment dicts.
        duration:     Total audio duration in seconds.
        avg_log_prob: Mean log-probability from Whisper (optional).

    Returns:
        QualityMetrics populated with all fields.
    """
    metrics = QualityMetrics()
    flags: List[str] = []

    # ── Gather all words ──────────────────────────────────────────────────────
    all_words: List[str] = []
    real_speech_duration = 0.0

    for seg in segments:
        text = seg.get("text_bn", "").strip()
        if is_silence_or_noise(text):
            flags.append(f"silence_marker: '{text[:30]}'")
            continue
        words = tokenize(text)
        all_words.extend(words)
        seg_dur = seg.get("end", 0) - seg.get("start", 0)
        if seg_dur > 0:
            real_speech_duration += seg_dur

    # ── Word-level repetition ─────────────────────────────────────────────────
    word_ratio, dominant_word = detect_word_repetition(all_words)
    metrics.repetition_ratio = word_ratio
    if word_ratio >= WORD_REPEAT_RATIO_THRESHOLD:
        flags.append(f"word_loop: '{dominant_word}' covers {word_ratio:.0%} of tokens")

    # ── N-gram repetition ─────────────────────────────────────────────────────
    ng_ratio, dominant_ng = detect_ngram_repetition(all_words, n=3)
    metrics.ngram_repeat_ratio = ng_ratio
    if ng_ratio >= NGRAM_REPEAT_RATIO_THRESHOLD:
        ng_str = " ".join(dominant_ng) if dominant_ng else ""
        flags.append(f"ngram_loop: '{ng_str}' covers {ng_ratio:.0%}")

    # ── Log probability ───────────────────────────────────────────────────────
    metrics.avg_log_prob = avg_log_prob
    if avg_log_prob is not None and avg_log_prob < -1.5:
        flags.append(f"low_confidence: avg_log_prob={avg_log_prob:.3f}")

    # ── Speech density ────────────────────────────────────────────────────────
    if duration > 0:
        metrics.speech_density = len(all_words) / duration

    # Very high density often means looping hallucination
    if metrics.speech_density > 8.0:
        flags.append(f"high_speech_density: {metrics.speech_density:.1f} words/sec")

    # ── Composite quality score ───────────────────────────────────────────────
    score = 1.0

    # Penalise word repetition
    score -= word_ratio * 0.5

    # Penalise n-gram repetition
    score -= ng_ratio * 0.3

    # Penalise low log-prob
    if avg_log_prob is not None:
        lp_penalty = max(0.0, (-avg_log_prob - 0.5) * 0.1)
        score -= min(lp_penalty, 0.2)

    # Penalise excessive speech density
    if metrics.speech_density > 8.0:
        score -= 0.15

    metrics.quality_score     = max(0.0, min(1.0, score))
    metrics.hallucination_flags = flags
    metrics.is_suspicious      = (
        word_ratio >= WORD_REPEAT_RATIO_THRESHOLD
        or ng_ratio >= NGRAM_REPEAT_RATIO_THRESHOLD
        or (avg_log_prob is not None and avg_log_prob < -2.0)
        or metrics.speech_density > 8.0
    )

    return metrics


def clean_segment_text(text: str, max_repeat: int = 2) -> str:
    """
    Apply lightweight text-level cleaning to a single segment.

    Steps:
      1. Strip surrounding whitespace.
      2. Collapse runs of identical consecutive words.
      3. Remove known silence marker patterns.
    """
    text = text.strip()
    # Remove silence markers inline
    for pat in _SILENCE_PATTERNS:
        text = pat.sub(" ", text)
    text = " ".join(text.split())          # normalise internal whitespace
    text = collapse_repetitions(text, max_repeat=max_repeat)
    return text.strip()
