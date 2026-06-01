"""
Stage 1 — Bengali Transcript Cleaner
=====================================
Reads raw Whisper JSONL transcripts from data/transcripts/
Writes cleaned JSONL to data/nlp/cleaned/
Checkpoints progress in checkpoints/nlp_cleaning.sqlite

Cleaning strategy (conservative — never destroys data):
  1. Strip known Whisper garbage tokens (music notes, [BLANK_AUDIO], etc.)
  2. Collapse consecutive duplicate word / N-gram repetition loops
     e.g.  "হলো হলো হলো হলো" → "হলো"
     e.g.  "আছে আছে তে এখন" → unchanged (mixed, not pure loop)
  3. Normalize Unicode (NFC) and fix excessive whitespace
  4. Compute a quality_score (0.0–1.0)
  5. Set flagged=True if quality is highly suspicious

Quality score heuristics (each component is 0–1, equally weighted):
  - word_rate_ok   : words-per-second in [0.5, 8.0] → good
  - rep_ok         : repetition density below threshold → good
  - length_ok      : segment has at least 2 words → good

Usage:
    python -m src.nlp.clean_bengali [--input-dir DATA/transcripts] \\
                                    [--output-dir DATA/nlp/cleaned] \\
                                    [--checkpoint checkpoints/nlp_cleaning.sqlite] \\
                                    [--pattern "*.jsonl"] \\
                                    [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("nlp.clean_bengali")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Whisper frequently hallucinates these tokens (language-agnostic + Bengali)
GARBAGE_TOKENS: List[str] = [
    # --- Universal Whisper artifacts ---
    "♪",
    "♫",
    "[BLANK_AUDIO]",
    "[MUSIC]",
    "[NOISE]",
    "[INAUDIBLE]",
    "(BLANK_AUDIO)",
    "(MUSIC)",
    "(NOISE)",
    "(INAUDIBLE)",
    "...",
    # --- Bengali-specific Whisper hallucinations ---
    # Whisper's YouTube training data causes it to hallucinate these:
    "সাবস্ক্রাইব",      # "subscribe" — very common hallucination
    "সাবসক্রাইব",       # alternate spelling
    "লাইক",              # "like"
    "কমেন্ট",            # "comment"
    "শেয়ার",             # "share"
    "চ্যানেল",           # "channel"
]

# Bengali greetings that Whisper often repeats at segment boundaries.
# We collapse runs of ≥2 of these into a single instance.
REPEATED_GREETINGS: List[str] = [
    "হ্যালো",    # hello
    "হলো",       # hello (rural)
    "হ্যাঁ",     # yes
    "নমস্কার",   # namaskar
    "আচ্ছা",     # achha
]

# Common ASR phonetic misspellings and dialect normalizations
REPLACEMENTS: Dict[str, str] = {
    "দাক্তা": "ডাক্তার",
    "দাক্তার": "ডাক্তার",
    "হাইস্কুড": "হাইস্কুল",
    "তেকে": "থেকে",
    "কেছিল": "গিয়েছিল",
    "খাথে": "খাতে",
    "এখনা": "খানা",
    "কথে": "কথা",
}

# Words per second thresholds — outside these bounds → suspicious
MIN_WPS: float = 0.3   # below this: near-silence or very long pause
MAX_WPS: float = 9.0   # above this: likely repetition loop or artifact

# A segment with repetition density > this is flagged
REPETITION_DENSITY_THRESHOLD: float = 0.5

# Max n-gram size to check for loops
MAX_NGRAM: int = 6

# ---------------------------------------------------------------------------
# SQLite Checkpoint
# ---------------------------------------------------------------------------

class NLPCleaningCheckpoint:
    """
    File-level checkpoint for the cleaning stage.

    Schema:
        file_id      TEXT  PRIMARY KEY  (stem of the JSONL filename)
        status       TEXT               'done' | 'error'
        input_path   TEXT
        output_path  TEXT
        segments_in  INTEGER
        segments_out INTEGER
        error        TEXT
        updated_at   TEXT
    """

    TABLE = "nlp_cleaning"

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    file_id      TEXT PRIMARY KEY,
                    status       TEXT NOT NULL,
                    input_path   TEXT,
                    output_path  TEXT,
                    segments_in  INTEGER,
                    segments_out INTEGER,
                    error        TEXT,
                    updated_at   TEXT NOT NULL
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.TABLE}_status "
                f"ON {self.TABLE}(status)"
            )
            conn.commit()

    def is_done(self, file_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT status FROM {self.TABLE} WHERE file_id = ?",
                (file_id,),
            ).fetchone()
        return bool(row) and row["status"] == "done"

    def mark_done(
        self,
        file_id: str,
        input_path: str,
        output_path: str,
        segments_in: int,
        segments_out: int,
    ) -> None:
        self._upsert(
            file_id=file_id,
            status="done",
            input_path=input_path,
            output_path=output_path,
            segments_in=segments_in,
            segments_out=segments_out,
            error=None,
        )

    def mark_error(self, file_id: str, input_path: str, error: str) -> None:
        self._upsert(
            file_id=file_id,
            status="error",
            input_path=input_path,
            output_path=None,
            segments_in=None,
            segments_out=None,
            error=error,
        )

    def _upsert(self, **kwargs: Any) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.TABLE}
                    (file_id, status, input_path, output_path,
                     segments_in, segments_out, error, updated_at)
                VALUES
                    (:file_id, :status, :input_path, :output_path,
                     :segments_in, :segments_out, :error, :updated_at)
                ON CONFLICT(file_id) DO UPDATE SET
                    status      = excluded.status,
                    input_path  = excluded.input_path,
                    output_path = excluded.output_path,
                    segments_in = excluded.segments_in,
                    segments_out= excluded.segments_out,
                    error       = excluded.error,
                    updated_at  = excluded.updated_at
                """,
                {**kwargs, "updated_at": now},
            )
            conn.commit()

    def stats(self) -> Dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT status, COUNT(*) AS n FROM {self.TABLE} GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}


# ---------------------------------------------------------------------------
# Text-cleaning helpers
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Apply NFC Unicode normalization (important for Bengali combining chars)."""
    return unicodedata.normalize("NFC", text)


def strip_garbage_tokens(text: str) -> str:
    """Remove known Whisper artifact tokens."""
    for token in GARBAGE_TOKENS:
        text = text.replace(token, " ")
    return text


def fix_whitespace(text: str) -> str:
    """Collapse multiple spaces / tabs / newlines into a single space."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def _make_ngrams(words: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def collapse_repetition_loops(text: str) -> Tuple[str, float]:
    """
    Detect and collapse consecutive repetition loops.

    Strategy: For each N-gram size from 1 to MAX_NGRAM, scan left-to-right
    and collapse any run of ≥3 identical consecutive N-grams down to one copy.
    Returns (cleaned_text, repetition_density).

    repetition_density = (removed_words) / max(total_words, 1)
    """
    if len(text) <= MAX_NGRAM * 5:
        # Avoid running complex n-gram logic on very short texts
        return text, 0.0

    words = text.split()
    if not words:
        return text, 0.0

    total_words = len(words)
    removed_words = 0

    for n in range(1, min(MAX_NGRAM + 1, len(words) + 1)):
        i = 0
        new_words: List[str] = []
        while i < len(words):
            # Try to detect a run of identical n-grams starting at i
            gram = tuple(words[i : i + n])
            if len(gram) < n:
                new_words.extend(words[i:])
                break

            run_len = 1
            j = i + n
            while j + n <= len(words) + 1:
                next_gram = tuple(words[j : j + n])
                if next_gram == gram:
                    run_len += 1
                    j += n
                else:
                    break

            if run_len >= 3:
                # Collapse: keep exactly one copy
                removed = (run_len - 1) * n
                removed_words += removed
                new_words.extend(gram)
                i = j
            else:
                new_words.append(words[i])
                i += 1

        words = new_words

    repetition_density = removed_words / max(total_words, 1)
    cleaned_text = " ".join(words)
    return cleaned_text, repetition_density


def collapse_greeting_runs(text: str) -> str:
    """
    Collapse runs of ≥2 identical Bengali greetings at the beginning of a segment.
    E.g. "হলো, হলো, হলো, হে" → "হলো, হে"
    Also handles comma-separated runs.
    """
    for greeting in REPEATED_GREETINGS:
        # Match "greeting," or "greeting " repeated 2+ times
        pattern = rf"(?:{re.escape(greeting)}[,\s]*?){{2,}}"
        text = re.sub(pattern, greeting + ", ", text)
    return text


def normalize_bengali_punctuation(text: str) -> str:
    """
    Normalize Bengali-specific punctuation:
    - Multiple দাঁড়ি (।) → single
    - Multiple consecutive commas → single
    - Fix spacing around punctuation
    """
    text = re.sub(r"।{2,}", "।", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r"\s+([,।?])", r"\1", text)  # remove space before punctuation
    text = re.sub(r"([,।?])(?=[^\s])", r"\1 ", text)  # ensure space after punctuation
    return text


def apply_asr_replacements(text: str) -> str:
    """
    Replace common Whisper phonetic misspellings and dialect words.
    Uses word-boundary regex to prevent replacing substrings.
    """
    for bad, good in REPLACEMENTS.items():
        # Match the word specifically (accounting for punctuation)
        pattern = rf"(?<![\u0980-\u09FF]){bad}(?![\u0980-\u09FF])"
        text = re.sub(pattern, good, text)
    return text


def clean_text(text: str) -> Tuple[str, float]:
    """
    Full cleaning pipeline for a single text field.
    Returns (cleaned_text, repetition_density).

    Pipeline order:
      1. Unicode NFC normalization
      2. Strip garbage tokens (Whisper artifacts)
      3. Apply ASR phonetic replacements (দাক্তা -> ডাক্তার)
      4. Collapse greeting runs
      5. Whitespace fix
      6. N-gram repetition loop collapse
      7. Bengali punctuation normalization
      8. Final whitespace fix
    """
    text = normalize_unicode(text)
    text = strip_garbage_tokens(text)
    text = apply_asr_replacements(text)
    text = collapse_greeting_runs(text)
    text = fix_whitespace(text)
    text, rep_density = collapse_repetition_loops(text)
    text = normalize_bengali_punctuation(text)
    text = fix_whitespace(text)
    return text, rep_density


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def compute_quality_score(
    text: str,
    start: float,
    end: float,
    rep_density: float,
) -> float:
    """
    Compute a quality score in [0.0, 1.0].

    Three equally-weighted components:
      1. word_rate_ok  : words/second is in [MIN_WPS, MAX_WPS]
      2. rep_ok        : repetition density below threshold
      3. length_ok     : segment has ≥ 2 words

    Score = mean(component_scores)
    """
    words = text.split()
    word_count = len(words)
    duration = max(end - start, 0.01)  # avoid division by zero
    wps = word_count / duration

    # Component 1: word rate
    if MIN_WPS <= wps <= MAX_WPS:
        word_rate_score = 1.0
    elif wps < MIN_WPS:
        # Linearly decay toward 0 as wps → 0
        word_rate_score = max(0.0, wps / MIN_WPS)
    else:
        # Linearly decay toward 0 as wps → MAX_WPS * 3
        word_rate_score = max(0.0, 1.0 - (wps - MAX_WPS) / (MAX_WPS * 2))

    # Component 2: repetition density (lower is better)
    rep_score = max(0.0, 1.0 - rep_density / REPETITION_DENSITY_THRESHOLD)
    rep_score = min(rep_score, 1.0)

    # Component 3: minimum word length
    length_score = 1.0 if word_count >= 2 else (0.5 if word_count == 1 else 0.0)

    quality = (word_rate_score + rep_score + length_score) / 3.0
    return round(quality, 4)


def should_flag(quality_score: float, rep_density: float) -> bool:
    """Flag a segment as suspicious if quality drops below 0.35 or rep_density > 0.5."""
    return quality_score < 0.35 or rep_density > REPETITION_DENSITY_THRESHOLD


# ---------------------------------------------------------------------------
# Segment processing
# ---------------------------------------------------------------------------

def process_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean a single segment dict in-place (returns new dict to preserve immutability).
    Adds: text_bn_raw, quality_score, flagged, repetition_density fields.
    """
    original_text = segment.get("text_bn", "")
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", 0.0))

    cleaned_text, rep_density = clean_text(original_text)
    quality = compute_quality_score(cleaned_text, start, end, rep_density)
    flagged = should_flag(quality, rep_density)

    return {
        **segment,
        "text_bn_raw": original_text,          # always preserve the original
        "text_bn": cleaned_text,               # overwrite with cleaned version
        "quality_score": quality,
        "repetition_density": round(rep_density, 4),
        "flagged": flagged,
    }


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

def process_file(input_path: Path, output_path: Path) -> Tuple[int, int]:
    """
    Stream-process a JSONL file line by line and write cleaned output.
    Each line is a JSON object representing one phone call.

    Returns (segments_in_total, segments_out_total).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segments_in = 0
    segments_out = 0

    # Use a temporary file to ensure atomicity — only rename on success.
    tmp_path = output_path.with_suffix(".jsonl.tmp")

    with (
        open(input_path, "r", encoding="utf-8") as fin,
        open(tmp_path, "w", encoding="utf-8") as fout,
    ):
        for line_no, raw_line in enumerate(fin, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed JSON at %s line %d: %s",
                    input_path.name,
                    line_no,
                    exc,
                )
                continue

            segments: List[Dict[str, Any]] = record.get("segments", [])
            segments_in += len(segments)

            cleaned_segments = [process_segment(seg) for seg in segments]
            segments_out += len(cleaned_segments)

            output_record = {
                **record,
                "segments": cleaned_segments,
                "nlp_stage": "cleaned",
                "nlp_cleaned_at": datetime.now(timezone.utc).isoformat(),
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    # Atomic rename — only visible to readers after complete write
    tmp_path.replace(output_path)
    return segments_in, segments_out


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

def run(
    input_dir: Path,
    output_dir: Path,
    checkpoint_db: Path,
    pattern: str,
    force: bool,
) -> None:
    """Discover and process all JSONL files under input_dir."""
    checkpoint = NLPCleaningCheckpoint(checkpoint_db)

    input_files = sorted(input_dir.rglob(pattern))
    if not input_files:
        logger.warning("No files matching '%s' found in %s", pattern, input_dir)
        return

    logger.info(
        "Found %d file(s) to process. Checkpoint DB: %s",
        len(input_files),
        checkpoint_db,
    )

    skipped = done = errors = 0

    with tqdm(input_files, desc="Cleaning Bengali transcripts", unit="file") as pbar:
        for input_path in pbar:
            # Preserve relative directory structure in output
            try:
                rel = input_path.relative_to(input_dir)
            except ValueError:
                rel = Path(input_path.name)

            file_id = str(rel)
            output_path = output_dir / rel

            pbar.set_postfix(file=input_path.name, refresh=False)

            if not force and checkpoint.is_done(file_id):
                skipped += 1
                logger.debug("SKIP (already done): %s", file_id)
                continue

            try:
                segments_in, segments_out = process_file(input_path, output_path)
                checkpoint.mark_done(
                    file_id=file_id,
                    input_path=str(input_path),
                    output_path=str(output_path),
                    segments_in=segments_in,
                    segments_out=segments_out,
                )
                done += 1
                logger.debug(
                    "DONE: %s — %d segments in / %d out",
                    file_id,
                    segments_in,
                    segments_out,
                )
            except Exception as exc:  # noqa: BLE001
                errors += 1
                err_msg = f"{type(exc).__name__}: {exc}"
                checkpoint.mark_error(file_id=file_id, input_path=str(input_path), error=err_msg)
                logger.error("ERROR processing %s: %s", file_id, err_msg)

    stats = checkpoint.stats()
    logger.info(
        "Run complete — done=%d  skipped=%d  errors=%d  | DB stats: %s",
        done,
        skipped,
        errors,
        stats,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clean_bengali",
        description="Stage 1 — Bengali transcript cleaning pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/transcripts"),
        help="Directory containing raw Whisper JSONL transcripts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/nlp/cleaned"),
        help="Directory to write cleaned JSONL files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/nlp_cleaning.sqlite"),
        help="Path to the SQLite checkpoint database.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="Glob pattern to match input files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process files even if already marked done in checkpoint DB.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint_db=args.checkpoint,
        pattern=args.pattern,
        force=args.force,
    )


if __name__ == "__main__":
    main()
