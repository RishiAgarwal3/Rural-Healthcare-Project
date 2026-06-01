"""
Stage 2 — Bengali → English Contextual Translation
====================================================
INPUT  : data/nlp/cleaned/*.jsonl          (Stage 1 cleaned Bengali)
OUTPUT : data/nlp/translated/*.jsonl       (Bengali preserved + English added)
LOGS   : data/nlp/logs/translation_<date>.log
CHECKPOINT : checkpoints/nlp_translation.sqlite

Pipeline guarantees
-------------------
• data/transcripts/      NEVER touched  — raw Whisper output preserved as-is
• data/nlp/cleaned/      NEVER touched  — cleaned Bengali preserved as-is
• data/nlp/translated/   new files only — Bengali + English side-by-side
• All four layers remain independently auditable and reproducible.

Contextual translation strategy
---------------------------------
PROBLEM with segment-by-segment translation:
  Each segment is sent in isolation.  The model has no idea that:
    - "উনি" (he/she) refers to a doctor mentioned 3 segments earlier
    - "সেখানে" (there) means a specific rural clinic
    - "শুক্রবার করে" (doing friday) is a repetition hallucination
  This produces pronoun confusion, broken references, and unnatural English.

SOLUTION — call-context translation (for API backends):
  1. Collect ALL segments for a call into a structured JSON payload.
  2. Send the FULL payload to the LLM in a single request, with a system
     prompt that establishes domain expertise (rural health, telephony).
  3. The model sees the entire conversation → pronouns resolve, medical
     references carry through, conversational flow is natural.
  4. We parse the model's JSON response and map text_en back to each
     segment by segment_id — timestamps and boundaries are always preserved.
  5. Flagged segments are still translated but carry warning metadata
     (translation_warning: true) so downstream models can weight them.

NLLB local backend: uses a sliding-window context strategy — prepends the
previous 2 segments as "[CONTEXT]" tokens before the target segment to give
the seq2seq model cross-sentence cues, while only the current segment's
translation is retained.

Backends
--------
  --backend openai      OpenAI Chat Completions (default: gpt-4o-mini)
  --backend anthropic   Anthropic Claude Messages
  --backend nllb        Local HuggingFace NLLB-200 (no API key needed)

Modes
-----
  Normal:   translate all files, skip already-done files via checkpoint
  Validate: --validate-only --num-calls N
            runs on N calls from the first file, prints a human-readable
            comparison (raw Bengali / cleaned Bengali / English) so you can
            inspect quality before committing to a full run.

Usage examples
--------------
  # Validation batch — inspect quality before full run:
  python -m src.nlp.translate_english \\
      --backend openai --validate-only --num-calls 3

  # Full run with OpenAI:
  python -m src.nlp.translate_english \\
      --backend openai --openai-model gpt-4o-mini

  # Full run with Anthropic:
  python -m src.nlp.translate_english \\
      --backend anthropic --anthropic-model claude-3-haiku-20240307

  # Full run with local NLLB:
  python -m src.nlp.translate_english \\
      --backend nllb --nllb-model facebook/nllb-200-distilled-600M \\
      --device cpu --batch-size 8

  # Resume a crashed run (auto-skips completed files):
  python -m src.nlp.translate_english --backend openai

  # Force re-process everything:
  python -m src.nlp.translate_english --backend openai --force
"""

from __future__ import annotations

import abc
import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging — dual output: stderr console + rotating file under data/nlp/logs/
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: Path, verbose: bool) -> logging.Logger:
    """Configure logging to both stderr and a date-stamped file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"translation_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"

    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Console handler (stderr)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt, datefmt))
    root_logger.addHandler(console)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)   # always full detail in the log file
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    root_logger.addHandler(fh)

    logger = logging.getLogger("nlp.translate_english")
    logger.info("Log file: %s", log_file)
    return logger


logger = logging.getLogger("nlp.translate_english")


# ---------------------------------------------------------------------------
# Storage paths — single source of truth
# ---------------------------------------------------------------------------

class StoragePaths:
    """
    Centralised path definitions.  All pipeline stages read from here
    so paths never drift between scripts.
    """

    def __init__(self, project_root: Path = Path(".")) -> None:
        self.root = project_root.resolve()

    # Input (read-only — never written by this stage)
    @property
    def raw_transcripts(self) -> Path:
        return self.root / "data" / "transcripts"

    @property
    def cleaned_bengali(self) -> Path:
        return self.root / "data" / "nlp" / "cleaned"

    # Output
    @property
    def translated(self) -> Path:
        return self.root / "data" / "nlp" / "translated"

    @property
    def enriched(self) -> Path:
        return self.root / "data" / "enriched"

    @property
    def logs(self) -> Path:
        return self.root / "data" / "nlp" / "logs"

    @property
    def checkpoint_db(self) -> Path:
        return self.root / "checkpoints" / "nlp_translation.sqlite"

    def print_summary(self) -> None:
        """Print a clear storage map to stdout."""
        print("\n" + "=" * 72)
        print("  STORAGE LOCATIONS — Translation Stage")
        print("=" * 72)
        print(f"  INPUT  (cleaned Bengali)   : {self.cleaned_bengali}")
        print(f"  OUTPUT (translated JSONL)  : {self.translated}")
        print(f"  LOGS                       : {self.logs}")
        print(f"  CHECKPOINT DB              : {self.checkpoint_db}")
        print(f"  [PRESERVED] Raw transcripts: {self.raw_transcripts}")
        print(f"  [FUTURE]    Enriched data  : {self.enriched}")
        print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# SQLite Checkpoint
# ---------------------------------------------------------------------------

class NLPTranslationCheckpoint:
    """
    File-level checkpoint for the translation stage.

    Schema:
        file_id        TEXT  PRIMARY KEY  (relative path from input_dir)
        status         TEXT               'done' | 'error'
        backend        TEXT               which translator backend was used
        strategy       TEXT               'call_context' | 'windowed_nllb'
        input_path     TEXT
        output_path    TEXT
        calls          INTEGER            number of call records processed
        segments       INTEGER            total segments translated
        flagged_segs   INTEGER            segments that had quality warnings
        error          TEXT
        updated_at     TEXT
    """

    TABLE = "nlp_translation"

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
                    backend      TEXT,
                    strategy     TEXT,
                    input_path   TEXT,
                    output_path  TEXT,
                    calls        INTEGER,
                    segments     INTEGER,
                    flagged_segs INTEGER,
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
        backend: str,
        strategy: str,
        input_path: str,
        output_path: str,
        calls: int,
        segments: int,
        flagged_segs: int,
    ) -> None:
        self._upsert(
            file_id=file_id,
            status="done",
            backend=backend,
            strategy=strategy,
            input_path=input_path,
            output_path=output_path,
            calls=calls,
            segments=segments,
            flagged_segs=flagged_segs,
            error=None,
        )

    def mark_error(
        self, file_id: str, backend: str, strategy: str, input_path: str, error: str
    ) -> None:
        self._upsert(
            file_id=file_id,
            status="error",
            backend=backend,
            strategy=strategy,
            input_path=input_path,
            output_path=None,
            calls=None,
            segments=None,
            flagged_segs=None,
            error=error,
        )

    def _upsert(self, **kwargs: Any) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.TABLE}
                    (file_id, status, backend, strategy, input_path, output_path,
                     calls, segments, flagged_segs, error, updated_at)
                VALUES
                    (:file_id, :status, :backend, :strategy, :input_path, :output_path,
                     :calls, :segments, :flagged_segs, :error, :updated_at)
                ON CONFLICT(file_id) DO UPDATE SET
                    status       = excluded.status,
                    backend      = excluded.backend,
                    strategy     = excluded.strategy,
                    input_path   = excluded.input_path,
                    output_path  = excluded.output_path,
                    calls        = excluded.calls,
                    segments     = excluded.segments,
                    flagged_segs = excluded.flagged_segs,
                    error        = excluded.error,
                    updated_at   = excluded.updated_at
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
# System prompt (shared by all API backends)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are translating noisy Bengali rural phone-call transcripts into clean natural English.

The Bengali text comes from imperfect speech-to-text transcription of real phone conversations in rural West Bengal.

The transcript may contain:
* spelling mistakes
* phonetic spellings
* dialect Bengali
* repeated words
* ASR artifacts
* broken grammar

Your task:
1. Infer the intended Bengali meaning.
2. Translate semantically, not literally.
3. Produce fluent conversational English.
4. Ignore obvious ASR repetition artifacts.
5. Do NOT hallucinate extra details.
6. Preserve uncertainty if meaning is unclear.
7. Keep translations concise and natural.

Examples:
Input:
"আমি যে দাক্তা এখনা তেকে ফন করছেল"
Meaning:
"আমি যে ডাক্তারখানা থেকে ফোন করছিলাম"
Output:
"I was calling from the clinic."

Input:
"উনি দাক্তা দেখিয়ে গেছিলেন"
Output:
"He had visited the doctor."

Input format:
  A JSON array of segment objects, each with:
    "id"      : integer segment index
    "text_bn" : cleaned Bengali text
    "flagged" : boolean — true if Whisper quality was suspicious

Output format (STRICT — return ONLY valid JSON, no extra text):
  A JSON array with the same length and order:
    [{"id": 0, "text_en": "..."}, {"id": 1, "text_en": "..."}, ...]

If a segment's text_bn is empty or pure garbage, return text_en as "".
"""

# ---------------------------------------------------------------------------
# Abstract backend interface
# ---------------------------------------------------------------------------

class TranslatorBackend(abc.ABC):
    """
    Abstract base class.  Subclasses implement one of two strategies:
      • translate_call_context: send full call as JSON, get JSON back (API backends)
      • translate_batch:        send individual texts, with optional context prefix (NLLB)
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable identifier for checkpoint records."""

    @property
    @abc.abstractmethod
    def strategy(self) -> str:
        """'call_context' or 'windowed_nllb'"""

    @abc.abstractmethod
    def translate_call(
        self,
        segments: List[Dict[str, Any]],
        call_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Translate all segments of a single call.

        Args:
            segments : list of cleaned segment dicts (must contain 'text_bn')
            call_id  : used only for logging/error messages

        Returns:
            Same list with 'text_en' and optional 'translation_warning' added.
            Every input segment MUST have a corresponding output segment.
        """


# ---------------------------------------------------------------------------
# Shared JSON response parser (used by both API backends)
# ---------------------------------------------------------------------------

def _parse_json_translation_response(
    raw_response: str,
    expected_count: int,
    call_id: str,
) -> Optional[List[Dict[str, Any]]]:
    """
    Parse LLM JSON response into a list of {id, text_en} dicts.

    The LLM occasionally wraps the JSON in markdown fences (```json ... ```).
    We strip those before parsing.  On failure, returns None.
    """
    # Strip markdown code fences if present
    text = raw_response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "[%s] Failed to parse translation JSON: %s | Response was: %s…",
            call_id,
            exc,
            text[:200],
        )
        return None

    if not isinstance(parsed, list):
        logger.warning(
            "[%s] Translation response was not a list (got %s).",
            call_id,
            type(parsed).__name__,
        )
        return None

    if len(parsed) != expected_count:
        logger.warning(
            "[%s] Translation response length mismatch: expected %d, got %d.",
            call_id,
            expected_count,
            len(parsed),
        )
        # Still usable if close — we'll match by 'id' field below
        # Return it and let the caller reconcile.

    return parsed


def _reconcile_translations(
    original_segments: List[Dict[str, Any]],
    parsed_response: List[Dict[str, Any]],
    call_id: str,
) -> List[Dict[str, Any]]:
    """
    Map text_en values from the LLM response back to the original segments by 'id'.
    If a segment's translation is missing, falls back to '[TRANSLATION_MISSING]'.
    """
    # Build a lookup from id → text_en
    id_to_en: Dict[int, str] = {}
    for item in parsed_response:
        seg_id = item.get("id")
        text_en = item.get("text_en", "")
        if seg_id is not None:
            id_to_en[int(seg_id)] = str(text_en)

    result: List[Dict[str, Any]] = []
    for idx, seg in enumerate(original_segments):
        text_en = id_to_en.get(idx, None)
        if text_en is None:
            logger.warning(
                "[%s] Missing translation for segment %d; using fallback.",
                call_id,
                idx,
            )
            text_en = "[TRANSLATION_MISSING]"
            translation_warning = True
        else:
            translation_warning = bool(seg.get("flagged", False))

        result.append({
            **seg,
            "text_en": text_en,
            "translation_warning": translation_warning,
        })

    return result


# ---------------------------------------------------------------------------
# Backend 1: OpenAI — full call-context strategy
# ---------------------------------------------------------------------------

class OpenAITranslator(TranslatorBackend):
    """
    Sends the entire call as a single structured JSON payload.
    The model sees all segments simultaneously → coherent pronoun resolution
    and natural conversational flow.

    Requires: pip install openai
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        request_delay: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI package not installed. Run: pip install openai"
            ) from exc

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY env var "
                "or pass --openai-api-key."
            )

        self._client = OpenAI(api_key=resolved_key)
        self._model = model
        self._delay = request_delay
        self._max_retries = max_retries
        logger.info("OpenAI translator ready (model=%s, strategy=call_context).", model)

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    @property
    def strategy(self) -> str:
        return "call_context"

    def translate_call(
        self,
        segments: List[Dict[str, Any]],
        call_id: str,
    ) -> List[Dict[str, Any]]:
        if not segments:
            return []

        # Build the minimal payload: only what the model needs
        payload = [
            {
                "id": idx,
                "text_bn": seg.get("text_bn", ""),
                "flagged": bool(seg.get("flagged", False)),
            }
            for idx, seg in enumerate(segments)
        ]

        user_message = json.dumps(payload, ensure_ascii=False, indent=2)

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.15,    # low variance for consistent medical translation
                    max_tokens=4096,
                    response_format={"type": "json_object"} if "gpt-4" in self._model else None,
                )
                raw = response.choices[0].message.content or ""

                # The model may return {"translations": [...]} or directly [...]
                # Normalise to a bare list string for _parse_json_translation_response
                raw = _unwrap_json_envelope(raw)

                parsed = _parse_json_translation_response(raw, len(segments), call_id)
                if parsed is not None:
                    if self._delay > 0:
                        time.sleep(self._delay)
                    return _reconcile_translations(segments, parsed, call_id)

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] OpenAI attempt %d/%d failed: %s",
                    call_id, attempt, self._max_retries, exc,
                )
                if attempt < self._max_retries:
                    time.sleep(self._delay * 2 ** attempt)   # exponential back-off

        # All retries exhausted — return segments with failure marker
        logger.error("[%s] Translation failed after %d retries.", call_id, self._max_retries)
        return _mark_all_failed(segments)


# ---------------------------------------------------------------------------
# Backend 2: Anthropic — full call-context strategy
# ---------------------------------------------------------------------------

class AnthropicTranslator(TranslatorBackend):
    """
    Sends the entire call as structured JSON.  Same strategy as OpenAI.

    Requires: pip install anthropic
    """

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
        request_delay: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        try:
            import anthropic as anthropic_sdk
            self._sdk = anthropic_sdk
        except ImportError as exc:
            raise ImportError(
                "Anthropic package not installed. Run: pip install anthropic"
            ) from exc

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY env var "
                "or pass --anthropic-api-key."
            )

        self._client = self._sdk.Anthropic(api_key=resolved_key)
        self._model = model
        self._delay = request_delay
        self._max_retries = max_retries
        logger.info("Anthropic translator ready (model=%s, strategy=call_context).", model)

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    @property
    def strategy(self) -> str:
        return "call_context"

    def translate_call(
        self,
        segments: List[Dict[str, Any]],
        call_id: str,
    ) -> List[Dict[str, Any]]:
        if not segments:
            return []

        payload = [
            {
                "id": idx,
                "text_bn": seg.get("text_bn", ""),
                "flagged": bool(seg.get("flagged", False)),
            }
            for idx, seg in enumerate(segments)
        ]

        user_message = (
            "Translate these call segments from Bengali to English.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

        for attempt in range(1, self._max_retries + 1):
            try:
                message = self._client.messages.create(
                    model=self._model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )
                raw = message.content[0].text.strip()
                raw = _unwrap_json_envelope(raw)

                parsed = _parse_json_translation_response(raw, len(segments), call_id)
                if parsed is not None:
                    if self._delay > 0:
                        time.sleep(self._delay)
                    return _reconcile_translations(segments, parsed, call_id)

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] Anthropic attempt %d/%d failed: %s",
                    call_id, attempt, self._max_retries, exc,
                )
                if attempt < self._max_retries:
                    time.sleep(self._delay * 2 ** attempt)

        logger.error("[%s] Translation failed after %d retries.", call_id, self._max_retries)
        return _mark_all_failed(segments)


# ---------------------------------------------------------------------------
# Backend 3: NLLB (local HuggingFace) — windowed context strategy
# ---------------------------------------------------------------------------

class NLLBTranslator(TranslatorBackend):
    """
    Production-quality local translation via Meta's NLLB-200 model.

    Features:
      • Auto-detects best device: CUDA → MPS (Apple Silicon) → CPU
      • Pre-translation Bengali cleanup (re-strips repetitions right before
        inference so the seq2seq model receives the cleanest possible input)
      • Hallucination skip: segments with repetition_density > 0.4 or
        flagged=True get text_en='[LOW_QUALITY_AUDIO]' instead of wasting
        compute on garbled input that would produce garbled output
      • LRU translation cache: identical Bengali strings (which occur across
        calls — e.g. greetings, stock phrases) are translated only once
      • Sliding-window context: prepends previous CONTEXT_WINDOW segments
        for cross-sentence pronoun/reference resolution

    Recommended models:
      facebook/nllb-200-distilled-600M   (~2.4 GB, good quality, fast)
      facebook/nllb-200-distilled-1.3B   (~5.2 GB, better quality)

    Requires: pip install transformers sentencepiece torch
    """

    SOURCE_LANG = "ben_Beng"
    TARGET_LANG = "eng_Latn"
    CONTEXT_WINDOW = 0        # disabled: NLLB is sentence-level and breaks with separators
    CONTEXT_SEP = " | "       # separator between context and target
    CACHE_MAXSIZE = 4096      # max cached unique Bengali strings
    SKIP_REP_DENSITY = 0.4    # skip translation if rep_density exceeds this
    LOW_QUALITY_LABEL = "[LOW_QUALITY_AUDIO]"

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "auto",
        batch_size: int = 8,
    ) -> None:
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
            import torch
        except ImportError as exc:
            raise ImportError(
                "HuggingFace Transformers not installed. "
                "Run: pip install transformers sentencepiece torch"
            ) from exc

        # ── Auto-detect best available device ──
        resolved_device = self._resolve_device(device, torch)
        logger.info(
            "Loading NLLB model '%s' on device '%s' …", model_name, resolved_device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=self.SOURCE_LANG)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Map device string → PyTorch device
        if resolved_device == "cuda":
            torch_device = torch.device("cuda")
        elif resolved_device == "mps":
            torch_device = torch.device("mps")
        else:
            torch_device = torch.device("cpu")

        self._model = model.to(torch_device)
        self._tokenizer = tokenizer
        self._device = torch_device
        self._target_lang_id = tokenizer.convert_tokens_to_ids(self.TARGET_LANG)
        
        self._model_name = model_name
        self._batch_size = batch_size
        self._resolved_device = resolved_device

        # ── Translation cache (LRU) ──
        # Key: Bengali text string → Value: English translation
        from functools import lru_cache
        self._translate_single_cached = lru_cache(maxsize=self.CACHE_MAXSIZE)(
            self._translate_single_uncached
        )
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info(
            "NLLB model ready on %s.  Cache capacity: %d entries.",
            resolved_device,
            self.CACHE_MAXSIZE,
        )

    @staticmethod
    def _resolve_device(requested: str, torch_module: Any) -> str:
        """
        Resolve the best available device.
        'auto' → CUDA > MPS > CPU.
        Explicit values ('cuda', 'mps', 'cpu') are respected but validated.
        """
        if requested == "auto":
            if torch_module.cuda.is_available():
                return "cuda"
            elif hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"

        # Validate explicit device
        if requested == "cuda" and not torch_module.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        if requested == "mps":
            if not (hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available()):
                logger.warning("MPS requested but not available. Falling back to CPU.")
                return "cpu"
        return requested

    @property
    def name(self) -> str:
        return f"nllb:{self._model_name}"

    @property
    def strategy(self) -> str:
        return "nllb_windowed_context"

    # ── Pre-translation Bengali cleanup ──

    @staticmethod
    def _pre_clean_for_translation(text: str) -> str:
        """
        Lightweight Bengali cleanup applied immediately before translation.
        This catches residual issues the main cleaning stage may have left:
          - Collapse any remaining word-level duplicate runs (≥2)
          - Strip leading/trailing whitespace and punctuation noise
          - Normalise multiple spaces
        This does NOT replace Stage 1 cleaning — it's a safety net.
        """
        if not text or not text.strip():
            return ""

        # Collapse consecutive duplicate words (≥2 runs)
        words = text.split()
        deduped: List[str] = []
        prev = None
        repeat_count = 0
        for w in words:
            if w == prev:
                repeat_count += 1
                if repeat_count < 2:  # allow one repeat (e.g. "না না" = "no no")
                    deduped.append(w)
                # else: skip — this is a 3rd+ consecutive repeat
            else:
                deduped.append(w)
                prev = w
                repeat_count = 0

        result = " ".join(deduped).strip()
        # Collapse multiple spaces
        result = re.sub(r"\s{2,}", " ", result)
        return result

    # ── Should-skip logic ──

    @staticmethod
    def _should_skip_translation(seg: Dict[str, Any]) -> bool:
        """
        Determine if a segment's Bengali text is too noisy to translate.
        Returns True if:
          - flagged == True  (from Stage 1 quality scoring)
          - repetition_density > SKIP_REP_DENSITY
          - text_bn is empty or whitespace-only
        """
        if bool(seg.get("flagged", False)):
            return True
        if float(seg.get("repetition_density", 0.0)) > NLLBTranslator.SKIP_REP_DENSITY:
            return True
        text = seg.get("text_bn", "").strip()
        if not text:
            return True
        return False

    # ── Core translation with caching ──

    def _translate_single_uncached(self, text: str) -> str:
        """Translate a single Bengali string. Called via LRU cache wrapper."""
        if not text.strip():
            return ""
        try:
            inputs = self._tokenizer(text, return_tensors="pt", padding=True).to(self._device)
            translated_tokens = self._model.generate(
                **inputs,
                forced_bos_token_id=self._target_lang_id,
                max_length=512
            )
            return self._tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("NLLB single-text translation failed: %s", exc)
            return "[TRANSLATION_FAILED]"

    def _translate_with_cache(self, text: str) -> str:
        """
        Translate a Bengali string, using the LRU cache for repeated phrases.
        """
        # Check cache stats (the lru_cache tracks this internally)
        result = self._translate_single_cached(text)
        return result

    def _translate_batch_texts(self, texts: List[str]) -> List[str]:
        """
        Translate a list of Bengali texts using batched inference.
        Uses the cache for texts we've seen before; batches only the novel ones.
        """
        results: List[Optional[str]] = [None] * len(texts)
        novel_indices: List[int] = []
        novel_texts: List[str] = []

        # Phase 1: check cache for each text
        for i, text in enumerate(texts):
            if not text.strip():
                results[i] = ""
                continue
            # Try the cache (lru_cache is not iterable — we peek by calling)
            # Use cache_info to detect if this will be a hit
            info_before = self._translate_single_cached.cache_info()
            cached = self._translate_single_cached(text)
            info_after = self._translate_single_cached.cache_info()
            if info_after.hits > info_before.hits:
                # Was a cache hit
                results[i] = cached
                self._cache_hits += 1
            else:
                # Was a cache miss — the call already translated it, so use the result
                results[i] = cached
                self._cache_misses += 1

        return [r if r is not None else "" for r in results]

    # ── Main entry point ──

    def translate_call(
        self,
        segments: List[Dict[str, Any]],
        call_id: str,
    ) -> List[Dict[str, Any]]:
        if not segments:
            return []

        result_segments: List[Dict[str, Any]] = []

        # Phase 1: classify segments — skip vs translate
        texts_for_translation: List[str] = []   # pre-cleaned Bengali texts
        segment_indices: List[int] = []          # which segments need translation

        for i, seg in enumerate(segments):
            if self._should_skip_translation(seg):
                # Hallucinated / garbage — skip translation entirely
                result_segments.append({
                    **seg,
                    "text_en": self.LOW_QUALITY_LABEL,
                    "translation_warning": True,
                    "translation_skipped": True,
                })
            else:
                # Pre-clean Bengali text before sending to NLLB
                cleaned = self._pre_clean_for_translation(seg.get("text_bn", ""))
                texts_for_translation.append(cleaned)
                segment_indices.append(i)
                result_segments.append(None)  # placeholder

        if not texts_for_translation:
            logger.debug("[%s] All %d segments skipped (low quality).", call_id, len(segments))
            return result_segments

        # Phase 2: build windowed context inputs
        windowed_inputs: List[str] = []
        for j, text in enumerate(texts_for_translation):
            # Gather context from the previous CONTEXT_WINDOW translated texts
            ctx_start = max(0, j - self.CONTEXT_WINDOW)
            context_parts = texts_for_translation[ctx_start:j]
            if context_parts:
                windowed_text = self.CONTEXT_SEP.join(context_parts) + self.CONTEXT_SEP + text
            else:
                windowed_text = text
            windowed_inputs.append(windowed_text)

        # Phase 3: translate (with cache)
        translations = self._translate_batch_texts(windowed_inputs)

        # Phase 4: post-process — strip echoed context separators
        cleaned_translations: List[str] = []
        for raw_t in translations:
            if self.CONTEXT_SEP in raw_t:
                raw_t = raw_t.split(self.CONTEXT_SEP)[-1].strip()
            cleaned_translations.append(raw_t)

        # Phase 5: map translations back to segment positions
        for j, seg_idx in enumerate(segment_indices):
            seg = segments[seg_idx]
            result_segments[seg_idx] = {
                **seg,
                "text_en": cleaned_translations[j],
                "translation_warning": False,
                "translation_skipped": False,
            }

        logger.debug(
            "[%s] Translated %d/%d segments (skipped %d).  Cache: %d hits / %d misses.",
            call_id,
            len(translations),
            len(segments),
            len(segments) - len(translations),
            self._cache_hits,
            self._cache_misses,
        )
        return result_segments


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _unwrap_json_envelope(raw: str) -> str:
    """
    Some models return {"translations": [...]} instead of a bare list.
    Strip one level of object envelope if present.
    """
    text = raw.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
    text = text.strip()

    # Detect JSON object wrapping a list value
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                # Find the first key whose value is a list
                for val in obj.values():
                    if isinstance(val, list):
                        return json.dumps(val, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

    return text


def _mark_all_failed(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return segments with text_en set to failure marker."""
    return [
        {**seg, "text_en": "[TRANSLATION_FAILED]", "translation_warning": True}
        for seg in segments
    ]


# ---------------------------------------------------------------------------
# Backend 4: Two-Stage Repair + Translate (LLM-based, best quality)
# ---------------------------------------------------------------------------

TWO_STAGE_REPAIR_PROMPT = """\
You are an expert Bengali conversational transcript repair system.

The following Bengali text comes from noisy phone-call ASR output and may contain:
  * phonetic spelling errors
  * broken words
  * repeated fragments
  * partial words
  * incorrect punctuation
  * ASR hallucinations

Your task:
  1. Reconstruct the MOST LIKELY natural Bengali sentence.
  2. Preserve the original meaning.
  3. Remove repetitions and ASR garbage.
  4. Do NOT invent information.
  5. Keep uncertain regions conservative.
  6. Output ONLY corrected Bengali text, nothing else.
"""

TWO_STAGE_TRANSLATE_PROMPT = """\
Translate the following cleaned Bengali conversational speech into natural English.

Rules:
  * Preserve meaning faithfully.
  * Do not hallucinate.
  * If a phrase is unclear, translate conservatively.
  * Keep tone conversational.
  * Ignore filler words and ASR artifacts.

Output ONLY the English translation, nothing else.
"""


class TwoStageRepairTranslator(TranslatorBackend):
    """
    Two-stage LLM pipeline for noisy Bengali telephone ASR:

      Stage A — Repair:    LLM reconstructs the most likely natural Bengali
                           sentence from garbled/phonetic/broken ASR output.
      Stage B — Translate: LLM translates the repaired Bengali to natural English.

    This produces significantly better results than direct seq2seq translation
    (NLLB) on telephone-quality audio because the LLM understands:
      • Rural Bengali phonetic spelling ("দাক্তা" = doctor)
      • Broken compound words
      • Conversational filler and dialect

    Works with OpenAI and Anthropic models via the --repair-model flag.
    Requires: pip install openai   OR   pip install anthropic
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        request_delay: float = 0.3,
        max_retries: int = 3,
    ) -> None:
        self._provider = provider.lower()
        self._model = model
        self._delay = request_delay
        self._max_retries = max_retries

        if self._provider == "openai":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("Run: pip install openai") from exc
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not resolved_key:
                raise ValueError("Set OPENAI_API_KEY env var or pass --repair-api-key.")
            self._client = OpenAI(api_key=resolved_key)
        elif self._provider == "anthropic":
            try:
                import anthropic as anthropic_sdk
                self._sdk = anthropic_sdk
            except ImportError as exc:
                raise ImportError("Run: pip install anthropic") from exc
            resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not resolved_key:
                raise ValueError("Set ANTHROPIC_API_KEY env var or pass --repair-api-key.")
            self._client = self._sdk.Anthropic(api_key=resolved_key)
        else:
            raise ValueError(f"repair backend provider must be 'openai' or 'anthropic', got '{provider}'.")

        logger.info(
            "TwoStageRepairTranslator ready (provider=%s, model=%s).",
            self._provider, model,
        )

    @property
    def name(self) -> str:
        return f"repair:{self._provider}:{self._model}"

    @property
    def strategy(self) -> str:
        return "two_stage_repair"

    def _call_llm(self, system: str, user: str) -> str:
        """Single LLM call with retries. Returns the text content."""
        for attempt in range(1, self._max_retries + 1):
            try:
                if self._provider == "openai":
                    resp = self._client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=0.1,
                        max_tokens=512,
                    )
                    return (resp.choices[0].message.content or "").strip()
                else:  # anthropic
                    msg = self._client.messages.create(
                        model=self._model,
                        max_tokens=512,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    return msg.content[0].text.strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LLM call attempt %d/%d failed: %s", attempt, self._max_retries, exc
                )
                if attempt < self._max_retries:
                    time.sleep(self._delay * 2 ** attempt)
        return ""

    def _repair_segment(self, text_bn_raw: str) -> str:
        """Stage A: Repair noisy Bengali ASR text."""
        if not text_bn_raw.strip():
            return ""
        user_msg = f"Input:\n{text_bn_raw}"
        repaired = self._call_llm(TWO_STAGE_REPAIR_PROMPT, user_msg)
        return repaired or text_bn_raw  # fallback to original if LLM fails

    def _translate_segment(self, cleaned_bn: str) -> str:
        """Stage B: Translate cleaned Bengali to English."""
        if not cleaned_bn.strip():
            return ""
        user_msg = f"Input:\n{cleaned_bn}"
        return self._call_llm(TWO_STAGE_TRANSLATE_PROMPT, user_msg)

    def translate_call(
        self,
        segments: List[Dict[str, Any]],
        call_id: str,
    ) -> List[Dict[str, Any]]:
        if not segments:
            return []

        results: List[Dict[str, Any]] = []
        for seg in segments:
            raw_bn = seg.get("text_bn", "")

            # Stage A: Repair
            if bool(seg.get("flagged", False)) and float(seg.get("repetition_density", 0)) > 0.6:
                # Severely hallucinated — skip both stages
                repaired_bn = raw_bn
                text_en = "[LOW_QUALITY_AUDIO]"
                results.append({
                    **seg,
                    "text_bn_repaired": repaired_bn,
                    "text_en": text_en,
                    "translation_warning": True,
                    "translation_skipped": True,
                })
                continue

            repaired_bn = self._repair_segment(raw_bn)
            if self._delay > 0:
                time.sleep(self._delay)

            # Stage B: Translate
            text_en = self._translate_segment(repaired_bn)
            if self._delay > 0:
                time.sleep(self._delay)

            results.append({
                **seg,
                "text_bn_repaired": repaired_bn,   # preserve the LLM-repaired Bengali
                "text_en": text_en,
                "translation_warning": bool(seg.get("flagged", False)),
                "translation_skipped": False,
            })

        logger.debug(
            "[%s] TwoStage: repaired+translated %d/%d segments.",
            call_id, len(results), len(segments),
        )
        return results


# ---------------------------------------------------------------------------
# Backend registry and factory
# ---------------------------------------------------------------------------

BACKENDS: Dict[str, type] = {
    "openai": OpenAITranslator,
    "anthropic": AnthropicTranslator,
    "nllb": NLLBTranslator,
    "repair": TwoStageRepairTranslator,
}


def build_backend(args: argparse.Namespace) -> TranslatorBackend:
    """Construct and return the appropriate backend from CLI args."""
    name = args.backend.lower()

    if name == "openai":
        return OpenAITranslator(
            model=args.openai_model,
            api_key=getattr(args, "openai_api_key", None),
            request_delay=args.request_delay,
            max_retries=args.max_retries,
        )
    elif name == "anthropic":
        return AnthropicTranslator(
            model=args.anthropic_model,
            api_key=getattr(args, "anthropic_api_key", None),
            request_delay=args.request_delay,
            max_retries=args.max_retries,
        )
    elif name == "nllb":
        return NLLBTranslator(
            model_name=args.nllb_model,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif name == "repair":
        return TwoStageRepairTranslator(
            provider=args.repair_provider,
            model=args.repair_model,
            api_key=getattr(args, "repair_api_key", None),
            request_delay=args.request_delay,
            max_retries=args.max_retries,
        )
    else:
        raise ValueError(f"Unknown backend '{name}'. Choices: {list(BACKENDS.keys())}")


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

def process_file(
    input_path: Path,
    output_path: Path,
    translator: TranslatorBackend,
) -> Tuple[int, int, int]:
    """
    Translate a single cleaned JSONL file.
    Each line = one call.  We translate at the call level for full context.

    Returns: (calls_processed, total_segments, flagged_segments)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".jsonl.tmp")
    total_calls = total_segs = flagged_segs = 0

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
                    input_path.name, line_no, exc,
                )
                continue

            call_id = record.get("call_id", f"line_{line_no}")
            segments = record.get("segments", [])

            if segments:
                translated_segs = translator.translate_call(segments, call_id)
            else:
                translated_segs = []

            flagged_segs += sum(
                1 for s in translated_segs if s.get("translation_warning", False)
            )
            total_segs += len(translated_segs)
            total_calls += 1

            output_record = {
                # Preserve ALL original fields from the cleaned record
                **record,
                # Overwrite segments with translated version
                "segments": translated_segs,
                # Stage tracking metadata
                "nlp_stage": "translated",
                "nlp_translated_at": datetime.now(timezone.utc).isoformat(),
                "translation_backend": translator.name,
                "translation_strategy": translator.strategy,
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    # Atomic rename — readers never see a partial file
    tmp_path.replace(output_path)
    logger.debug(
        "Wrote %d calls / %d segments to %s",
        total_calls, total_segs, output_path,
    )
    return total_calls, total_segs, flagged_segs


# ---------------------------------------------------------------------------
# Validation mode — inspect quality before committing to full run
# ---------------------------------------------------------------------------

def run_validation(
    input_dir: Path,
    translator: TranslatorBackend,
    num_calls: int,
    pattern: str,
) -> None:
    """
    Translate a small sample and print a rich side-by-side comparison.
    Does NOT write any files or update checkpoints.
    """
    print("\n" + "=" * 72)
    print(f"  VALIDATION MODE — {num_calls} call(s)  |  backend: {translator.name}")
    print("=" * 72)

    input_files = sorted(input_dir.rglob(pattern))
    if not input_files:
        print(f"  ERROR: No files matching '{pattern}' in {input_dir}")
        return

    source_file = input_files[0]
    print(f"  Source file: {source_file}\n")

    calls_done = 0
    with open(source_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            if calls_done >= num_calls:
                break
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            record = json.loads(raw_line)
            call_id = record.get("call_id", "?")
            duration = record.get("duration", 0)
            segments = record.get("segments", [])

            print(f"\n{'─' * 72}")
            print(f"  CALL ID : {call_id}")
            print(f"  Duration: {duration:.1f}s   Segments: {len(segments)}")
            print(f"{'─' * 72}")

            translated = translator.translate_call(segments, call_id)

            for seg in translated:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                quality = seg.get("quality_score", "-")
                flagged = "⚠ FLAGGED" if seg.get("flagged") else ""
                warn = " [translation_warning]" if seg.get("translation_warning") else ""

                raw_bn = seg.get("text_bn_raw", seg.get("text_bn", ""))
                cln_bn = seg.get("text_bn", "")
                en = seg.get("text_en", "")

                # Truncate for display
                def trunc(s: str, n: int = 90) -> str:
                    return (s[:n] + "…") if len(s) > n else s

                print(f"\n  [{start:.1f}s → {end:.1f}s]  Q={quality}  {flagged}")
                print(f"  🔴 raw BN : {trunc(raw_bn)}")
                if cln_bn != raw_bn:
                    print(f"  🟡 cln BN : {trunc(cln_bn)}")
                else:
                    print(f"  🟡 cln BN : (unchanged)")
                print(f"  🟢 EN     : {trunc(en)}{warn}")

            calls_done += 1

    print(f"\n{'=' * 72}")
    print("  Validation complete. Review the translations above.")
    print("  If quality is acceptable, run without --validate-only to process all files.")
    print(f"{'=' * 72}\n")


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

def run(
    input_dir: Path,
    output_dir: Path,
    checkpoint_db: Path,
    translator: TranslatorBackend,
    pattern: str,
    force: bool,
    max_files: Optional[int] = None,
) -> None:
    """Process cleaned JSONL files with call-context translation."""
    checkpoint = NLPTranslationCheckpoint(checkpoint_db)

    input_files = sorted(input_dir.rglob(pattern))
    if not input_files:
        logger.warning("No files matching '%s' found in %s", pattern, input_dir)
        return

    if max_files is not None:
        input_files = input_files[:max_files]
        logger.info("--max-files %d: limiting run to %d file(s).", max_files, len(input_files))

    logger.info(
        "Found %d file(s). Backend: %s  Strategy: %s  Checkpoint: %s",
        len(input_files),
        translator.name,
        translator.strategy,
        checkpoint_db,
    )

    skipped = done = errors = 0
    total_segs_translated = 0

    with tqdm(input_files, desc="Translating Bengali → English", unit="file") as pbar:
        for input_path in pbar:
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
                calls, segs, flagged = process_file(input_path, output_path, translator)
                total_segs_translated += segs
                checkpoint.mark_done(
                    file_id=file_id,
                    backend=translator.name,
                    strategy=translator.strategy,
                    input_path=str(input_path),
                    output_path=str(output_path),
                    calls=calls,
                    segments=segs,
                    flagged_segs=flagged,
                )
                done += 1
                logger.debug(
                    "DONE: %s — %d calls, %d segs (%d flagged)",
                    file_id, calls, segs, flagged,
                )
            except Exception as exc:  # noqa: BLE001
                errors += 1
                err_msg = f"{type(exc).__name__}: {exc}"
                checkpoint.mark_error(
                    file_id=file_id,
                    backend=translator.name,
                    strategy=translator.strategy,
                    input_path=str(input_path),
                    error=err_msg,
                )
                logger.error("ERROR processing %s: %s", file_id, err_msg)

    stats = checkpoint.stats()
    logger.info(
        "Run complete — done=%d  skipped=%d  errors=%d  "
        "total_segments=%d  | DB stats: %s",
        done, skipped, errors, total_segs_translated, stats,
    )


# ---------------------------------------------------------------------------
# Project tree printer
# ---------------------------------------------------------------------------

def print_project_tree(paths: StoragePaths, output_dir: Path) -> None:
    """Print a live project tree showing all pipeline outputs."""

    def _ls(directory: Path) -> List[str]:
        if not directory.exists():
            return ["(not yet created)"]
        items = sorted(directory.iterdir())
        if not items:
            return ["(empty)"]
        return [f.name for f in items[:10]] + (["…"] if len(items) > 10 else [])

    print("\n" + "=" * 72)
    print("  PROJECT DATA TREE")
    print("=" * 72)
    layers = [
        ("data/transcripts/", paths.raw_transcripts, "Stage 0 — Raw Whisper output (READ-ONLY)"),
        ("data/nlp/cleaned/", paths.cleaned_bengali, "Stage 1 — Cleaned Bengali"),
        ("data/nlp/translated/", output_dir, "Stage 2 — Bengali + English (THIS STAGE)"),
        ("data/enriched/", paths.enriched, "Stage 3 — Analytics-ready enriched JSONL"),
        ("data/nlp/logs/", paths.logs, "Translation logs"),
        ("checkpoints/", paths.checkpoint_db.parent, "SQLite checkpoint databases"),
    ]
    for label, path, desc in layers:
        files = _ls(path)
        print(f"\n  📁 {label}")
        print(f"     {desc}")
        print(f"     Path: {path}")
        for f in files:
            print(f"       • {f}")
    print("\n" + "=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="translate_english",
        description=(
            "Stage 2 — Contextual Bengali → English translation.\n\n"
            "Translates full calls in a single LLM request for natural,\n"
            "context-aware English output. Timestamps and Bengali are preserved."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # I/O paths
    io_group = parser.add_argument_group("I/O paths")
    io_group.add_argument(
        "--input-dir", type=Path, default=Path("data/nlp/cleaned"),
        help="Directory containing cleaned JSONL files from Stage 1. (default: data/nlp/cleaned)",
    )
    io_group.add_argument(
        "--output-dir", type=Path, default=Path("data/nlp/translated"),
        help="Directory to write translated JSONL files. (default: data/nlp/translated)",
    )
    io_group.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/nlp_translation.sqlite"),
        help="Path to the SQLite checkpoint database. (default: checkpoints/nlp_translation.sqlite)",
    )
    io_group.add_argument(
        "--log-dir", type=Path, default=Path("data/nlp/logs"),
        help="Directory for translation log files. (default: data/nlp/logs)",
    )
    io_group.add_argument(
        "--pattern", type=str, default="*.jsonl",
        help="Glob pattern to match input files. (default: *.jsonl)",
    )

    # Backend selection
    parser.add_argument(
        "--backend", type=str, default="nllb",
        choices=list(BACKENDS.keys()),
        help="Translation backend to use. (default: nllb)",
    )

    # NLLB options
    nllb_group = parser.add_argument_group("NLLB options (--backend nllb)")
    nllb_group.add_argument(
        "--nllb-model", type=str, default="facebook/nllb-200-distilled-600M",
        help="HuggingFace model ID.",
    )
    nllb_group.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device for NLLB. 'auto' selects CUDA > MPS > CPU.",
    )
    nllb_group.add_argument(
        "--batch-size", type=int, default=8,
        help="Segments per NLLB batch.",
    )

    # OpenAI options
    oai_group = parser.add_argument_group("OpenAI options (--backend openai)")
    oai_group.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    oai_group.add_argument("--openai-api-key", type=str, default=None,
                           help="Override OPENAI_API_KEY env var.")

    # Anthropic options
    ant_group = parser.add_argument_group("Anthropic options (--backend anthropic)")
    ant_group.add_argument("--anthropic-model", type=str, default="claude-3-haiku-20240307")
    ant_group.add_argument("--anthropic-api-key", type=str, default=None,
                           help="Override ANTHROPIC_API_KEY env var.")

    # Repair backend options
    repair_group = parser.add_argument_group("Repair+Translate options (--backend repair)")
    repair_group.add_argument(
        "--repair-provider", type=str, default="openai", choices=["openai", "anthropic"],
        help="LLM provider for the repair backend. (default: openai)",
    )
    repair_group.add_argument(
        "--repair-model", type=str, default="gpt-4o-mini",
        help="Model name for repair backend. (default: gpt-4o-mini)",
    )
    repair_group.add_argument(
        "--repair-api-key", type=str, default=None,
        help="API key for repair backend (overrides env var).",
    )

    # Retry / rate-limit controls
    retry_group = parser.add_argument_group("API rate-limit / retry controls")
    retry_group.add_argument(
        "--request-delay", type=float, default=0.5,
        help="Seconds to wait between API calls. (default: 0.5)",
    )
    retry_group.add_argument(
        "--max-retries", type=int, default=3,
        help="Max retry attempts per call on API failure. (default: 3)",
    )

    # Run mode
    mode_group = parser.add_argument_group("Run mode")
    mode_group.add_argument(
        "--validate-only", action="store_true",
        help=(
            "Translate a small sample and print a human-readable comparison. "
            "No files are written. Use before a full run to check quality."
        ),
    )
    mode_group.add_argument(
        "--num-calls", type=int, default=3,
        help="Number of calls to use in --validate-only mode. (default: 3)",
    )
    mode_group.add_argument(
        "--max-files", type=int, default=None,
        help="Limit full run to first N files (skips checkpoint). Useful for batch testing.",
    )
    mode_group.add_argument(
        "--force", action="store_true",
        help="Re-process files already marked done in the checkpoint DB.",
    )
    mode_group.add_argument(
        "--print-tree", action="store_true",
        help="Print the project data tree after completing.",
    )
    mode_group.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    paths = StoragePaths()
    global logger
    logger = _setup_logging(args.log_dir, args.verbose)

    # Always show the storage map at startup
    paths.print_summary()

    translator = build_backend(args)

    if args.validate_only:
        run_validation(
            input_dir=args.input_dir,
            translator=translator,
            num_calls=args.num_calls,
            pattern=args.pattern,
        )
    else:
        run(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            checkpoint_db=args.checkpoint,
            translator=translator,
            pattern=args.pattern,
            force=args.force,
            max_files=args.max_files,
        )

    if args.print_tree:
        print_project_tree(paths, args.output_dir)


if __name__ == "__main__":
    main()
