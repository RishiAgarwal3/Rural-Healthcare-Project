#!/usr/bin/env python3
"""
=============================================================================
  FinAI Voice Analytics — Improved Extraction Pipeline
=============================================================================
  Reads the existing enriched JSONL and applies a hybrid extraction engine
  to aggressively reduce "reply_too_vague_to_classify" values across all key clinical fields.

  Strategy:
    1. Exact regex patterns            → high confidence (0.85–1.0)
    2. rapidfuzz fuzzy phrase matching → medium confidence (0.70–0.85)
    3. Contextual / fallback inference → low confidence (0.55–0.70)
    4. If confidence < threshold       → keep unknown, set needs_manual_review

  Output:
    data/processed_calls_improved.csv
    data/processed_calls_improved.parquet  (optional)

  Usage:
    .venv/bin/python scripts/improve_extraction.py
    .venv/bin/python scripts/improve_extraction.py --no-parquet
    .venv/bin/python scripts/improve_extraction.py --debug

=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Optional rapidfuzz import — graceful fallback if not installed
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz, process as rf_process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parents[1]
ENRICHED_JSONL = PROJECT_ROOT / "data" / "enriched" / "calls.jsonl"
RULES_YAML     = PROJECT_ROOT / "config" / "extraction_rules.yaml"
OUT_CSV        = PROJECT_ROOT / "data" / "processed_calls_improved.csv"
OUT_PARQUET    = PROJECT_ROOT / "data" / "processed_calls_improved.parquet"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("improve_extraction")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UNKNOWN_SENTINEL = {"reply_too_vague_to_classify", "null", "", None}


# =============================================================================
# RULES LOADER
# =============================================================================

def load_rules(path: Path) -> Dict[str, Any]:
    """Load and parse the YAML extraction rules file."""
    with open(path, "r", encoding="utf-8") as f:
        rules = yaml.safe_load(f)
    log.info(f"Loaded extraction rules from {path.name}")
    return rules


# =============================================================================
# CORE EXTRACTION PRIMITIVES
# =============================================================================

def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation noise."""
    text = text.lower()
    text = re.sub(r"['\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def regex_match(text: str, patterns: List[str]) -> Optional[float]:
    """
    Try all patterns as case-insensitive regex against normalised text.
    Returns confidence 0.90 if matched, else None.
    """
    t = _normalise(text)
    for pat in patterns:
        try:
            if re.search(pat, t, re.IGNORECASE):
                return 0.90
        except re.error:
            # Fall back to simple substring search for malformed patterns
            if pat.lower() in t:
                return 0.88
    return None


def negation_guard(text: str, guards: List[str]) -> bool:
    """
    Returns True if any negation phrase is found in the text, which should
    cancel a positive match.
    """
    t = _normalise(text)
    for g in guards:
        try:
            if re.search(g, t, re.IGNORECASE):
                return True
        except re.error:
            if g.lower() in t:
                return True
    return False


def fuzzy_match(
    text: str,
    phrases: List[str],
    threshold: int = 82,
) -> Optional[Tuple[str, float]]:
    """
    Use rapidfuzz partial_ratio and token_sort_ratio to find the best match.
    Returns (best_phrase, normalised_confidence) or None.
    """
    if not HAS_RAPIDFUZZ or not phrases:
        return None
    t = _normalise(text)
    best_score = 0
    best_phrase = None
    for phrase in phrases:
        p = _normalise(phrase)
        score = max(
            fuzz.partial_ratio(p, t),
            fuzz.token_sort_ratio(p, t),
        )
        if score > best_score:
            best_score = score
            best_phrase = phrase
    if best_score >= threshold and best_phrase is not None:
        # Normalise rapidfuzz 0-100 score to 0.70-0.85 confidence range
        conf = 0.70 + (best_score - threshold) / (100 - threshold) * 0.15
        return best_phrase, round(min(conf, 0.85), 3)
    return None


def keyword_in_text(text: str, keyword_list: List[str]) -> List[str]:
    """
    Return all keywords from keyword_list that appear verbatim in text.
    Case-insensitive substring search.
    """
    t = _normalise(text)
    found = []
    for kw in keyword_list:
        if kw.lower() in t:
            found.append(kw)
    return found


# =============================================================================
# FIELD EXTRACTORS
# =============================================================================

def extract_recovery_status(
    text: str, rules: Dict, existing: str, conv_type: str
) -> Tuple[str, float]:
    """
    Extract recovery_status from transcript text.
    Returns (value, confidence).
    Only replaces existing if existing is in UNKNOWN_SENTINEL.
    """
    if existing not in UNKNOWN_SENTINEL:
        return existing, 0.90  # preserve confident existing value

    rec_rules = rules.get("recovery_status", {})
    candidates = ["fully_recovered", "worsening", "unresolved", "improving"]

    t = _normalise(text)

    for label in candidates:
        label_rules = rec_rules.get(label, {})
        patterns  = label_rules.get("patterns", [])
        fuzzy_phr = label_rules.get("fuzzy_phrases", [])
        neg_guard = label_rules.get("negation_guard", [])
        base_conf = float(label_rules.get("confidence", 0.85))

        # Step 1: Regex
        conf = regex_match(text, patterns)
        if conf is not None:
            if not negation_guard(text, neg_guard):
                return label, round(base_conf, 3)

        # Step 2: Fuzzy
        result = fuzzy_match(text, fuzzy_phr)
        if result is not None:
            _, fconf = result
            if not negation_guard(text, neg_guard):
                return label, round(fconf, 3)

    # Step 3: Contextual inference
    ctx = rec_rules.get("context_inference", {})
    # If conv_type is a pure outreach/unreachable type, unknown is correct
    unknown_types = ctx.get("unknown_if_informational_outreach", {}).get(
        "applies_to_conv_types", []
    )
    if conv_type in unknown_types:
        return "reply_too_vague_to_classify", 0.0

    # If no signal found: return unknown
    return "reply_too_vague_to_classify", 0.0


def extract_patient_compliance(
    text: str, rules: Dict, existing: str
) -> Tuple[str, float]:
    """Extract patient_compliance."""
    if existing not in UNKNOWN_SENTINEL:
        return existing, 0.90

    comp_rules = rules.get("patient_compliance", {})
    candidates = ["compliant", "non_compliant", "partial"]

    for label in candidates:
        label_rules = comp_rules.get(label, {})
        patterns  = label_rules.get("patterns", [])
        fuzzy_phr = label_rules.get("fuzzy_phrases", [])
        neg_guard = label_rules.get("negation_guard", [])
        base_conf = float(label_rules.get("confidence", 0.80))

        conf = regex_match(text, patterns)
        if conf is not None:
            if not negation_guard(text, neg_guard):
                return label, round(base_conf, 3)

        result = fuzzy_match(text, fuzzy_phr)
        if result is not None:
            _, fconf = result
            if not negation_guard(text, neg_guard):
                return label, round(fconf, 3)

    return "reply_too_vague_to_classify", 0.0


def extract_call_outcome(
    text: str, rules: Dict, existing: str, recovery_status: str
) -> Tuple[str, float]:
    """Extract call_outcome.  Priority order matters — more specific first."""
    if existing not in UNKNOWN_SENTINEL:
        return existing, 0.85

    outcome_rules = rules.get("call_outcome", {})
    # Priority order: most specific → least specific
    priority = [
        "patient_unreachable",
        "escalation_recommended",
        "referred_elsewhere",
        "follow_up_scheduled",
        "advised_clinic_visit",
        "medication_continued",
        "medication_stopped",
        "no_action_needed",
        "informational",
    ]

    for label in priority:
        label_rules = outcome_rules.get(label, {})
        patterns  = label_rules.get("patterns", [])
        fuzzy_phr = label_rules.get("fuzzy_phrases", [])
        base_conf = float(label_rules.get("confidence", 0.75))

        conf = regex_match(text, patterns)
        if conf is not None:
            return label, round(base_conf, 3)

        result = fuzzy_match(text, fuzzy_phr)
        if result is not None:
            _, fconf = result
            return label, round(fconf, 3)

    # Contextual fallback: if patient is fully recovered → no_action_needed
    if recovery_status == "fully_recovered":
        return "no_action_needed", 0.65

    return "informational", 0.55


def extract_tests_recommended(
    text: str, rules: Dict, existing: List[str]
) -> Tuple[List[str], float]:
    """Extract tests_recommended list."""
    # Only extend/fill if existing is empty
    if existing and any(
        t not in ["", "reply_too_vague_to_classify"] for t in existing
    ):
        return existing, 0.85

    test_rules = rules.get("tests_recommended", {})
    general_kws = test_rules.get("general", [])
    neg_guard   = test_rules.get("negation_guard", [])

    t_norm = _normalise(text)
    found = []

    for kw in general_kws:
        try:
            if re.search(r"\b" + re.escape(kw) + r"\b", t_norm, re.IGNORECASE):
                # Check negation in context window around the keyword
                idx = t_norm.find(kw.lower())
                window = t_norm[max(0, idx - 50): idx + 80]
                if not negation_guard(window, neg_guard):
                    found.append(kw)
        except re.error:
            if kw.lower() in t_norm:
                found.append(kw)

    if found:
        return found, 0.80
    return [], 0.0


def extract_advice_given(
    text: str, rules: Dict, existing: List[str]
) -> Tuple[List[str], float]:
    """Extract advice_given list."""
    if existing and any(a not in ["", "reply_too_vague_to_classify"] for a in existing):
        return existing, 0.85

    advice_rules = rules.get("advice_given", {})
    t_norm = _normalise(text)
    found = []

    for advice_label, kw_list in advice_rules.items():
        for kw in kw_list:
            try:
                if re.search(r"\b" + re.escape(kw.lower()) + r"\b", t_norm, re.IGNORECASE):
                    found.append(advice_label)
                    break  # one keyword per label is sufficient
            except re.error:
                if kw.lower() in t_norm:
                    found.append(advice_label)
                    break

    if found:
        return list(set(found)), 0.80
    return [], 0.0


def extract_conditions(
    text: str, rules: Dict, existing_screened: List[str], existing_reported: List[str]
) -> Tuple[List[str], List[str], float]:
    """
    Extract conditions_screened and conditions_reported.
    Heuristic: screening questions from clinic → screened.
    Patient confirms condition → reported.
    """
    cond_rules = rules.get("conditions", {})
    t_norm = _normalise(text)

    found_conditions = []
    for cond_label, aliases in cond_rules.items():
        for alias in aliases:
            if alias.lower() in t_norm:
                found_conditions.append(cond_label)
                break

    new_screened = list(existing_screened) if existing_screened else []
    new_reported = list(existing_reported) if existing_reported else []

    for cond in found_conditions:
        aliases = cond_rules.get(cond, [])
        # Build an alternation group safe for use inside regex
        alias_group = "|".join(re.escape(a) for a in aliases)

        # --- Check if it looks like a screening question ---
        # Patterns use direct string concat to avoid .format() issues with {0,40}
        is_screened = False
        screened_pats = [
            r"do you have.{0,40}(" + alias_group + r")",
            r"any.{0,20}(" + alias_group + r").{0,20}problem",
            r"checked?.{0,40}(" + alias_group + r")",
            r"issues?.{0,10}like.{0,60}",
        ]
        for pat in screened_pats:
            try:
                if re.search(pat, t_norm, re.IGNORECASE):
                    is_screened = True
                    break
            except re.error:
                pass

        # --- Check if patient confirmed the condition ---
        is_reported = False
        reported_pats = [
            r"(yes.{0,20}|i have.{0,20}|he has.{0,20}|she has.{0,20})(" + alias_group + r")",
            r"(" + alias_group + r").{0,30}(problem|issue|hurts|pain)",
        ]
        for pat in reported_pats:
            try:
                if re.search(pat, t_norm, re.IGNORECASE):
                    is_reported = True
                    break
            except re.error:
                pass

        # Default: found anywhere → screened (conservative)
        if cond not in new_screened:
            new_screened.append(cond)
        if is_reported and cond not in new_reported:
            new_reported.append(cond)

    conf = 0.80 if found_conditions else 0.0
    return new_screened, new_reported, conf



def extract_medicines(
    text: str, rules: Dict, existing: List[str]
) -> Tuple[List[str], float]:
    """Extract medicines list."""
    # If already has specific medicines (not just 'unspecified_medicine'), keep
    meaningful = [m for m in existing if m not in {"", "unspecified_medicine", "reply_too_vague_to_classify"}]
    if meaningful:
        return existing, 0.85

    med_rules = rules.get("medicines", {})
    t_norm = _normalise(text)
    found = []

    for med_label, aliases in med_rules.items():
        for alias in aliases:
            if alias.lower() in t_norm:
                found.append(med_label)
                break

    # Fallback: if text mentions "medicine" generically without specific name
    if not found and re.search(r"\b(medicine|medication|tablet|drug|oshudh|dawai|dawa)\b", t_norm):
        found = ["unspecified_medicine"]

    if found:
        return found, 0.75
    return existing, 0.0


def extract_clinic_name(
    text: str, rules: Dict, existing: Optional[str], location_ref: Optional[str]
) -> Tuple[Optional[str], float]:
    """Extract clinic name if currently null/unknown."""
    if existing and existing not in UNKNOWN_SENTINEL:
        return existing, 0.90

    clinic_rules = rules.get("clinic_names", {})
    t_norm = _normalise(text)

    for canonical_name, aliases in clinic_rules.items():
        for alias in aliases:
            if alias.lower() in t_norm:
                return canonical_name, 0.88

    # Fallback: if location_ref matches, apply default
    default_by_loc = rules.get("default_clinic_by_location", {})
    if location_ref:
        for loc_key, clinic_val in default_by_loc.items():
            if loc_key.lower() in (location_ref or "").lower():
                return clinic_val, 0.65  # low confidence inference

    return existing, 0.0


def extract_conversation_type(
    text: str, rules: Dict, existing: str
) -> Tuple[str, float]:
    """Extract conversation_type if unknown."""
    if existing not in UNKNOWN_SENTINEL:
        return existing, 0.85

    ct_rules = rules.get("conversation_type", {})
    t_norm = _normalise(text)

    priority = [
        "unreachable", "test_follow_up", "specialist_referral",
        "chronic_condition_follow_up", "appointment_scheduling",
        "medication_inquiry", "recovery_check", "informational_outreach",
    ]
    for label in priority:
        patterns = ct_rules.get(label, [])
        for pat in patterns:
            if pat.lower() in t_norm:
                return label, 0.80

    return "routine_follow_up", 0.55  # safest default


def extract_urgency(
    text: str, rules: Dict, existing: str, recovery_status: str
) -> Tuple[str, float]:
    """Extract or refine urgency_level."""
    # Always re-check for high urgency even if existing is set
    urg_rules = rules.get("urgency", {})

    high_patterns = urg_rules.get("high", {}).get("patterns", [])
    if regex_match(text, high_patterns) is not None:
        return "high", 0.90

    medium_patterns = urg_rules.get("medium", {}).get("patterns", [])
    if existing == "low" and recovery_status in {"unresolved", "worsening"}:
        # Upgrade from low to medium if patient isn't recovering
        return "medium", 0.72

    if existing not in UNKNOWN_SENTINEL:
        return existing, 0.85

    if regex_match(text, medium_patterns) is not None:
        return "medium", 0.80

    return "low", 0.70  # safe default for routine calls


# =============================================================================
# RECORD PROCESSOR
# =============================================================================

def process_record(record: Dict[str, Any], rules: Dict) -> Dict[str, Any]:
    """
    Apply all improved extractors to a single enriched record.
    Modifies in place and returns the updated record.
    """
    # --- Get raw translation text ---
    text = record.get("text", {}).get("raw_translation", "") or ""
    if not text:
        # Try from segments
        segs = record.get("segments", [])
        text = " ".join(
            s.get("text_en", "") or "" for s in segs if s.get("text_en")
        )

    if not text.strip():
        return record  # Nothing to do without text

    # --- Shortcuts to nested dicts ---
    patient_status = record.setdefault("patient_status", {})
    clinical       = record.setdefault("clinical", {})
    call_meta      = record.setdefault("call_meta", {})
    clinic_info    = record.setdefault("clinic", {})
    quality        = record.setdefault("quality", {})
    conf_scores    = quality.setdefault("confidence_scores", {})

    # --- Extract each field ---

    # Conversation type (needed for recovery context)
    conv_type, ct_conf = extract_conversation_type(
        text, rules, call_meta.get("conversation_type", "reply_too_vague_to_classify")
    )
    call_meta["conversation_type"] = conv_type
    conf_scores["conversation_type"] = ct_conf

    # Recovery status
    rec_status, rec_conf = extract_recovery_status(
        text, rules,
        str(patient_status.get("recovery_status", "reply_too_vague_to_classify")),
        conv_type,
    )
    patient_status["recovery_status"] = rec_status
    conf_scores["recovery_status"] = rec_conf

    # Patient compliance
    comp, comp_conf = extract_patient_compliance(
        text, rules,
        str(patient_status.get("patient_compliance", "reply_too_vague_to_classify")),
    )
    patient_status["patient_compliance"] = comp
    conf_scores["patient_compliance"] = comp_conf

    # Call outcome
    outcome, out_conf = extract_call_outcome(
        text, rules,
        str(call_meta.get("call_outcome", "reply_too_vague_to_classify")),
        rec_status,
    )
    call_meta["call_outcome"] = outcome
    conf_scores["call_outcome"] = out_conf

    # Urgency
    urgency, urg_conf = extract_urgency(
        text, rules,
        str(call_meta.get("urgency_level", "reply_too_vague_to_classify")),
        rec_status,
    )
    call_meta["urgency_level"] = urgency
    conf_scores["urgency_level"] = urg_conf

    # Clinic name
    clinic_name, clinic_conf = extract_clinic_name(
        text, rules,
        clinic_info.get("name"),
        clinic_info.get("location_reference"),
    )
    clinic_info["name"] = clinic_name
    conf_scores["clinic_name"] = clinic_conf

    # Tests recommended
    existing_tests = clinical.get("tests_recommended", []) or []
    tests, test_conf = extract_tests_recommended(text, rules, existing_tests)
    clinical["tests_recommended"] = tests
    conf_scores["tests_recommended"] = test_conf

    # Advice given
    existing_advice = patient_status.get("advice_given", []) or []
    advice, adv_conf = extract_advice_given(text, rules, existing_advice)
    patient_status["advice_given"] = advice
    conf_scores["advice_given"] = adv_conf

    # Conditions
    existing_screened = clinical.get("conditions_screened", []) or []
    existing_reported = clinical.get("conditions_reported", []) or []
    new_screened, new_reported, cond_conf = extract_conditions(
        text, rules, existing_screened, existing_reported
    )
    clinical["conditions_screened"] = new_screened
    clinical["conditions_reported"] = new_reported
    conf_scores["conditions"] = cond_conf

    # Medicines
    existing_meds = clinical.get("medicines", []) or []
    meds, med_conf = extract_medicines(text, rules, existing_meds)
    clinical["medicines"] = meds
    conf_scores["medicines"] = med_conf

    # --- needs_manual_review: if any core field has low confidence ---
    min_thresh = rules.get("confidence_thresholds", {}).get("manual_review_below", 0.60)
    core_fields = ["recovery_status", "patient_compliance", "call_outcome"]
    low_conf_fields = [
        f for f in core_fields
        if conf_scores.get(f, 1.0) < min_thresh
        and conf_scores.get(f, 0.0) > 0.0  # exclude 0.0 (genuinely unknown)
    ]
    if low_conf_fields:
        quality["needs_manual_review"] = True
        existing_warnings = quality.get("extraction_warnings", []) or []
        for f in low_conf_fields:
            warn = f"LOW_CONF_{f.upper()}: {conf_scores[f]:.2f}"
            if warn not in existing_warnings:
                existing_warnings.append(warn)
        quality["extraction_warnings"] = existing_warnings
    # else: preserve existing manual review status

    return record


# =============================================================================
# FLATTEN RECORD → CSV ROW
# =============================================================================

def flatten_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested JSON record to a flat dict for CSV/DataFrame export."""
    patient_status = rec.get("patient_status", {})
    clinical       = rec.get("clinical", {})
    call_meta      = rec.get("call_meta", {})
    clinic_info    = rec.get("clinic", {})
    quality        = rec.get("quality", {})
    conf_scores    = quality.get("confidence_scores", {})

    def _join(lst: Any) -> str:
        if isinstance(lst, list):
            return "|".join(str(x) for x in lst if x)
        return str(lst) if lst is not None else ""

    return {
        "call_id":              rec.get("call_id", ""),
        "call_date":            rec.get("call_date"),
        "file_id":              rec.get("file_id", ""),
        "duration_seconds":     rec.get("duration_seconds", 0.0),
        "patient_name":         rec.get("patient", {}).get("name"),
        "clinic_name":          clinic_info.get("name"),
        "clinic_location":      clinic_info.get("location_reference"),
        "conversation_type":    call_meta.get("conversation_type", "reply_too_vague_to_classify"),
        "call_outcome":         call_meta.get("call_outcome", "reply_too_vague_to_classify"),
        "urgency_level":        call_meta.get("urgency_level", "low"),
        "follow_up_required":   call_meta.get("follow_up_required", False),
        "referral_needed":      call_meta.get("referral_needed", False),
        "appointment_status":   call_meta.get("appointment_status", "reply_too_vague_to_classify"),
        "clinic_closed_notice": call_meta.get("clinic_closed_notice", False),
        "family_health_mentions": patient_status.get("family_health_mentions", False),
        "recovery_status":      patient_status.get("recovery_status", "reply_too_vague_to_classify"),
        "patient_compliance":   patient_status.get("patient_compliance", "reply_too_vague_to_classify"),
        "doctor_specializations": _join(clinical.get("doctor_specializations", [])),
        "conditions_screened":  _join(clinical.get("conditions_screened", [])),
        "conditions_reported":  _join(clinical.get("conditions_reported", [])),
        "medicines":            _join(clinical.get("medicines", [])),
        "tests_recommended":    _join(clinical.get("tests_recommended", [])),
        "advice_given":         _join(patient_status.get("advice_given", [])),
        "needs_manual_review":  quality.get("needs_manual_review", False),
        "extraction_warnings":  _join(quality.get("extraction_warnings", [])),
        "conf_patient_name":    conf_scores.get("patient_name", 0.0),
        "conf_recovery_status": conf_scores.get("recovery_status", 0.0),
        "conf_compliance":      conf_scores.get("patient_compliance", 0.0),
        "conf_call_outcome":    conf_scores.get("call_outcome", 0.0),
        "segment_count":        rec.get("segment_count", 1),
        "flagged_segment_ratio": quality.get("flagged_segment_ratio", 0.0),
        "low_quality_segments": quality.get("low_quality_segments", 0),
        "structured_summary":   rec.get("text", {}).get("structured_summary", ""),
    }


# =============================================================================
# METRICS REPORTER
# =============================================================================

def compute_metrics(before_df: pd.DataFrame, after_df: pd.DataFrame) -> None:
    """Print extraction improvement statistics."""
    print("\n" + "=" * 70)
    print("  📊  EXTRACTION IMPROVEMENT METRICS")
    print("=" * 70)

    target_fields = [
        "recovery_status", "patient_compliance", "call_outcome",
        "advice_given", "tests_recommended", "conditions_reported",
        "clinic_name", "medicines",
    ]

    rows = []
    for field in target_fields:
        if field not in before_df.columns or field not in after_df.columns:
            continue
        def unknown_pct(s: pd.Series) -> float:
            return (
                s.isin(["reply_too_vague_to_classify", "", None]).sum()
                + s.isna().sum()
            ) / len(s) * 100

        bef = unknown_pct(before_df[field])
        aft = unknown_pct(after_df[field])
        delta = bef - aft
        rows.append({
            "Field": field,
            "Before (% unknown)": f"{bef:.1f}%",
            "After (% unknown)":  f"{aft:.1f}%",
            "Improvement":        f"▼ {delta:.1f}pp" if delta > 0 else f"— {delta:.1f}pp",
        })

    # Print table
    print(f"\n{'Field':<30} {'Before':>15} {'After':>14} {'Δ':>14}")
    print("-" * 75)
    for r in rows:
        print(
            f"{r['Field']:<30} "
            f"{r['Before (% unknown)']:>15} "
            f"{r['After (% unknown)']:>14} "
            f"{r['Improvement']:>14}"
        )

    # Manual review count
    review_count = after_df["needs_manual_review"].astype(str).str.lower().eq("true").sum()
    print(f"\n  🔍  Rows requiring manual review : {review_count} / {len(after_df)}")

    # Overall coverage
    total_unknowns_before = sum(
        (before_df[f].isin(["reply_too_vague_to_classify", "", None]) | before_df[f].isna()).sum()
        for f in target_fields if f in before_df.columns
    )
    total_unknowns_after = sum(
        (after_df[f].isin(["reply_too_vague_to_classify", "", None]) | after_df[f].isna()).sum()
        for f in target_fields if f in after_df.columns
    )
    total_possible = len(after_df) * len([f for f in target_fields if f in after_df.columns])
    coverage_before = 100 - total_unknowns_before / total_possible * 100
    coverage_after  = 100 - total_unknowns_after  / total_possible * 100

    print(f"\n  📈  Overall extraction coverage  : {coverage_before:.1f}% → {coverage_after:.1f}% "
          f"(+{coverage_after - coverage_before:.1f}pp)")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Improve extraction quality on enriched JSONL")
    p.add_argument("--input-jsonl", type=Path, default=ENRICHED_JSONL,
                   help="Path to enriched calls.jsonl")
    p.add_argument("--output-csv", type=Path, default=OUT_CSV,
                   help="Output CSV path")
    p.add_argument("--output-parquet", type=Path, default=OUT_PARQUET,
                   help="Output Parquet path")
    p.add_argument("--no-parquet", action="store_true",
                   help="Skip Parquet output")
    p.add_argument("--rules", type=Path, default=RULES_YAML,
                   help="Path to extraction_rules.yaml")
    p.add_argument("--debug", action="store_true",
                   help="Enable debug logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if not HAS_RAPIDFUZZ:
        log.warning(
            "rapidfuzz not installed — fuzzy matching disabled. "
            "Install with: pip install rapidfuzz"
        )

    # --- Load rules ---
    rules = load_rules(args.rules)
    min_conf = rules.get("confidence_thresholds", {}).get("min_for_assignment", 0.55)

    # --- Load existing flat CSV for before-metrics ---
    before_csv = PROJECT_ROOT / "data" / "enriched" / "enriched_flat.csv"
    if before_csv.exists():
        before_df = pd.read_csv(before_csv)
        log.info(f"Loaded before-snapshot: {len(before_df)} rows")
    else:
        before_df = None
        log.warning("No baseline enriched_flat.csv found — skipping before/after comparison")

    # --- Load enriched JSONL ---
    if not args.input_jsonl.exists():
        log.error(f"Input JSONL not found: {args.input_jsonl}")
        sys.exit(1)

    records = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log.warning(f"Skipping malformed line: {e}")

    log.info(f"Loaded {len(records)} records from {args.input_jsonl.name}")

    # --- Process records ---
    t0 = time.time()
    processed = []
    for rec in tqdm(records, desc="Improving extraction", unit="call"):
        processed.append(process_record(rec, rules))

    elapsed = time.time() - t0
    log.info(f"Processed {len(processed)} records in {elapsed:.1f}s "
             f"({len(processed)/elapsed:.0f} records/sec)")

    # --- Flatten to DataFrame ---
    rows = [flatten_record(rec) for rec in processed]
    after_df = pd.DataFrame(rows)
    log.info(f"Flattened to DataFrame: {after_df.shape}")

    # --- Save CSV ---
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    after_df.to_csv(args.output_csv, index=False)
    log.info(f"Saved CSV → {args.output_csv}")

    # --- Save Parquet ---
    if not args.no_parquet:
        try:
            after_df.to_parquet(args.output_parquet, index=False)
            log.info(f"Saved Parquet → {args.output_parquet}")
        except Exception as e:
            log.warning(f"Parquet export failed: {e}")

    # --- Metrics ---
    if before_df is not None:
        compute_metrics(before_df, after_df)

    print(f"\n  ✅  Done! Output → {args.output_csv}")


if __name__ == "__main__":
    main()
