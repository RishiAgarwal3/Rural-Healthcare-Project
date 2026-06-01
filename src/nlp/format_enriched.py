"""
Stage 3 — Healthcare NLP Extraction & Analytics Enrichment
============================================================
Reads translated JSONL from data/nlp/translated/
Writes fully-structured, analytics-ready JSONL to data/enriched/

Architecture:
  • Rule-based extractor  — handles ~80% of fields reliably (zero API cost)
  • LLM-assisted extractor — optional, handles weak fields (--llm-extraction)
  • Confidence scoring     — per-field reliability estimate (0.0–1.0)
  • Manual review flagging — uncertain calls marked needs_manual_review=True
  • SQLite checkpoint      — fully resumable

Output schema (one JSON object per JSONL line, one call per object):

{
  "call_id": "1767241522.10067",
  "call_date": "2026-01-01",
  ...
  "patient": { "name": "Amara Bibi", ... },
  "clinic":  { "name": "Maya Bhavan", ... },
  "call_meta": { "conversation_type": "routine_follow_up", ... },
  "clinical": { "doctor_specializations": [...], ... },
  "patient_status": { "recovery_status": "fully_recovered", ... },
  "quality": { "confidence_scores": {...}, "needs_manual_review": false, ... },
  "text": { "raw_translation": "...", "structured_summary": null },
  "segments": [...]
}

Usage:
    python -m src.nlp.format_enriched
    python -m src.nlp.format_enriched --llm-extraction --llm-backend openai
    python -m src.nlp.format_enriched --demo            # process 20 sample calls
    python -m src.nlp.format_enriched --export-csv      # also write flat CSV
    python -m src.nlp.format_enriched --force           # re-process all files
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sqlite3
import time
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
logger = logging.getLogger("nlp.format_enriched")

# ---------------------------------------------------------------------------
# Pydantic (optional, used for schema validation)
# ---------------------------------------------------------------------------
try:
    from pydantic import BaseModel, Field
    _PYDANTIC = True
except ImportError:
    _PYDANTIC = False
    logger.debug("Pydantic not installed — skipping schema validation.")

# ============================================================================
# PART 1 — TAXONOMIES & ENUMS  (exact spec from user)
# ============================================================================

# conversation_type — pick ONE primary type
CONVERSATION_TYPES = {
    "routine_follow_up",
    "medication_inquiry",
    "appointment_scheduling",
    "specialist_referral",
    "unreachable",
    "informational_outreach",
    "test_follow_up",
    "recovery_check",
    "chronic_condition_follow_up",
}

# call_outcome
CALL_OUTCOMES = {
    "no_action_needed",
    "informational",
    "follow_up_scheduled",
    "referred_elsewhere",
    "patient_unreachable",
    "advised_clinic_visit",
    "medication_continued",
    "medication_stopped",
    "escalation_recommended",
}

# recovery_status
RECOVERY_STATUSES = {
    "fully_recovered",
    "improving",
    "unresolved",
    "worsening",
    "not_discussed",
    "unknown",
}

# patient_compliance
COMPLIANCE_STATUSES = {
    "compliant",
    "partial",
    "non_compliant",
    "not_applicable",
    "unknown",
}

# urgency — HIGH only for severe/worsening/untreated issues
URGENCY_LEVELS = {"low", "medium", "high"}

# appointment_status
APPOINTMENT_STATUSES = {
    "scheduled",
    "invited",
    "refused",
    "not_required",
    "unknown",
}

# Canonical condition categories
CONDITIONS = {
    "blood_pressure", "diabetes", "thyroid", "eye", "dental", "heart",
    "iron_deficiency", "bone_degeneration", "kidney", "skin_condition",
    "respiratory", "stomach", "nerve", "anemia",
}

# Doctor specializations
DOCTOR_SPECIALIZATIONS = {
    "general", "eye", "dental", "heart", "orthopedic",
    "skin", "diabetology", "gynaecology",
}

# Advice types
ADVICE_TYPES = {
    "visit_clinic", "continue_medication", "consult_specialist", "get_tested",
    "diet_restriction", "exercise", "do_not_buy_medicine_outside", "do_not_ignore",
}

# ============================================================================
# PART 2 — RULE-BASED EXTRACTORS
# ============================================================================

# ---------------------------------------------------------------------------
# 2a. Patient Name Extractor
# ---------------------------------------------------------------------------

# Patterns: "speaking from X's house", "Am I speaking with X?", "speaking to someone at X's house"
_NAME_PATTERNS: List[re.Pattern] = [
    # "speaking from X's house" — highest confidence
    re.compile(
        r"speaking (?:from|to someone at|with someone from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'?s? house",
        re.IGNORECASE,
    ),
    # "Am I speaking with X?" — direct address
    re.compile(
        r"Am I speaking with\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\?",
        re.IGNORECASE,
    ),
    # "speaking from X's house" (simpler form)
    re.compile(
        r"from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'?s? house",
        re.IGNORECASE,
    ),
    # "I am Mabuda Paik." — self-identification
    re.compile(
        r"Yes,\s+I am\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\.",
        re.IGNORECASE,
    ),
    # "X had consulted / visited" — patient reference in third person
    re.compile(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:had consulted|had visited|visited our|consulted a doctor)",
        re.IGNORECASE,
    ),
]

# Words that are NOT names
_NAME_BLOCKLIST = {
    "Krishna", "Chandrapur", "Maya", "Bhavan", "Fridays", "December",
    "Chennai", "Hello", "Doctors", "Doctor", "Okay", "Please", "Thank",
    "Yes", "No", "Am", "Is", "He", "She", "Someone", "Anyone", "Everyone",
}


def extract_patient_name(text: str) -> Tuple[Optional[str], float]:
    """
    Extract patient name from translated call text using regex heuristics.
    Returns (name, confidence).
    """
    for pat in _NAME_PATTERNS:
        match = pat.search(text)
        if match:
            candidate = match.group(1).strip()
            parts = candidate.split()
            # Filter blocklist, pronouns, multi-word non-name fragments
            if any(p in _NAME_BLOCKLIST for p in parts):
                continue
            if any(p.lower() in ("someone", "anyone", "not", "from", "at") for p in parts):
                continue
            if len(parts) > 4 or len(candidate) < 3:
                continue
            # Names: each word starts with uppercase
            if not all(p[0].isupper() for p in parts if p):
                continue
            return candidate, 0.88
    return None, 0.0


# ---------------------------------------------------------------------------
# 2b. Clinic Info Extractor
# ---------------------------------------------------------------------------

_CLINIC_NAME_PATTERNS = [
    re.compile(r"Maya\s+Bhavan", re.IGNORECASE),
    re.compile(r"our clinic", re.IGNORECASE),
    re.compile(r"clinic near", re.IGNORECASE),
]

_LOCATION_PATTERNS = [
    re.compile(r"Krishna\s+Chandrapur\s+High\s+School", re.IGNORECASE),
    re.compile(r"near\s+([A-Z][A-Za-z\s]+High School)", re.IGNORECASE),
]

_CLOSED_ON_PATTERNS = [
    re.compile(r"closed on\s+(\w+days?)", re.IGNORECASE),
    re.compile(r"closed on\s+(Fridays?|Saturdays?|Sundays?)", re.IGNORECASE),
    re.compile(r"it is closed on\s+(\w+)", re.IGNORECASE),
    re.compile(r"skip\s+(\w+day)", re.IGNORECASE),
]


def extract_clinic_info(text: str) -> Dict[str, Any]:
    """Extract clinic name, location reference, and closure days."""
    clinic_name = None
    if re.search(r"Maya\s+Bhavan", text, re.IGNORECASE):
        clinic_name = "Maya Bhavan"

    location_ref = None
    for pat in _LOCATION_PATTERNS:
        m = pat.search(text)
        if m:
            location_ref = "Krishna Chandrapur High School"
            break

    closed_on: List[str] = []
    for pat in _CLOSED_ON_PATTERNS:
        for m in pat.finditer(text):
            day = m.group(1).strip().title()
            if day not in closed_on:
                closed_on.append(day)

    return {
        "name": clinic_name,
        "location_reference": location_ref,
        "closed_on": closed_on,
    }


# ---------------------------------------------------------------------------
# 2c. Doctor Specialization Extractor
# ---------------------------------------------------------------------------

_SPECIALIZATION_MAP: Dict[str, str] = {
    "eye specialist": "eye",
    "eye doctor": "eye",
    "eye": "eye",
    "dental specialist": "dental",
    "dental": "dental",
    "dentist": "dental",
    "heart specialist": "heart",
    "cardiologist": "heart",
    "general medicine": "general",
    "general doctor": "general",
    "bone": "orthopedic",
    "orthopedic": "orthopedic",
    "gynaecologist": "gynaecology",
    "gynecologist": "gynaecology",
    "diabetologist": "diabetology",
}


def extract_doctor_specializations(text: str) -> List[str]:
    """Extract all mentioned doctor specializations."""
    found: List[str] = []
    text_lower = text.lower()
    for phrase, canonical in _SPECIALIZATION_MAP.items():
        if phrase in text_lower and canonical not in found:
            found.append(canonical)
    # If no specialists mentioned but clinic is mentioned, assume general
    if not found and re.search(r"(clinic|doctor|checked|consult)", text, re.IGNORECASE):
        found.append("general")
    return found


# ---------------------------------------------------------------------------
# 2d. Conditions Extractor
# ---------------------------------------------------------------------------

_CONDITION_MAP: Dict[str, str] = {
    "blood pressure": "blood_pressure",
    "bp": "blood_pressure",
    "hypertension": "blood_pressure",
    "sugar": "diabetes",
    "diabetes": "diabetes",
    "diabetic": "diabetes",
    "thyroid": "thyroid",
    "eye": "eye",
    "eye problem": "eye",
    "eye issue": "eye",
    "eye drops": "eye",
    "dental": "dental",
    "teeth": "dental",
    "heart": "heart",
    "cardiac": "heart",
    "iron deficiency": "iron_deficiency",
    "anemia": "anemia",
    "bone degeneration": "bone_degeneration",
    "bone": "bone_degeneration",
    "kidney": "kidney",
    "skin": "skin",
    "skin issue": "skin",
    "skin problem": "skin",
    "breathing": "respiratory",
    "respiratory": "respiratory",
    "stomach": "stomach",
    "nerve": "nerve",
    "nerve pain": "nerve",
    "waist": "nerve",
    "back pain": "nerve",
}

# Conditions mentioned in a screening context ("Do you have X?")
_SCREENING_PATTERN = re.compile(
    r"(?:any|have|having|do you have|no)\s+"
    r"(blood pressure|sugar|thyroid|eye|dental|heart|kidney|stomach|breathing)[^\.]",
    re.IGNORECASE,
)

# Conditions actually reported by patient
_REPORTED_PATTERN = re.compile(
    r"(?:I have|he has|she has|pain|problem|issue|suffering from|diagnosed with)\s+"
    r"([a-z\s]+?)(?:\.|,|$)",
    re.IGNORECASE,
)


def extract_conditions(text: str) -> Tuple[List[str], List[str]]:
    """
    Returns (conditions_screened, conditions_reported).

    RULE: If transcript only asks about/mentions a condition, it goes to screened.
    RULE: If patient explicitly says they HAVE a condition, it goes to reported.
    RULE: Do NOT infer conditions from symptoms (e.g. 'pain' does NOT imply arthritis).
    """
    screened: List[str] = []
    reported: List[str] = []
    text_lower = text.lower()

    # Sugar → diabetes normalization
    if "sugar" in text_lower:
        _CONDITION_MAP["sugar"] = "diabetes"
    # Pressure → blood_pressure normalization
    if re.search(r"\bpressure\b", text_lower):
        _CONDITION_MAP["pressure"] = "blood_pressure"

    for phrase, canonical in _CONDITION_MAP.items():
        if phrase in text_lower:
            if canonical not in screened:
                screened.append(canonical)

    # Reported detection — stricter, explicit patient statements
    for pat in [
        re.compile(r"I have\s+([a-z\s,]+?)(?:\.|,|\band\b)", re.IGNORECASE),
        re.compile(r"(?:he|she) has\s+([a-z\s]+?)\s+(?:problem|issue|pain|condition)", re.IGNORECASE),
        re.compile(r"suffering from\s+([a-z\s]+?)(?:\.|,)", re.IGNORECASE),
        re.compile(r"diagnosed with\s+([a-z\s]+?)(?:\.|,)", re.IGNORECASE),
    ]:
        for m in pat.finditer(text):
            fragment = m.group(1).lower().strip()
            for phrase, canonical in _CONDITION_MAP.items():
                if phrase in fragment and canonical not in reported:
                    reported.append(canonical)

    # Remove from screened if it's actually reported (avoid double-counting)
    screened = [c for c in screened if c not in reported]
    return screened, reported


# ---------------------------------------------------------------------------
# 2d-b. Symptoms Extractor  (separate from conditions per extraction rules)
# ---------------------------------------------------------------------------

# Only extract symptoms EXPLICITLY STATED by patient, not asked in screening
_SYMPTOM_MAP: Dict[str, str] = {
    "pain in the fields":  "exertional_pain",
    "waist pain":          "waist_pain",
    "back pain":           "back_pain",
    "arm pain":            "limb_pain",
    "leg pain":            "limb_pain",
    "pain in her arms":    "limb_pain",
    "pain in her legs":    "limb_pain",
    "nerve pain":          "nerve_pain",
    "pain from the waist": "waist_pain",
    "breathing issue":     "breathing_issue",
    "breathing problem":   "breathing_issue",
    "breathing issues":    "breathing_issue",
    "swelling":            "swelling",
    "fever":               "fever",
    "stomach problem":     "stomach_complaint",
    "stomach issue":       "stomach_complaint",
    "skin issue":          "skin_complaint",
    "skin problem":        "skin_complaint",
    "eye problem":         "eye_complaint",
    "eye issue":           "eye_complaint",
    "dental issue":        "dental_complaint",
    "dental problem":      "dental_complaint",
    "iron deficiency":     "iron_deficiency",
    "pain when I work":    "exertional_pain",
    "pain": "pain",  # Generic — only extracted from patient-speech context
}

# Screening-question patterns to STRIP before symptom extraction
_SCREENING_QUESTION_RE = re.compile(
    r"(?:do you have|does he have|does she have|any|have you|is there|no)\s+"
    r"(?:blood pressure|sugar|thyroid|eye|dental|heart|kidney|stomach|breathing)"
    r"[^.?]*?(?:\?|$)",
    re.IGNORECASE,
)


def extract_symptoms(text: str) -> List[str]:
    """
    Extract ONLY explicitly reported symptoms.
    RULE: Screening questions are NOT symptoms.
    RULE: 'If only pain is mentioned, do NOT infer arthritis or any diagnosis.'
    RULE: Prefer under-extraction.
    """
    # Remove screening question fragments so they don't confuse symptom matching
    cleaned = _SCREENING_QUESTION_RE.sub("", text.lower())
    found: List[str] = []
    for phrase, canonical in _SYMPTOM_MAP.items():
        if phrase in cleaned and canonical not in found:
            found.append(canonical)
    return found


# ---------------------------------------------------------------------------

_RECOVERY_PATTERNS: List[Tuple[str, str, float]] = [
    # (regex_pattern, recovery_status, confidence)
    # ── Positive health confirmations (common in screening calls) ──
    (r"everyone\s+(?:is\s+)?(?:healthy|fine|well|okay)", "fully_recovered", 0.70),
    (r"(?:yes|yeah).*(?:everyone|all).*(?:healthy|fine|well|doing well)", "fully_recovered", 0.68),
    (r"all\s+(?:five|four|three|two|six|seven)?\s*members.*(?:are\s+)?(?:well|fine|healthy)", "fully_recovered", 0.70),
    (r"(?:he|she)\s+is\s+(?:well|fine|healthy|okay|alright)", "fully_recovered", 0.75),
    (r"(?:he|she)\s+is\s+doing\s+(?:well|fine|okay|good)", "fully_recovered", 0.78),
    (r"is\s+(?:he|she)\s+(?:well|okay|fine|healthy)\?.*(?:yes|yeah|ha|haan)", "fully_recovered", 0.72),
    (r"as\s+long\s+as\s+(?:he|she)\s+is\s+doing\s+well", "fully_recovered", 0.72),
    (r"doing\s+(?:well|fine|okay)\s+now", "fully_recovered", 0.80),
    (r"doing\s+moderately\s+well", "improving", 0.78),
    # ── Original patterns ──
    (r"fully recovered", "fully_recovered", 0.95),
    (r"completely (?:well|recovered|healed|cured)", "fully_recovered", 0.92),
    (r"completely resolved", "fully_recovered", 0.90),
    (r"no (?:further )?problems?", "fully_recovered", 0.80),
    (r"(?:cured|healed|better now)", "fully_recovered", 0.82),
    (r"(?:cured now|no issues)", "fully_recovered", 0.88),
    (r"doing (?:well|fine|okay)(?! now)", "fully_recovered", 0.72),
    (r"getting better|improving|somewhat fine|moderately well", "improving", 0.85),
    (r"doing (?:moderately|somewhat|a bit) (?:well|fine|okay)", "improving", 0.80),
    (r"improved a bit", "improving", 0.82),
    (r"has it improved", "improving", 0.65),
    (r"same condition|still (?:not |the same|persists?|ongoing|present)", "unresolved", 0.88),
    (r"(?:no|not) better|still there|still having", "unresolved", 0.85),
    (r"problem (?:returned|is back|recurred)", "unresolved", 0.88),
    (r"still (?:the )?problem", "unresolved", 0.80),
    (r"pain (?:is still|persists|in her)", "unresolved", 0.82),
    # worsening — explicit deterioration signals
    (r"(?:getting )?worse|deteriorating|day by day", "worsening", 0.90),
    (r"will get worse|ignore.*worse", "worsening", 0.85),
    # NOTE: 'stable' is NOT in the user schema — removed
]


def extract_recovery_status(text: str) -> Tuple[str, float]:
    """Determine patient recovery status from text patterns."""
    text_lower = text.lower()
    best_status = "unknown"
    best_conf = 0.0
    for pattern, status, conf in _RECOVERY_PATTERNS:
        if re.search(pattern, text_lower):
            if conf > best_conf:
                best_status = status
                best_conf = conf
                
    # Rule 3 enforcement: If the patient says the issue still exists, recovery_status cannot be "fully_recovered"
    if best_status == "fully_recovered" or best_status == "improving":
        for pattern, status, conf in _RECOVERY_PATTERNS:
            if status == "unresolved" and re.search(pattern, text_lower):
                best_status = "unresolved"
                best_conf = 0.99

    # Fallback: If no recovery pattern matched, check if this is a purely
    # informational/referral call where recovery was never discussed.
    if best_status == "unknown":
        # Check if the call even asked about health status
        health_inquiry = re.search(
            r"how (?:are|is) (?:you|he|she)|(?:are you|is he|is she) (?:doing|feeling|well|okay|fine)"
            r"|how.*doing|how.*feeling|how.*now|recovered|getting better"
            r"|everyone.*(?:healthy|well|fine|doing well)|is\s+(?:he|she)\s+okay",
            text_lower,
        )
        if not health_inquiry:
            best_status = "not_discussed"
            best_conf = 0.60

    return best_status, best_conf


# ---------------------------------------------------------------------------
# 2f. Patient Compliance Extractor
# ---------------------------------------------------------------------------

_COMPLIANCE_PATTERNS: List[Tuple[str, str, float]] = [
    (r"still taking (?:the )?medi(?:cine|cation)", "compliant", 0.92),
    (r"still on (?:the )?medi(?:cine|cation)", "compliant", 0.90),
    (r"taking (?:the )?medicine", "compliant", 0.85),
    (r"medication (?:is )?ongoing", "compliant", 0.88),
    # Additional compliance signals
    (r"(?:are you|is he|is she)\s+still\s+(?:taking|on|using)", "compliant", 0.80),
    (r"medicine\s+(?:is\s+)?(?:still\s+)?(?:ongoing|continuing|running)", "compliant", 0.85),
    (r"still\s+(?:using|on)\s+(?:the\s+)?(?:eye drops|drops|tablets?|medicine)", "compliant", 0.85),
    # Past-tense compliance: patient took/used medicine
    (r"took\s+(?:the\s+)?medicine\s+(?:for|and)", "compliant", 0.72),
    (r"(?:he|she|you|i)\s+took\s+medicine", "compliant", 0.70),
    (r"(?:and\s+)?took\s+medicine", "compliant", 0.68),
    (r"(?:we\s+)?tried\s+the\s+medicine", "compliant", 0.68),
    (r"visited\s+(?:the\s+|a\s+|our\s+)?doctor\s+(?:here\s+)?(?:for|and)\s+.*(?:took|given|prescribed)\s+medicine", "compliant", 0.72),
    (r"as\s+long\s+as\s+(?:he|she)\s+takes\s+(?:the\s+)?medicine", "compliant", 0.80),
    (r"(?:the\s+)?medicine\s+kept\s+it\s+under\s+control", "compliant", 0.75),
    (r"once\s+(?:he|she)\s+stops.*(?:problem|issue)\s+returns", "partial", 0.72),
    (r"(?:the\s+)?medicine\s+(?:didn'?t|did not)\s+(?:reduce|work|help)", "compliant", 0.65),
    (r"it\s+(?:didn'?t|did not)\s+work\s+(?:at all\s+)?for", "compliant", 0.62),
    (r"(?:got|gets)\s+a\s+little\s+better\s+and\s+then", "partial", 0.68),
    (r"for\s+(?:pain|an?\s+(?:issue|problem)).*took\s+medicine", "compliant", 0.70),
    (r"prescribed|gave\s+(?:dietary\s+)?restrictions", "compliant", 0.65),
    (r"(?:the\s+)?medicine.*under\s+control.*(?:increased|worse)\s+later", "partial", 0.72),
    (r"consulted\s+(?:someone\s+)?locally\s+and", "non_compliant", 0.68),
    # Explicit non-compliance
    (r"stopped (?:the )?medi(?:cine|cation|seeing the doctor)", "non_compliant", 0.92),
    (r"stopped seeing the doctor", "non_compliant", 0.85),
    (r"(?:no|not|haven't) (?:taking|taken|come|visited)", "non_compliant", 0.78),
    (r"didn't come|you (?:didn't|haven't) come", "non_compliant", 0.80),
    (r"medicine (?:is )?finished|finished the medicine", "non_compliant", 0.82),
    (r"(?:not\s+taking|no\s+longer\s+taking|stopped\s+taking)", "non_compliant", 0.88),
    (r"(?:no|not)\s+(?:medicine|medication|tablets?)(?:\s+now)?", "non_compliant", 0.75),
    (r"didn't\s+(?:take|buy|get)\s+(?:the\s+)?(?:medicine|medication)", "non_compliant", 0.82),
    (r"(?:you|he|she)\s+(?:didn't|haven't)\s+come\b", "non_compliant", 0.72),
    (r"you\s+should\s+(?:get\s+a\s+checkup|come)\s+(?:done\s+)?before\s+buying.*medicine\s+from\s+outside", "non_compliant", 0.80),
    (r"(?:medicines?\s+(?:is\s+|are\s+)?finished|are\s+the\s+medicines?\s+finished)", "non_compliant", 0.82),
    (r"consulted\s+(?:elsewhere|another|different|other)\s+(?:doctor)?", "non_compliant", 0.70),
    (r"we\s+consulted\s+(?:a\s+)?doctor\s+elsewhere", "non_compliant", 0.72),
    (r"for (?:a|one|two) (?:week|month)s?", "partial", 0.65),
]

# Patterns indicating the clinic is just advertising medicine availability
# (NOT an actual patient compliance discussion)
_CLINIC_OFFER_RE = re.compile(
    r"medicines?\s+(?:are\s+)?(?:available|provided|free)|"
    r"(?:we\s+)?provide\s+medicines?|"
    r"(?:good\s+)?doctors?\s+(?:are\s+)?available|"
    r"if\s+(?:he|she|you|anyone)\s+needs?\s+(?:the\s+same\s+)?medicine|"
    r"medicines?\s+and\s+prescriptions?\s+.*(?:here|provided)|"
    r"consultation\s+fee",
    re.IGNORECASE,
)

# Patterns indicating actual patient-level medicine discussion
_PATIENT_MEDICINE_RE = re.compile(
    r"(?:still\s+taking|took\s+medicine|stopped\s+the|medication\s+ongoing|"
    r"medicine\s+finished|are\s+the\s+medicines?\s+finished|"
    r"tried\s+the\s+medicine|medicine\s+didn|"
    r"(?:is\s+the\s+)?medication\s+still|how\s+many\s+days.*medicine|"
    r"(?:you|he|she|i)\s+took|(?:no\s+longer|stopped)\s+taking|"
    r"should\s+.*(?:come|checkup).*before\s+buying|"
    r"takes\s+(?:the\s+)?medicine|medicine\s+kept\s+it|"
    r"once\s+(?:he|she)\s+stops|under\s+control|"
    r"consulted\s+(?:someone\s+)?locally|consulted\s+(?:a\s+)?doctor\s+elsewhere|"
    r"(?:it\s+)?didn'?t\s+(?:work|reduce|help)|prescribed|"
    r"(?:for|and)\s+took\s+medicine|given\s+medicine)",
    re.IGNORECASE,
)


def extract_compliance(text: str) -> Tuple[str, float]:
    """Extract patient medication/appointment compliance."""
    text_lower = text.lower()
    for pattern, status, conf in _COMPLIANCE_PATTERNS:
        if re.search(pattern, text_lower):
            return status, conf

    # Fallback: distinguish between actual medicine discussion and clinic advertising
    has_patient_medicine = bool(_PATIENT_MEDICINE_RE.search(text))
    has_clinic_offer = bool(_CLINIC_OFFER_RE.search(text))

    # If medicine is only mentioned in a clinic-offer context, it's not applicable
    if has_clinic_offer and not has_patient_medicine:
        return "not_applicable", 0.60

    # Check if medication was discussed at all
    medicine_discussed = re.search(
        r"\b(?:medicine|medication|tablets?|prescription|dose|eye drops?|drops|pill|capsule)"
        r"|\b(?:taking|stopped|finished|ongoing|still on|still taking)"
        r"|\b(?:checkup|check-up|check up)\b",
        text_lower,
    )
    if not medicine_discussed:
        return "not_applicable", 0.65
    return "unknown", 0.0


# ---------------------------------------------------------------------------
# 2g. Conversation Type & Call Outcome Extractor
# ---------------------------------------------------------------------------

_CONV_PATTERNS: List[Tuple[str, str, float]] = [
    # Unreachable first — clear signal
    (r"(?:not at home|no one here|couldn't reach|no one available)", "unreachable", 0.95),
    # Medicine-focused calls
    (r"(?:how many days|still taking|stopped the medicine|medication|prescription|dose)", "medication_inquiry", 0.80),
    # Specialist referral
    (r"(?:specialist|referred|heart specialist|eye specialist|dental specialist)", "specialist_referral", 0.88),
    # Test follow-up
    (r"(?:blood test|x.?ray|report|test result)", "test_follow_up", 0.85),
    # Chronic condition follow-up
    (r"(?:sugar|blood pressure|thyroid|diabetes).*(?:check|control|managing|still)", "chronic_condition_follow_up", 0.78),
    # Appointment scheduling
    (r"(?:book|come (?:on|this|next) (?:Monday|Tuesday|Wednesday|Thursday|Saturday|Sunday|week)|when can you come)", "appointment_scheduling", 0.82),
    # Recovery check — generic follow-up with explicit recovery question
    (r"(?:how (?:are|is) (?:you|he|she) (?:doing|feeling|now)|fully recovered|recovered)", "recovery_check", 0.78),
    # Informational outreach — no specific patient issue
    (r"(?:doctors are available|you can visit|please inform your neighbors|general information)", "informational_outreach", 0.72),
    # Routine follow-up — default for clinic-initiated calls
    (r"(?:calling from the|I am calling from|Maya Bhavan|clinic near).*(?:visit|seen|consulted|checked)", "routine_follow_up", 0.70),
]

_OUTCOME_PATTERNS: List[Tuple[str, str, float]] = [
    # Patient unreachable
    (r"(?:not at home|no one here|couldn't reach)", "patient_unreachable", 0.95),
    # Explicitly referred
    (r"(?:referred|consult (?:a |the )?specialist|see (?:a |the )? \w+ specialist|consult a doctor there)", "referred_elsewhere", 0.88),
    # Escalation indicators
    (r"(?:worsening|getting worse|day by day|don't ignore|neglect|urgent)", "escalation_recommended", 0.85),
    # Medication stopped
    (r"(?:stopped (?:the )?medi(?:cine|cation)|medicine (?:is )?finished|stopped seeing the doctor)", "medication_stopped", 0.88),
    # Medication continued
    (r"(?:still taking|medication (?:is )?ongoing|continue (?:the )?medi)", "medication_continued", 0.88),
    # Follow-up scheduled
    (r"(?:I will|okay I will|yes I will|definitely) (?:come|visit)", "follow_up_scheduled", 0.85),
    (r"(?:please come|you should come|come (?:this|next) week|come back)", "advised_clinic_visit", 0.80),
    # No action needed
    (r"(?:fully recovered|no issues|no problems|all well|stay well)(?!.*please come)", "no_action_needed", 0.78),
    # Informational
    (r"(?:doctors are available|you can visit|please inform)", "informational", 0.70),
]


def extract_conversation_type(text: str) -> Tuple[str, float]:
    text_lower = text.lower()
    for pattern, conv_type, conf in _CONV_PATTERNS:
        if re.search(pattern, text_lower):
            return conv_type, conf
    return "routine_follow_up", 0.50  # safe default for this dataset


def extract_call_outcome(text: str) -> Tuple[str, float]:
    text_lower = text.lower()
    for pattern, outcome, conf in _OUTCOME_PATTERNS:
        if re.search(pattern, text_lower):
            return outcome, conf
    return "informational", 0.50


# ---------------------------------------------------------------------------
# 2h. Urgency Level Extractor
# ---------------------------------------------------------------------------

_URGENCY_PATTERNS: List[Tuple[str, str, float]] = [
    # HIGH — persistent untreated serious issue / worsening
    (r"(?:worsening|getting worse|day by day|deteriorat)", "high", 0.90),
    (r"(?:don't ignore|do not ignore|neglect).*(?:worse|serious|problem)", "high", 0.85),
    (r"persistent.*(?:pain|swelling|bleeding|fever)", "high", 0.82),
    (r"(?:severe|very bad|extreme|unbearable)", "high", 0.88),
    # MEDIUM — unresolved but not acute
    (r"(?:still (?:has|having)|not better|same condition|still the problem)", "medium", 0.78),
    (r"(?:please come|must come|need to come) this week", "medium", 0.70),
    (r"(?:don't ignore|do not ignore)", "medium", 0.72),
    (r"problem (?:persists|returned|is back)", "medium", 0.75),
]


def extract_urgency(text: str) -> Tuple[str, float]:
    text_lower = text.lower()
    for pattern, level, conf in _URGENCY_PATTERNS:
        if re.search(pattern, text_lower):
            return level, conf
    return "low", 0.85


# ---------------------------------------------------------------------------
# 2i. Follow-up, Appointment, Referral, Advice Extractors
# ---------------------------------------------------------------------------

_FOLLOWUP_PATTERNS = [
    r"please (?:come|visit|get (?:checked|tested))",
    r"you (?:should|must|need to) come",
    r"come (?:back|this|next|in a)",
    r"don't (?:neglect|ignore)",
    r"come (?:and|to) consult",
]

_REFERRAL_PATTERNS = [
    r"(?:sent|refer(?:red)?) to (?:a |the )?specialist",
    r"see (?:a |the )?specialist",
    r"consult (?:a |the )?\w+ specialist",
    r"(?:consult|see) (?:a )?doctor (?:there|elsewhere|in)",
]

_ADVICE_MAP: List[Tuple[str, str]] = [
    (r"(?:please )?(?:come|visit) (?:the )?clinic", "visit_clinic"),
    (r"(?:continue|keep taking|still take) (?:the )?medi", "continue_medication"),
    (r"consult (?:a )?specialist", "consult_specialist"),
    (r"(?:get|have) (?:a |the )?(?:blood test|test|check-?up|checked)", "get_tested"),
    (r"diet(?:ary)? restriction", "diet_restriction"),
    (r"exercise", "exercise"),
    (r"don't (?:buy|purchase) medicine (?:from )?outside", "do_not_buy_medicine_outside"),
    (r"don't (?:ignore|neglect)", "do_not_ignore"),
]


def extract_followup_required(text: str) -> Tuple[bool, float]:
    text_lower = text.lower()
    for pat in _FOLLOWUP_PATTERNS:
        if re.search(pat, text_lower):
            return True, 0.85
    return False, 0.75


def extract_referral_needed(text: str) -> Tuple[bool, float]:
    text_lower = text.lower()
    for pat in _REFERRAL_PATTERNS:
        if re.search(pat, text_lower, re.IGNORECASE):
            return True, 0.85
    return False, 0.80


def extract_advice_given(text: str) -> List[str]:
    advice: List[str] = []
    text_lower = text.lower()
    for pat, label in _ADVICE_MAP:
        if re.search(pat, text_lower) and label not in advice:
            advice.append(label)
    return advice


def extract_appointment_status(text: str) -> Tuple[str, float]:
    text_lower = text.lower()
    if re.search(r"(?:yes|okay|sure|definitely|I will) (?:come|visit|be there)", text_lower):
        return "scheduled", 0.85
    if re.search(r"(?:come|please visit|you should come|come (?:this|next) week)", text_lower):
        return "invited", 0.80
    if re.search(r"(?:not coming|can't come|refused|no need)", text_lower):
        return "refused", 0.75
    if re.search(r"(?:fully recovered|no issues|no problems|all well)", text_lower):
        return "not_required", 0.70
    return "unknown", 0.40


def extract_family_health_mentions(text: str) -> bool:
    return bool(re.search(
        r"(?:family|members?|everyone|rest of|all \w+ of|other members?)",
        text, re.IGNORECASE,
    ))


def extract_clinic_closed_notice(text: str) -> bool:
    return bool(re.search(
        r"closed on|skip (?:that )?day|not open on|no (?:clinic|doctors?) on",
        text, re.IGNORECASE,
    ))


# ---------------------------------------------------------------------------
# 2j. Medicines & Tests Extractor (weak — requires LLM for accuracy)
# ---------------------------------------------------------------------------

_MEDICINE_PATTERNS = [
    re.compile(r"\b(?:drops?|medicine|tablet|capsule|injection|insulin)\b", re.IGNORECASE),
    re.compile(r"eye drops", re.IGNORECASE),
    re.compile(r"(?:taking|prescribed|on)\s+\w+\s+(?:mg|ml|tablet)", re.IGNORECASE),
]

_TEST_PATTERNS = [
    re.compile(r"\b(?:blood test|X-ray|x ray|photo|scan|echo|ECG|urine test|test)\b", re.IGNORECASE),
]


def extract_medicines(text: str) -> List[str]:
    """Conservative medicine extraction — only returns generic types, not drug names."""
    found: List[str] = []
    if re.search(r"eye drops?", text, re.IGNORECASE):
        found.append("eye_drops")
    if re.search(r"\b(?:medicine|tablet|capsule)\b", text, re.IGNORECASE):
        found.append("unspecified_medicine")
    return found


def extract_tests_recommended(text: str) -> List[str]:
    tests: List[str] = []
    text_lower = text.lower()
    if re.search(r"blood test", text_lower):
        tests.append("blood_test")
    if re.search(r"x.?ray|photo", text_lower):
        tests.append("x_ray")
    if re.search(r"check.?up", text_lower):
        tests.append("check_up")
    return tests


# ---------------------------------------------------------------------------
# 2k. Rule-Based Structured Summary Generator
# ---------------------------------------------------------------------------

def _build_rule_based_summary(
    patient_name: Optional[str],
    recovery_status: str,
    compliance: str,
    conditions_reported: List[str],
    symptoms: List[str],
    advice_given: List[str],
    outcome: str,
    followup: bool,
) -> str:
    """
    Generate a concise professional CRM-style summary without LLM.
    Format: patient condition / current status / advice / next step.
    """
    parts: List[str] = []

    # Patient reference
    ref = patient_name if patient_name else "Patient"

    # Condition context
    if conditions_reported:
        conds = ", ".join(c.replace("_", " ") for c in conditions_reported)
        parts.append(f"{ref} has reported condition(s): {conds}.")
    elif symptoms:
        syms = ", ".join(s.replace("_", " ") for s in symptoms[:3])
        parts.append(f"{ref} reported symptom(s): {syms}.")
    else:
        parts.append(f"{ref} was previously seen at the clinic.")

    # Recovery status
    status_map = {
        "fully_recovered": "Patient reports full recovery with no remaining issues.",
        "improving":        "Patient is showing improvement but not fully recovered.",
        "unresolved":       "Patient's condition remains unresolved.",
        "worsening":        "Patient's condition is worsening and requires attention.",
        "not_discussed":    "Recovery was not discussed during this call.",
        "unknown":          "Current recovery status could not be determined from the transcript.",
    }
    parts.append(status_map.get(recovery_status, "Recovery status unknown."))

    # Compliance
    if compliance == "non_compliant":
        parts.append("Patient has stopped medication or missed follow-up visits.")
    elif compliance == "compliant":
        parts.append("Patient is continuing prescribed medication.")
    elif compliance == "partial":
        parts.append("Patient has taken medication partially or for a limited period.")
    elif compliance == "not_applicable":
        parts.append("Medication compliance was not applicable to this call.")

    # Advice
    advice_phrases = {
        "visit_clinic":              "advised to visit the clinic",
        "continue_medication":       "advised to continue medication",
        "consult_specialist":        "referred to a specialist",
        "get_tested":                "advised to get tested",
        "do_not_buy_medicine_outside": "advised not to buy medicine externally",
        "do_not_ignore":             "warned not to ignore the condition",
        "diet_restriction":          "advised dietary restrictions",
        "exercise":                  "advised exercise",
    }
    advice_strs = [advice_phrases[a] for a in advice_given if a in advice_phrases]
    if advice_strs:
        parts.append("Caller " + "; ".join(advice_strs) + ".")

    # Next step
    if outcome == "follow_up_scheduled":
        parts.append("Patient confirmed they will visit the clinic.")
    elif outcome == "advised_clinic_visit":
        parts.append("Patient was advised to visit the clinic for a follow-up.")
    elif outcome == "escalation_recommended":
        parts.append("Case may require urgent follow-up or escalation.")
    elif outcome == "patient_unreachable":
        parts.append("Patient was not reachable during this call.")
    elif outcome == "medication_stopped":
        parts.append("Patient has discontinued medication; follow-up recommended.")
    elif outcome == "referred_elsewhere":
        parts.append("Patient has been referred to another provider.")
    elif outcome == "no_action_needed":
        parts.append("No immediate action required.")

    return " ".join(parts)


# ============================================================================
# PART 3 — CONFIDENCE SCORING & QUALITY CONTROL
# ============================================================================

# Fields and their default reliability thresholds
# Below this confidence → trigger needs_manual_review
MANUAL_REVIEW_THRESHOLDS: Dict[str, float] = {
    "patient_name":     0.50,
    "recovery_status":  0.50,
    "patient_compliance": 0.40,
    "urgency_level":    0.40,
}

LOW_QUALITY_SEGMENT_THRESHOLD = 0.45  # quality_score below this = low quality
FLAGGED_RATIO_THRESHOLD = 0.4         # if >40% of segments flagged → manual review


def compute_needs_manual_review(
    confidence_scores: Dict[str, float],
    flagged_ratio: float,
    warnings: List[str],
) -> bool:
    """Determine if a call needs human review."""
    for field, threshold in MANUAL_REVIEW_THRESHOLDS.items():
        if confidence_scores.get(field, 1.0) < threshold:
            return True
    if flagged_ratio > FLAGGED_RATIO_THRESHOLD:
        return True
    if any("CRITICAL" in w or "EMPTY" in w for w in warnings):
        return True
    return False


def compute_extraction_warnings(
    text: str,
    patient_name: Optional[str],
    recovery_status: str,
    segments: List[Dict],
) -> List[str]:
    """Generate a list of extraction quality warnings."""
    warnings: List[str] = []
    if not patient_name:
        warnings.append("PATIENT_NAME_NOT_FOUND: could not identify patient name.")
    if recovery_status == "unknown":
        warnings.append("RECOVERY_STATUS_UNKNOWN: status not inferable from text.")
    if not text or len(text) < 30:
        warnings.append("EMPTY_OR_SHORT_TRANSLATION: text is too short for reliable extraction.")
    low_q = sum(1 for s in segments if (s.get("quality_score") or 1.0) < LOW_QUALITY_SEGMENT_THRESHOLD)
    if low_q > 0:
        warnings.append(f"LOW_QUALITY_SEGMENTS: {low_q} segment(s) with quality_score < {LOW_QUALITY_SEGMENT_THRESHOLD}.")
    return warnings


# ============================================================================
# PART 4 — FULL EXTRACTION ENGINE
# ============================================================================

def build_full_translation(segments: List[Dict]) -> str:
    """Concatenate all English segment text into one document for extraction."""
    parts = []
    for seg in segments:
        text = seg.get("text_en") or ""
        if text and text not in ("[LOW_QUALITY_AUDIO]", "[TRANSLATION_FAILED]"):
            parts.append(text.strip())
    return " ".join(parts)


def count_flagged_ratio(segments: List[Dict]) -> float:
    if not segments:
        return 0.0
    flagged = sum(1 for s in segments if s.get("flagged", False))
    return round(flagged / len(segments), 4)


def extract_all(
    call_id: str,
    segments: List[Dict],
    call_date: Optional[str],
) -> Dict[str, Any]:
    """
    Full extraction pipeline for one call.
    Runs all rule-based extractors and assembles the enriched record.
    Under-extracts rather than over-extracts — null/unknown preferred on uncertainty.
    """
    full_text = build_full_translation(segments)
    flagged_ratio = count_flagged_ratio(segments)

    # --- Patient ---
    patient_name, name_conf = extract_patient_name(full_text)

    # --- Clinic ---
    clinic_info = extract_clinic_info(full_text)

    # --- Clinical ---
    specializations = extract_doctor_specializations(full_text)
    conditions_screened, conditions_reported = extract_conditions(full_text)
    symptoms = extract_symptoms(full_text)
    medicines = extract_medicines(full_text)
    tests = extract_tests_recommended(full_text)

    # --- Patient Status ---
    recovery_status, recovery_conf = extract_recovery_status(full_text)
    compliance, compliance_conf = extract_compliance(full_text)
    family_health = extract_family_health_mentions(full_text)
    advice_given = extract_advice_given(full_text)

    # --- Call Meta ---
    conv_type, conv_conf = extract_conversation_type(full_text)
    outcome, outcome_conf = extract_call_outcome(full_text)
    urgency, urgency_conf = extract_urgency(full_text)
    followup, followup_conf = extract_followup_required(full_text)
    referral, referral_conf = extract_referral_needed(full_text)
    appt_status, appt_conf = extract_appointment_status(full_text)
    clinic_notice = extract_clinic_closed_notice(full_text)

    # --- Structured Summary (rule-based fallback) ---
    structured_summary = _build_rule_based_summary(
        patient_name=patient_name,
        recovery_status=recovery_status,
        compliance=compliance,
        conditions_reported=conditions_reported,
        symptoms=symptoms,
        advice_given=advice_given,
        outcome=outcome,
        followup=followup,
    )

    # --- Quality ---
    confidence_scores = {
        "patient_name":       round(name_conf, 3),
        "recovery_status":    round(recovery_conf, 3),
        "patient_compliance": round(compliance_conf, 3),
        "conversation_type":  round(conv_conf, 3),
        "call_outcome":       round(outcome_conf, 3),
        "urgency_level":      round(urgency_conf, 3),
        "follow_up_required": round(followup_conf, 3),
        "referral_needed":    round(referral_conf, 3),
        "appointment_status": round(appt_conf, 3),
        # Weak fields — rule-based cannot reliably extract these
        "symptoms":           0.55 if symptoms else 0.10,
        "medicines":          0.25 if medicines else 0.10,
        "tests_recommended":  0.60 if tests else 0.10,
    }

    warnings = compute_extraction_warnings(
        full_text, patient_name, recovery_status, segments,
    )
    needs_review = compute_needs_manual_review(
        confidence_scores, flagged_ratio, warnings,
    )
    low_q_count = sum(
        1 for s in segments
        if (s.get("quality_score") or 1.0) < LOW_QUALITY_SEGMENT_THRESHOLD
    )

    return {
        "patient": {
            "name": patient_name,
        },
        "clinic": clinic_info,
        "call_meta": {
            "conversation_type":   conv_type,
            "call_outcome":        outcome,
            "urgency_level":       urgency,
            "follow_up_required":  followup,
            "appointment_status":  appt_status,
            "clinic_closed_notice": clinic_notice,
            "referral_needed":     referral,
        },
        "clinical": {
            "doctor_specializations": specializations,
            "conditions_screened":    conditions_screened,
            "conditions_reported":    conditions_reported,
            "symptoms":               symptoms,
            "medicines":              medicines,
            "tests_recommended":      tests,
            "diagnoses_mentioned":    [],  # Only populated by LLM extractor
        },
        "patient_status": {
            "recovery_status":        recovery_status,
            "patient_compliance":     compliance,
            "family_health_mentions": family_health,
            "advice_given":           advice_given,
        },
        "quality": {
            "confidence_scores":     confidence_scores,
            "needs_manual_review":   needs_review,
            "extraction_warnings":   warnings,
            "low_quality_segments":  low_q_count,
            "flagged_segment_ratio": flagged_ratio,
        },
        "text": {
            "raw_translation":   full_text,
            "structured_summary": structured_summary,
        },
    }


# ============================================================================
# PART 5 — OPTIONAL LLM-ASSISTED EXTRACTION
# ============================================================================

LLM_EXTRACTION_PROMPT = """\
You are an expert healthcare conversation intelligence system.

You will receive translated rural healthcare follow-up call transcripts from clinics in India.

Your job is to convert each transcript into a STRICT structured medical-contact JSON object.

The transcripts are noisy, partially translated, conversational, repetitive, and may contain unclear grammar.

Your task is NOT to summarize casually.
Your task is to perform structured healthcare NLP extraction.

---

## CORE OBJECTIVES

For every transcript:

1. Identify:
   * patient name
   * clinic name
   * clinic location
   * doctor types
   * diseases
   * symptoms
   * medicines
   * tests
   * advice
   * follow-up requirements
   * urgency
   * recovery status
   * compliance
   * appointment intent
   * referrals
   * conversation category

2. Infer meaning conservatively.
   Do NOT hallucinate facts not present in transcript.

3. Use normalized categories whenever possible.

4. Output STRICT VALID JSON ONLY.

5. Every field must exist even if null or empty.

6. Include confidence scores.

7. Flag uncertain cases for manual review.

---

## IMPORTANT EXTRACTION RULES

GENERAL RULES:

* Never invent diagnoses.
* Never assume medicine names unless explicitly stated.
* Distinguish: symptom vs disease vs diagnosis vs screening topic.
* If the transcript only mentions "pain", do not infer arthritis.
* If "sugar" is mentioned, normalize to diabetes.
* If "pressure" is mentioned, normalize to blood_pressure.
* If transcript says "eye issue", classify under: conditions_screened = ["eye"]
  NOT diagnosis.
* Never infer recovery if the transcript does not explicitly indicate improvement or recovery.
* If symptoms are only mentioned as screening questions ("Do you have eye or dental problems?"), do NOT record them as confirmed symptoms.
* If the patient says the issue still exists, recovery_status cannot be "fully_recovered".
* If medication was stopped before recovery, patient_compliance should be "non_compliant" or "unknown".
* Prefer "unknown" over guessing.
* Structured summaries must remain conservative and evidence-based.

---

## PATIENT NAME RULES

Examples:
"Am I speaking from Ruddin Sardar's house?"
→ patient.name = "Ruddin Sardar"

If no reliable patient identity: patient.name = null

---

## CLINIC RULES

Extract: clinic.name, location reference, closed days.

Examples:
"Maya Bhavan clinic" → clinic.name = "Maya Bhavan"
"closed on Fridays" → closed_on = ["Fridays"]

---

## CONVERSATION TYPE

Allowed values ONLY:
routine_follow_up | medication_inquiry | appointment_scheduling | specialist_referral
| unreachable | informational_outreach | test_follow_up | recovery_check
| chronic_condition_follow_up

Pick ONE primary type.

---

## CALL OUTCOME

Allowed values ONLY:
no_action_needed | informational | follow_up_scheduled | referred_elsewhere
| patient_unreachable | advised_clinic_visit | medication_continued
| medication_stopped | escalation_recommended

---

## RECOVERY STATUS

Allowed values ONLY: fully_recovered | improving | unresolved | worsening | unknown

"same condition" → unresolved
"better now" → improving
"fully fine" → fully_recovered

---

## PATIENT COMPLIANCE

Allowed values ONLY: compliant | partial | non_compliant | unknown

Stopped medicines early → non_compliant
Took medicines regularly → compliant

---

## URGENCY

Allowed values ONLY: low | medium | high

Use HIGH ONLY if:
* severe unresolved symptoms
* referral urgency
* worsening condition
* persistent untreated issue

---

## MEDICAL ENTITY RULES

symptoms: pain | breathing_issue | swelling | fever | etc.
conditions_reported: diabetes | blood_pressure | thyroid | skin_condition | bone_degeneration | iron_deficiency
doctor_specializations: general | eye | dental | orthopedic | heart | skin
medicines: Only if explicitly discussed.
tests_recommended: blood_test | x_ray | check_up

---

## FOLLOW-UP DETECTION

Set follow_up_required = true if:
* patient asked to revisit
* advised to return
* doctor review recommended
* unresolved issue exists

---

## MANUAL REVIEW CONDITIONS

Set needs_manual_review = true if:
* patient identity unclear
* conflicting medical info
* recovery unclear
* extremely vague transcript
* low extraction confidence
* noisy translation

---

## CONFIDENCE SCORING

Every major extracted field must receive 0.0 to 1.0 confidence score.
Lower confidence when: inference required | transcript vague | translation ambiguous

---

## STRUCTURED SUMMARY RULE

Generate a concise professional medical CRM summary.
Format: patient condition / current status / advice / next step.

Example:
"Patient previously visited for skin-related issue. Symptoms improved after medication but
condition recurred after stopping treatment. Caller advised patient to revisit clinic for
follow-up consultation."

Keep summary factual and concise.

---

## IMPORTANT FINAL RULE

Prefer UNDER-extraction over hallucination.
If uncertain: use null | use unknown | lower confidence | flag manual review.
Never fabricate medical information.

---

## INPUT TRANSCRIPT

{text}

---

## OUTPUT SCHEMA (return ONLY this JSON, no other text)

{
  "call_id": "{call_id}",
  "patient": { "name": null },
  "clinic": { "name": null, "location_reference": null, "closed_on": [] },
  "call_meta": {
    "conversation_type": "",
    "call_outcome": "",
    "urgency_level": "",
    "follow_up_required": false,
    "appointment_status": "",
    "clinic_closed_notice": false,
    "referral_needed": false
  },
  "clinical": {
    "doctor_specializations": [],
    "conditions_screened": [],
    "conditions_reported": [],
    "symptoms": [],
    "medicines": [],
    "tests_recommended": [],
    "diagnoses_mentioned": []
  },
  "patient_status": {
    "recovery_status": "",
    "patient_compliance": "",
    "family_health_mentions": false,
    "advice_given": []
  },
  "quality": {
    "confidence_scores": {},
    "needs_manual_review": false,
    "extraction_warnings": []
  },
  "text": {
    "raw_translation": "",
    "structured_summary": ""
  }
}
"""


def llm_extract_all(
    text: str,
    call_id: str,
    provider: str,
    model: str,
    client: Any,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Call LLM to extract the full schema.
    The LLM is given the user's complete extraction system prompt and returns
    the full schema JSON. We merge its output over the rule-based result.
    Returns a dict with all extractable fields (None values preserved as fallback).
    """
    prompt = LLM_EXTRACTION_PROMPT.format(text=text[:4000], call_id=call_id)

    for attempt in range(1, max_retries + 1):
        try:
            if provider == "openai":
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1500,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content.strip()
            else:  # anthropic
                msg = client.messages.create(
                    model=model,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = msg.content[0].text.strip()

            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            parsed = json.loads(raw)
            return parsed

        except (json.JSONDecodeError, KeyError, Exception) as exc:  # noqa: BLE001
            logger.warning(
                "[%s] LLM extraction attempt %d/%d failed: %s",
                call_id, attempt, max_retries, exc,
            )
            if attempt < max_retries:
                time.sleep(1.0 * attempt)

    logger.error("[%s] LLM extraction failed after %d retries.", call_id, max_retries)
    return {}


# Keep old name for backward compat
llm_extract_weak_fields = llm_extract_all


def build_llm_client(provider: str, model: str, api_key: Optional[str]) -> Any:
    """Instantiate and return the LLM client."""
    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Run: pip install openai") from exc
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY or pass --llm-api-key.")
        return OpenAI(api_key=key)
    elif provider == "anthropic":
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Run: pip install anthropic") from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Set ANTHROPIC_API_KEY or pass --llm-api-key.")
        return anthropic.Anthropic(api_key=key)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# ============================================================================
# PART 6 — ENRICHED RECORD BUILDER
# ============================================================================

def _build_enriched_segment(segment: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Reshape a translated segment dict into the enriched segment schema."""
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", 0.0))
    return {
        "segment_id":          idx,
        "start":               start,
        "end":                 end,
        "duration":            round(max(end - start, 0.0), 4),
        "text_bn":             segment.get("text_bn", ""),
        "text_bn_raw":         segment.get("text_bn_raw", None),
        "text_bn_repaired":    segment.get("text_bn_repaired", None),
        "text_en":             segment.get("text_en", None),
        "quality_score":       segment.get("quality_score", None),
        "repetition_density":  segment.get("repetition_density", None),
        "flagged":             bool(segment.get("flagged", False)),
        "translation_warning": bool(segment.get("translation_warning", False)),
    }


def build_enriched_record(
    record: Dict[str, Any],
    file_id: str,
    date: Optional[str],
    llm_client: Optional[Any] = None,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Build the complete enriched record for a single call.
    Rule-based extraction always runs first.
    If llm_client is provided, LLM extraction replaces/supplements weak fields.
    LLM always wins when it returns a non-null, non-empty value (it uses the full prompt spec).
    """
    call_id = record.get("call_id", "UNKNOWN")
    raw_segments = record.get("segments", [])
    enriched_segments = [_build_enriched_segment(s, i) for i, s in enumerate(raw_segments)]

    extracted = extract_all(call_id, raw_segments, date)

    # LLM full-schema extraction (optional)
    if llm_client is not None:
        full_text = extracted["text"]["raw_translation"]
        llm = llm_extract_all(full_text, call_id, llm_provider, llm_model, llm_client)

        if llm:
            # Patient — use LLM name if rule-based failed
            llm_patient = llm.get("patient") or {}
            if llm_patient.get("name") and not extracted["patient"].get("name"):
                extracted["patient"]["name"] = llm_patient["name"]

            # Clinic
            llm_clinic = llm.get("clinic") or {}
            if llm_clinic.get("name"):
                extracted["clinic"]["name"] = llm_clinic["name"]
            if llm_clinic.get("location_reference"):
                extracted["clinic"]["location_reference"] = llm_clinic["location_reference"]
            if llm_clinic.get("closed_on"):
                extracted["clinic"]["closed_on"] = llm_clinic["closed_on"]

            # Call meta — LLM wins on all enum fields (it has full rule context)
            llm_meta = llm.get("call_meta") or {}
            for field in ("conversation_type", "call_outcome", "urgency_level",
                          "follow_up_required", "appointment_status",
                          "clinic_closed_notice", "referral_needed"):
                val = llm_meta.get(field)
                if val is not None and val != "" and val != []:
                    extracted["call_meta"][field] = val

            # Clinical — LLM adds symptoms, diagnoses, medicines
            llm_clinical = llm.get("clinical") or {}
            if llm_clinical.get("symptoms"):
                extracted["clinical"]["symptoms"] = llm_clinical["symptoms"]
            if llm_clinical.get("diagnoses_mentioned"):
                extracted["clinical"]["diagnoses_mentioned"] = llm_clinical["diagnoses_mentioned"]
            if llm_clinical.get("medicines"):
                extracted["clinical"]["medicines"] = llm_clinical["medicines"]
            # Only override conditions if LLM found more
            if llm_clinical.get("conditions_reported"):
                extracted["clinical"]["conditions_reported"] = llm_clinical["conditions_reported"]
            if llm_clinical.get("conditions_screened"):
                extracted["clinical"]["conditions_screened"] = llm_clinical["conditions_screened"]
            if llm_clinical.get("tests_recommended"):
                extracted["clinical"]["tests_recommended"] = llm_clinical["tests_recommended"]
            if llm_clinical.get("doctor_specializations"):
                extracted["clinical"]["doctor_specializations"] = llm_clinical["doctor_specializations"]

            # Patient status
            llm_ps = llm.get("patient_status") or {}
            for field in ("recovery_status", "patient_compliance",
                          "family_health_mentions", "advice_given"):
                val = llm_ps.get(field)
                if val is not None and val != "" and val != []:
                    extracted["patient_status"][field] = val

            # Text — LLM summary overrides rule-based
            llm_text = llm.get("text") or {}
            if llm_text.get("structured_summary"):
                extracted["text"]["structured_summary"] = llm_text["structured_summary"]

            # Quality — merge LLM confidence scores on top
            llm_quality = llm.get("quality") or {}
            llm_conf = llm_quality.get("confidence_scores") or {}
            if llm_conf:
                extracted["quality"]["confidence_scores"].update(llm_conf)
            # LLM needs_manual_review — OR with rule-based (stricter)
            if llm_quality.get("needs_manual_review"):
                extracted["quality"]["needs_manual_review"] = True
            # Add LLM warnings to existing
            for w in (llm_quality.get("extraction_warnings") or []):
                if w not in extracted["quality"]["extraction_warnings"]:
                    extracted["quality"]["extraction_warnings"].append(w)

            # Mark LLM-boosted confidence for key weak fields
            extracted["quality"]["confidence_scores"].setdefault("symptoms", 0.75)
            extracted["quality"]["confidence_scores"].setdefault("medicines", 0.70)
            extracted["quality"]["confidence_scores"].setdefault("diagnoses_mentioned", 0.72)

    return {
        "call_id":             call_id,
        "call_date":           date,
        "file_id":             file_id,
        "duration_seconds":    float(record.get("duration", 0.0)),
        "segment_count":       len(enriched_segments),
        "source_language":     record.get("language", "bn"),
        "target_language":     "en",
        "translation_backend": record.get("translation_backend", None),
        "nlp_stage":           "enriched",
        "nlp_enriched_at":     datetime.now(timezone.utc).isoformat(),
        # Extracted structured fields
        **extracted,
        # Full enriched segment array (preserves all upstream data)
        "segments":            enriched_segments,
    }


# ============================================================================
# PART 7 — CSV FLAT EXPORT
# ============================================================================

CSV_FIELDS = [
    "call_id", "call_date", "file_id", "duration_seconds",
    "patient_name",
    "clinic_name", "clinic_location",
    "conversation_type", "call_outcome", "urgency_level",
    "follow_up_required", "referral_needed", "appointment_status",
    "clinic_closed_notice", "family_health_mentions",
    "recovery_status", "patient_compliance",
    "doctor_specializations", "conditions_screened", "conditions_reported",
    "medicines", "tests_recommended", "advice_given",
    "needs_manual_review", "extraction_warnings",
    "conf_patient_name", "conf_recovery_status", "conf_compliance",
    "segment_count", "flagged_segment_ratio", "low_quality_segments",
    "structured_summary",
]


def flatten_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a nested enriched record to a single-level dict for CSV export."""
    conf = record.get("quality", {}).get("confidence_scores", {})
    return {
        "call_id":              record["call_id"],
        "call_date":            record.get("call_date"),
        "file_id":              record.get("file_id"),
        "duration_seconds":     record.get("duration_seconds"),
        "patient_name":         (record.get("patient") or {}).get("name"),
        "clinic_name":          (record.get("clinic") or {}).get("name"),
        "clinic_location":      (record.get("clinic") or {}).get("location_reference"),
        "conversation_type":    (record.get("call_meta") or {}).get("conversation_type"),
        "call_outcome":         (record.get("call_meta") or {}).get("call_outcome"),
        "urgency_level":        (record.get("call_meta") or {}).get("urgency_level"),
        "follow_up_required":   (record.get("call_meta") or {}).get("follow_up_required"),
        "referral_needed":      (record.get("call_meta") or {}).get("referral_needed"),
        "appointment_status":   (record.get("call_meta") or {}).get("appointment_status"),
        "clinic_closed_notice": (record.get("call_meta") or {}).get("clinic_closed_notice"),
        "family_health_mentions": (record.get("patient_status") or {}).get("family_health_mentions"),
        "recovery_status":      (record.get("patient_status") or {}).get("recovery_status"),
        "patient_compliance":   (record.get("patient_status") or {}).get("patient_compliance"),
        "doctor_specializations": "|".join((record.get("clinical") or {}).get("doctor_specializations", [])),
        "conditions_screened":  "|".join((record.get("clinical") or {}).get("conditions_screened", [])),
        "conditions_reported":  "|".join((record.get("clinical") or {}).get("conditions_reported", [])),
        "medicines":            "|".join((record.get("clinical") or {}).get("medicines", [])),
        "tests_recommended":    "|".join((record.get("clinical") or {}).get("tests_recommended", [])),
        "advice_given":         "|".join((record.get("patient_status") or {}).get("advice_given", [])),
        "needs_manual_review":  (record.get("quality") or {}).get("needs_manual_review"),
        "extraction_warnings":  " | ".join((record.get("quality") or {}).get("extraction_warnings", [])),
        "conf_patient_name":    conf.get("patient_name"),
        "conf_recovery_status": conf.get("recovery_status"),
        "conf_compliance":      conf.get("patient_compliance"),
        "segment_count":        record.get("segment_count"),
        "flagged_segment_ratio": (record.get("quality") or {}).get("flagged_segment_ratio"),
        "low_quality_segments": (record.get("quality") or {}).get("low_quality_segments"),
        "structured_summary":   (record.get("text") or {}).get("structured_summary"),
    }


# ============================================================================
# PART 8 — DEMO MODE (process 20 inline sample calls)
# ============================================================================

DEMO_CALLS = [
    {"call_id": "1767241522.10067", "text": "Hello, are you speaking from Amara Bibi's house? I am calling from the clinic near Krishna Chandrapur High School. You visited earlier, right? Are you completely well now, with no issues? Are you still taking medicine? No? Okay. Besides general medicine doctors, eye and dental specialists also sit here. If anyone in your family needs to consult them, you can visit. Stay well."},
    {"call_id": "1767242157.10067", "text": "Am I speaking with someone from Ruddin Sardar's house? I am calling from the Maya Bhavan clinic near Krishna Chandrapur High School. He had consulted a doctor from our clinic earlier. Please have him checked; good doctors are available here at Maya Bhavan near Krishna Chandrapur High School. It is closed on Fridays."},
    {"call_id": "1767242346.10067", "text": "Hello, are you speaking from Ebadul Haldar's house? I am calling from the clinic near Krishna Chandrapur High School. Ebadul Haldar had visited our clinic for a health issue. How is he doing now? Is he consulting another doctor now? Which heart specialist is he seeing? I was asking if he is consulting a heart specialist now. As long as he is doing well, that's fine. Stay well."},
    {"call_id": "1767242787.10067", "text": "Are you speaking from Ajai Naskar's house? I am calling from Maya Bhavan clinic near Krishna Chandrapur High School. Is he okay? If he needs the same medicine, it is available here; you would have to come here. The eye drops that were given... I can't find the prescription. Doctors are available here, I'm not sure if you know. Anyway, stay well."},
    {"call_id": "1767243167.10067", "text": "Hello, am I speaking to someone at Halek Molla's house? I am calling from the clinic near Krishna Chandrapur High School—yes, from Maya Bhavan. He had consulted a doctor here. How is he now? Is he doing well? Has the issue he was seen for completely resolved? Yes, he is doing moderately well now. Does he have any other issues like sugar, blood pressure, eye, or dental problems? No. If there are any problems, he goes there to get checked. Okay, that's fine. We just want everyone to stay healthy. If needed, please get checked."},
    {"call_id": "1767243351.10067", "text": "Hello, am I speaking with Mabuda Paik? Yes, I am Mabuda Paik. Did you visit the clinic near Krishna Chandrapur High School? Yes, I did. How are you feeling now? Do you do any heavy physical work? I have pain when I work in the fields, otherwise, I sit at home. Did the doctor mention if the pain is due to bone degeneration? No, the doctor didn't say what is causing the nerve pain from the waist down to the back. You should come back and ask the doctor exactly what is causing this issue—if it's bone degeneration, whether you need exercises or dietary restrictions. Don't spend money on an X-ray/photo unnecessarily unless the doctor explicitly asks for it. Talk to him first. Okay, I will go in a day or two. Remember it's closed tomorrow, so come the day after. Stay well."},
    {"call_id": "1767243880.10067", "text": "Hello. Have you fully recovered from the issue you were seen for? Are the other members of your family healthy? Do they have any eye, dental, blood sugar, blood pressure, or thyroid issues? Please inform your neighbors as well so they can get checked if needed. Stay well."},
    {"call_id": "1767243895.10067", "text": "Hello, namaskar. Are you speaking from Paritan Sheikh's house? Doctors are available here, so please contact them if needed. Okay, stay well."},
    {"call_id": "1767244064.10067", "text": "Hello, am I speaking with someone from Momina Molla's house? I am calling from the clinic near Krishna Chandrapur High School. You visited earlier for a skin issue. Are you consulting someone else for that now? You aren't getting relief from the current doctor? Are the rest of your family members healthy? You can come and get checked. Stay well."},
    {"call_id": "1767244258.10067", "text": "Hello, I am not at home. Did someone visit the doctor from here recently? No, we just wanted to ask about the issue for which they were seen. Thank you."},
    {"call_id": "1767244560.10067", "text": "Hello. Is the person who was seen for an issue doing well now? Is the medication still ongoing or stopped? You didn't come in December; you should get a checkup done before buying more medicine from outside. Please come this week or next week for a check-up. Is everyone else at home doing well? Yes, all five members of my family are well. Good, if there are any issues, let the doctor know."},
    {"call_id": "1767244575.10067", "text": "Am I speaking to someone at Ashidul Sardar's house? I am calling from Maya Bhavan clinic near Krishna Chandrapur High School. How is Ashidul doing now? I have shown him to many doctors elsewhere. If there is an issue, you can come and get a blood test done here. Where are you calling from? Maya Bhavan near Krishna Chandrapur."},
    {"call_id": "1767244694.10067", "text": "Hello. Have you fully recovered from the issue you were seen for with no further problems? My brother is currently working in Chennai. Okay, if there are any issues, have him consult a doctor there."},
    {"call_id": "1767244726.10067", "text": "Hello. How many days did you take the medicine? For a week. Has it improved a bit? Are the rest of your family members healthy—no blood pressure, sugar, thyroid, eye, or dental issues? I have iron deficiency, breathing issues, and some stomach problems. Did you see a doctor? Yes, I saw one a week ago. Okay, as long as you are well with that medicine, that's fine. If you feel any issues, we have eye and dental specialists here. Do regular specialists sit here? Only eye, dental, and general medicine doctors sit here for these treatments. Okay."},
    {"call_id": "1767244923.10067", "text": "Hello, am I speaking with Arva Haldar? You were seen for a skin issue, right? Has the issue completely recovered? I took medicine from there for two months. If the problem returned after stopping the medicine, you should come back and show the doctor again. Okay."},
    {"call_id": "1767245318.10067", "text": "Hello, am I speaking with Surekha Nalia's house? I am calling from Maya Bhavan clinic near Krishna Chandrapur High School. You visited us a few days ago for an issue. How are you now? I am doing somewhat fine, I am currently somewhere else. Has the eye problem completely healed? No. Are you still taking the medicine or have you stopped? Stopped. Why did you stop seeing the doctor if the problem is still there? As long as I took the drops it was fine. You need to come and tell the doctor; don't neglect your eyes. Is everyone else at home healthy—no eye, dental, blood pressure, or sugar issues?"},
    {"call_id": "1767245449.10067", "text": "Hello, I am calling from the Maya Bhavan clinic near Krishna Chandrapur High School. Jabi Debi had visited our doctor here for an issue. How is she doing now? I will come and see the doctor there. Okay, please remember that it is closed on Fridays, so skip that day when you come. Yes, I will definitely come. Stay well."},
    {"call_id": "1767245552.10067", "text": "Hello, namaskar. Am I speaking with someone from Chandana Mondal's house? I am calling from the Maya Bhavan clinic near Krishna Chandrapur High School. You had consulted a doctor here a few months ago for pain. How are you now? Is everyone healthy? Okay, stay well. If you ever feel any problems, doctors are available here."},
    {"call_id": "1767245575.10067", "text": "Hello, am I speaking with Jahanara Aboidro's house? I am calling from Maya Bhavan clinic near Krishna Chandrapur High School. She consulted a doctor here a few months ago. How is she now? She is in the same condition. Is the pain in her arms and legs still not better? No. How many times did she see the doctor here? I can't say exactly. Is the medicine finished? Yes. Since the problem persists, please come back and consult the doctor again. Don't ignore it, or it will get worse day by day. It could be due to age-related bone degeneration."},
    {"call_id": "1767245578.10067", "text": "Hello. Are you completely cured now with no issues? Do you have any eye or dental problems? I have dental issues, no eye issues. We provide medicines and prescriptions for dental issues here if needed."},
]


def run_demo(output_dir: Path) -> None:
    """Process the 20 built-in sample calls and write demo output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "demo_enriched.jsonl"
    csv_path = output_dir / "demo_enriched.csv"

    print(f"\n{'='*70}")
    print("  DEMO MODE — processing 20 sample healthcare calls")
    print(f"{'='*70}\n")

    records: List[Dict] = []

    for call in DEMO_CALLS:
        # Simulate minimal segment structure for demo calls
        demo_segments = [{
            "text_en": call["text"],
            "text_bn": "",
            "start": 0.0,
            "end": 60.0,
            "quality_score": 0.85,
            "repetition_density": 0.0,
            "flagged": False,
        }]
        fake_record = {
            "call_id": call["call_id"],
            "duration": 60.0,
            "segments": demo_segments,
            "translation_backend": "demo",
        }
        enriched = build_enriched_record(fake_record, "demo", "2026-01-01")
        records.append(enriched)

        # Print summary
        p = enriched.get("patient", {})
        m = enriched.get("call_meta", {})
        ps = enriched.get("patient_status", {})
        print(f"  📞 {call['call_id']}")
        print(f"     Patient    : {p.get('name') or '(not found)'}")
        print(f"     Recovery   : {ps.get('recovery_status')} | Compliance: {ps.get('patient_compliance')}")
        print(f"     Outcome    : {m.get('call_outcome')} | Urgency: {m.get('urgency_level')}")
        print(f"     Review?    : {'⚠️  YES' if enriched['quality']['needs_manual_review'] else '✅ no'}")
        warnings = enriched["quality"]["extraction_warnings"]
        if warnings:
            for w in warnings:
                print(f"     ⚠  {w}")
        print()

    # Write JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write flat CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(flatten_for_csv(rec))

    needs_review = sum(1 for r in records if r["quality"]["needs_manual_review"])
    fully_recovered = sum(
        1 for r in records
        if r["patient_status"]["recovery_status"] == "fully_recovered"
    )
    unresolved = sum(
        1 for r in records
        if r["patient_status"]["recovery_status"] == "unresolved"
    )

    print(f"{'='*70}")
    print(f"  SUMMARY — {len(records)} calls processed")
    print(f"  Fully recovered  : {fully_recovered}")
    print(f"  Unresolved       : {unresolved}")
    print(f"  Needs review     : {needs_review}")
    print(f"  Output JSONL     : {out_path}")
    print(f"  Output CSV       : {csv_path}")
    print(f"{'='*70}\n")


# ============================================================================
# PART 9 — SQLITE CHECKPOINT
# ============================================================================

class NLPEnrichmentCheckpoint:
    TABLE = "nlp_enrichment_v2"

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
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    file_id     TEXT PRIMARY KEY,
                    status      TEXT NOT NULL,
                    input_path  TEXT,
                    output_path TEXT,
                    calls       INTEGER,
                    segments    INTEGER,
                    error       TEXT,
                    updated_at  TEXT NOT NULL
                )
            """)
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.TABLE}_status "
                f"ON {self.TABLE}(status)"
            )
            conn.commit()

    def is_done(self, file_id: str, expected_calls: Optional[int] = None) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT status, calls FROM {self.TABLE} WHERE file_id = ?", (file_id,)
            ).fetchone()
        if not row or row["status"] != "done":
            return False
        if expected_calls is not None and row["calls"] != expected_calls:
            return False
        return True

    def mark_done(self, file_id: str, input_path: str, output_path: str, calls: int, segments: int) -> None:
        self._upsert(file_id=file_id, status="done", input_path=input_path,
                     output_path=output_path, calls=calls, segments=segments, error=None)

    def mark_error(self, file_id: str, input_path: str, error: str) -> None:
        self._upsert(file_id=file_id, status="error", input_path=input_path,
                     output_path=None, calls=None, segments=None, error=error)

    def _upsert(self, **kwargs: Any) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(f"""
                INSERT INTO {self.TABLE}
                    (file_id, status, input_path, output_path, calls, segments, error, updated_at)
                VALUES
                    (:file_id, :status, :input_path, :output_path, :calls, :segments, :error, :updated_at)
                ON CONFLICT(file_id) DO UPDATE SET
                    status=excluded.status, input_path=excluded.input_path,
                    output_path=excluded.output_path, calls=excluded.calls,
                    segments=excluded.segments, error=excluded.error,
                    updated_at=excluded.updated_at
            """, {**kwargs, "updated_at": now})
            conn.commit()

    def stats(self) -> Dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT status, COUNT(*) AS n FROM {self.TABLE} GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}


# ============================================================================
# PART 10 — FILE PROCESSING & MAIN PIPELINE RUNNER
# ============================================================================

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def infer_date(path: Path) -> Optional[str]:
    for part in reversed(path.parts):
        m = _DATE_RE.search(part)
        if m:
            return m.group(1)
    return None


def process_file(
    input_path: Path,
    output_path: Path,
    file_id: str,
    csv_path: Optional[Path] = None,
    llm_client: Optional[Any] = None,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
) -> Tuple[int, int]:
    """Stream-process one translated JSONL file into enriched output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".jsonl.tmp")
    date = infer_date(input_path)
    calls = total_segments = 0

    csv_rows: List[Dict] = []

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
                logger.warning("Skipping malformed JSON at %s line %d: %s",
                               input_path.name, line_no, exc)
                continue

            enriched = build_enriched_record(
                record, file_id, date,
                llm_client=llm_client,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
            fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            if csv_path is not None:
                csv_rows.append(flatten_for_csv(enriched))
            calls += 1
            total_segments += enriched.get("segment_count", 0)

    tmp_path.replace(output_path)

    # Append CSV rows
    if csv_path is not None and csv_rows:
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(csv_rows)

    return calls, total_segments


def run(
    input_dir: Path,
    output_dir: Path,
    checkpoint_db: Path,
    pattern: str,
    force: bool,
    export_csv: bool = False,
    llm_client: Optional[Any] = None,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
) -> None:
    """Discover and process all translated JSONL files."""
    checkpoint = NLPEnrichmentCheckpoint(checkpoint_db)
    input_files = sorted(input_dir.rglob(pattern))
    if not input_files:
        logger.warning("No files matching '%s' found in %s", pattern, input_dir)
        return

    csv_path = output_dir / "enriched_flat.csv" if export_csv else None
    if csv_path and csv_path.exists():
        csv_path.unlink()  # Start fresh for this run

    llm_label = f"{llm_provider}:{llm_model}" if llm_client else "rule-based only"
    logger.info("Found %d file(s). Extraction: %s  Checkpoint: %s",
                len(input_files), llm_label, checkpoint_db)

    skipped = done = errors = 0

    with tqdm(input_files, desc="Enrichment formatting", unit="file") as pbar:
        for input_path in pbar:
            try:
                rel = input_path.relative_to(input_dir)
            except ValueError:
                rel = Path(input_path.name)

            file_id = str(rel)
            output_path = output_dir / rel
            pbar.set_postfix(file=input_path.name, refresh=False)

            # Count the calls in the input file to detect if it has changed
            expected_calls = None
            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    expected_calls = sum(1 for line in f if line.strip())
            except Exception:
                pass

            if not force and checkpoint.is_done(file_id, expected_calls=expected_calls):
                skipped += 1
                continue

            try:
                calls, total_segments = process_file(
                    input_path, output_path, file_id,
                    csv_path=csv_path,
                    llm_client=llm_client,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                )
                checkpoint.mark_done(
                    file_id=file_id, input_path=str(input_path),
                    output_path=str(output_path), calls=calls, segments=total_segments,
                )
                done += 1
                logger.debug("DONE: %s — %d calls, %d segments", file_id, calls, total_segments)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                err_msg = f"{type(exc).__name__}: {exc}"
                checkpoint.mark_error(file_id=file_id, input_path=str(input_path), error=err_msg)
                logger.error("ERROR processing %s: %s", file_id, err_msg)

    stats = checkpoint.stats()
    logger.info("Run complete — done=%d  skipped=%d  errors=%d  | DB stats: %s",
                done, skipped, errors, stats)

    # ── Fallback CSV export ────────────────────────────────────────────────────
    # If --export-csv was requested but all files were skipped (already checkpointed),
    # the CSV was never written. Regenerate it from the existing enriched JSONL outputs.
    if csv_path and not csv_path.exists() and skipped > 0:
        logger.info("CSV not found but files were skipped — regenerating from existing enriched JSONL …")
        csv_rows: List[Dict] = []
        for input_path in input_files:
            try:
                rel = input_path.relative_to(input_dir)
            except ValueError:
                rel = Path(input_path.name)
            output_path = output_dir / rel
            if not output_path.exists():
                logger.warning("Enriched JSONL not found for skipped file: %s", output_path)
                continue
            with open(output_path, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        record = json.loads(raw_line)
                        csv_rows.append(flatten_for_csv(record))
                    except (json.JSONDecodeError, KeyError) as exc:
                        logger.warning("Skipping malformed record during CSV rebuild: %s", exc)
        if csv_rows:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as cf:
                writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(csv_rows)
            logger.info("Flat CSV rebuilt from existing enriched JSONL: %s  (%d rows)", csv_path, len(csv_rows))
        else:
            logger.warning("No enriched records found to rebuild CSV — run with --force to reprocess.")

    if csv_path and csv_path.exists():
        logger.info("Flat CSV export: %s", csv_path)


# ============================================================================
# PART 11 — CLI
# ============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="format_enriched",
        description=(
            "Stage 3 — Healthcare NLP extraction & analytics enrichment.\n\n"
            "Transforms translated Bengali call transcripts into structured,\n"
            "analytics-ready JSON with entity extraction and confidence scoring."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    io = parser.add_argument_group("I/O paths")
    io.add_argument("--input-dir", type=Path, default=Path("data/nlp/translated"),
                    help="Translated JSONL input directory. (default: data/nlp/translated)")
    io.add_argument("--output-dir", type=Path, default=Path("data/enriched"),
                    help="Enriched JSONL output directory. (default: data/enriched)")
    io.add_argument("--checkpoint", type=Path, default=Path("checkpoints/nlp_enrichment.sqlite"))
    io.add_argument("--pattern", type=str, default="*.jsonl",
                    help="Glob pattern for input files.")

    mode = parser.add_argument_group("Run mode")
    mode.add_argument("--demo", action="store_true",
                      help="Process the 20 built-in sample calls and print a report.")
    mode.add_argument("--force", action="store_true",
                      help="Re-process files already marked done in checkpoint DB.")
    mode.add_argument("--export-csv", action="store_true",
                      help="Also write a flat CSV file (data/enriched/enriched_flat.csv).")
    mode.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")

    llm = parser.add_argument_group("LLM-assisted extraction (optional)")
    llm.add_argument("--llm-extraction", action="store_true",
                     help="Enable LLM extraction for weak fields (symptoms, diagnoses, summary).")
    llm.add_argument("--llm-provider", type=str, default="openai",
                     choices=["openai", "anthropic"],
                     help="LLM provider for weak-field extraction. (default: openai)")
    llm.add_argument("--llm-model", type=str, default="gpt-4o-mini",
                     help="Model name. (default: gpt-4o-mini)")
    llm.add_argument("--llm-api-key", type=str, default=None,
                     help="API key (overrides env var).")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.demo:
        demo_dir = args.output_dir / "demo"
        run_demo(demo_dir)
        return

    llm_client = None
    if args.llm_extraction:
        logger.info("LLM extraction enabled: %s / %s", args.llm_provider, args.llm_model)
        llm_client = build_llm_client(args.llm_provider, args.llm_model, args.llm_api_key)

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint_db=args.checkpoint,
        pattern=args.pattern,
        force=args.force,
        export_csv=args.export_csv,
        llm_client=llm_client,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )


if __name__ == "__main__":
    main()
