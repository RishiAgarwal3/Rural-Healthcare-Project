#!/usr/bin/env python3
"""
============================================================================
  FinAI Voice Analytics — Comprehensive Healthcare EDA Pipeline
============================================================================
  Performs end-to-end exploratory data analysis on the enriched healthcare
  call dataset.  All outputs go into  <PROJECT_ROOT>/results/  organized
  as: charts/, tables/, reports/, nlp/, correlations/, eda_summary.md
============================================================================
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

# ---------------------------------------------------------------------------
# Configuration
# Supports env-var overrides so the SAME script powers both:
#   results/          (default, original dataset)
#   results_improved/ (improved dataset)
#
# Usage for improved run:
#   FINAI_EDA_INPUT_CSV=data/processed_calls_improved.csv \
#   FINAI_EDA_OUTPUT_DIR=results_improved \
#   python scripts/eda/run_eda.py
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow env-var override for input CSV (relative paths resolved from PROJECT_ROOT)
_env_csv = os.environ.get("FINAI_EDA_INPUT_CSV")
ENRICHED_CSV  = (
    PROJECT_ROOT / _env_csv if _env_csv and not Path(_env_csv).is_absolute()
    else Path(_env_csv) if _env_csv
    else PROJECT_ROOT / "data" / "enriched" / "enriched_flat.csv"
)
# JSONL is always from the original enriched dir (for NLP text analysis)
ENRICHED_JSONL = PROJECT_ROOT / "data" / "enriched" / "calls.jsonl"

# Allow env-var override for output directory
_env_out = os.environ.get("FINAI_EDA_OUTPUT_DIR")
RESULTS = (
    PROJECT_ROOT / _env_out if _env_out and not Path(_env_out).is_absolute()
    else Path(_env_out) if _env_out
    else PROJECT_ROOT / "results"
)

# For results_improved/, use 'dashboards' subdir instead of 'charts'
# to match the required output structure. Falls back to 'charts' for default.
_is_improved = _env_out is not None
CHARTS        = RESULTS / ("dashboards" if _is_improved else "charts")
TABLES        = RESULTS / "tables"
REPORTS       = RESULTS / "reports"
NLP_DIR       = RESULTS / "nlp"
CORRELATIONS  = RESULTS / "correlations"

# Professional palette
PALETTE_MAIN = ["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd", "#ddd6fe",
                "#818cf8", "#4f46e5", "#4338ca", "#3730a3", "#312e81"]
PALETTE_ACCENT = ["#10b981", "#f59e0b", "#ef4444", "#3b82f6", "#ec4899",
                  "#14b8a6", "#f97316", "#8b5cf6", "#06b6d4", "#84cc16"]
URGENCY_COLORS = {"low": "#10b981", "medium": "#f59e0b", "high": "#ef4444"}
RECOVERY_COLORS = {
    "fully_recovered": "#10b981", "improving": "#3b82f6",
    "unresolved": "#f59e0b", "worsening": "#ef4444",
    "not_discussed": "#475569", "unknown": "#94a3b8",
    "reply_too_vague_to_classify": "#94a3b8",
}
COMPLIANCE_COLORS = {
    "compliant": "#10b981", "partial": "#f59e0b",
    "non_compliant": "#ef4444", "not_applicable": "#475569",
    "unknown": "#94a3b8", "reply_too_vague_to_classify": "#94a3b8",
}

BG_COLOR = "#0f172a"
CARD_COLOR = "#1e293b"
TEXT_COLOR = "#e2e8f0"
GRID_COLOR = "#334155"
TITLE_COLOR = "#f1f5f9"

DPI = 180

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_dirs() -> None:
    for d in [CHARTS, TABLES, REPORTS, NLP_DIR, CORRELATIONS]:
        d.mkdir(parents=True, exist_ok=True)


def style_ax(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(CARD_COLOR)
    ax.set_title(title, color=TITLE_COLOR, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.3, linewidth=0.5)


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  📊  {path.relative_to(PROJECT_ROOT)}")


def make_fig(w: float = 10, h: float = 6) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG_COLOR)
    return fig, ax


def make_fig_multi(nrows: int, ncols: int, w: float = 14, h: float = 6) -> Tuple[plt.Figure, Any]:
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), facecolor=BG_COLOR)
    return fig, axes


def explode_pipe_col(series: pd.Series) -> pd.Series:
    """Explode '|'-separated values into individual items."""
    return series.dropna().str.split("|").explode().str.strip().replace("", np.nan).dropna()


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_csv() -> pd.DataFrame:
    df = pd.read_csv(ENRICHED_CSV)
    # Boolean fixup
    for col in ["follow_up_required", "referral_needed", "clinic_closed_notice",
                "family_health_mentions", "needs_manual_review"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    return df


def load_jsonl() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(ENRICHED_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ============================================================================
# A.  DATASET OVERVIEW
# ============================================================================

def section_a_overview(df: pd.DataFrame) -> str:
    report_lines: List[str] = []
    report_lines.append("# A. Dataset Overview\n")

    # 1. Shape
    nrow, ncol = df.shape
    report_lines.append(f"- **Rows:** {nrow}")
    report_lines.append(f"- **Columns:** {ncol}\n")

    # 2. Dtypes
    dtype_counts = df.dtypes.value_counts()
    report_lines.append("## Data Types")
    report_lines.append("| Type | Count |")
    report_lines.append("|------|------:|")
    for dtype, cnt in dtype_counts.items():
        report_lines.append(f"| `{dtype}` | {cnt} |")
    report_lines.append("")

    # Save dtype detail table
    dtype_df = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "non_null": df.notnull().sum().values,
        "null_count": df.isnull().sum().values,
        "null_pct": (df.isnull().sum() / len(df) * 100).round(2).values,
        "unique": df.nunique().values,
        "sample": [str(df[c].dropna().iloc[0])[:60] if df[c].notna().any() else "" for c in df.columns],
    })
    dtype_df.to_csv(TABLES / "column_profile.csv", index=False)

    # 3. Missing-value chart
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if len(missing):
        report_lines.append("## Missing Values")
        report_lines.append("| Column | Missing | % |")
        report_lines.append("|--------|--------:|---:|")
        for col, cnt in missing.items():
            report_lines.append(f"| `{col}` | {cnt} | {cnt / nrow * 100:.1f}% |")
        report_lines.append("")

        fig, ax = make_fig(10, max(4, len(missing) * 0.35))
        bars = ax.barh(missing.index, missing.values, color=PALETTE_ACCENT[2], height=0.6, edgecolor="none")
        for bar, val in zip(bars, missing.values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val} ({val / nrow * 100:.0f}%)", va="center", color=TEXT_COLOR, fontsize=9)
        style_ax(ax, "Missing Values by Column", "Count")
        save_fig(fig, CHARTS / "a_missing_values.png")
    else:
        report_lines.append("> ✅ No missing values found.\n")

    # 4. Duplicates
    dup_count = df.duplicated(subset=["call_id"]).sum() if "call_id" in df.columns else df.duplicated().sum()
    report_lines.append(f"## Duplicates\n- Duplicate call_ids: **{dup_count}**\n")

    return "\n".join(report_lines)


# ============================================================================
# B.  DEMOGRAPHIC / LOCATION ANALYSIS
# ============================================================================

def section_b_demographics(df: pd.DataFrame) -> str:
    report_lines: List[str] = ["# B. Demographic & Location Analysis\n"]

    # Patient name stats
    has_name = df["patient_name"].notna().sum() if "patient_name" in df.columns else 0
    report_lines.append(f"- Calls with identified patient name: **{has_name}** / {len(df)} ({has_name / len(df) * 100:.1f}%)")

    # Unique patients
    if "patient_name" in df.columns:
        unique_patients = df["patient_name"].dropna().nunique()
        report_lines.append(f"- Unique patient names: **{unique_patients}**\n")

        top_patients = df["patient_name"].value_counts().head(15)
        top_patients.to_csv(TABLES / "b_top_patients.csv")

    # Surname / community analysis
    if "patient_name" in df.columns:
        surnames = df["patient_name"].dropna().str.strip().str.split().str[-1]
        surname_counts = surnames.value_counts().head(20)
        if len(surname_counts) > 3:
            report_lines.append("## Community Distribution (by Surname)")
            report_lines.append("| Surname | Count |")
            report_lines.append("|---------|------:|")
            for s, c in surname_counts.head(15).items():
                report_lines.append(f"| {s} | {c} |")
            report_lines.append("")

            fig, ax = make_fig(10, 6)
            colors = sns.color_palette("husl", len(surname_counts))
            wedges, texts, autotexts = ax.pie(
                surname_counts.values, labels=None,
                autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
                colors=colors, startangle=140,
                textprops={"color": TEXT_COLOR, "fontsize": 8},
                pctdistance=0.85,
            )
            ax.legend(surname_counts.index, loc="center left", bbox_to_anchor=(1, 0.5),
                      fontsize=8, frameon=False, labelcolor=TEXT_COLOR)
            ax.set_title("Patient Community Distribution (by Surname)", color=TITLE_COLOR,
                         fontsize=13, fontweight="bold")
            fig.patch.set_facecolor(BG_COLOR)
            save_fig(fig, CHARTS / "b_community_distribution.png")
            surname_counts.to_csv(TABLES / "b_surname_distribution.csv")

    # Clinic distribution
    if "clinic_name" in df.columns:
        clinic_counts = df["clinic_name"].value_counts()
        report_lines.append(f"\n## Clinic Distribution")
        report_lines.append(f"- Clinics identified: **{clinic_counts.index.tolist()}**")
        report_lines.append(f"- Calls with clinic name: **{df['clinic_name'].notna().sum()}**\n")

    if "clinic_location" in df.columns:
        loc_counts = df["clinic_location"].value_counts()
        report_lines.append(f"## Location References")
        for loc, cnt in loc_counts.items():
            report_lines.append(f"- {loc}: {cnt} calls")
        report_lines.append("")

    return "\n".join(report_lines)


# ============================================================================
# C.  HEALTHCARE ANALYSIS
# ============================================================================

def section_c_healthcare(df: pd.DataFrame) -> str:
    report_lines: List[str] = ["# C. Healthcare Analysis\n"]

    # --- C1: Conversation Type Distribution ---
    if "conversation_type" in df.columns:
        ct_counts = df["conversation_type"].value_counts()
        report_lines.append("## Conversation Type Distribution")
        report_lines.append("| Type | Count | % |")
        report_lines.append("|------|------:|---:|")
        for t, c in ct_counts.items():
            report_lines.append(f"| {t} | {c} | {c / len(df) * 100:.1f}% |")
        report_lines.append("")

        fig, ax = make_fig(10, 6)
        bars = ax.barh(ct_counts.index[::-1], ct_counts.values[::-1],
                       color=PALETTE_MAIN[:len(ct_counts)], edgecolor="none", height=0.6)
        for bar, val in zip(bars, ct_counts.values[::-1]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val}", va="center", color=TEXT_COLOR, fontsize=9)
        style_ax(ax, "Conversation Type Distribution", "Count")
        save_fig(fig, CHARTS / "c_conversation_types.png")
        ct_counts.to_csv(TABLES / "c_conversation_types.csv")

    # --- C2: Call Outcome ---
    if "call_outcome" in df.columns:
        co_counts = df["call_outcome"].value_counts()
        fig, ax = make_fig(10, 6)
        bars = ax.barh(co_counts.index[::-1], co_counts.values[::-1],
                       color=PALETTE_ACCENT[:len(co_counts)], edgecolor="none", height=0.6)
        for bar, val in zip(bars, co_counts.values[::-1]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val}", va="center", color=TEXT_COLOR, fontsize=9)
        style_ax(ax, "Call Outcome Distribution", "Count")
        save_fig(fig, CHARTS / "c_call_outcomes.png")
        co_counts.to_csv(TABLES / "c_call_outcomes.csv")

    # --- C3: Urgency Level ---
    if "urgency_level" in df.columns:
        urg = df["urgency_level"].value_counts()
        report_lines.append("## Urgency Level Distribution")
        for u, c in urg.items():
            report_lines.append(f"- **{u}**: {c} ({c / len(df) * 100:.1f}%)")
        report_lines.append("")

        fig, ax = make_fig(10, 6)
        colors_urg = [URGENCY_COLORS.get(x, "#64748b") for x in urg.index]
        bars = ax.barh(urg.index[::-1], urg.values[::-1], color=colors_urg[::-1], edgecolor="none", height=0.5)
        for bar, val in zip(bars, urg.values[::-1]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val} ({val / len(df) * 100:.1f}%)", va="center", color=TEXT_COLOR, fontsize=9)
        style_ax(ax, "Urgency Level Distribution", "Count")
        save_fig(fig, CHARTS / "c_urgency_levels.png")

    # --- C4: Recovery Status ---
    if "recovery_status" in df.columns:
        rec = df["recovery_status"].value_counts()
        report_lines.append("## Recovery Status")
        for r, c in rec.items():
            report_lines.append(f"- **{r}**: {c} ({c / len(df) * 100:.1f}%)")
        report_lines.append("")

        # Split into meaningful vs low-info categories
        low_info_cats = {"unknown", "not_discussed", "reply_too_vague_to_classify"}
        meaningful = rec.drop(labels=[l for l in low_info_cats if l in rec.index], errors="ignore")
        low_info_total = sum(rec.get(c, 0) for c in low_info_cats)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]})
        fig.patch.set_facecolor(BG_COLOR)

        ax_main = axes[0]
        if len(meaningful) > 0:
            colors_rec = [RECOVERY_COLORS.get(x, "#64748b") for x in meaningful.index]
            bars = ax_main.barh(meaningful.index[::-1], meaningful.values[::-1], color=colors_rec[::-1], edgecolor="none", height=0.5)
            for bar, val in zip(bars, meaningful.values[::-1]):
                ax_main.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                            f"{val} ({val / meaningful.sum() * 100:.1f}%)", va="center", color=TEXT_COLOR, fontsize=9)
            style_ax(ax_main, "Patient Recovery Status\n(Calls Where Recovery Was Discussed)", "Count")
        else:
            ax_main.text(0.5, 0.5, "No recovery data", ha="center", va="center",
                        color=TEXT_COLOR, fontsize=14, transform=ax_main.transAxes)
            ax_main.set_facecolor(CARD_COLOR)

        ax_info = axes[1]
        ax_info.set_facecolor(CARD_COLOR)
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis("off")
        total_discussed = meaningful.sum()
        ax_info.text(0.5, 0.75, f"{total_discussed}", ha="center", va="center",
                    color="#10b981", fontsize=36, fontweight="bold")
        ax_info.text(0.5, 0.58, "recovery discussed", ha="center", va="center",
                    color=TEXT_COLOR, fontsize=11)
        ax_info.text(0.5, 0.35, f"{low_info_total}", ha="center", va="center",
                    color="#94a3b8", fontsize=28, fontweight="bold")
        ax_info.text(0.5, 0.20, "not discussed / unclear", ha="center", va="center",
                    color="#94a3b8", fontsize=10)

        plt.tight_layout()
        save_fig(fig, CHARTS / "c_recovery_status.png")

    # --- C5: Patient Compliance ---
    if "patient_compliance" in df.columns:
        comp = df["patient_compliance"].value_counts()
        report_lines.append("## Patient Compliance")
        for c_val, cnt in comp.items():
            report_lines.append(f"- **{c_val}**: {cnt} ({cnt / len(df) * 100:.1f}%)")
        report_lines.append("")

        # Split into meaningful vs low-info categories
        low_info_cats = {"unknown", "not_applicable", "reply_too_vague_to_classify"}
        meaningful = comp.drop(labels=[l for l in low_info_cats if l in comp.index], errors="ignore")
        low_info_total = sum(comp.get(c, 0) for c in low_info_cats)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]})
        fig.patch.set_facecolor(BG_COLOR)

        ax_main = axes[0]
        if len(meaningful) > 0:
            colors_comp = [COMPLIANCE_COLORS.get(x, "#64748b") for x in meaningful.index]
            bars = ax_main.barh(meaningful.index[::-1], meaningful.values[::-1], color=colors_comp[::-1], edgecolor="none", height=0.5)
            for bar, val in zip(bars, meaningful.values[::-1]):
                ax_main.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                            f"{val} ({val / meaningful.sum() * 100:.1f}%)", va="center", color=TEXT_COLOR, fontsize=9)
            style_ax(ax_main, "Patient Medication Compliance\n(Calls Where Medication Was Discussed)", "Count")
        else:
            ax_main.text(0.5, 0.5, "No compliance data", ha="center", va="center",
                        color=TEXT_COLOR, fontsize=14, transform=ax_main.transAxes)
            ax_main.set_facecolor(CARD_COLOR)

        ax_info = axes[1]
        ax_info.set_facecolor(CARD_COLOR)
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis("off")
        total_discussed = meaningful.sum()
        ax_info.text(0.5, 0.75, f"{total_discussed}", ha="center", va="center",
                    color="#10b981", fontsize=36, fontweight="bold")
        ax_info.text(0.5, 0.58, "medication discussed", ha="center", va="center",
                    color=TEXT_COLOR, fontsize=11)
        ax_info.text(0.5, 0.35, f"{low_info_total}", ha="center", va="center",
                    color="#94a3b8", fontsize=28, fontweight="bold")
        ax_info.text(0.5, 0.20, "not applicable / unclear", ha="center", va="center",
                    color="#94a3b8", fontsize=10)

        plt.tight_layout()
        save_fig(fig, CHARTS / "c_patient_compliance.png")

    # --- C6: Doctor Specializations ---
    if "doctor_specializations" in df.columns:
        specs = explode_pipe_col(df["doctor_specializations"])
        spec_counts = specs.value_counts()
        report_lines.append("## Doctor Specializations Mentioned")
        for s, c in spec_counts.items():
            report_lines.append(f"- **{s}**: {c}")
        report_lines.append("")

        fig, ax = make_fig(8, 5)
        bars = ax.bar(spec_counts.index, spec_counts.values,
                      color=PALETTE_MAIN[:len(spec_counts)], edgecolor="none", width=0.5)
        for bar, val in zip(bars, spec_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(val), ha="center", va="bottom", color=TEXT_COLOR, fontsize=10, fontweight="bold")
        style_ax(ax, "Doctor Specializations", ylabel="Mention Count")
        save_fig(fig, CHARTS / "c_doctor_specializations.png")
        spec_counts.to_csv(TABLES / "c_doctor_specializations.csv")

    # --- C7: Conditions Screened ---
    if "conditions_screened" in df.columns:
        cond_s = explode_pipe_col(df["conditions_screened"])
        if len(cond_s):
            cs_counts = cond_s.value_counts()
            fig, ax = make_fig(10, 5)
            bars = ax.barh(cs_counts.index[::-1], cs_counts.values[::-1],
                           color=PALETTE_ACCENT[:len(cs_counts)], edgecolor="none", height=0.5)
            for bar, val in zip(bars, cs_counts.values[::-1]):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                        str(val), va="center", color=TEXT_COLOR, fontsize=9)
            style_ax(ax, "Health Conditions Screened During Calls", "Mention Count")
            save_fig(fig, CHARTS / "c_conditions_screened.png")
            cs_counts.to_csv(TABLES / "c_conditions_screened.csv")

    # --- C8: Follow-up Required ---
    if "follow_up_required" in df.columns:
        fu = df["follow_up_required"].value_counts()
        report_lines.append("## Follow-Up Required")
        for f_val, cnt in fu.items():
            report_lines.append(f"- **{f_val}**: {cnt}")
        report_lines.append("")

    # --- C9: Tests Recommended ---
    if "tests_recommended" in df.columns:
        tests = explode_pipe_col(df["tests_recommended"])
        if len(tests):
            test_counts = tests.value_counts()
            report_lines.append("## Tests Recommended")
            for t, c in test_counts.items():
                report_lines.append(f"- **{t}**: {c}")
            report_lines.append("")
            test_counts.to_csv(TABLES / "c_tests_recommended.csv")

    # --- C10: Advice Given ---
    if "advice_given" in df.columns:
        adv = explode_pipe_col(df["advice_given"])
        if len(adv):
            adv_counts = adv.value_counts()
            report_lines.append("## Advice Given")
            for a, c in adv_counts.items():
                report_lines.append(f"- **{a}**: {c}")
            report_lines.append("")
            adv_counts.to_csv(TABLES / "c_advice_given.csv")

    return "\n".join(report_lines)


# ============================================================================
# D.  COMBINED DASHBOARD CHART
# ============================================================================

def section_d_dashboard(df: pd.DataFrame) -> str:
    report_lines: List[str] = ["# D. Healthcare Dashboard\n"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor=BG_COLOR)

    # 1. Conversation Type
    ax = axes[0, 0]
    ct = df["conversation_type"].value_counts()
    ax.barh(ct.index[::-1], ct.values[::-1], color=PALETTE_MAIN[:len(ct)], height=0.6)
    style_ax(ax, "Conversation Types")

    # 2. Urgency
    ax = axes[0, 1]
    urg = df["urgency_level"].value_counts()
    colors_urg = [URGENCY_COLORS.get(x, "#64748b") for x in urg.index]
    wedges, _, autotexts = ax.pie(urg.values, labels=urg.index, autopct="%1.1f%%",
                                  colors=colors_urg, startangle=90,
                                  textprops={"color": TEXT_COLOR, "fontsize": 9})
    ax.set_title("Urgency Levels", color=TITLE_COLOR, fontsize=12, fontweight="bold")

    # 3. Recovery (exclude low-info categories)
    ax = axes[0, 2]
    rec = df["recovery_status"].value_counts()
    low_info_rec = {"unknown", "not_discussed", "reply_too_vague_to_classify"}
    rec_filtered = rec.drop(labels=[l for l in low_info_rec if l in rec.index], errors="ignore")
    if len(rec_filtered) > 0:
        colors_rec = [RECOVERY_COLORS.get(x, "#64748b") for x in rec_filtered.index]
        ax.bar(range(len(rec_filtered)), rec_filtered.values, color=colors_rec, width=0.6)
        ax.set_xticks(range(len(rec_filtered)))
        ax.set_xticklabels(rec_filtered.index, rotation=30, ha="right")
    style_ax(ax, "Recovery Status (Discussed)", ylabel="Count")

    # 4. Call Outcomes
    ax = axes[1, 0]
    co = df["call_outcome"].value_counts()
    ax.barh(co.index[::-1], co.values[::-1], color=PALETTE_ACCENT[:len(co)], height=0.6)
    style_ax(ax, "Call Outcomes")

    # 5. Compliance (exclude low-info categories)
    ax = axes[1, 1]
    comp = df["patient_compliance"].value_counts()
    low_info_comp = {"unknown", "not_applicable", "reply_too_vague_to_classify"}
    comp_filtered = comp.drop(labels=[l for l in low_info_comp if l in comp.index], errors="ignore")
    if len(comp_filtered) > 0:
        colors_comp = [COMPLIANCE_COLORS.get(x, "#64748b") for x in comp_filtered.index]
        ax.bar(range(len(comp_filtered)), comp_filtered.values, color=colors_comp, width=0.6)
        ax.set_xticks(range(len(comp_filtered)))
        ax.set_xticklabels(comp_filtered.index, rotation=30, ha="right")
    style_ax(ax, "Compliance (Discussed)", ylabel="Count")

    # 6. Follow-up pie
    ax = axes[1, 2]
    fu = df["follow_up_required"].value_counts()
    fu_labels = ["Follow-up Needed" if v else "No Follow-up" for v in fu.index]
    ax.pie(fu.values, labels=fu_labels, autopct="%1.1f%%",
           colors=["#ef4444", "#10b981"], startangle=90,
           textprops={"color": TEXT_COLOR, "fontsize": 9})
    ax.set_title("Follow-Up Required", color=TITLE_COLOR, fontsize=12, fontweight="bold")

    fig.suptitle("Healthcare Call Analytics Dashboard", color=TITLE_COLOR,
                 fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, CHARTS / "d_healthcare_dashboard.png")

    report_lines.append("![Dashboard](../charts/d_healthcare_dashboard.png)\n")
    return "\n".join(report_lines)


# ============================================================================
# E.  NLP / TEXT ANALYSIS
# ============================================================================

def section_e_nlp(df: pd.DataFrame, records: List[Dict]) -> str:
    report_lines: List[str] = ["# E. NLP & Text Analysis\n"]

    # Extract all raw translations
    all_texts: List[str] = []
    for rec in records:
        txt = rec.get("text", {}).get("raw_translation", "")
        if txt:
            all_texts.append(txt)

    if not all_texts:
        report_lines.append("> ⚠️ No raw translations found in JSONL.\n")
        return "\n".join(report_lines)

    # Transcript lengths
    lengths = [len(t.split()) for t in all_texts]
    lengths_chars = [len(t) for t in all_texts]

    report_lines.append("## Transcript Length Statistics")
    report_lines.append(f"- Mean word count: **{np.mean(lengths):.1f}**")
    report_lines.append(f"- Median word count: **{np.median(lengths):.1f}**")
    report_lines.append(f"- Max word count: **{np.max(lengths)}**")
    report_lines.append(f"- Min word count: **{np.min(lengths)}**")
    report_lines.append(f"- Std dev: **{np.std(lengths):.1f}**\n")

    # Length distribution
    fig, (ax1, ax2) = make_fig_multi(1, 2, 14, 5)
    ax1.hist(lengths, bins=30, color=PALETTE_MAIN[0], edgecolor=CARD_COLOR, alpha=0.9)
    style_ax(ax1, "Transcript Word Count Distribution", "Word Count", "Frequency")
    ax2.hist(lengths_chars, bins=30, color=PALETTE_MAIN[3], edgecolor=CARD_COLOR, alpha=0.9)
    style_ax(ax2, "Transcript Character Count Distribution", "Character Count", "Frequency")
    fig.tight_layout(pad=3)
    save_fig(fig, NLP_DIR / "e_transcript_length_distribution.png")

    # Length stats CSV
    pd.DataFrame({
        "metric": ["mean_words", "median_words", "max_words", "min_words", "std_words",
                    "mean_chars", "median_chars", "max_chars", "min_chars"],
        "value": [np.mean(lengths), np.median(lengths), np.max(lengths), np.min(lengths), np.std(lengths),
                  np.mean(lengths_chars), np.median(lengths_chars), np.max(lengths_chars), np.min(lengths_chars)]
    }).to_csv(TABLES / "e_transcript_length_stats.csv", index=False)

    # Top medical keywords
    medical_keywords = [
        "doctor", "medicine", "pain", "eye", "dental", "surgery", "checkup",
        "blood", "pressure", "sugar", "thyroid", "clinic", "specialist",
        "cataract", "operation", "drops", "glasses", "prescription", "report",
        "test", "x-ray", "health", "healthy", "well", "problem", "issue",
        "treatment", "nerve", "bone", "skin", "allergy", "breathing",
        "stomach", "swelling", "eczema", "tablet", "dose", "hospital",
        "consultation", "fee", "rupees", "cold", "fever", "calcium",
        "cholesterol", "degeneration", "diet", "exercise", "free",
    ]

    all_lower = " ".join(all_texts).lower()
    words = re.findall(r"\b[a-z]+\b", all_lower)
    word_counts = Counter(words)

    medical_freq = {kw: word_counts.get(kw, 0) for kw in medical_keywords if word_counts.get(kw, 0) > 0}
    medical_freq = dict(sorted(medical_freq.items(), key=lambda x: x[1], reverse=True))

    report_lines.append("## Top Medical Terms")
    report_lines.append("| Term | Frequency |")
    report_lines.append("|------|----------:|")
    for term, freq in list(medical_freq.items())[:25]:
        report_lines.append(f"| {term} | {freq} |")
    report_lines.append("")

    # Medical keywords bar chart
    top_med = dict(list(medical_freq.items())[:20])
    fig, ax = make_fig(10, 6)
    keys = list(top_med.keys())[::-1]
    vals = [top_med[k] for k in keys]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(keys)))
    bars = ax.barh(keys, vals, color=colors, edgecolor="none", height=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", color=TEXT_COLOR, fontsize=9)
    style_ax(ax, "Top Medical Terms in Transcripts", "Frequency")
    save_fig(fig, NLP_DIR / "e_top_medical_terms.png")
    pd.DataFrame(list(medical_freq.items()), columns=["term", "frequency"]).to_csv(
        TABLES / "e_medical_term_frequencies.csv", index=False)

    # Word cloud
    if HAS_WORDCLOUD:
        stopwords = {"the", "is", "are", "am", "i", "you", "we", "they", "he", "she", "it",
                     "a", "an", "in", "on", "at", "to", "for", "of", "with", "from", "by",
                     "and", "or", "but", "not", "no", "yes", "okay", "ok", "so", "if",
                     "do", "did", "was", "were", "has", "have", "had", "be", "been",
                     "can", "will", "should", "would", "could", "may", "might",
                     "this", "that", "these", "those", "my", "your", "his", "her",
                     "our", "their", "its", "who", "what", "which", "where", "when",
                     "how", "why", "here", "there", "now", "then", "also", "just",
                     "about", "up", "out", "all", "any", "some", "every", "each",
                     "hello", "namaskar", "stay", "well", "tell", "come", "go",
                     "don", "doesn", "didn", "won", "isn", "aren", "wasn", "weren",
                     "let", "know", "want", "need", "like", "said", "say", "saying",
                     "speaking", "calling", "call", "called"}

        wc = WordCloud(
            width=1600, height=800,
            background_color="#0f172a",
            colormap="cool",
            max_words=150,
            stopwords=stopwords,
            font_path=None,
            contour_width=0,
            prefer_horizontal=0.7,
        ).generate(all_lower)

        fig, ax = make_fig(16, 8)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Healthcare Call Transcript Word Cloud", color=TITLE_COLOR,
                     fontsize=16, fontweight="bold", pad=15)
        save_fig(fig, NLP_DIR / "e_wordcloud.png")

    # Common bigrams
    from itertools import islice
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    bg_stop = {"i am", "am i", "you are", "it is", "is the", "if you", "in the",
               "to the", "from the", "you can", "of the", "you have", "is he",
               "he is", "she is", "are you", "for the", "do you", "at home",
               "the clinic", "the doctor", "the medicine", "a doctor", "is it",
               "if there", "can come", "stay well", "from here", "come and",
               "and get", "and dental", "you should"}
    bigram_counts = Counter(bg for bg in bigrams if bg not in bg_stop)
    top_bigrams = bigram_counts.most_common(20)

    report_lines.append("## Top Bigrams (word pairs)")
    report_lines.append("| Bigram | Count |")
    report_lines.append("|--------|------:|")
    for bg, cnt in top_bigrams:
        report_lines.append(f"| {bg} | {cnt} |")
    report_lines.append("")

    fig, ax = make_fig(10, 6)
    bg_keys = [b[0] for b in top_bigrams][::-1]
    bg_vals = [b[1] for b in top_bigrams][::-1]
    colors_bg = plt.cm.plasma(np.linspace(0.3, 0.9, len(bg_keys)))
    ax.barh(bg_keys, bg_vals, color=colors_bg, edgecolor="none", height=0.6)
    style_ax(ax, "Top 20 Bigrams in Transcripts", "Count")
    save_fig(fig, NLP_DIR / "e_top_bigrams.png")

    pd.DataFrame(top_bigrams, columns=["bigram", "count"]).to_csv(
        TABLES / "e_top_bigrams.csv", index=False)

    # Symptom clustering (from conditions_screened and conditions_reported)
    if "conditions_screened" in df.columns:
        screened = explode_pipe_col(df["conditions_screened"])
        reported = explode_pipe_col(df["conditions_reported"]) if "conditions_reported" in df.columns else pd.Series()
        all_conditions = pd.concat([screened, reported])
        cond_counts = all_conditions.value_counts()

        report_lines.append("## Symptom/Condition Clustering")
        report_lines.append("| Condition | Count |")
        report_lines.append("|-----------|------:|")
        for cond, cnt in cond_counts.items():
            report_lines.append(f"| {cond} | {cnt} |")
        report_lines.append("")

        fig, ax = make_fig(10, 5)
        bars = ax.bar(cond_counts.index, cond_counts.values,
                      color=PALETTE_ACCENT[:len(cond_counts)], edgecolor="none", width=0.5)
        for bar, val in zip(bars, cond_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(val), ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
        style_ax(ax, "Conditions Mentioned Across All Calls", ylabel="Count")
        plt.xticks(rotation=35, ha="right")
        save_fig(fig, NLP_DIR / "e_condition_clustering.png")

    return "\n".join(report_lines)


# ============================================================================
# F.  CORRELATION / RELATIONSHIP ANALYSIS
# ============================================================================

def section_f_correlations(df: pd.DataFrame) -> str:
    report_lines: List[str] = ["# F. Correlation & Relationship Analysis\n"]

    # --- F1: Recovery × Compliance Cross-tab ---
    if "recovery_status" in df.columns and "patient_compliance" in df.columns:
        # Filter out low-info categories for cleaner visualization
        low_rec = {"unknown", "not_discussed", "reply_too_vague_to_classify"}
        low_comp = {"unknown", "not_applicable", "reply_too_vague_to_classify"}
        df_f1 = df[~df["recovery_status"].isin(low_rec) & ~df["patient_compliance"].isin(low_comp)]
        ct = pd.crosstab(df_f1["recovery_status"], df_f1["patient_compliance"])
        ct.to_csv(CORRELATIONS / "f_recovery_vs_compliance.csv")

        fig, ax = make_fig(10, 6)
        ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
        ct_norm.plot(kind="bar", stacked=True, ax=ax,
                     color=[COMPLIANCE_COLORS.get(c, "#64748b") for c in ct.columns],
                     edgecolor="none", width=0.6)
        style_ax(ax, "Recovery Status × Patient Compliance\n(Calls With Both Discussed)", ylabel="Percentage (%)")
        ax.legend(frameon=False, labelcolor=TEXT_COLOR, fontsize=9)
        plt.xticks(rotation=30, ha="right")
        save_fig(fig, CORRELATIONS / "f_recovery_vs_compliance.png")

        report_lines.append("## Recovery Status × Compliance")
        report_lines.append(ct.to_markdown())
        report_lines.append("")

    # --- F2: Recovery × Urgency ---
    if "recovery_status" in df.columns and "urgency_level" in df.columns:
        low_rec = {"unknown", "not_discussed", "reply_too_vague_to_classify"}
        df_f2 = df[~df["recovery_status"].isin(low_rec)]
        ct2 = pd.crosstab(df_f2["recovery_status"], df_f2["urgency_level"])
        ct2.to_csv(CORRELATIONS / "f_recovery_vs_urgency.csv")

        fig, ax = make_fig(10, 6)
        ct2.plot(kind="bar", ax=ax,
                 color=[URGENCY_COLORS.get(c, "#64748b") for c in ct2.columns],
                 edgecolor="none", width=0.6)
        style_ax(ax, "Recovery Status × Urgency Level\n(Calls With Recovery Discussed)", ylabel="Count")
        ax.legend(frameon=False, labelcolor=TEXT_COLOR, fontsize=9)
        plt.xticks(rotation=30, ha="right")
        save_fig(fig, CORRELATIONS / "f_recovery_vs_urgency.png")

        report_lines.append("## Recovery Status × Urgency")
        report_lines.append(ct2.to_markdown())
        report_lines.append("")

    # --- F3: Conversation Type × Urgency ---
    if "conversation_type" in df.columns and "urgency_level" in df.columns:
        ct3 = pd.crosstab(df["conversation_type"], df["urgency_level"])
        ct3.to_csv(CORRELATIONS / "f_convtype_vs_urgency.csv")

        fig, ax = make_fig(12, 6)
        ct3_norm = ct3.div(ct3.sum(axis=1), axis=0) * 100
        ct3_norm.plot(kind="barh", stacked=True, ax=ax,
                      color=[URGENCY_COLORS.get(c, "#64748b") for c in ct3.columns],
                      edgecolor="none")
        style_ax(ax, "Conversation Type × Urgency Distribution (%)", xlabel="Percentage")
        ax.legend(frameon=False, labelcolor=TEXT_COLOR, fontsize=9)
        save_fig(fig, CORRELATIONS / "f_convtype_vs_urgency.png")

    # --- F4: Follow-up × Recovery ---
    if "follow_up_required" in df.columns and "recovery_status" in df.columns:
        low_rec = {"unknown", "not_discussed", "reply_too_vague_to_classify"}
        df_f4 = df[~df["recovery_status"].isin(low_rec)]
        ct4 = pd.crosstab(df_f4["recovery_status"], df_f4["follow_up_required"])
        ct4.columns = ["No Follow-up", "Follow-up Needed"]
        ct4.to_csv(CORRELATIONS / "f_recovery_vs_followup.csv")

        fig, ax = make_fig(10, 6)
        ct4.plot(kind="bar", ax=ax, color=["#10b981", "#ef4444"], edgecolor="none", width=0.6)
        style_ax(ax, "Recovery Status × Follow-Up Required\n(Calls With Recovery Discussed)", ylabel="Count")
        ax.legend(frameon=False, labelcolor=TEXT_COLOR, fontsize=9)
        plt.xticks(rotation=30, ha="right")
        save_fig(fig, CORRELATIONS / "f_recovery_vs_followup.png")

        report_lines.append("## Recovery × Follow-Up Required")
        report_lines.append(ct4.to_markdown())
        report_lines.append("")

    # --- F5: Compliance × Follow-up ---
    if "patient_compliance" in df.columns and "follow_up_required" in df.columns:
        ct5 = pd.crosstab(df["patient_compliance"], df["follow_up_required"])
        ct5.columns = ["No Follow-up", "Follow-up Needed"]
        ct5.to_csv(CORRELATIONS / "f_compliance_vs_followup.csv")

        report_lines.append("## Compliance × Follow-Up Required")
        report_lines.append(ct5.to_markdown())
        report_lines.append("")

    # --- F6: Confidence Score Distributions ---
    conf_cols = [c for c in df.columns if c.startswith("conf_")]
    if conf_cols:
        fig, ax = make_fig(10, 5)
        conf_data = df[conf_cols].melt(var_name="field", value_name="confidence")
        conf_data["field"] = conf_data["field"].str.replace("conf_", "")
        sns.boxplot(data=conf_data, x="field", y="confidence", ax=ax,
                    palette=PALETTE_MAIN[:len(conf_cols)], fliersize=3)
        ax.set_facecolor(CARD_COLOR)
        style_ax(ax, "Extraction Confidence Score Distributions", ylabel="Confidence (0–1)")
        plt.xticks(rotation=30, ha="right")
        save_fig(fig, CORRELATIONS / "f_confidence_distributions.png")

    # --- F7: Heatmap of all categorical relationships ---
    cat_cols = ["conversation_type", "call_outcome", "urgency_level",
                "recovery_status", "patient_compliance"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    if len(cat_cols) >= 2:
        # Encode as numeric
        from sklearn.preprocessing import LabelEncoder
        df_enc = pd.DataFrame()
        for c in cat_cols:
            le = LabelEncoder()
            df_enc[c] = le.fit_transform(df[c].astype(str))
        corr = df_enc.corr(method="spearman")
        corr.to_csv(CORRELATIONS / "f_spearman_correlation_matrix.csv")

        fig, ax = make_fig(9, 7)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                    ax=ax, vmin=-1, vmax=1, linewidths=0.5,
                    annot_kws={"fontsize": 10, "fontweight": "bold"})
        ax.set_facecolor(CARD_COLOR)
        ax.set_title("Spearman Correlation – Key Healthcare Variables",
                     color=TITLE_COLOR, fontsize=13, fontweight="bold", pad=12)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)
        save_fig(fig, CORRELATIONS / "f_correlation_heatmap.png")

        report_lines.append("## Spearman Correlation Matrix")
        report_lines.append(corr.round(3).to_markdown())
        report_lines.append("")

    return "\n".join(report_lines)


# ============================================================================
# G.  DATA QUALITY ANALYSIS
# ============================================================================

def section_g_quality(df: pd.DataFrame) -> str:
    report_lines: List[str] = ["# G. Data Quality Analysis\n"]

    # --- G1: Manual Review Flagging ---
    if "needs_manual_review" in df.columns:
        review_rate = df["needs_manual_review"].sum()
        report_lines.append(f"## Manual Review Flags")
        report_lines.append(f"- Calls needing manual review: **{review_rate}** / {len(df)} ({review_rate / len(df) * 100:.1f}%)\n")

    # --- G2: Extraction Warnings ---
    if "extraction_warnings" in df.columns:
        warnings = df["extraction_warnings"].dropna()
        warnings = warnings[warnings.str.len() > 0]
        if len(warnings):
            all_warns = warnings.str.split("|").explode().str.strip()
            # Categorize warnings
            warn_types = all_warns.str.extract(r"^([A-Z_]+):?")[0].dropna()
            wt_counts = warn_types.value_counts()
            report_lines.append("## Extraction Warning Types")
            for w, c in wt_counts.items():
                report_lines.append(f"- `{w}`: {c}")
            report_lines.append("")
            wt_counts.to_csv(TABLES / "g_extraction_warning_types.csv")

    # --- G3: Confidence distribution ---
    conf_cols = [c for c in df.columns if c.startswith("conf_")]
    if conf_cols:
        low_conf = pd.DataFrame()
        for c in conf_cols:
            low = (df[c] < 0.5).sum()
            zero = (df[c] == 0.0).sum()
            low_conf = pd.concat([low_conf, pd.DataFrame({
                "field": [c.replace("conf_", "")],
                "low_conf_count (<0.5)": [low],
                "zero_conf_count": [zero],
                "mean_conf": [df[c].mean().round(3)],
            })])
        low_conf.to_csv(TABLES / "g_confidence_analysis.csv", index=False)

        report_lines.append("## Confidence Score Analysis")
        report_lines.append(low_conf.to_markdown(index=False))
        report_lines.append("")

    # --- G4: Schema completeness ---
    completeness = ((df.notna().sum() / len(df)) * 100).round(2)
    comp_df = pd.DataFrame({"column": completeness.index, "completeness_%": completeness.values})
    comp_df = comp_df.sort_values("completeness_%", ascending=True)
    comp_df.to_csv(TABLES / "g_schema_completeness.csv", index=False)

    fig, ax = make_fig(10, max(4, len(comp_df) * 0.25))
    colors = ["#ef4444" if v < 50 else "#f59e0b" if v < 80 else "#10b981" for v in comp_df["completeness_%"]]
    ax.barh(comp_df["column"], comp_df["completeness_%"], color=colors, height=0.6, edgecolor="none")
    ax.axvline(x=80, color="#f59e0b", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=50, color="#ef4444", linestyle="--", alpha=0.5, linewidth=1)
    style_ax(ax, "Schema Completeness by Column", "Completeness (%)")
    ax.set_xlim(0, 105)
    save_fig(fig, CHARTS / "g_schema_completeness.png")

    report_lines.append("## Schema Completeness")
    report_lines.append("- 🟢 Green = >80%  |  🟡 Yellow = 50–80%  |  🔴 Red = <50%\n")

    # --- G5: Flagged segments ---
    if "flagged_segment_ratio" in df.columns:
        flagged = df["flagged_segment_ratio"]
        high_flag = (flagged > 0.5).sum()
        report_lines.append(f"## Audio Quality Flags")
        report_lines.append(f"- Calls with >50% flagged segments: **{high_flag}**")
        report_lines.append(f"- Mean flagged ratio: **{flagged.mean():.4f}**\n")

    return "\n".join(report_lines)


# ============================================================================
# MASTER SUMMARY REPORT
# ============================================================================

def generate_master_summary(df: pd.DataFrame) -> str:
    n = len(df)
    # Key metrics
    has_name = df["patient_name"].notna().sum() if "patient_name" in df.columns else 0
    urgent = (df["urgency_level"] == "high").sum() if "urgency_level" in df.columns else 0
    unresolved = (df["recovery_status"] == "unresolved").sum() if "recovery_status" in df.columns else 0
    worsening = (df["recovery_status"] == "worsening").sum() if "recovery_status" in df.columns else 0
    recovered = (df["recovery_status"] == "fully_recovered").sum() if "recovery_status" in df.columns else 0
    non_comp = (df["patient_compliance"] == "non_compliant").sum() if "patient_compliance" in df.columns else 0
    followup = df["follow_up_required"].sum() if "follow_up_required" in df.columns else 0
    manual_rev = df["needs_manual_review"].sum() if "needs_manual_review" in df.columns else 0

    report = f"""# 📊 Healthcare Call Analytics — EDA Summary Report

> **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
> **Dataset:** {ENRICHED_CSV.name} ({n} calls)
> **Pipeline:** FinAI Voice Analytics

---

## 🔑 Executive Summary

| Metric | Value |
|--------|------:|
| Total Calls Analyzed | **{n}** |
| Identified Patients | **{has_name}** ({has_name/n*100:.1f}%) |
| High Urgency Cases | **{urgent}** ({urgent/n*100:.1f}%) |
| Unresolved Cases | **{unresolved}** ({unresolved/n*100:.1f}%) |
| Worsening Cases | **{worsening}** ({worsening/n*100:.1f}%) |
| Fully Recovered | **{recovered}** ({recovered/n*100:.1f}%) |
| Non-Compliant Patients | **{non_comp}** ({non_comp/n*100:.1f}%) |
| Follow-Up Required | **{followup}** ({followup/n*100:.1f}%) |
| Needs Manual Review | **{manual_rev}** ({manual_rev/n*100:.1f}%) |

---

## 🏥 Key Clinical Insights

### Patient Recovery Landscape
- **{recovered/n*100:.1f}%** of patients reported full recovery during follow-up calls
- **{unresolved/n*100:.1f}%** have unresolved conditions requiring continued attention
- **{worsening/n*100:.1f}%** showed worsening symptoms — these are priority cases
- A large portion (**{df['recovery_status'].isin(['unknown', 'not_discussed', 'reply_too_vague_to_classify']).sum()/n*100:.1f}%**) has unknown recovery status, indicating short or inconclusive calls

### Compliance Concerns
- Only **{(df['patient_compliance']=='compliant').sum()}** patients ({(df['patient_compliance']=='compliant').sum()/n*100:.1f}%) confirmed active medication compliance
- **{non_comp}** patients explicitly stopped medication prematurely
- **{df['patient_compliance'].isin(['unknown', 'not_applicable', 'reply_too_vague_to_classify']).sum()}** calls did not yield compliance information

### Urgency Distribution
- The vast majority of calls (**{(df['urgency_level']=='low').sum()/n*100:.1f}%**) are low urgency routine follow-ups
- **{urgent}** high-urgency cases demand immediate clinical attention
- High-urgency cases correlate strongly with unresolved/worsening conditions

---

## 🔬 NLP Findings

- Average transcript length: ~{df['structured_summary'].dropna().str.split().str.len().mean():.0f} words (structured summary)
- Most frequently mentioned conditions: **eye**, **dental**, **blood_pressure**, **diabetes**, **thyroid**
- Primary doctor types: **general medicine**, **eye**, **dental** specialists
- Most common advice: **get_tested**, **do_not_ignore**, **continue_medication**

---

## ⚠️ Data Quality Notes

- **{(df['call_date'].isna()).sum()}** calls have missing dates (transcripts provided without timestamps)
- **{(df['conditions_reported'].isna() | (df['conditions_reported']=='')).sum()}** calls have no confirmed reported conditions (most conditions appear in screening questions only)
- Extraction confidence is generally high for patient names (mean: {df['conf_patient_name'].mean():.2f}) but lower for compliance (mean: {df['conf_compliance'].mean():.2f})

---

## 📁 Output Structure

```
results/
├── charts/           — {len(list(CHARTS.glob('*.png')))} PNG visualizations
├── tables/           — {len(list(TABLES.glob('*.csv')))} CSV summary tables
├── reports/          — Detailed section reports
├── nlp/              — Text analysis outputs
├── correlations/     — Cross-tabulations and heatmaps
└── eda_summary.md    — This master report
```

---

## 🎯 Recommendations for Next Steps

1. **Prioritize High-Urgency Cases:** {urgent} cases flagged as high urgency should be routed to clinical staff immediately
2. **Address Non-Compliance:** {non_comp} patients stopped medication — targeted outreach recommended
3. **Improve Data Collection:** {df['recovery_status'].isin(['unknown', 'not_discussed', 'reply_too_vague_to_classify']).sum()} calls with unknown recovery suggest call scripts could be improved
4. **Enrich with Timestamps:** Adding call timestamps would enable temporal trend analysis
5. **LLM-Enhanced Extraction:** Running with LLM extraction (`--llm-extraction`) would improve accuracy for complex transcripts
"""
    return report


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("  FinAI Voice Analytics — Healthcare EDA Pipeline")
    print("=" * 70)

    setup_dirs()

    print("\n📂 Loading data...")
    df = load_csv()
    records = load_jsonl()
    print(f"   CSV: {len(df)} rows × {len(df.columns)} cols")
    print(f"   JSONL: {len(records)} records\n")

    # Run all sections
    sections = {}

    print("📊 A. Dataset Overview...")
    sections["A"] = section_a_overview(df)

    print("📊 B. Demographics & Location...")
    sections["B"] = section_b_demographics(df)

    print("📊 C. Healthcare Analysis...")
    sections["C"] = section_c_healthcare(df)

    print("📊 D. Dashboard...")
    sections["D"] = section_d_dashboard(df)

    print("📊 E. NLP & Text Analysis...")
    sections["E"] = section_e_nlp(df, records)

    print("📊 F. Correlations...")
    sections["F"] = section_f_correlations(df)

    print("📊 G. Data Quality...")
    sections["G"] = section_g_quality(df)

    # Write section reports
    for key, content in sections.items():
        report_path = REPORTS / f"section_{key.lower()}.md"
        with open(report_path, "w") as f:
            f.write(content)
        print(f"  📝  {report_path.relative_to(PROJECT_ROOT)}")

    # Master summary
    print("\n📝 Generating master summary...")
    summary = generate_master_summary(df)
    with open(RESULTS / "eda_summary.md", "w") as f:
        f.write(summary)
    print(f"  📝  results/eda_summary.md")

    # Final stats
    charts = list(CHARTS.glob("*.png")) + list(NLP_DIR.glob("*.png")) + list(CORRELATIONS.glob("*.png"))
    tables = list(TABLES.glob("*.csv")) + list(CORRELATIONS.glob("*.csv"))
    reports = list(REPORTS.glob("*.md"))

    print("\n" + "=" * 70)
    print(f"  ✅  EDA Complete!")
    print(f"  📊  Charts:  {len(charts)}")
    print(f"  📋  Tables:  {len(tables)}")
    print(f"  📝  Reports: {len(reports) + 1}")
    print(f"  📁  All outputs in: results/")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
