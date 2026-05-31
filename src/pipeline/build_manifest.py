from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


# Columns we prefer to keep for downstream pipeline + reporting
KEEP_COLS = [
    # identifiers
    "Unique Id",
    "Lead ID",
    "Recording Filename",

    # audio
    "recording_location",
    "length_in_sec",

    # timestamps
    "Date",
    "Hour",
    "Start Ts",
    "End Ts",
    "entry_date",
    "modify_date",

    # agent / ops metadata (useful later)
    "Interaction Type",
    "User",
    "Agent Name",
    "User Group",
    "List Name",
    "List ID",
    "Campaign Name",
    "Dialer Disposition",
    "Disposition Category",
    "Agent Disposition",
    "Hangup Reason",
    "status",
    "bucket_category_name",
]


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_manifest(input_xlsx: Path, out_dir: Path) -> pd.DataFrame:
    if not input_xlsx.exists():
        raise FileNotFoundError(f"Input file not found: {input_xlsx}")

    # Read Excel
    df = pd.read_excel(input_xlsx, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    # Ensure required column exists
    if "recording_location" not in df.columns:
        raise ValueError("Missing required column: recording_location")

    # Keep only columns that exist (don’t crash if some are absent)
    keep = [c for c in KEEP_COLS if c in df.columns]
    df = df[keep].copy()

    # Basic cleanup: normalize recording_location
    df["recording_location"] = df["recording_location"].astype(str).str.strip()
    df.loc[df["recording_location"].isin(["", "nan", "None", "null"]), "recording_location"] = pd.NA

    # Parse timestamps if present
    for ts_col in ["Start Ts", "End Ts", "entry_date", "modify_date"]:
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", dayfirst=True)

    # length_in_sec numeric
    if "length_in_sec" in df.columns:
        df["length_in_sec"] = pd.to_numeric(df["length_in_sec"], errors="coerce")

    # Create a stable call_id
    uid_col = _first_existing(df, ["Unique Id", "Lead ID"])
    if uid_col:
        df["call_id"] = df[uid_col].astype(str)
    else:
        # fallback if neither exists (should be rare)
        parts = []
        if "Recording Filename" in df.columns:
            parts.append(df["Recording Filename"].astype(str))
        if "Start Ts" in df.columns:
            parts.append(df["Start Ts"].astype(str))
        df["call_id"] = ("|".join(["fallback"] + ["x"] * len(parts))) if not parts else (parts[0] if len(parts) == 1 else (parts[0] + "|" + parts[1]))

    # Drop rows with no recording URL
    before = len(df)
    df = df.dropna(subset=["recording_location"]).copy()
    after_dropna = len(df)

    # De-duplication strategy:
    # Prefer Unique Id; otherwise fall back to (Recording Filename + Start Ts)
    if "Unique Id" in df.columns:
        df = df.drop_duplicates(subset=["Unique Id"], keep="first")
    elif "Recording Filename" in df.columns and "Start Ts" in df.columns:
        df = df.drop_duplicates(subset=["Recording Filename", "Start Ts"], keep="first")
    else:
        df = df.drop_duplicates(subset=["recording_location"], keep="first")

    after_dedup = len(df)

    # Add simple flags useful later
    df["has_audio_url"] = df["recording_location"].notna()
    if "length_in_sec" in df.columns:
        df["is_long_call"] = df["length_in_sec"].fillna(0) > 20 * 60  # >20 minutes

    # Reorder: call_id first
    cols = ["call_id"] + [c for c in df.columns if c != "call_id"]
    df = df[cols]

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "calls_manifest.parquet", index=False)
    df.to_csv(out_dir / "calls_manifest.csv", index=False)

    print("Manifest built ✅")
    print(f"Input rows: {before}")
    print(f"After dropping missing URLs: {after_dropna}")
    print(f"After de-duplication: {after_dedup}")
    print(f"Saved: {out_dir / 'calls_manifest.parquet'}")
    print(f"Saved: {out_dir / 'calls_manifest.csv'}")

    return df


def main():
    ap = argparse.ArgumentParser(description="Build calls manifest from Excel export")
    ap.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to Excel export (.xlsx)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="data/manifest",
        help="Output directory for manifest files",
    )
    args = ap.parse_args()

    build_manifest(Path(args.input), Path(args.out))


if __name__ == "__main__":
    main()
