from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import httpx
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm

from src.utils.sqlite_checkpoint import SQLiteCheckpoint


DEFAULT_HEADERS = {
    # This matched your successful curl approach (browser-like UA)
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*",
    "Connection": "keep-alive",
}


def _safe_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:200]


def _infer_ext_from_url(url: str) -> str:
    path = urlparse(url).path
    if "." in path:
        ext = path.rsplit(".", 1)[-1].lower()
        if 1 <= len(ext) <= 5:
            return "." + ext
    return ".bin"


def _date_folder(row: pd.Series) -> str:
    # Prefer manifest's Date column; normalize to YYYY-MM-DD
    if "Date" in row and pd.notna(row["Date"]):
        d = pd.to_datetime(row["Date"], errors="coerce")
        if pd.notna(d):
            return d.date().isoformat()

    # fallback: try Start Ts date
    if "Start Ts" in row and pd.notna(row["Start Ts"]):
        d = pd.to_datetime(row["Start Ts"], errors="coerce")
        if pd.notna(d):
            return d.date().isoformat()

    return "unknown_date"


@retry(wait=wait_random_exponential(min=1, max=15), stop=stop_after_attempt(5))
def _download_one(url: str, out_path: Path, timeout_s: float = 60.0) -> tuple[int, int]:
    """
    Downloads url to out_path via streaming.
    Returns (http_status, bytes_written).
    Retries are handled by tenacity for exceptions / transient failures.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(
        headers=DEFAULT_HEADERS,
        follow_redirects=True,
        timeout=httpx.Timeout(timeout_s, connect=20.0),
    ) as client:
        with client.stream("GET", url) as r:
            status = r.status_code
            r.raise_for_status()

            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            bytes_written = 0

            with open(tmp_path, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)

            tmp_path.replace(out_path)
            return status, bytes_written


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if manifest_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(manifest_path)
    elif manifest_path.suffix.lower() == ".csv":
        df = pd.read_csv(manifest_path)
    else:
        raise ValueError("Manifest must be .parquet or .csv")

    required = {"call_id", "recording_location"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    return df


def main():
    ap = argparse.ArgumentParser(description="Download call recordings from manifest")
    ap.add_argument("--manifest", type=str, default="data/manifest/calls_manifest.parquet")
    ap.add_argument("--out_dir", type=str, default="data/audio_raw")
    ap.add_argument("--db", type=str, default="checkpoints/downloads.sqlite")
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    db_path = Path(args.db)

    df = load_manifest(manifest_path)

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    ckpt = SQLiteCheckpoint(db_path=db_path, table="downloads")

    total = len(df)
    skipped = 0
    ok = 0
    failed = 0

    for _, row in tqdm(df.iterrows(), total=total, desc="Downloading"):
        call_id = str(row["call_id"])
        url = str(row["recording_location"]).strip()

        if ckpt.is_done(call_id):
            skipped += 1
            continue

        # Decide output path
        date_folder = _date_folder(row)
        ext = _infer_ext_from_url(url)
        filename = _safe_filename(call_id) + ext
        out_path = out_dir / date_folder / filename

        # If file exists already, mark done (safety)
        if out_path.exists() and out_path.stat().st_size > 0:
            ckpt.upsert(call_id, status="done", http_status=200, bytes_=out_path.stat().st_size, path=str(out_path), url=url)
            ok += 1
            continue

        ckpt.upsert(call_id, status="started", path=str(out_path), url=url)

        try:
            status, bytes_written = _download_one(url=url, out_path=out_path, timeout_s=args.timeout)
            ckpt.upsert(call_id, status="done", http_status=status, bytes_=bytes_written, path=str(out_path), url=url)
            ok += 1
        except Exception as e:
            # clean partial
            part = out_path.with_suffix(out_path.suffix + ".part")
            if part.exists():
                try:
                    part.unlink()
                except Exception:
                    pass

            ckpt.upsert(call_id, status="failed", http_status=None, bytes_=0, path=str(out_path), url=url, error=str(e)[:500])
            failed += 1

    print("Download stage finished ✅")
    print(f"Total considered: {total}")
    print(f"Downloaded/marked done: {ok}")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed: {failed}")
    print(f"Checkpoint DB: {db_path}")


if __name__ == "__main__":
    main()
