from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils.sqlite_checkpoint import SQLiteCheckpoint


AUDIO_EXTS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".ogg",
    ".flac"
}


def normalize_date_folder(folder_name: str) -> str:
    try:
        d = pd.to_datetime(folder_name, errors="coerce")

        if pd.notna(d):
            return d.date().isoformat()

    except Exception:
        pass

    return folder_name


def ffmpeg_convert(in_path: Path, out_path: Path) -> None:
    """
    Convert audio into:
    - mono
    - 16kHz
    - PCM 16-bit WAV

    Optimized for Whisper ASR.
    """

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tmp_path = out_path.with_suffix(
        out_path.suffix + ".part"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",

        "-i",
        str(in_path),

        "-vn",

        "-ac", "1",

        "-ar", "16000",

        "-c:a", "pcm_s16le",

        "-f", "wav",

        str(tmp_path),
    ]

    print(f"\nConverting: {in_path.name}")

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if res.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed:\n{res.stderr}"
        )

    tmp_path.replace(out_path)

    print(f"Finished: {out_path.name}")


def main():

    ap = argparse.ArgumentParser(
        description="Normalize audio for Whisper"
    )

    ap.add_argument(
        "--in_dir",
        type=str,
        default="data/audio_raw"
    )

    ap.add_argument(
        "--out_dir",
        type=str,
        default="data/audio_wav16k"
    )

    ap.add_argument(
        "--db",
        type=str,
        default="checkpoints/normalization.sqlite"
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=0
    )

    args = ap.parse_args()

    in_dir = Path(args.in_dir)

    out_dir = Path(args.out_dir)

    db_path = Path(args.db)

    if not in_dir.exists():
        raise FileNotFoundError(
            f"Missing input folder: {in_dir}"
        )

    ckpt = SQLiteCheckpoint(
        db_path=db_path,
        table="normalization"
    )

    files = [
        p for p in in_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTS
    ]

    files.sort()

    if args.limit > 0:
        files = files[:args.limit]

    if len(files) == 0:
        print("No audio files found.")
        return

    ok = 0
    skipped = 0
    failed = 0

    for in_path in tqdm(
        files,
        desc="Normalizing"
    ):

        key = in_path.stem

        if ckpt.is_done(key):
            skipped += 1
            continue

        date_folder = normalize_date_folder(
            in_path.parent.name
        )

        out_path = (
            out_dir
            / date_folder
            / f"{key}.wav"
        )

        try:

            ffmpeg_convert(
                in_path,
                out_path
            )

            ckpt.upsert(
                key,
                status="done",
                http_status=200,
                bytes_=out_path.stat().st_size,
                path=str(out_path),
                url=str(in_path)
            )

            ok += 1

        except Exception as e:

            ckpt.upsert(
                key,
                status="failed",
                error=str(e)[:500],
                path=str(out_path),
                url=str(in_path)
            )

            failed += 1

            print("\nFAILED:")
            print(str(e))

    print("\nNormalization Complete ✅")

    print(f"Success: {ok}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()