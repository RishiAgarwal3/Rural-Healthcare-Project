from pathlib import Path
import json
import sqlite3

from faster_whisper import WhisperModel
from tqdm import tqdm


# ----------- PATHS -----------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIO_ROOT = PROJECT_ROOT / "data/audio_wav16k"
TRANSCRIPTS_ROOT = PROJECT_ROOT / "data/transcripts"
CHECKPOINT_DB = PROJECT_ROOT / "checkpoints/transcription.sqlite"


# ----------- CHECKPOINT SETUP -----------

def init_db():
    conn = sqlite3.connect(CHECKPOINT_DB)

    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS transcription (
            path TEXT PRIMARY KEY,
            status TEXT,
            error TEXT
        )
    """)

    conn.commit()

    return conn


def already_done(conn, path):
    cur = conn.cursor()

    cur.execute(
        "SELECT status FROM transcription WHERE path=?",
        (str(path),)
    )

    row = cur.fetchone()

    return row and row[0] == "done"


def mark_status(conn, path, status, error=None):
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO transcription (
            path,
            status,
            error
        )
        VALUES (?, ?, ?)
    """, (
        str(path),
        status,
        error
    ))

    conn.commit()


# ----------- WHISPER HELPERS -----------

def transcribe_bengali(model, wav_path):
    segments, info = model.transcribe(
        str(wav_path),

        language="bn",
        task="transcribe",

        vad_filter=True,

        beam_size=7,
        temperature=0.0,

        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,

        condition_on_previous_text=False
    )

    return list(segments), info


# ----------- BUILD SEGMENTS -----------

def build_segment_objects(segments_bn):
    final_segments = []

    for seg in segments_bn:
        final_segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text_bn": seg.text.strip()
        })

    return final_segments


# ----------- MAIN -----------

def run():

    print("A) Initializing DB...")
    conn = init_db()

    print("B) Ensuring transcripts folder exists...")
    TRANSCRIPTS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("C) Scanning wav files...")

    wav_files = sorted(
        AUDIO_ROOT.rglob("*.wav")
    )

    print(f"D) Found {len(wav_files)} wav files")

    print("E) Loading Whisper model...")

    model = WhisperModel(
        "large-v3",
        device="cpu",
        compute_type="int8"
    )

    print("F) Model loaded successfully")

    for wav in tqdm(wav_files, desc="Transcribing"):

        print(f"\n→ Processing: {wav.name}")

        if already_done(conn, wav):
            print("   Skipping (already done)")
            continue

        try:

            # ---------- DATE-BASED JSONL ----------

            date_folder = wav.parent.name

            transcript_path = (
                TRANSCRIPTS_ROOT
                / f"{date_folder}.jsonl"
            )

            transcript_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # ---------- CALL ID ----------

            call_id = wav.stem

            # ---------- TRANSCRIBE ----------

            print("   Running Bengali transcription...")

            segments_bn, info = transcribe_bengali(
                model,
                wav
            )

            # ---------- BUILD SEGMENTS ----------

            final_segments = build_segment_objects(
                segments_bn
            )

            # ---------- RECORD ----------

            record = {

                "call_id": call_id,

                "path": str(
                    wav.relative_to(PROJECT_ROOT)
                ),

                "duration": round(
                    info.duration,
                    2
                ),

                "language": info.language,

                "segment_count": len(final_segments),

                "segments": final_segments
            }

            # ---------- WRITE JSONL ----------

            with transcript_path.open(
                "a",
                encoding="utf-8"
            ) as f:

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

            mark_status(
                conn,
                wav,
                "done"
            )

            print("   Done")

        except Exception as e:

            print("   FAILED:", str(e))

            mark_status(
                conn,
                wav,
                "failed",
                str(e)[:500]
            )

    conn.close()

    print("G) Finished run()")


if __name__ == "__main__":
    run()
