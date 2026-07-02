"""Download the `jeroenherczeg/eu-ai-act` parquet from the Hugging Face Hub.

A small (~548 KB) test corpus: the real EU AI Act already parsed into structured
chunks. Reliable from Demetra (HF CDN, no EUR-Lex 202 anti-bot gate). Use this to
exercise the full pipeline; see scripts/ingest_hf.py.
"""

from __future__ import annotations

from pathlib import Path

from src.ingestion.hf_loader import HF_REPO, PARQUET_FILE

DEST = Path("data/raw") / PARQUET_FILE


def download() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size > 0:
        print(f"Already downloaded: {DEST} ({DEST.stat().st_size} bytes)")
        return

    from huggingface_hub import hf_hub_download

    print(f"Downloading {HF_REPO}/{PARQUET_FILE} from the Hugging Face Hub...")
    path = hf_hub_download(repo_id=HF_REPO, filename=PARQUET_FILE, repo_type="dataset")
    data = Path(path).read_bytes()
    DEST.write_bytes(data)
    print(f"Saved to {DEST} ({len(data)} bytes)")


if __name__ == "__main__":
    download()
