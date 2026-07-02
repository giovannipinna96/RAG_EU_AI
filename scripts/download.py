"""Download the EU AI Act (Regulation 2024/1689) HTML from EUR-Lex.

NOTE: EUR-Lex serves this document via an async renderer that returns HTTP 202
with an empty body until generation completes, and gates non-browser clients.
This script polls and validates the payload, but if it keeps failing use the
Hugging Face mirror instead: `uv run python scripts/download_hf.py`.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx

URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689"
DEST = Path("data/raw/eu_ai_act.html")
MIN_BYTES = 200_000  # the full Act is multiple MB; anything smaller is a stub
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def download(max_polls: int = 10, delay: float = 3.0) -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size >= MIN_BYTES:
        print(f"Already downloaded: {DEST} ({DEST.stat().st_size} bytes)")
        return

    print("Downloading EU AI Act from EUR-Lex...")
    with httpx.Client(follow_redirects=True, timeout=60, headers={"User-Agent": USER_AGENT}) as c:
        for attempt in range(max_polls):
            resp = c.get(URL)
            if resp.status_code == 200 and len(resp.content) >= MIN_BYTES:
                DEST.write_text(resp.text, encoding="utf-8")
                print(f"Saved to {DEST} ({len(resp.text) // 1024} KB)")
                return
            print(f"  poll {attempt}: status={resp.status_code} "
                  f"bytes={len(resp.content)} — retrying")
            time.sleep(delay)

    raise SystemExit(
        "EUR-Lex did not return the full document (likely 202/anti-bot). "
        "Use the Hugging Face mirror instead:\n"
        "  uv run python scripts/download_hf.py && uv run python scripts/ingest_hf.py"
    )


if __name__ == "__main__":
    download()
