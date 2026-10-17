"""
Build the knowledge base from real Alibaba Cloud help-center pages.

    venv/bin/python -m scripts.build_kb              # fetch + chunk + embed
    venv/bin/python -m scripts.build_kb --no-embed   # fetch + chunk only (BM25 works)
    venv/bin/python -m scripts.build_kb --embed-only # re-embed existing chunks

Outputs data/kb/chunks.json and data/kb/vectors.npy. Raw HTML is cached in
data/kb/cache/ (gitignored) so re-chunking never re-downloads.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.kb import KB_DIR, chunk_article, extract_article  # noqa: E402

CACHE = KB_DIR / "cache"
UA = "TicketCoach-KB-builder/1.0 (+https://github.com/xungirl/TicketCoach)"


def fetch(url: str) -> tuple:
    """(final_url, html), cached on disk by URL."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    path = CACHE / f"{key}.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        return d["final_url"], d["html"]
    r = httpx.get(url, headers={"User-Agent": UA}, follow_redirects=True, timeout=30)
    r.raise_for_status()
    path.write_text(json.dumps({"final_url": str(r.url), "html": r.text}, ensure_ascii=False), encoding="utf-8")
    time.sleep(1)  # be polite to the help center
    return str(r.url), r.text


def build_chunks() -> list:
    sources = json.loads((KB_DIR / "sources.json").read_text(encoding="utf-8"))
    chunks, seen = [], set()
    for src in sources:
        try:
            final_url, html = fetch(src["url"])
        except Exception as e:
            print(f"  ✗ {src['url']}: {e}")
            continue
        art = extract_article(html)
        if final_url in seen or not art["sections"]:
            print(f"  - skip {src['url']} ({'duplicate' if final_url in seen else 'no article body'})")
            continue
        seen.add(final_url)
        doc_chunks = chunk_article(art)
        for c in doc_chunks:
            c.update(url=final_url, title=art["title"], business_type=src["business_type"])
        chunks.extend(doc_chunks)
        print(f"  ✓ {art['title'][:40]:40s} {len(doc_chunks):3d} chunks")
    return chunks


def embed_chunks(chunks: list) -> None:
    import numpy as np
    from core import llm

    vecs = np.asarray(llm.embed([c["text"] for c in chunks]), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
    np.save(KB_DIR / "vectors.npy", vecs)
    print(f"embedded {len(chunks)} chunks -> vectors.npy {vecs.shape}")
    print("embedding usage:", llm.METRICS.snapshot()["steps"].get("embed"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--embed-only", action="store_true")
    args = ap.parse_args()

    chunks_path = KB_DIR / "chunks.json"
    if args.embed_only:
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    else:
        chunks = build_chunks()
        chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
        docs = len({c["url"] for c in chunks})
        print(f"\n{len(chunks)} chunks from {docs} docs -> {chunks_path}")
        # Old vectors no longer line up with the new chunks.
        (KB_DIR / "vectors.npy").unlink(missing_ok=True)

    if not args.no_embed:
        try:
            embed_chunks(chunks)
        except Exception as e:
            print(f"\n⚠ embedding failed ({str(e)[:160]})\n  KB still works in BM25-only mode; "
                  f"rerun with --embed-only once the embedding API is available.")


if __name__ == "__main__":
    main()
