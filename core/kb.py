"""
Knowledge base: real Alibaba Cloud help-center docs, searchable two ways.

    build (offline, scripts/build_kb.py):
        fetch HTML -> extract article body -> chunk by heading -> embed
        -> data/kb/chunks.json (+ vectors.npy)

    query (online, search()):
        BM25 keyword search  ─┐
                              ├─ Reciprocal Rank Fusion -> top-k chunks
        vector (cosine) search┘

Why hybrid: BM25 nails exact terms ("Arrearage", "serverHold", "502"),
vectors catch paraphrases ("连不上服务器" ~ "SSH 远程连接失败"). If the
embedding API is down — or the index was built without vectors — search
degrades to BM25 only instead of failing.

Everything here is hand-rolled (no vector DB, no jieba): ~1k chunks fit in
memory and a numpy dot product is all a cosine search needs at this size.
"""

import json
import logging
import math
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

log = logging.getLogger("ticketcoach.kb")

KB_DIR = Path(__file__).parent.parent / "data" / "kb"
ALLOWED_DOC_HOSTS = ("help.aliyun.com", "www.alibabacloud.com")

# ---------------------------------------------------------------------------
# HTML -> article text
# ---------------------------------------------------------------------------


class _ArticleParser(HTMLParser):
    """Collects text inside <div class="markdown-body"> as (heading, text) sections."""

    BLOCK = {"p", "div", "li", "tr", "br", "pre", "table", "ul", "ol", "section"}
    SKIP = {"script", "style", "noscript", "svg", "button"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title = False
        self._depth = 0          # div depth inside the article; 0 = outside
        self._skip = 0
        self._heading = None     # tag name while inside h1-h4
        self._buf = []
        self.sections = []       # [(heading, text)]
        self._cur_heading = ""
        self._cur_lines = []

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        cls = dict(attrs).get("class", "") or ""
        if self._depth == 0:
            if tag == "div" and "markdown-body" in cls.split():
                self._depth = 1
            return
        if tag == "div":
            self._depth += 1
        if tag in self.SKIP:
            self._skip += 1
        elif tag in ("h1", "h2", "h3", "h4"):
            self._flush_line()
            self._heading = tag
        elif tag in self.BLOCK:
            self._flush_line()
            if tag == "li":
                self._buf.append("- ")
        elif tag in ("td", "th"):
            self._buf.append(" | ")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if self._depth == 0:
            return
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == self._heading:
            text = "".join(self._buf).strip()
            self._buf = []
            self._heading = None
            if text:
                self._close_section()
                self._cur_heading = text
        elif tag in self.BLOCK:
            self._flush_line()
        if tag == "div":
            self._depth -= 1
            if self._depth == 0:
                self._flush_line()
                self._close_section()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._depth and not self._skip:
            self._buf.append(data)

    def _flush_line(self):
        if self._heading:
            return
        line = re.sub(r"\s+", " ", "".join(self._buf)).strip(" |")
        self._buf = []
        if line and line != "-":
            self._cur_lines.append(line)

    def _close_section(self):
        text = "\n".join(self._cur_lines).strip()
        if text:
            self.sections.append((self._cur_heading, text))
        self._cur_lines = []


def extract_article(html: str) -> dict:
    """{"title", "sections": [(heading, text)]} from a help-center page."""
    p = _ArticleParser()
    p.feed(html)
    p.close()
    # Prefer the visible <h1>; some pages put a long description in <title>.
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.S)
    title = re.sub(r"<[^>]+>", "", h1.group(1)) if h1 else p.title
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"\s*[-_|]\s*阿里云帮助中心.*$", "", title)
    return {"title": title, "sections": p.sections}


def chunk_article(article: dict, max_chars: int = 600) -> list:
    """Split each section on line boundaries into <= max_chars chunks.

    Every chunk is prefixed with "title > heading" (a contextual header) so a
    chunk like "检查安全组规则..." still says WHICH product / problem it is about.
    """
    chunks = []
    for heading, text in article["sections"]:
        header = article["title"] + (f" > {heading}" if heading else "")
        buf = ""
        for line in text.split("\n"):
            if buf and len(buf) + len(line) + 1 > max_chars:
                chunks.append({"heading": heading, "text": f"{header}\n{buf}"})
                buf = ""
            # hard-split a single overlong line
            while len(line) > max_chars:
                chunks.append({"heading": heading, "text": f"{header}\n{line[:max_chars]}"})
                line = line[max_chars:]
            buf = f"{buf}\n{line}" if buf else line
        if buf.strip():
            chunks.append({"heading": heading, "text": f"{header}\n{buf}"})
    return chunks


# ---------------------------------------------------------------------------
# BM25 (hand-rolled; CJK bigrams + ASCII words)
# ---------------------------------------------------------------------------

_ASCII = re.compile(r"[a-z0-9][a-z0-9_.\-]*")
_CJK = re.compile(r"[一-鿿]+")
# Question / filler words. They are cut out of a CJK run BEFORE bigramming,
# so "应该怎么配置" yields just "配置" instead of junk like "该怎" / "么配".
_STOP = re.compile("|".join(sorted("""
怎么 怎样 如何 什么 为什么 为啥 应该 可以 能不能 是不是 是否 一下 一直 还是 的话 请问
我们 你们 他们 这个 那个 这些 一些 已经 但是 而且 然后 就是 没有 不是 需要 因为 所以
吗 呢 吧 啊 了 的 么 呀 哦 嘛 我 你 他 她 它 是 在 有 和 与 及 或 都 也 就 还 又 被 把 让 给 要 会 能 对
""".split(), key=len, reverse=True)))


def tokenize(text: str) -> list:
    """No Chinese word segmenter: overlapping character bigrams are a strong,
    dependency-free baseline for BM25 on Chinese (single chars for 1-char runs)."""
    text = text.lower()
    tokens = _ASCII.findall(text)
    for run in _CJK.findall(text):
        for part in _STOP.split(run):
            if part:
                tokens.extend([part] if len(part) == 1 else [part[i:i + 2] for i in range(len(part) - 1)])
    return tokens


class BM25:
    def __init__(self, docs: list, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list:
        q = [t for t in set(tokenize(query)) if t in self.idf]
        out = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def rrf(rankings: list, k: int = 60) -> dict:
    """Reciprocal Rank Fusion: score(d) = sum 1 / (k + rank). Rank-based, so
    BM25 scores and cosine similarities never need to be on the same scale."""
    fused = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


class KnowledgeBase:
    def __init__(self, chunks: list, vectors=None, embed_fn=None):
        self.chunks = chunks
        self.bm25 = BM25([c["text"] for c in chunks])
        self.vectors = vectors           # np.ndarray (n, d), L2-normalized, or None
        self.embed_fn = embed_fn         # list[str] -> list[list[float]]

    @classmethod
    def load(cls, kb_dir: Path = KB_DIR, embed_fn=None) -> "KnowledgeBase":
        chunks_path = kb_dir / "chunks.json"
        chunks = json.loads(chunks_path.read_text(encoding="utf-8")) if chunks_path.exists() else []
        vectors = None
        vec_path = kb_dir / "vectors.npy"
        if chunks and vec_path.exists():
            import numpy as np
            vectors = np.load(vec_path)
            if len(vectors) != len(chunks):
                log.warning("kb_vectors_stale", extra={"fields": {
                    "chunks": len(chunks), "vectors": len(vectors)}})
                vectors = None
        return cls(chunks, vectors, embed_fn)

    def _vector_ranking(self, query: str, n: int) -> Optional[list]:
        if self.vectors is None or self.embed_fn is None:
            return None
        import numpy as np
        try:
            q = np.asarray(self.embed_fn([query])[0], dtype=np.float32)
        except Exception as e:  # embedding API down -> keyword-only, don't fail the user
            log.warning("kb_embed_failed", extra={"fields": {"error": str(e)[:200]}})
            return None
        q /= (np.linalg.norm(q) or 1.0)
        sims = self.vectors @ q
        return [int(i) for i in np.argsort(-sims)[:n]]

    def search(self, query: str, k: int = 5, business_type: Optional[str] = None,
               mode: str = "hybrid") -> list:
        """Top-k chunks. mode: hybrid | bm25 | vector. Each hit says which
        retrieval actually ran (so evals / logs can see a degraded search)."""
        if not self.chunks or not query.strip():
            return []
        pool = 30
        rankings, used = [], []
        if mode in ("hybrid", "bm25"):
            s = self.bm25.scores(query)
            ranking = [i for i in sorted(range(len(s)), key=lambda i: -s[i])[:pool] if s[i] > 0]
            rankings.append(ranking)
            used.append("bm25")
        if mode in ("hybrid", "vector"):
            vr = self._vector_ranking(query, pool)
            if vr is not None:
                rankings.append(vr)
                used.append("vector")
        if not rankings or not any(rankings):
            return []

        fused = rrf(rankings)
        if business_type:
            # Soft filter: boost the matching product line instead of hiding
            # everything else (questions often span products).
            for i in fused:
                if self.chunks[i].get("business_type") == business_type:
                    fused[i] *= 1.5
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [dict(self.chunks[i], id=i, score=round(fused[i], 5), retrieval="+".join(used))
                for i in top]


_KB: Optional[KnowledgeBase] = None


def get_kb() -> KnowledgeBase:
    """Process-wide KB, loaded lazily on first use."""
    global _KB
    if _KB is None:
        from core import llm
        _KB = KnowledgeBase.load(embed_fn=llm.embed)
        log.info("kb_loaded", extra={"fields": {
            "chunks": len(_KB.chunks), "vectors": _KB.vectors is not None}})
    return _KB
