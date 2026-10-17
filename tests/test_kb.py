"""Knowledge base: extraction, tokenizer, BM25, RRF, graceful degradation."""

import numpy as np

from core.kb import BM25, KnowledgeBase, chunk_article, extract_article, rrf, tokenize

PAGE = """<html><head><title>长描述 - 阿里云帮助中心</title></head><body>
<nav>导航 菜单 不要</nav>
<h1>能ping通ECS实例但端口不通的排查方法</h1>
<div class="markdown-body">
  <h2>问题现象</h2><p>客户端提示 Connection timed out。</p>
  <h2>排查步骤</h2><ul><li>检查安全组入方向规则</li><li>检查防火墙</li></ul>
  <script>var junk = 1;</script>
</div>
<footer>版权所有</footer></body></html>"""


def test_extract_article_keeps_body_only():
    a = extract_article(PAGE)
    assert a["title"] == "能ping通ECS实例但端口不通的排查方法"
    headings = [h for h, _ in a["sections"]]
    assert headings == ["问题现象", "排查步骤"]
    text = " ".join(t for _, t in a["sections"])
    assert "安全组" in text and "导航" not in text and "junk" not in text and "版权" not in text


def test_chunks_carry_contextual_header():
    chunks = chunk_article(extract_article(PAGE))
    assert chunks[1]["text"].startswith("能ping通ECS实例但端口不通的排查方法 > 排查步骤\n")


def test_long_section_is_split():
    art = {"title": "T", "sections": [("h", "\n".join(["句子" * 50] * 10))]}
    chunks = chunk_article(art, max_chars=300)
    assert len(chunks) > 1 and all(len(c["text"]) <= 300 + len("T > h\n") for c in chunks)


def test_tokenizer_drops_question_words():
    assert tokenize("回源host应该怎么配置") == ["host", "回源", "配置"]


def test_bm25_prefers_rare_matching_terms():
    bm = BM25(["安全组 规则 放行 端口", "账单 欠费 停机", "怎么 怎么 怎么 端口"])
    s = bm.scores("安全组端口")
    assert s[0] > s[2] > s[1] == 0


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [2, 1, 4]])
    assert max(fused, key=fused.get) in (1, 2) and fused[2] > fused[3]


def _kb(embed_fn):
    chunks = [{"text": "ECS SSH 连接超时 安全组", "title": "A", "business_type": "ecs"},
              {"text": "域名 解析 不生效 DNS", "title": "B", "business_type": "dns"},
              {"text": "退款 退订 五天无理由", "title": "C", "business_type": "bill"}]
    vecs = np.eye(3, dtype=np.float32)
    return KnowledgeBase(chunks, vecs, embed_fn)


def test_hybrid_uses_vectors_for_paraphrase():
    kb = _kb(lambda texts: [[0.0, 0.0, 1.0]])  # "embedding" says: refund doc
    hits = kb.search("钱能拿回来吗", k=1)
    assert hits[0]["title"] == "C" and hits[0]["retrieval"] == "bm25+vector"


def test_hybrid_degrades_to_bm25_when_embedding_fails():
    def broken(texts):
        raise RuntimeError("Arrearage")
    hits = _kb(broken).search("域名解析不生效", k=1)
    assert hits[0]["title"] == "B" and hits[0]["retrieval"] == "bm25"


def test_business_type_boost():
    kb = _kb(None)
    hits = kb.search("ECS 域名", k=2, business_type="dns")
    assert hits[0]["title"] == "B"
