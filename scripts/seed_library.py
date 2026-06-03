"""
Offline batch pre-generation of a script library.

This simulates the enterprise pattern: generate training scripts ahead of time
(slow, offline) and store them, so the app can serve them INSTANTLY later
instead of running the 30-60s pipeline on every user click.

Run once: python scripts/seed_library.py
Output: data/library.json
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import run_pipeline

# Varied Alibaba Cloud scenarios for a representative library
SEED_PARAMS = [
    {"business_type": "云服务器ECS/实例连接", "emotion": "愤怒", "difficulty": "高"},
    {"business_type": "账单计费/异常扣费", "emotion": "不满", "difficulty": "中"},
    {"business_type": "域名服务/解析与续费", "emotion": "平静", "difficulty": "中"},
    {"business_type": "云数据库RDS/连接与性能", "emotion": "不满", "difficulty": "高"},
    {"business_type": "对象存储OSS/数据与权限", "emotion": "平静", "difficulty": "中"},
]

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "library.json")


def main():
    library = []
    for i, params in enumerate(SEED_PARAMS, 1):
        print(f"[seed] {i}/{len(SEED_PARAMS)} generating: {params}")
        try:
            result = run_pipeline(params)
            library.append(result)
            print(f"[seed]   ok: {result['script'].get('title', '')[:40]} "
                  f"(score {result['review'].get('overall_score')})")
        except Exception as e:
            print(f"[seed]   FAILED: {e}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(library, f, ensure_ascii=False, indent=2)
    print(f"[seed] wrote {len(library)} entries to {OUT}")


if __name__ == "__main__":
    main()
