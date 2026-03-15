"""
虎菇婆訂單向量資料庫 — 語意搜尋介面
E5 模型規格：查詢字串自動加 "query: " 前綴

使用方式：
    python restaurant-tiger/query_tiger.py
"""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

DB = Path(__file__).parent / "vector_db_tiger"


def get_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-small"
    )
    client = chromadb.PersistentClient(path=str(DB))
    return client.get_collection("tiger_orders", embedding_function=ef)


def search(query: str, n: int = 5) -> list[dict]:
    col = get_collection()
    # E5 規格：查詢加 "query: " 前綴
    results = col.query(
        query_texts=[f"query: {query}"],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "similarity": round(1 - dist, 4),
            "items":      meta.get("items", ""),
            "dining":     meta.get("dining", ""),
            "platform":   meta.get("platform", ""),
            "amount":     meta.get("amount", 0),
            "order_date": meta.get("order_date", ""),
            "temperature": meta.get("temperature"),
            "rainfall":   meta.get("rainfall"),
        })
    return output


if __name__ == "__main__":
    print("虎菇婆訂單語意搜尋（輸入 q 離開）\n")
    while True:
        query = input("搜尋：").strip()
        if query.lower() in ("q", "quit", "exit", ""):
            break
        hits = search(query)
        print(f"\n  Top {len(hits)} 結果：")
        for i, h in enumerate(hits, 1):
            print(
                f"  [{i}] 相似度={h['similarity']:.4f} | "
                f"{h['order_date']} | {h['dining']} | "
                f"{h['amount']}元 | 氣溫{h['temperature']}℃ | "
                f"品項：{h['items'][:60]}"
            )
        print()
