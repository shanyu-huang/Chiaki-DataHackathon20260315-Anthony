"""
美業訪談向量資料庫 — 查詢介面

執行方式：
    python query.py

範例查詢：
    討厭被推銷的消費者
    30歲以上高雄地區
    願意購買精油的族群
    環境乾淨安靜的需求
"""

from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

DB = Path(__file__).parent / "vector_db"

def get_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    client = chromadb.PersistentClient(path=str(DB))
    return client.get_collection("beauty_interviews", embedding_function=ef)


def search(query: str, n: int = 5) -> list[dict]:
    """
    輸入任意中文描述，回傳語意最相近的受訪者。
    相似度 1.0 = 完全一致；0.0 = 毫無關聯。
    """
    col = get_collection()
    results = col.query(query_texts=[query], n_results=n)

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "similarity":        round(1 - dist, 3),
            "age":               meta["age"],
            "region":            meta["region"],
            "service":           meta["service"][:60],
            "why_chosen":        meta["why_chosen"][:80],
            "essential_oil":     meta["essential_oil"][:60],
            "return_conditions": meta["return_conditions"][:100],
        })
    return output


def print_results(results: list[dict], query: str):
    print(f'\n── 查詢：「{query}」 ──')
    for i, r in enumerate(results, 1):
        print(
            f"\n[{i}] 相似度 {r['similarity']} ｜ {r['age']}歲 ｜ {r['region']}\n"
            f"  消費：{r['service']}\n"
            f"  選擇原因：{r['why_chosen']}\n"
            f"  回訪條件：{r['return_conditions']}"
        )
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("  美業訪談向量資料庫 — 語意查詢介面")
    print("  輸入 'q' 離開")
    print("=" * 50)

    while True:
        q = input("\n查詢：").strip()
        if q.lower() in ("q", "quit", "exit"):
            break
        if not q:
            continue
        results = search(q, n=5)
        print_results(results, q)
