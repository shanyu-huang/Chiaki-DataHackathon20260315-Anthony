"""
美業游擊訪談 — 資料處理全流程
執行順序：CSV 解析 → 清理 → JSONL → 向量資料庫

執行方式：
    pip install -r requirements.txt
    python pipeline.py
"""

import csv
import json
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
BASE  = Path(__file__).parent
RAW   = BASE / "美業消費者_游擊訪談.csv"
JSONL = BASE / "interviews_clean.jsonl"
DB    = BASE / "vector_db"

# ── 工具函式 ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """移除多餘換行與空白"""
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def normalize_age(raw: str) -> str:
    """統一年齡格式：'31 ～ 35 歲' → '31-35'"""
    raw = raw.strip()
    raw = re.sub(r"\s*[～~]\s*", "-", raw)
    raw = raw.replace("歲以上", "+").replace(" 歲", "").replace("歲", "")
    return raw.strip()

def normalize_region(raw: str) -> str:
    """萃取縣市層級：'新北市永和區' → '新北市'"""
    raw = clean_text(raw)
    match = re.match(r"([\u4e00-\u9fff]{2,4}[市縣])", raw)
    return match.group(1) if match else raw

# ── Step 1：解析 CSV ──────────────────────────────────────────────────────────

def parse_csv(path: Path) -> list[dict]:
    """
    使用 Python 內建 csv.reader 解析含換行的原始 CSV。
    跳過年齡欄為空或非數字的列（這些是 CSV 換行的殘留行）。
    """
    records = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # 跳過標頭

        for i, row in enumerate(reader, start=1):
            if len(row) < 5:
                continue

            age = normalize_age(row[0])
            # 略過沒有數字的殘留行（換行符造成的假列）
            if not re.search(r"\d", age):
                continue

            while len(row) < 6:
                row.append("")

            r = {
                "id":                f"R{i:03d}",
                "age":               age,
                "region":            normalize_region(row[1]),
                "region_detail":     clean_text(row[1]),
                "service":           clean_text(row[2]),
                "why_chosen":        clean_text(row[3]),
                "essential_oil":     clean_text(row[4]),
                "return_conditions": clean_text(row[5]),
            }

            # 合併全文供 embedding 使用
            r["text"] = (
                f"[年齡:{r['age']}] [地區:{r['region']}] "
                f"消費項目：{r['service']} "
                f"選擇原因：{r['why_chosen']} "
                f"精油購買：{r['essential_oil']} "
                f"回訪條件：{r['return_conditions']}"
            )
            records.append(r)

    return records

# ── Step 2：儲存為 JSONL ───────────────────────────────────────────────────────

def save_jsonl(records: list[dict], path: Path):
    """
    每位受訪者獨立一行，格式緊湊。
    相較原始 CSV 可減少約 40% 的 token 使用量。
    """
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ {len(records)} 筆記錄 → {path.name}")

# ── Step 3：建立向量資料庫 ────────────────────────────────────────────────────

def build_vector_db(records: list[dict], db_path: Path):
    """
    使用多語系模型將每筆訪談轉為向量，存入 ChromaDB。
    首次執行會下載模型（約 280MB），之後直接讀取本地快取。
    """
    print("  正在載入多語系 Embedding 模型（首次需下載約 280MB）...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    client = chromadb.PersistentClient(path=str(db_path))

    # 若重跑則清除舊資料
    try:
        client.delete_collection("beauty_interviews")
    except Exception:
        pass

    col = client.create_collection(
        name="beauty_interviews",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    meta_keys = ["age", "region", "region_detail", "service",
                 "why_chosen", "essential_oil", "return_conditions"]

    col.add(
        ids=       [r["id"]   for r in records],
        documents= [r["text"] for r in records],
        metadatas= [{k: r[k] for k in meta_keys} for r in records],
    )

    print(f"  ✓ {len(records)} 個向量 → {db_path.name}/")
    return col

# ── 主程式 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n══ Step 1：解析 & 清理 CSV ══")
    records = parse_csv(RAW)
    print(f"  ✓ 解析出 {len(records)} 位受訪者")

    print("\n══ Step 2：儲存 JSONL ══")
    save_jsonl(records, JSONL)

    print("\n══ Step 3：建立向量資料庫 ══")
    build_vector_db(records, DB)

    print("\n✅ 完成！執行 python query.py 開始查詢。")
