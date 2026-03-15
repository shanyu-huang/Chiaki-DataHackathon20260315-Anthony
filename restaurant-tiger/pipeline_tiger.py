"""
虎菇婆餐廳訂單 × 氣象資料 — 向量資料庫建立流程
Embedding 模型：intfloat/multilingual-e5-small（384 維）

執行方式：
    pip install -r requirements.txt
    python restaurant-tiger/pipeline_tiger.py
"""

import json
import re
from pathlib import Path

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent
ORDERS  = BASE / "虎菇婆_訂單"
WEATHER = BASE / "氣象資料"
JSONL   = BASE / "orders_clean.jsonl"
DB      = BASE / "vector_db_tiger"

TEMP_CSV = WEATHER / "板橋逐時氣溫.csv"
RAIN_CSV = WEATHER / "永和逐時降雨.csv"

# ── 工具函式 ──────────────────────────────────────────────────────────────────

def clean(val) -> str:
    """將值轉為乾淨字串，NaN 回傳空字串"""
    if pd.isna(val):
        return ""
    text = str(val)
    text = re.sub(r"[\r\n]+", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def parse_order_time(raw: str) -> pd.Timestamp | None:
    """解析 '2025/11/24 11:23:08AM' 格式"""
    try:
        return pd.to_datetime(raw, format="%Y/%m/%d %I:%M:%S%p")
    except Exception:
        try:
            return pd.to_datetime(raw)
        except Exception:
            return None


# ── Step 1：讀取訂單 XLSX ─────────────────────────────────────────────────────

def load_orders() -> pd.DataFrame:
    files = sorted(ORDERS.glob("*.xlsx"))
    frames = []
    for f in files:
        df = pd.read_excel(f, dtype={"訂單ID": str})
        df["_source_file"] = f.stem
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"  ✓ {len(combined)} 筆訂單（{len(files)} 個 xlsx）")
    return combined


# ── Step 2：讀取氣象資料 ──────────────────────────────────────────────────────

def load_weather() -> dict[pd.Timestamp, dict]:
    """回傳 {整點時間戳 → {temp, rain}} 的查詢字典"""
    # 板橋氣溫
    temp_df = pd.read_csv(TEMP_CSV, encoding="utf-8-sig")
    temp_df["dt"] = pd.to_datetime(temp_df["日期時間"])
    temp_df = temp_df.set_index("dt")["氣溫(℃)"]

    # 永和降雨
    rain_df = pd.read_csv(RAIN_CSV, encoding="utf-8-sig")
    rain_df["dt"] = pd.to_datetime(rain_df["日期時間"])
    rain_df = rain_df.set_index("dt")

    # 優先使用 降水量(mm)，其次 雨量(mm)
    rain_col = "降水量(mm)" if "降水量(mm)" in rain_df.columns else "雨量(mm)"
    rain_series = rain_df[rain_col]

    lookup: dict[pd.Timestamp, dict] = {}
    for ts in temp_df.index.union(rain_series.index):
        lookup[ts] = {
            "temp": round(float(temp_df.get(ts, float("nan"))), 1)
                    if ts in temp_df.index else None,
            "rain": round(float(rain_series.get(ts, float("nan"))), 1)
                    if ts in rain_series.index else None,
        }

    print(f"  ✓ 氣象資料：{len(lookup)} 個整點時段")
    return lookup


def match_weather(ts: pd.Timestamp | None, lookup: dict) -> dict:
    """取最近整點氣象（±30 分鐘內）"""
    if ts is None:
        return {"temp": None, "rain": None}
    rounded = ts.floor("h")
    return lookup.get(rounded, {"temp": None, "rain": None})


# ── Step 3：組合記錄 ──────────────────────────────────────────────────────────

def build_records(orders: pd.DataFrame, weather: dict) -> list[dict]:
    records = []
    for i, row in orders.iterrows():
        ts = parse_order_time(clean(row.get("下單時間", "")))
        w  = match_weather(ts, weather)

        items   = clean(row.get("品項", ""))
        dining  = clean(row.get("用餐型態", ""))
        platform = clean(row.get("點餐平台", ""))
        tx_type = clean(row.get("交易型態", ""))
        note    = clean(row.get("備註", ""))
        amount  = row.get("總金額", 0)
        if pd.isna(amount):
            amount = 0
        amount = int(amount)

        # E5 模型需 "passage: " 前綴
        text_body = (
            f"品項：{items} "
            f"用餐：{dining} "
            f"點餐平台：{platform} "
            f"交易：{tx_type} "
            f"金額：{amount}元"
        )
        if w["temp"] is not None:
            text_body += f" 氣溫：{w['temp']}℃"
        if w["rain"] is not None:
            text_body += f" 降雨：{w['rain']}mm"
        if note:
            text_body += f" 備註：{note}"

        rec = {
            "id":         f"O{i:05d}",
            "order_id":   clean(row.get("訂單ID", "")),
            "text":       "passage: " + text_body,
            "items":      items,
            "dining":     dining,
            "platform":   platform,
            "tx_type":    tx_type,
            "amount":     amount,
            "order_date": ts.strftime("%Y-%m-%d") if ts else "",
            "order_hour": ts.hour if ts else -1,
            "temperature": w["temp"] if w["temp"] is not None else -99.0,
            "rainfall":    w["rain"] if w["rain"] is not None else -1.0,
            "source_file": clean(row.get("_source_file", "")),
        }
        records.append(rec)

    print(f"  ✓ 組合 {len(records)} 筆記錄")
    return records


# ── Step 4：儲存 JSONL ────────────────────────────────────────────────────────

def save_jsonl(records: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ JSONL → {path.name}")


# ── Step 5：建立向量資料庫 ────────────────────────────────────────────────────

def build_vector_db(records: list[dict], db_path: Path):
    print("  正在載入 multilingual-e5-small（首次需下載約 120MB）...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-small"
    )

    client = chromadb.PersistentClient(path=str(db_path))

    try:
        client.delete_collection("tiger_orders")
    except Exception:
        pass

    col = client.create_collection(
        name="tiger_orders",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    meta_keys = ["order_id", "items", "dining", "platform", "tx_type",
                 "amount", "order_date", "order_hour",
                 "temperature", "rainfall", "source_file"]

    # ChromaDB 批次上傳（每批 500）
    batch_size = 500
    for start in range(0, len(records), batch_size):
        batch = records[start: start + batch_size]
        col.add(
            ids=       [r["id"]   for r in batch],
            documents= [r["text"] for r in batch],
            metadatas= [{k: r[k] for k in meta_keys} for r in batch],
        )
        print(f"    上傳 {start + len(batch)}/{len(records)} 筆...")

    print(f"  ✓ {len(records)} 個向量 → {db_path.name}/")
    return col


# ── 主程式 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n══ Step 1：讀取訂單 XLSX ══")
    orders = load_orders()

    print("\n══ Step 2：讀取氣象資料 ══")
    weather = load_weather()

    print("\n══ Step 3：組合記錄 ══")
    records = build_records(orders, weather)

    print("\n══ Step 4：儲存 JSONL ══")
    save_jsonl(records, JSONL)

    print("\n══ Step 5：建立向量資料庫 ══")
    build_vector_db(records, DB)

    print("\n✅ 完成！執行 python restaurant-tiger/query_tiger.py 開始查詢。")
