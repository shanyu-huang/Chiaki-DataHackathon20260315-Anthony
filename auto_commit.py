"""
auto_commit.py — 黑客松自動提交腳本
====================================
由 Claude Code Stop Hook 觸發，在每次對話結束後：
1. 偵測本 repo 有無未提交變更
2. 根據變更內容自動生成說明訊息
3. git add → commit → push

執行方式（手動測試）：
    python auto_commit.py
"""

import subprocess
import sys
import re
from datetime import datetime
from pathlib import Path

# Windows 終端機 UTF-8 輸出（避免中文/Emoji 編碼錯誤）
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).parent
BRANCH = "master"

# ── 不納入自動提交的路徑 ─────────────────────────────────────────────────────
EXCLUDE_PATTERNS = [
    "vector_db/",
    ".claude/",
    "__pycache__/",
    "*.pyc",
    ".cache/",
]


def run(cmd: list[str], cwd=None) -> tuple[int, str, str]:
    """執行指令，回傳 (returncode, stdout, stderr)"""
    result = subprocess.run(
        cmd, cwd=cwd or REPO,
        capture_output=True, text=True, encoding="utf-8"
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def get_changed_files() -> list[str]:
    """取得所有未提交的變更檔案清單"""
    _, out, _ = run(["git", "status", "--porcelain"])
    files = []
    for line in out.splitlines():
        if line.strip():
            # 格式：XY path 或 XY "path"
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                path = parts[1].strip('"').split(" -> ")[-1]  # 處理 rename
                files.append(path)
    return files


def categorize_files(files: list[str]) -> dict[str, list[str]]:
    """將變更檔案分類"""
    cats = {
        "session_notes": [],
        "pipeline_code": [],
        "data_files": [],
        "config": [],
        "other": [],
    }
    for f in files:
        if "session_notes/" in f:
            cats["session_notes"].append(f)
        elif f.endswith((".py", ".txt", ".sh")):
            cats["pipeline_code"].append(f)
        elif f.endswith((".csv", ".jsonl", ".json")) and "settings" not in f:
            cats["data_files"].append(f)
        elif f in (".gitignore", "requirements.txt") or "settings" in f:
            cats["config"].append(f)
        else:
            cats["other"].append(f)
    return cats


def extract_session_topic(files: list[str]) -> str:
    """從 session_notes 檔名萃取主題"""
    topics = []
    for f in files:
        # e.g. session_notes/20260315_02_三維商業分析_獲客_體驗_回頭客.md
        match = re.search(r"\d{8}_\d+_(.+)\.md", f)
        if match:
            topics.append(match.group(1).replace("_", " "))
    return "、".join(topics) if topics else ""


def build_commit_message(cats: dict[str, list[str]], all_files: list[str]) -> str:
    """根據分類自動生成 commit 訊息"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    # ── 主旨行（50 字以內） ────────────────────────────────────────────
    subject_parts = []
    if cats["session_notes"]:
        topic = extract_session_topic(cats["session_notes"])
        if topic:
            subject_parts.append(f"docs: {topic}")
        else:
            subject_parts.append(f"docs: 新增 {len(cats['session_notes'])} 篇討論筆記")
    if cats["pipeline_code"]:
        subject_parts.append(f"feat: 更新分析腳本")
    if cats["data_files"]:
        subject_parts.append(f"data: 更新資料檔案")
    if cats["config"]:
        subject_parts.append(f"chore: 更新設定")
    if not subject_parts:
        subject_parts.append("chore: 自動同步")

    subject = " | ".join(subject_parts)
    lines.append(subject)
    lines.append("")

    # ── 內文（詳細說明） ───────────────────────────────────────────────
    lines.append(f"自動提交時間：{now}")
    lines.append("")

    if cats["session_notes"]:
        lines.append("📝 討論筆記：")
        for f in cats["session_notes"]:
            fname = Path(f).name
            lines.append(f"  - {fname}")
        lines.append("")

    if cats["pipeline_code"]:
        lines.append("🔧 程式碼變更：")
        for f in cats["pipeline_code"]:
            lines.append(f"  - {f}")
        lines.append("")

    if cats["data_files"]:
        lines.append("📊 資料檔案：")
        for f in cats["data_files"]:
            lines.append(f"  - {f}")
        lines.append("")

    if cats["config"]:
        lines.append("⚙️ 設定變更：")
        for f in cats["config"]:
            lines.append(f"  - {f}")
        lines.append("")

    if cats["other"]:
        lines.append("📁 其他：")
        for f in cats["other"]:
            lines.append(f"  - {f}")
        lines.append("")

    lines.append("Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
    return "\n".join(lines)


def main():
    print("🔍 檢查 Git 狀態...")

    # 確認在 repo 目錄內
    code, _, err = run(["git", "rev-parse", "--git-dir"])
    if code != 0:
        print("⚠️  不在 Git repo 內，略過自動提交。")
        sys.exit(0)

    # 取得變更檔案
    changed = get_changed_files()
    if not changed:
        print("✅ 無未提交的變更，略過。")
        sys.exit(0)

    print(f"  發現 {len(changed)} 個變更檔案")

    # 分類
    cats = categorize_files(changed)

    # git add（排除特定路徑）
    print("📦 加入暫存區...")
    exclude_args = []
    for pat in EXCLUDE_PATTERNS:
        exclude_args += ["--", f":!{pat}"]

    # 先 add all，再 reset 排除的
    run(["git", "add", "-A"])
    for pat in EXCLUDE_PATTERNS:
        run(["git", "reset", "HEAD", "--", pat])

    # 確認暫存區有無內容
    code, staged, _ = run(["git", "diff", "--cached", "--name-only"])
    if not staged:
        print("✅ 暫存區為空（可能全是排除路徑），略過提交。")
        sys.exit(0)

    # 生成訊息
    msg = build_commit_message(cats, changed)
    print("\n📝 提交訊息預覽：")
    print("─" * 50)
    print(msg)
    print("─" * 50)

    # Commit
    code, out, err = run(["git", "commit", "-m", msg])
    if code != 0:
        print(f"❌ Commit 失敗：{err}")
        sys.exit(1)
    print(f"\n✅ Commit 成功")

    # Push
    print(f"🚀 推送至 origin/{BRANCH}...")
    code, out, err = run(["git", "push", "origin", BRANCH])
    if code != 0:
        print(f"❌ Push 失敗：{err}")
        sys.exit(1)
    print(f"✅ 已同步至 GitHub")


if __name__ == "__main__":
    main()
