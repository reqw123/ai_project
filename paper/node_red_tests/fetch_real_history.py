"""
唯讀抓取 Node-RED 目前真實累積的 v2_daily_history（個體化基線的原始資料），
存成一份快照 JSON，給互動式 UI 預覽頁面當「複製真實資料再竄改」的起點。

⚠️ 全程唯讀：只用 open(...).read() 讀取，從頭到尾不會寫回
C:\\Users\\homec\\.node-red\\context\\global\\global.json 這個檔案，
不會動到 Node-RED 正在使用的任何 context 資料。

用法：
    python fetch_real_history.py

會在同資料夾產生 real_history_snapshot.json，接著跑：
    node render_baseline_ui_preview.js
渲染出來的頁面會內嵌這份快照，並提供「複製真實資料 + 倍率竄改」模式。
"""
import json
from pathlib import Path

# Node-RED file-based 全域 context 的實際位置（跟 paper/_tools/1_read_context.py
# 用的是同一個路徑，已確認這台機器上這個檔案真實存在）
REAL_CONTEXT_PATH = Path(r"C:\Users\homec\.node-red\context\global\global.json")
OUTPUT_PATH = Path(__file__).parent / "real_history_snapshot.json"


def main():
    if not REAL_CONTEXT_PATH.exists():
        print(f"⚠ 找不到 {REAL_CONTEXT_PATH}，略過（互動頁面會停用「複製真實資料」模式）")
        return

    with open(REAL_CONTEXT_PATH, encoding="utf-8") as f:
        ctx = json.load(f)

    history = ctx.get("v2_daily_history", [])
    raw_settings = ctx.get("v2_user_settings", {})
    excluded = ctx.get("v2_excluded_dates", [])

    # 只留 computeBaseline() 真正會用到的欄位（baseline_days）。v2_user_settings
    # 裡還有 discord_webhook、cat_name/cat_breed/cat_birth_* 等個人資料/機密，
    # 這份快照會被整包內嵌進 baseline_ui_preview.html（靜態檔案，可能被複製、
    # 分享），絕對不能把這些欄位一起帶出來。
    settings = {"baseline_days": raw_settings.get("baseline_days", 7)}

    snapshot = {
        "source_path": str(REAL_CONTEXT_PATH),
        "fetched_at_note": "此快照為唯讀抓取當下的複本，不會隨真實資料變動而自動更新，"
                            "要看最新的真實資料需重新執行本腳本。",
        "history": history,
        "settings": settings,
        "excluded_dates": excluded,
    }

    OUTPUT_PATH.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已抓取 {len(history)} 天真實資料（唯讀，未動到原始檔案）-> {OUTPUT_PATH}")
    print(f"baseline_days 設定: {settings.get('baseline_days', 7)}")
    print(f"排除日期: {excluded if excluded else '無'}")
    print("\n接著執行: node render_baseline_ui_preview.js")


if __name__ == "__main__":
    main()
