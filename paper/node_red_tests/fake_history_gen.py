"""
在 Python 端捏造「個體化基線」用的假每日彙整資料。

跟 synthetic_data.js（JS 端的假資料產生器）做同一件事、欄位形狀完全一致
（對齊 Python behavior_tracker.get_today_stats 的輸出），只是換成 Python
寫，方便你直接在這裡改參數（不用碰 JS）。

用法：
    python fake_history_gen.py

會在同資料夾產生 fake_scenarios.json，接著跑：
    node render_baseline_ui_preview.js
render_baseline_ui_preview.js 偵測到這個檔案存在時，會優先讀它來產生
UI 預覽（而不是用內建的 JS 假資料），所以你只要改這支檔案的參數、
重新執行這兩個指令，就能直接在 baseline_ui_preview.html 看到結果。
"""
import json
import random
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "fake_scenarios.json"

# ═══════════════════════════════════════════════════════
#  使用者設定區 —— 改這裡的參數/情境即可
# ═══════════════════════════════════════════════════════
# walk_min_base / lick_min_base：分鐘為單位的「平均每天時長」基準值
# scratch_sec_base / shake_sec_base：秒為單位的「平均每天時長」基準值
# 實際產生時每天會在基準值的 0.6~1.4 倍之間隨機浮動，模擬天與天之間的自然差異。
SCENARIOS = [
    dict(
        name="normal_21d",
        title="情境 1：10 天正常資料",
        note="Python 端生成，可調整下面參數觀察基線變化（預期 sanity_ok=true）",
        seed=20260718, start_date="2026-06-01", day_count=15,
        walk_min_base=90, lick_min_base=35, scratch_sec_base=50, shake_sec_base=15,
    ),
    dict(
        name="excess_lick_7d",
        title="情境 2：7 天過度舔舐異常資料",
        note="lick_min_base 刻意調高到 480 分鐘，預期觸發 sanity warning",
        seed=999, start_date="2026-07-01", day_count=7,
        walk_min_base=90, lick_min_base=480, scratch_sec_base=25, shake_sec_base=15,
    ),
    dict(
        name="insufficient_4d",
        title="情境 3：只有 4 天資料",
        note="天數低於 baseline_days=7 門檻，預期直接回傳「資料不足」",
        seed=42, start_date="2026-08-01", day_count=4,
        walk_min_base=90, lick_min_base=35, scratch_sec_base=25, shake_sec_base=15,
    ),
    # 想加新情境，直接照上面格式在這裡多加一個 dict 即可，
    # render_baseline_ui_preview.js 會自動把 fake_scenarios.json 裡的每一個
    # 情境都畫成一塊面板，不用改 JS。
]
# ═══════════════════════════════════════════════════════


# 各行為「平均單次長度」（秒），用來把時長換算成次數：count = round(time / 這裡的值)。
AVG_BOUT_LEN_SEC = {"walk": 90, "lick": 120, "scratch": 30, "shake": 8, "stop": 600}


def make_synthetic_day(date: str, rng: random.Random, *,
                        walk_min_base=90, lick_min_base=35,
                        scratch_sec_base=25, shake_sec_base=15,
                        monitoring_hours_range=(6, 10)):
    """純公式版：先用 base 參數 * 隨機倍率決定當天各行為時長，monitoring_seconds/
    active_time/次數/各百分比欄位都直接用公式從這些時長算出來，彼此保證一致。
    這是測試用的靜態假資料產生器，不需要真的模擬逐幀切換事件。"""
    lo, hi = monitoring_hours_range
    monitoring_seconds = round(rng.uniform(lo, hi) * 3600)

    walk_time = round(walk_min_base * 60 * rng.uniform(0.7, 1.3))
    lick_time = round(lick_min_base * 60 * rng.uniform(0.7, 1.3))
    scratch_time = round(scratch_sec_base * rng.uniform(0.6, 1.4))
    shake_time = round(shake_sec_base * rng.uniform(0.5, 1.5))

    active_non_rest_time = walk_time + lick_time + scratch_time + shake_time
    stop_time = max(0, monitoring_seconds - active_non_rest_time)

    walk_count = max(1, round(walk_time / AVG_BOUT_LEN_SEC["walk"]))
    lick_count = max(1, round(lick_time / AVG_BOUT_LEN_SEC["lick"]))
    scratch_count = max(0, round(scratch_time / AVG_BOUT_LEN_SEC["scratch"]))
    shake_count = max(0, round(shake_time / AVG_BOUT_LEN_SEC["shake"]))
    stop_count = max(1, round(stop_time / AVG_BOUT_LEN_SEC["stop"]))

    return {
        "date": date,
        "monitoring_seconds": monitoring_seconds,
        "active_time": active_non_rest_time,
        "walk_time": walk_time, "walk_count": walk_count,
        "lick_time": lick_time, "lick_count": lick_count,
        "scratch_time": scratch_time, "scratch_count": scratch_count,
        "shake_time": shake_time, "shake_count": shake_count,
        "stop_time": stop_time, "stop_count": stop_count,
        "active_ratio": round(active_non_rest_time / monitoring_seconds * 100, 2) if monitoring_seconds else 0,
        "rest_ratio": round(stop_time / monitoring_seconds * 100, 2) if monitoring_seconds else 0,
        "lick_pct_of_day": round(lick_time / monitoring_seconds * 100, 3) if monitoring_seconds else 0,
        "scratch_pct_of_day": round(scratch_time / monitoring_seconds * 100, 3) if monitoring_seconds else 0,
        "lick_pct_of_active": round(lick_time / active_non_rest_time * 100, 3) if active_non_rest_time > 0 else 0,
        "scratch_pct_of_active": round(scratch_time / active_non_rest_time * 100, 3) if active_non_rest_time > 0 else 0,
    }


def make_synthetic_history(seed: int, start_date: str, day_count: int, **day_kwargs):
    from datetime import date as _date, timedelta
    rng = random.Random(seed)
    y, m, d = (int(x) for x in start_date.split("-"))
    start = _date(y, m, d)
    return [
        make_synthetic_day((start + timedelta(days=i)).isoformat(), rng, **day_kwargs)
        for i in range(day_count)
    ]


def main():
    out = {}
    for sc in SCENARIOS:
        history = make_synthetic_history(
            sc["seed"], sc["start_date"], sc["day_count"],
            walk_min_base=sc["walk_min_base"], lick_min_base=sc["lick_min_base"],
            scratch_sec_base=sc["scratch_sec_base"], shake_sec_base=sc["shake_sec_base"],
        )
        out[sc["name"]] = {"title": sc["title"], "note": sc["note"], "history": history}

    OUTPUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已產生 {len(out)} 個情境 -> {OUTPUT_PATH}")
    for name, sc in out.items():
        print(f"  [{name}] {len(sc['history'])} 天")
    print("\n接著執行: node render_baseline_ui_preview.js")


if __name__ == "__main__":
    main()
