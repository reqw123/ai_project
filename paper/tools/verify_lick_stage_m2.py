"""在真實影片上驗證 lick_stage 第一/第二階段（M1+M2）—— headless，不開視窗、不連 Node-RED。

用途（說明書要求的 shadow comparison）：確認「來源時間契約、五種 frame_state、
事件表 lick_events / 視窗表 lick_window_summary」在真正的 YOLO+ST-GCN 管線裡
跑得動且數值合理。

影片來源（優先序）：
  1. 環境變數 TEST_VIDEO_PATH  ← settings_window.py 的「🎬 影片路徑」欄位會塞這個
     （檔案或資料夾皆可；資料夾會遞迴掃描所有 .mp4/.mov/.avi/.mkv，每支各一個 session）
  2. 命令列第一個參數：  python tools/verify_lick_stage_m2.py "D:\\clips\\lick_31.mp4"
  3. config 的 ModelPaths.VIDEO_INPUT（預設影片）

輸出：
  <輸出資料夾>/lick_m2_verify.db  （SQLite；另有同名 .lick_events.csv / .lick_window_summary.csv）
  輸出資料夾優先序：環境變數 LICK_VERIFY_OUT_DIR ＞ 預設
  paper/cat_monitoring_system/plugins/lick_stage/verify_output/
  終端印出每個 session 的事件 / 視窗 / 不變式檢查。

可由 settings_window.py →「🧩 獨立腳本工具」下拉選單選取本檔、填「🎬 影片路徑」、
按「▶ 執行所選腳本」啟動；輸出即時顯示在下方終端機面板。
"""

import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cat_monitoring_system"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from config import ModelPaths, STGCNConfig, YOLOConfig
from processors.frame_processor import FrameProcessor
from plugins.lick_stage import LickStagePlugin
from plugins.lick_stage.ext_body_zones import ExtBodyZonePlugin

_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")
_DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "cat_monitoring_system"
    / "plugins"
    / "lick_stage"
    / "verify_output"
)


def _resolve_video_source() -> str:
    env = os.getenv("TEST_VIDEO_PATH", "").strip().strip('"')
    if env:
        return env
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return sys.argv[1].strip()
    return str(ModelPaths.VIDEO_INPUT)


def _collect_videos(src: str) -> list:
    p = Path(src)
    if p.is_dir():
        vids = sorted(
            q for q in p.rglob("*") if q.suffix.lower() in _VIDEO_EXTS
        )
        return [str(q) for q in vids]
    if p.is_file():
        return [str(p)]
    return []


def _process_one(video_path: str, db_path: str) -> str:
    session_id = "S_{}_{}".format(
        Path(video_path).stem[:40].replace(" ", "_"), int(time.time())
    )
    fp = FrameProcessor(
        yolo_model_path=ModelPaths.YOLO_MODEL,
        stgcn_model_path=ModelPaths.STGCN_MODEL,
        video_path=video_path,
        nodered_url=None,
        imgsz=YOLOConfig.IMAGE_SIZE,
        conf_thres=YOLOConfig.CONFIDENCE_THRESHOLD,
        sequence_length=STGCNConfig.SEQUENCE_LENGTH,
        overlay=False,
        normalize=True,
        kp_ema_alpha=STGCNConfig.KP_EMA_ALPHA,
    )
    # 用可辨識的 session_id（含影片檔名），方便同一個 DB 裡多支影片分開查
    fp._plugin_session_id = session_id
    fp.register_plugin(LickStagePlugin(nodered_url=None, storage_db_path=db_path))
    fp.register_plugin(ExtBodyZonePlugin(nodered_enabled=False, mqtt_enabled=False))

    t0 = time.time()
    n = 0
    while True:
        ret, frame = fp.cap.read()
        if not ret:
            break
        fp.process(frame)
        n += 1
        if n % 300 == 0:
            print(f"    ...{n} 幀")
    fp.finish_plugin_sessions()
    try:
        fp.cleanup()
    except Exception:
        pass
    dt = time.time() - t0
    print(
        f"  處理完成：{n} 幀，{dt:.1f}s（{n / max(dt, 1e-6):.1f} fps，"
        f"來源 {fp._plugin_source_fps:.1f} fps）"
    )
    return session_id


def _report(db_path: str) -> None:
    if not os.path.exists(db_path):
        print("\n⚠ 沒有產生 DB 檔 —— storage 初始化可能失敗（設 logging DEBUG 看原因）")
        return
    conn = sqlite3.connect(db_path)
    print("\n" + "=" * 78)
    for (sid,) in conn.execute(
        "SELECT session_id FROM lick_sessions ORDER BY started_at"
    ).fetchall():
        _report_session(conn, sid)
    conn.close()
    base = os.path.splitext(db_path)[0]
    print(f"\nCSV：{base}.lick_events.csv / {base}.lick_window_summary.csv")


def _report_session(conn, sid: str) -> None:
    s = conn.execute(
        "SELECT video_id, source_fps, event_count, window_count, closed_cleanly, "
        "started_at, finished_at FROM lick_sessions WHERE session_id=?", (sid,)
    ).fetchone()
    print(f"\n── session {sid} ──")
    print(f"   video={s[0]}  src_fps={s[1]}  events={s[2]}  windows={s[3]}  "
          f"closed_cleanly={s[4]}  finished_at={s[6]}")

    evs = conn.execute(
        "SELECT zone_l1, zone_l2, round(start_source_ts,2), round(end_source_ts,2), "
        "round(duration_sec,3), zone_switch_count, round(assigned_ratio,2), "
        "unknown_reason_mode FROM lick_events WHERE session_id=? ORDER BY id", (sid,)
    ).fetchall()
    print(f"   lick_events（{len(evs)} 筆）: l1/l2/start/end/dur/switch/assigned_ratio/reason")
    for r in evs[:30]:
        print(f"     {r}")
    if len(evs) > 30:
        print(f"     ...（省略 {len(evs) - 30} 筆）")

    wins = conn.execute(
        "SELECT round(window_start,1), round(window_end,1), round(observed_sec,2), "
        "round(no_cat_sec,2), round(stgcn_lick_sec,3), round(assigned_zone_sec,3), "
        "round(unassigned_lick_sec,3), bout_count, round(zone_coverage_ratio,3) "
        "FROM lick_window_summary WHERE session_id=? ORDER BY id", (sid,)
    ).fetchall()
    print(f"   lick_window_summary（{len(wins)} 筆）: win/observed/no_cat/stgcn/assigned/unassigned/bouts/coverage")
    max_err = 0.0
    for r in wins:
        print(f"     [{r[0]}-{r[1]}] obs={r[2]} no_cat={r[3]} | "
              f"stgcn={r[4]} asg={r[5]} unasg={r[6]} | bouts={r[7]} cov={r[8]}")
        max_err = max(max_err, abs(r[4] - (r[5] + r[6])))

    tot = conn.execute(
        "SELECT round(sum(stgcn_lick_sec),3), round(sum(assigned_zone_sec),3), "
        "round(sum(unassigned_lick_sec),3), round(sum(no_cat_sec),3) "
        "FROM lick_window_summary WHERE session_id=?", (sid,)
    ).fetchone()
    if tot and tot[0]:
        cov = tot[1] / tot[0] if tot[0] else 0.0
        print(f"   全片: stgcn={tot[0]}s  assigned={tot[1]}s  unassigned={tot[2]}s  "
              f"no_cat={tot[3]}s  coverage={cov:.1%}")
    flag = "✓ OK" if max_err < 1e-3 else "✗ 超標"
    print(f"   不變式 |stgcn - (assigned+unassigned)| 最大誤差: {max_err:.6f}  {flag}")


def main() -> None:
    src = _resolve_video_source()
    videos = _collect_videos(src)
    if not videos:
        print(f"❌ 找不到影片：{src}\n"
              f"   （設環境變數 TEST_VIDEO_PATH，或帶命令列參數，或確認 ModelPaths.VIDEO_INPUT）")
        sys.exit(1)

    out_dir = Path(os.getenv("LICK_VERIFY_OUT_DIR", "").strip() or _DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(out_dir / "lick_m2_verify.db")
    for p in (db_path, db_path + "-wal", db_path + "-shm"):
        if os.path.exists(p):
            os.remove(p)

    print(f"影片來源 : {src}")
    print(f"待處理   : {len(videos)} 支")
    print(f"輸出 DB  : {db_path}\n")

    for i, v in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {v}")
        try:
            _process_one(v, db_path)
        except Exception as exc:
            print(f"  ⚠ 這支影片處理失敗（略過）：{exc}")

    _report(db_path)


if __name__ == "__main__":
    main()
