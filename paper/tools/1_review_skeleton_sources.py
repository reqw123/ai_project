"""
逐支檢視 ST-GCN 骨架資料（skeletons/<split>/<類別>/*.json）的影片來源：照 JSON 裡記錄的
video_path 打開原始影片播放，疊上 JSON 裡存的骨架、偵測框與逐幀標籤，用來人工確認
「這筆骨架資料是從哪支影片來的、標註區間對不對、骨架有沒有抓歪」。純檢視，不修改任何骨架檔。

畫面：
  上方資訊列：[第幾支/共幾支] split / 類別 / JSON 檔名 / 是否固定 / 時間 / 幀數
  左上標籤  ：目前這一幀的標註（未標註＝灰色）
  下方時間軸：彩色＝JSON 裡標註過的區間（可點擊或拖曳跳轉）
  右下角    ：影片檔名（中文檔名也能顯示）
  畫面邊框  ：目前這一幀在標註區間內時，邊框換成該類別的顏色

啟動時在終端機問要看哪些（split／類別／檔名關鍵字／是否只看標記過的），每一題直接 Enter＝全部，
所以連按 Enter 就是讀取全部骨架資料（train → val → test，各自依類別、檔名排序）。

按鍵：
  Space   播放／暫停              a / d   上一幀／下一幀（會自動暫停）
  [ / ]   倒退／快轉 1 秒          j       跳到下一段標註區間的開頭
  2 / 1   下一支／上一支（n / p 也可以）  r   從頭播放
  k       骨架顯示開關            m       標記／取消標記這支（記在 skeletons/_review_marks.csv，之後再處理）
  o       在檔案總管中顯示這支影片  Ctrl + / Ctrl -   放大／縮小視窗
  q / Esc 結束
影片播完會停在最後一幀，不會自動跳下一支（按 2 繼續）。
"""
import bisect
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "cat_monitoring_system"))
from utils.skeleton_splits import SPLITS, UNASSIGNED, VIDEO_ROOT, iter_skeleton_files, split_of, class_folder_of  # noqa: E402
from utils.video_name_overlay import draw_video_name_label  # noqa: E402
from utils.constants import (  # noqa: E402
    EAR_DISTANCE_SKELETON_EDGES, EAR_DISTANCE_EDGE_COLORS, EAR_DISTANCE_KP_COLORS,
)
import _window_zoom  # noqa: E402  Ctrl+加號／減號縮放視窗（tools/_window_zoom.py）

# ==================== 設定 ====================
SKELETON_ROOT = Path(__file__).resolve().parents[1] / "skeletons"
RULES_FILE = SKELETON_ROOT / "_split_rules.json"      # 固定名單（gcn_dataset_manager.py 模式 2 維護）
MARKS_FILE = SKELETON_ROOT / "_review_marks.csv"      # 按 m 標記的清單
WINDOW_NAME = "Skeleton Source Review"
MAX_DISP_W, MAX_DISP_H = 1280, 720
KP_CONF_DRAW = 0.2                                     # 關鍵點信心低於此值不畫
CLASS_ORDER = ["walk", "lick", "scratch", "shake", "stop"]
# 各類別顏色（BGR）：時間軸色塊、標籤底色、區間內的邊框
CLASS_COLORS = {
    "walk": (80, 200, 80), "lick": (0, 200, 255), "scratch": (255, 150, 40),
    "shake": (200, 80, 255), "stop": (90, 90, 230),
}
UNLABELED_COLOR = (110, 110, 110)
TOPBAR_H, TIMELINE_H = 30, 34


def _natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _fmt_t(sec):
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


def _load_fixed():
    try:
        fixed = json.loads(RULES_FILE.read_text(encoding="utf-8")).get("fixed", {})
        return {v: sp for sp, vs in fixed.items() for v in vs}
    except (OSError, ValueError):
        return {}


def _load_marks():
    if not MARKS_FILE.exists():
        return {}
    marks = {}
    with open(MARKS_FILE, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("action") == "unmark":
                marks.pop(row["video_id"], None)
            else:
                marks[row["video_id"]] = row
    return marks


def _write_mark(item, action, frame_idx, t_sec):
    new = not MARKS_FILE.exists()
    with open(MARKS_FILE, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "action", "video_id", "split", "class", "frame", "time_sec", "json", "video_path"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), action, item["id"], item["split"],
                    item["cls"], frame_idx, f"{t_sec:.2f}", str(item["json"]), item["video_path"]])


# ==================== 選擇要看哪些 ====================
def build_playlist():
    files = iter_skeleton_files(SKELETON_ROOT)
    if not files:
        raise SystemExit(f"❌ {SKELETON_ROOT} 裡沒有骨架 JSON")
    counts = {}
    for p in files:
        key = (split_of(p), class_folder_of(p.stem))
        counts[key] = counts.get(key, 0) + 1
    print("=" * 64)
    print("骨架資料影片來源檢視")
    print("=" * 64)
    print(f"骨架資料夾：{SKELETON_ROOT}（共 {len(files)} 支）")
    splits_present = [s for s in list(SPLITS) + [UNASSIGNED] if any(k[0] == s for k in counts)]
    for s in splits_present:
        per = "  ".join(f"{c} {counts.get((s, c), 0)}" for c in CLASS_ORDER if counts.get((s, c)))
        print(f"  {s:<10} {per}")

    def ask(prompt, choices):
        while True:
            ans = input(prompt).strip().lower()
            if not ans:
                return None
            if ans in choices:
                return ans
            print(f"  請輸入 {' / '.join(choices)}，或直接 Enter＝全部")

    # 每一題直接 Enter＝全部（連按 Enter＝讀取全部影片）
    sp = ask(f"\n要看哪個 split？（{'/'.join(splits_present)}，Enter＝全部）：", splits_present)
    cl = ask(f"要看哪個類別？（{'/'.join(CLASS_ORDER)}，Enter＝全部）：", CLASS_ORDER)
    kw = input("檔名包含（例如 stop_5、_hq，Enter＝不篩選）：").strip().lower()
    only_marked = input("只看之前按 m 標記過的？（y／Enter＝否）：").strip().lower() == "y"

    marks = _load_marks()
    items = []
    for p in files:
        split, cls = split_of(p), class_folder_of(p.stem)
        if (sp and split != sp) or (cl and cls != cl) or (kw and kw not in p.stem.lower()):
            continue
        if only_marked and p.stem not in marks:
            continue
        items.append({"id": p.stem, "json": p, "split": split, "cls": cls})
    order = {s: i for i, s in enumerate(list(SPLITS) + [UNASSIGNED])}
    items.sort(key=lambda it: (order[it["split"]], CLASS_ORDER.index(it["cls"]) if it["cls"] in CLASS_ORDER else 99,
                               _natural(it["id"])))
    if not items:
        raise SystemExit("沒有符合條件的骨架資料。")
    print(f"\n→ 共 {len(items)} 支，開始播放（2／1＝下一支／上一支、Space＝暫停、q／Esc＝結束，其餘按鍵見腳本開頭）")
    return items


# ==================== 單支影片的資料 ====================
def load_item(item):
    d = json.loads(item["json"].read_text(encoding="utf-8"))
    md = d.get("video_metadata", {})
    frames = d.get("frames", [])
    item["video_path"] = md.get("video_path", "")
    item["frames"] = frames
    item["ts"] = [f.get("timestamp", i / 30.0) or 0.0 for i, f in enumerate(frames)]
    item["target_fps"] = md.get("target_fps", 30) or 30
    # 標註區間（秒）：優先用 action_intervals（以 JSON 幀索引記錄），沒有就從逐幀 label 推
    ivs = []
    if d.get("action_intervals"):
        for iv in d["action_intervals"]:
            s, e = int(iv["start"]), int(iv["end"])
            if 0 <= s < len(frames):
                ivs.append((item["ts"][s], item["ts"][min(e, len(frames) - 1)], iv["action"]))
    else:
        cur, start = None, 0
        for i, f in enumerate(frames + [{"label": None}]):
            lab = f.get("label")
            lab = None if lab in (None, "unannotated") else lab
            if lab != cur:
                if cur:
                    ivs.append((item["ts"][start], item["ts"][i - 1], cur))
                cur, start = lab, i
    item["intervals"] = ivs
    item["n_labeled"] = sum(1 for f in frames if f.get("label", "unannotated") != "unannotated")
    return item


def find_video(item):
    p = item["video_path"]
    if p and os.path.exists(p):
        return p
    # 影片被搬走時：先在 模型專用/<split>/<類別>/ 依檔名找（唯一一支才用；正常情況下
    # gcn_dataset_manager 搬動時會同步更新 JSON，不會走到這裡），再到原本資料夾的上一層找
    name = Path(p).name if p else f"{item['id']}.mp4"
    cands = list(VIDEO_ROOT.glob(f"*/*/{name}")) if VIDEO_ROOT.exists() else []
    if len(cands) == 1:
        return str(cands[0])
    base = Path(p).parent.parent if p else None
    if base and base.exists():
        for cand in [base / item["cls"] / name] + list(base.glob(f"*/{name}")):
            if cand.exists():
                return str(cand)
    return None


def json_frame_at(item, t_sec):
    """時間 t 對應的 JSON 幀（取時間最接近的一幀；超出半幀以上就視為沒有對應）。"""
    ts = item["ts"]
    if not ts:
        return None
    i = bisect.bisect_left(ts, t_sec)
    best = min((j for j in (i - 1, i) if 0 <= j < len(ts)), key=lambda j: abs(ts[j] - t_sec))
    return item["frames"][best] if abs(ts[best] - t_sec) <= 0.75 / item["target_fps"] else None


# ==================== 繪圖 ====================
def draw_skeleton(img, jf, scale, ox, oy, show_skeleton):
    if jf is None or not jf.get("detected"):
        return
    bb = jf.get("bbox")
    if bb and len(bb) == 4:
        x1, y1, x2, y2 = [int(v * scale) for v in bb]
        cv2.rectangle(img, (x1 + ox, y1 + oy), (x2 + ox, y2 + oy), (200, 200, 200), 1, cv2.LINE_AA)
    if not show_skeleton:
        return
    kd = {k["joint_id"]: k for k in jf.get("keypoints", [])}
    lw = max(1, int(img.shape[1] / 480))
    for ei, (a, b) in enumerate(EAR_DISTANCE_SKELETON_EDGES):
        ka, kb = kd.get(a), kd.get(b)
        if ka and kb and ka["conf"] > KP_CONF_DRAW and kb["conf"] > KP_CONF_DRAW:
            cv2.line(img, (int(ka["x"] * scale) + ox, int(ka["y"] * scale) + oy),
                     (int(kb["x"] * scale) + ox, int(kb["y"] * scale) + oy),
                     EAR_DISTANCE_EDGE_COLORS[ei % len(EAR_DISTANCE_EDGE_COLORS)], lw, cv2.LINE_AA)
    r = max(3, int(img.shape[1] / 300))
    for j, k in kd.items():
        if k["conf"] > KP_CONF_DRAW:
            c = (int(k["x"] * scale) + ox, int(k["y"] * scale) + oy)
            cv2.circle(img, c, r + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, c, r, EAR_DISTANCE_KP_COLORS[j % len(EAR_DISTANCE_KP_COLORS)], -1, cv2.LINE_AA)


def put_text(img, text, org, fs, color, th=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)


class Player:
    def __init__(self, items):
        self.items = items
        self.idx = 0
        self.fixed = _load_fixed()
        self.marks = _load_marks()
        self.show_skeleton = True
        self.zoom = 1.0
        self.seek_to = None           # 滑鼠點時間軸要跳去的秒數
        self.dragging = False
        self.timeline_rect = None     # (x0, y0, w, h)，在顯示畫面座標
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

    # ── 滑鼠：點擊／拖曳時間軸跳轉 ──
    def _on_mouse(self, event, x, y, flags, _param):
        if self.timeline_rect is None:
            return
        tx, ty, tw, th = self.timeline_rect
        inside = tx <= x <= tx + tw and ty - 6 <= y <= ty + th + 6
        if event == cv2.EVENT_LBUTTONDOWN and inside:
            self.dragging = True
        if event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        if (event == cv2.EVENT_LBUTTONDOWN and inside) or (event == cv2.EVENT_MOUSEMOVE and self.dragging):
            self.seek_to = max(0.0, min(1.0, (x - tx) / max(tw, 1))) * self.duration

    def run(self):
        while 0 <= self.idx < len(self.items):
            step = self.play_one(self.items[self.idx])
            if step == "quit":
                break
            self.idx = max(0, min(len(self.items) - 1, self.idx + step)) if step else self.idx
        cv2.destroyAllWindows()
        if self.marks:
            print(f"\n已標記 {len(self.marks)} 支（{MARKS_FILE}）：{', '.join(sorted(self.marks, key=_natural))}")

    def play_one(self, item):
        load_item(item)
        vpath = find_video(item)
        fixed = self.fixed.get(item["id"])
        print(f"\n[{self.idx + 1}/{len(self.items)}] {item['split']} / {item['cls']} / {item['id']}"
              f"{'  （固定在 ' + fixed + '）' if fixed else ''}")
        print(f"    影片：{item['video_path']}" + ("" if vpath else "  ⚠ 找不到影片，只顯示骨架"))
        print(f"    標註區間：" + ("、".join(f"{a} {_fmt_t(s)}~{_fmt_t(e)}" for s, e, a in item["intervals"]) or "無"))

        cap = cv2.VideoCapture(vpath) if vpath else None
        if cap is not None and cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            cap, fps, n = None, float(item["target_fps"]), max(1, len(item["frames"]))
            vw, vh = 1280, 720
        self.duration = (n - 1) / fps if n > 1 else 0.0
        base_scale = min(MAX_DISP_W / vw, MAX_DISP_H / vh, 1.0)

        pos, paused, frame, last_read = 0, False, None, -1
        delay = max(1, int(1000 / fps))
        while True:
            if self.seek_to is not None:
                pos = int(round(self.seek_to * fps))
                self.seek_to = None
            pos = max(0, min(n - 1, pos))
            if pos != last_read:
                if cap is not None:
                    if pos != last_read + 1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    ok, fr = cap.read()
                    frame = fr if ok else (frame if frame is not None else np.zeros((vh, vw, 3), np.uint8))
                else:
                    frame = np.full((vh, vw, 3), 30, np.uint8)
                last_read = pos

            t = pos / fps
            jf = json_frame_at(item, t)
            disp = self.render(item, frame, jf, t, pos, n, fixed, base_scale * self.zoom)
            cv2.imshow(WINDOW_NAME, disp)

            key = cv2.waitKey(delay if not paused else 30) & 0xFF
            zoom = _window_zoom.poll(WINDOW_NAME)
            if zoom:
                self.zoom = _window_zoom.clamp_scale(self.zoom, zoom, step=0.10, lo=0.5, hi=2.0)
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return "quit"
            if key in (ord("q"), 27):
                return "quit"
            if key in (ord("2"), ord("n")):
                return 1
            if key in (ord("1"), ord("p")):
                return -1
            if key == ord(" "):
                paused = not paused
                if pos >= n - 1 and not paused:
                    pos, last_read = 0, -1          # 已經在最後一幀又按播放 → 從頭
            elif key == ord("a"):
                paused, pos = True, pos - 1
            elif key == ord("d"):
                paused, pos = True, pos + 1
            elif key == ord("["):
                pos -= int(fps)
            elif key == ord("]"):
                pos += int(fps)
            elif key == ord("r"):
                pos, paused = 0, False
            elif key == ord("j"):
                nxt = [s for s, _, _ in item["intervals"] if s > t + 1e-3]
                if nxt:
                    pos = int(round(nxt[0] * fps))
            elif key == ord("k"):
                self.show_skeleton = not self.show_skeleton
            elif key == ord("m"):
                if item["id"] in self.marks:
                    self.marks.pop(item["id"])
                    _write_mark(item, "unmark", pos, t)
                    print(f"    取消標記 {item['id']}")
                else:
                    self.marks[item["id"]] = {}
                    _write_mark(item, "mark", pos, t)
                    print(f"    ★ 已標記 {item['id']}（第 {pos} 幀，{_fmt_t(t)}）")
            elif key == ord("o") and vpath:
                subprocess.Popen(["explorer", "/select,", os.path.normpath(vpath)])
            elif not paused and not self.dragging:
                if pos >= n - 1:
                    paused = True                   # 播完停在最後一幀，不自動換下一支
                else:
                    pos += 1

    def render(self, item, frame, jf, t, pos, n, fixed, scale):
        h, w = frame.shape[:2]
        dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
        video = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        s = max(0.6, dw / 960.0)
        top, bot = int(TOPBAR_H * s), int(TIMELINE_H * s)
        canvas = np.zeros((dh + top + bot, dw, 3), np.uint8)
        canvas[top:top + dh] = video
        draw_skeleton(canvas, jf, scale, 0, top, self.show_skeleton)

        # 目前這一幀的標註：在區間內就用類別顏色畫邊框＋左上標籤
        lab = jf.get("label") if jf else None
        lab = None if lab in (None, "unannotated") else lab
        col = CLASS_COLORS.get(lab, UNLABELED_COLOR)
        if lab:
            cv2.rectangle(canvas, (0, top), (dw - 1, top + dh - 1), col, max(2, int(3 * s)))
        tag = f" {lab.upper()} " if lab else " unlabeled "
        (tw, th_), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, max(1, int(2 * s)))
        cv2.rectangle(canvas, (8, top + 8), (8 + tw + 8, top + 8 + th_ + 12), col, -1)
        cv2.putText(canvas, tag, (12, top + 8 + th_ + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (0, 0, 0),
                    max(1, int(2 * s)), cv2.LINE_AA)
        if jf is not None and not jf.get("detected"):
            put_text(canvas, "NO DETECT", (12, top + 8 + th_ + 40), 0.6 * s, (0, 0, 255), max(1, int(s)))
        elif jf is None:
            put_text(canvas, "no JSON frame", (12, top + 8 + th_ + 40), 0.55 * s, (0, 160, 255), max(1, int(s)))

        # 上方資訊列
        mark = "  * MARKED" if item["id"] in self.marks else ""
        info = (f"[{self.idx + 1}/{len(self.items)}] {item['split'].upper()} / {item['cls']} / {item['id']}.json"
                f"{'  | FIXED:' + fixed if fixed else ''}  | {_fmt_t(t)} / {_fmt_t(self.duration)}"
                f"  | frame {pos + 1}/{n}  | labeled {item['n_labeled']}/{len(item['frames'])}{mark}")
        cv2.putText(canvas, info, (8, int(top * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.46 * s,
                    (120, 220, 255) if not mark else (80, 220, 255), max(1, int(s)), cv2.LINE_AA)

        # 下方時間軸（彩色＝標註區間）
        mx = int(8 * s)
        ty, tbh = top + dh + int(8 * s), bot - int(16 * s)
        tw_ = dw - mx * 2
        self.timeline_rect = (mx, ty, tw_, tbh)
        cv2.rectangle(canvas, (mx, ty), (mx + tw_, ty + tbh), (55, 55, 55), -1)
        dur = max(self.duration, 1e-6)
        for s0, s1, act in item["intervals"]:
            x0, x1 = mx + int(s0 / dur * tw_), mx + int(s1 / dur * tw_)
            cv2.rectangle(canvas, (x0, ty + 2), (max(x0 + 1, x1), ty + tbh - 2), CLASS_COLORS.get(act, (160, 160, 160)), -1)
        cx = mx + int(min(t, dur) / dur * tw_)
        cv2.line(canvas, (cx, ty - 3), (cx, ty + tbh + 3), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (mx, ty), (mx + tw_, ty + tbh), (130, 130, 130), 1)

        draw_video_name_label(canvas, item.get("video_path") or item["id"], self.idx, len(self.items),
                              ui_scale=s, bottom_offset=bot)
        return canvas


def main():
    items = build_playlist()
    Player(items).run()


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\n已結束。")
