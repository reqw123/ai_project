"""
單張圖片手動姿態分類與歸檔工具
=======================================================
讀取 SOURCE_FOLDER 底下所有圖片，用 YOLO-Pose 畫出骨架/bbox 作為人工判斷
的視覺輔助（單張圖片沒有時間序列可用，不會像 1_classify_and_sort_videos.py
那樣跑 ST-GCN 自動分類），開啟 GUI 逐張顯示；使用者依畫面判斷貓咪姿態後，
按 A/B/C/D/E（對應 BEHAVIOR_CLASSES 順序：walk/lick/scratch/shake/stop）
把該圖片的原始檔案搬進 <SOURCE_FOLDER>/image_sort/<behavior>/。

兩階段 GUI：python 1_classify_and_sort_images.py
  1. 逐張人工分類視窗（上述 A-E 快捷鍵）；Esc／關閉視窗即結束這一階段。
  2. 分類結束後跳出「批次序號命名」視窗（_rename_gui.py，以獨立子行程啟動，壞掉也不影響分類），列出五個 image_sort/<behavior>
     資料夾各自的檔案，可對選定類別的檔案改名成「<行為>_<序號>」（起始序號預設接在資料夾內
     既有最大序號之後，目標檔名被未選取的檔案佔用時會擋下，不會覆蓋其他來源的圖片）；
     直接關閉視窗＝跳過。
"""
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

# 加入系統路徑，讓 detectors/utils 等主專案模組可以被 import
sys.path.insert(0, str(Path(__file__).parent.parent / "cat_monitoring_system"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import YOLOConfig as _YOLOConfig
from detectors.keypoint_detector import KeypointDetector
from utils.constants import (
    BEHAVIOR_CLASSES,
    EAR_DISTANCE_SKELETON_EDGES as SKELETON_EDGES,
    EAR_DISTANCE_KP_COLORS as KP_COLORS,
    EAR_DISTANCE_EDGE_COLORS as EDGE_COLORS,
    BLACK,
    COLOR_HEAD,
)

# ==================== 使用者設定區 ====================
SOURCE_FOLDER = r"C:\Users\homec\OneDrive\圖片\Screenshots\f"  # 待分類圖片所在資料夾（單層，不含子資料夾）

# 若設定 TEST_VIDEO_PATH 環境變數且指向資料夾，優先使用該資料夾（覆蓋上面寫死的
# SOURCE_FOLDER，對應 settings_window.py 的「🎬 影片路徑」欄位；本腳本處理的是圖片
# 資料夾而非單一影片檔，所以只在填的是資料夾路徑時才生效）
_env_test_video = os.getenv("TEST_VIDEO_PATH", "").strip()
if _env_test_video and os.path.isdir(_env_test_video):
    SOURCE_FOLDER = _env_test_video

YOLO_MODEL_PATH = r"C:\ai_project\yolo_models\v11s_149.pt"

# 若設定 YOLO_MODEL_PATH 環境變數，優先使用該模型路徑（覆蓋上面寫死的 YOLO_MODEL_PATH，
# 對應 settings_window.py 的「🧠 模型路徑」欄位）
_env_yolo_model = os.getenv("YOLO_MODEL_PATH", "").strip()
if _env_yolo_model:
    YOLO_MODEL_PATH = _env_yolo_model

INFERENCE_DEVICE = "cuda"
YOLO_IMGSZ = _YOLOConfig.IMAGE_SIZE  # 跟主系統同步（設定視窗 yolo.image_size／環境變數 CAT_MONITORING_YOLO_IMAGE_SIZE，預設 640）
YOLO_CONF_THRESHOLD = 0.5  # YOLO bbox 偵測信心門檻（不是關鍵點 kp 信心門檻）
import os as _os
_env_yolo_conf = _os.getenv("CAT_MONITORING_YOLO_CONFIDENCE_THRESHOLD", "").strip()  # 設定視窗「⚙ 額外設定」可覆寫這個 bbox 信心門檻；環境變數名同 config.py 的 YOLOConfig.CONFIDENCE_THRESHOLD
if _env_yolo_conf:
    try:
        YOLO_CONF_THRESHOLD = float(_env_yolo_conf)
    except ValueError:
        print(f"⚠ 環境變數 CAT_MONITORING_YOLO_CONFIDENCE_THRESHOLD={_env_yolo_conf!r} 不是數字，沿用預設 {YOLO_CONF_THRESHOLD}")
DRAW_KP_CONF_THRESHOLD = 0.5  # 畫骨架線段/關鍵點圓點用門檻（>此值才畫），純視覺輔助不影響分類
import os as _os
_env_kp_conf = _os.getenv("CAT_MONITORING_KP_CONF_THRES", "").strip()  # 設定視窗「⚙ 額外設定」可覆寫這個關鍵點（kp）信心門檻；環境變數名同 config.py 的 AnomalyDetectionConfig.KP_CONF_THRES
if _env_kp_conf:
    try:
        DRAW_KP_CONF_THRESHOLD = float(_env_kp_conf)
    except ValueError:
        print(f"⚠ 環境變數 CAT_MONITORING_KP_CONF_THRES={_env_kp_conf!r} 不是數字，沿用預設 {DRAW_KP_CONF_THRESHOLD}")

# 分類搬檔資料夾建立在「來源資料夾」底下（<SOURCE_FOLDER>/image_sort/<behavior>）。
# 1_run_video_inference.py 的 Shift+A~E 也是同一組字母對應，但直接移進行為資料夾（見 move_video_to_class_folder()）。
IMAGE_SORT_FOLDER_NAME = "image_sort"

# A-E 對應 BEHAVIOR_CLASSES 順序：A=walk  B=lick  C=scratch  D=shake  E=stop
SORT_LETTERS = ("A", "B", "C", "D", "E")
LETTER_TO_BEHAVIOR = dict(zip(SORT_LETTERS, BEHAVIOR_CLASSES))

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 批次序號命名視窗是「完全獨立的選用擴充」：以獨立子行程執行 _rename_gui.py，本腳本不 import 它，
# 也不共用任何程式碼。它壞掉（語法錯誤／檔案遺失／例外／當機）都只會讓子行程失敗，這裡只印警告，
# 不影響已完成的分類與歸檔。（另一支 1_classify_and_sort_videos.py 有一份刻意獨立維護的相同啟動器，
# 不抽成共用模組，以免兩支腳本又因為共用程式碼而互相牽連。）
_RENAME_GUI_SCRIPT = Path(__file__).resolve().parent / "_rename_gui.py"


def run_optional_rename_window(dest_folders, exts, fresh_files):
    try:
        payload = json.dumps({
            "dest_folders": {name: str(path) for name, path in dest_folders.items()},
            "exts": sorted(exts),
            "fresh_files": [str(p) for p in fresh_files],
        })
        result = subprocess.run([sys.executable, str(_RENAME_GUI_SCRIPT)], input=payload, text=True)
        if result.returncode != 0:
            print(f"⚠ 批次命名視窗異常結束（exit code {result.returncode}），已略過；分類結果不受影響")
    except Exception as exc:  # noqa: BLE001 — 選用功能，任何錯誤都不該影響本腳本
        print(f"⚠ 無法啟動批次命名視窗（{type(exc).__name__}: {exc}），已略過；分類結果不受影響")


# ==================== 工具函式 ====================
def list_images(folder):
    p = Path(folder)
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    return sorted(files, key=lambda f: f.name.lower())


def move_image_to_behavior_folder(image_path, behavior_name):
    """把圖片搬到 <來源資料夾>/image_sort/<behavior_name>/，檔名重複時自動加流水號後綴。"""
    src = Path(image_path)
    dest_dir = src.parent / IMAGE_SORT_FOLDER_NAME / behavior_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    n = 1
    while dest.exists():
        dest = dest_dir / f"{src.stem}_{n}{src.suffix}"
        n += 1
    shutil.move(str(src), str(dest))
    return dest


def draw_skeleton_overlay(frame, kpts, kpt_conf, bbox, bbox_conf, conf_thresh=DRAW_KP_CONF_THRESHOLD):
    """畫骨架線段、關鍵點與 bbox，純視覺輔助給人工判斷用，不影響分類結果。"""
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BLACK, 4, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_HEAD, 2, cv2.LINE_AA)
        if bbox_conf is not None:
            label = f"{float(bbox_conf):.2f}"
            cv2.putText(
                frame, label, (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HEAD, 1, cv2.LINE_AA,
            )

    if kpts is None:
        return frame

    for ei, (a, b) in enumerate(SKELETON_EDGES):
        if float(kpt_conf[a]) > conf_thresh and float(kpt_conf[b]) > conf_thresh:
            pa = (int(kpts[a][0]), int(kpts[a][1]))
            pb = (int(kpts[b][0]), int(kpts[b][1]))
            col = EDGE_COLORS[ei] if ei < len(EDGE_COLORS) else (180, 180, 180)
            cv2.line(frame, pa, pb, col, 2, cv2.LINE_AA)

    for i in range(min(17, len(kpts))):
        if float(kpt_conf[i]) > conf_thresh:
            cx, cy = int(kpts[i][0]), int(kpts[i][1])
            col = KP_COLORS[i] if i < len(KP_COLORS) else (200, 200, 200)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 0), -1)
            cv2.circle(frame, (cx, cy), 3, col, -1)

    return frame


def build_preview(image_path, keypoint_detector):
    """跑 YOLO 偵測並畫上骨架輔助圖，回傳 (preview_bgr, has_cat)；讀圖失敗回傳 (None, False)。"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None, False

    # 每張圖片彼此獨立、無時間序列關係，偵測前重置追蹤狀態，避免上一張圖片
    # 鎖定的 bbox 影響這張圖片的多貓篩選結果。
    keypoint_detector.reset_track()
    kpts, kpt_conf, bbox, bbox_conf = keypoint_detector.detect(img)
    has_cat = kpts is not None
    preview = draw_skeleton_overlay(img.copy(), kpts, kpt_conf, bbox, bbox_conf)
    if not has_cat:
        banner_h = min(max(48, preview.shape[0] // 10), 100)
        cv2.rectangle(preview, (0, 0), (preview.shape[1], banner_h), (0, 0, 170), -1)
        cv2.putText(
            preview, "NO CAT DETECTED - JUDGE BY EYE",
            (12, max(30, int(banner_h * 0.65))),
            cv2.FONT_HERSHEY_SIMPLEX, max(0.45, min(preview.shape[:2]) / 900),
            (255, 255, 255), 2, cv2.LINE_AA,
        )
    return preview, has_cat


# ==================== GUI ====================
def launch_sort_gui(image_paths, keypoint_detector):
    """逐張顯示 YOLO 骨架輔助圖，讓使用者依 A-E 手動分類、搬檔到對應行為資料夾。

    回傳這次實際搬進各行為資料夾的檔案路徑 list（給第二階段命名視窗標「＊」用）；
    GUI 根本沒能開起來（沒圖片／缺 Pillow）時回傳 None。"""
    if not image_paths:
        print("⚠ 沒有待分類的圖片")
        return None

    try:
        import tkinter as tk
        from PIL import Image, ImageOps, ImageTk
    except ImportError as exc:
        print(f"⚠ 無法開啟分類 GUI：{exc}")
        print("請先安裝 Pillow：pip install pillow")
        return None

    remaining = list(image_paths)
    moved_files = []  # 這次實際搬進各行為資料夾的檔案（目的地路徑）
    current_index = 0
    current_photo = None
    preview_cache: dict = {}  # 已算過 YOLO 偵測的圖片快取，前後翻頁不用重跑
    sorted_counts = {name: 0 for name in BEHAVIOR_CLASSES}
    resize_job = None

    root = tk.Tk()
    root.title("圖片姿態手動分類（YOLO 骨架輔助）")
    root.geometry("1400x950")
    root.minsize(800, 600)
    root.configure(bg="#111827")
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(2, weight=1, minsize=300)
    try:
        root.state("zoomed")
    except tk.TclError:
        pass

    title_label = tk.Label(
        root, text="圖片姿態手動分類",
        font=("Microsoft JhengHei UI", 13, "bold"), fg="#F9FAFB", bg="#111827",
    )
    title_label.grid(row=0, column=0, sticky="ew", pady=(5, 0))

    status_label = tk.Label(
        root, text="", font=("Microsoft JhengHei UI", 9), fg="#CBD5E1", bg="#111827",
    )
    status_label.grid(row=1, column=0, sticky="ew", pady=(0, 3))

    image_frame = tk.Frame(root, bg="#030712", highlightthickness=2, highlightbackground="#374151")
    image_frame.grid(row=2, column=0, sticky="nsew", padx=6, pady=2)
    image_frame.grid_propagate(False)

    image_label = tk.Label(
        image_frame, text="正在載入圖片...",
        font=("Microsoft JhengHei UI", 14), fg="#E5E7EB", bg="#030712",
    )
    image_label.pack(fill="both", expand=True, padx=2, pady=2)

    source_label = tk.Label(
        root, text="", font=("Microsoft JhengHei UI", 9), fg="#D1D5DB", bg="#1F2937",
        anchor="w", justify="left", padx=7, pady=3,
    )
    source_label.grid(row=3, column=0, sticky="ew", padx=6, pady=(3, 2))

    help_label = tk.Label(
        root,
        text="快捷鍵：← 上一張　→／Space 下一張　A walk　B lick　C scratch　D shake　E stop　Esc 關閉",
        font=("Microsoft JhengHei UI", 8), fg="#94A3B8", bg="#111827",
    )
    help_label.grid(row=4, column=0, sticky="ew", pady=(0, 2))

    button_frame = tk.Frame(root, bg="#111827")
    button_frame.grid(row=5, column=0, sticky="ew", padx=6, pady=(1, 6))
    for column in range(2 + len(SORT_LETTERS)):
        button_frame.grid_columnconfigure(column, weight=1, uniform="actions")

    def make_button(parent, text, command, bg, column):
        btn = tk.Button(
            parent, text=text, command=command,
            font=("Microsoft JhengHei UI", 9, "bold"), fg="white", bg=bg,
            activeforeground="white", activebackground=bg,
            relief="flat", bd=0, padx=5, pady=4, cursor="hand2",
        )
        btn.grid(row=0, column=column, sticky="ew", padx=3)
        return btn

    def current_path():
        if not remaining:
            return None
        return remaining[current_index]

    def render_current():
        nonlocal current_photo

        path = current_path()
        if path is None:
            image_label.configure(image="", text="所有圖片已分類完成")
            source_label.configure(text="")
            summary = "  ".join(f"{name}:{sorted_counts[name]}" for name in BEHAVIOR_CLASSES)
            status_label.configure(text=f"完成！本次分類統計 → {summary}")
            current_photo = None
            return

        if path not in preview_cache:
            preview_cache[path] = build_preview(path, keypoint_detector)
        preview, has_cat = preview_cache[path]

        summary = "  ".join(f"{name}:{sorted_counts[name]}" for name in BEHAVIOR_CLASSES)
        status_label.configure(
            text=(
                f"剩餘 {len(remaining)} 張　｜　第 {current_index + 1} / {len(remaining)} 張　｜　"
                f"{'✅ 偵測到貓' if has_cat else '⚠ 未偵測到貓'}　｜　已分類 → {summary}"
            ),
            fg="#CBD5E1" if has_cat else "#FCA5A5",
        )
        source_label.configure(text=f"原始圖片：{path}")

        if preview is None:
            current_photo = None
            image_label.configure(image="", text=f"無法讀取圖片\n{path}")
            return

        try:
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            display_img = ImageOps.exif_transpose(Image.fromarray(rgb)).convert("RGB")

            max_w = max(500, image_frame.winfo_width() - 8)
            max_h = max(350, image_frame.winfo_height() - 8)
            resampling = getattr(Image, "Resampling", Image).LANCZOS

            scale = min(max_w / display_img.width, max_h / display_img.height)
            display_size = (
                max(1, int(display_img.width * scale)),
                max(1, int(display_img.height * scale)),
            )
            if display_size != display_img.size:
                display_img = display_img.resize(display_size, resampling)
            current_photo = ImageTk.PhotoImage(display_img)
            image_label.configure(image=current_photo, text="")
        except Exception as exc:
            current_photo = None
            image_label.configure(image="", text=f"無法顯示預覽圖\n{path}\n\n{exc}")

    def previous_image(event=None):
        nonlocal current_index
        if remaining:
            current_index = (current_index - 1) % len(remaining)
            render_current()

    def next_image(event=None):
        nonlocal current_index
        if remaining:
            current_index = (current_index + 1) % len(remaining)
            render_current()

    def sort_current(letter):
        nonlocal current_index

        path = current_path()
        if path is None:
            return
        behavior_name = LETTER_TO_BEHAVIOR[letter]

        try:
            dest = move_image_to_behavior_folder(path, behavior_name)
            moved_files.append(dest)
            print(f"  [{letter}] {path.name} → {behavior_name} ({dest})")
        except Exception as exc:
            status_label.configure(text=f"⚠ 搬移失敗：{exc}", fg="#FCA5A5")
            return

        preview_cache.pop(path, None)
        del remaining[current_index]
        sorted_counts[behavior_name] += 1
        if remaining:
            current_index = current_index % len(remaining)
        render_current()

    previous_button = make_button(button_frame, "← 上一張", previous_image, "#2563EB", 0)
    next_button = make_button(button_frame, "下一張 →", next_image, "#2563EB", 1)
    for i, letter in enumerate(SORT_LETTERS):
        behavior_name = LETTER_TO_BEHAVIOR[letter]
        make_button(
            button_frame, f"{letter} {behavior_name}",
            lambda l=letter: sort_current(l), "#15803D", 2 + i,
        )

    root.bind("<Left>", previous_image)
    root.bind("<Right>", next_image)
    root.bind("<space>", next_image)
    root.bind("<Escape>", lambda event: root.destroy())
    for letter in SORT_LETTERS:
        root.bind(f"<Key-{letter}>", lambda event, l=letter: sort_current(l))
        root.bind(f"<Key-{letter.lower()}>", lambda event, l=letter: sort_current(l))

    def rerender_after_resize():
        nonlocal resize_job
        resize_job = None
        render_current()

    def schedule_rerender(event):
        nonlocal resize_job
        if event.widget is not root:
            return
        if resize_job is not None:
            root.after_cancel(resize_job)
        resize_job = root.after(120, rerender_after_resize)

    root.bind("<Configure>", schedule_rerender)

    root.after(100, render_current)
    root.mainloop()
    return moved_files


# ==================== 主流程 ====================
def main():
    src = Path(SOURCE_FOLDER)
    if not src.is_dir():
        print(f"❌ 來源資料夾不存在: {SOURCE_FOLDER}")
        return

    images = list_images(src)
    if not images:
        print(f"❌ 找不到圖片: {SOURCE_FOLDER}")
        return

    print("=" * 60)
    print("圖片姿態手動分類工具")
    print("=" * 60)
    print(f"來源資料夾: {SOURCE_FOLDER}")
    print(f"待分類圖片共 {len(images)} 張")
    print(f"分類搬檔位置: {src / IMAGE_SORT_FOLDER_NAME}\\<behavior>")
    print("按鍵對應：" + "　".join(f"{k}={v}" for k, v in LETTER_TO_BEHAVIOR.items()))
    print("=" * 60)

    print("\n初始化 YOLO-Pose 模型...")
    keypoint_detector = KeypointDetector(
        YOLO_MODEL_PATH, device=INFERENCE_DEVICE, imgsz=YOLO_IMGSZ, conf_thres=YOLO_CONF_THRESHOLD
    )
    print("✅ 模型載入成功\n")

    for name in BEHAVIOR_CLASSES:
        (src / IMAGE_SORT_FOLDER_NAME / name).mkdir(parents=True, exist_ok=True)

    moved_files = launch_sort_gui(images, keypoint_detector)
    print("\n✓ 分類作業結束。")
    if moved_files is None:  # 分類 GUI 沒能開起來，不需要接著開命名視窗
        return

    print(f"本次歸檔 {len(moved_files)} 張。開啟批次序號命名視窗（直接關閉視窗即跳過）...")
    run_optional_rename_window(
        {name: str(src / IMAGE_SORT_FOLDER_NAME / name) for name in BEHAVIOR_CLASSES},
        SUPPORTED_IMAGE_EXTS,
        moved_files,
    )
    print("結束。")


if __name__ == "__main__":
    main()
