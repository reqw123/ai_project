from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog
import os

# ==================== 設定 ====================
IMGSZ = 640
CONF_THRES = 0.50
KP_CONF_THRES = 0.50
DEVIATION_THRES = 0.60
TOTAL_KPTS = 17

# ==================== 顏色 ====================
GREEN = (0, 255, 0)
RED   = (0, 0, 255)
BLUE  = (255, 0, 0)

COLOR_HEAD = (255, 255, 0)
COLOR_BODY = (0, 255, 0)
COLOR_TAIL = (255, 0, 255)
COLOR_KPT  = (0, 0, 255)

COLOR_LEFT_FRONT  = (255, 0, 255)   # 洋紅色 (magenta)
COLOR_RIGHT_FRONT = (0, 255, 255)   # 青色 (cyan)
COLOR_LEFT_HIND   = (255, 165, 0)   # 橙色 (orange)
COLOR_RIGHT_HIND  = (0, 255, 0)     # 綠色 (green)

# ==================== 骨架連結 ====================
HEAD_LINKS  = [(0,1),(0,2),(1,2)]
BODY_LINKS  = [(0,3),(3,4),(4,5)]
TAIL_LINKS  = [(5,14),(14,15),(15,16)]
LEFT_FRONT_LINKS  = [(3,6),(6,7)]
RIGHT_FRONT_LINKS = [(3,8),(8,9)]
LEFT_HIND_LINKS   = [(5,10),(10,11)]
RIGHT_HIND_LINKS  = [(5,12),(12,13)]

model = YOLO(r"C:\cat_pose\v11s_59.pt")

# ==================== 推論與繪製函式 ====================
def run_inference(img_path: str):
    if not img_path or not os.path.exists(img_path):
        print(f"❌ 找不到圖片檔案：{img_path}")
        return None

    try:
        img = cv2.imread(img_path)
    except FileNotFoundError:
        print(f"❌ 找不到圖片檔案：{img_path}")
        return None
    except Exception as e:
        print(f"❌ 讀取圖片失敗：{img_path}，原因：{e}")
        return None

    if img is None:
        print(f"❌ 無法讀取圖片：{img_path}")
        return None

    results = model(img, conf=CONF_THRES, imgsz=IMGSZ)[0]

    # 依解析度動態縮放繪圖參數
    h_img, w_img = img.shape[:2]
    _scale = max(w_img, h_img) / 640
    font_scale = max(0.4, 0.5 * _scale)
    font_thick = max(1, int(round(1.5 * _scale)))
    line_thick = max(1, int(round(2   * _scale)))
    box_thick  = max(1, int(round(2   * _scale)))
    circle_r   = max(3, int(round(4   * _scale)))
    pad        = max(4, int(4 * _scale))

    if results.keypoints is None:
        print("❌ 模型沒有 keypoints，這不是姿態模型")
        return None

    vis = img.copy()

    # 繪製檢測框
    if results.boxes:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, f"id_{cls_id}")
            label = f"{cls_name} {conf:.2f}"

            cv2.rectangle(vis, (x1, y1), (x2, y2), BLUE, box_thick)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            cv2.rectangle(vis, (x1, y1 - th - pad * 2), (x1 + tw + pad, y1), BLUE, -1)
            cv2.putText(vis, label, (x1 + pad // 2, y1 - pad // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)

    # 繪製骨架
    kpts     = results.keypoints.xy[0].cpu().numpy()
    kpt_conf = results.keypoints.conf[0].cpu().numpy()

    def _draw_links(link_list, color):
        for s, e in link_list:
            if kpt_conf[s] > KP_CONF_THRES and kpt_conf[e] > KP_CONF_THRES:
                cv2.line(vis, tuple(kpts[s].astype(int)), tuple(kpts[e].astype(int)), color, line_thick)

    _draw_links(HEAD_LINKS,        COLOR_HEAD)
    _draw_links(BODY_LINKS,        COLOR_BODY)
    _draw_links(TAIL_LINKS,        COLOR_TAIL)
    _draw_links(LEFT_FRONT_LINKS,  COLOR_LEFT_FRONT)
    _draw_links(RIGHT_FRONT_LINKS, COLOR_RIGHT_FRONT)
    _draw_links(LEFT_HIND_LINKS,   COLOR_LEFT_HIND)
    _draw_links(RIGHT_HIND_LINKS,  COLOR_RIGHT_HIND)

    for i, (x, y) in enumerate(kpts):
        if kpt_conf[i] > KP_CONF_THRES:
            cv2.circle(vis, (int(x), int(y)), circle_r, COLOR_KPT, -1)

    print(f"\n=== Keypoint Index & Coordinates ({img_path}) ===")
    for i, (x, y) in enumerate(kpts):
        print(f"Index {i}: ({x:.1f}, {y:.1f}) conf: {kpt_conf[i]:.2f}")

    # 縮放顯示視窗至 1920×1080 以內
    h, w = vis.shape[:2]
    scale = min(1920 / w, 1080 / h, 1.0)
    if scale < 1.0:
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return vis


def ask_image_path(_tk_root):
    _tk_root.deiconify()
    _tk_root.lift()
    _tk_root.focus_force()
    new_path = filedialog.askopenfilename(
        title="選擇圖片",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
    )
    _tk_root.withdraw()
    return new_path

# ==================== 主互動迴圈 ====================
# 隱藏 tkinter 根視窗
_tk_root = tk.Tk()
_tk_root.withdraw()

WINDOW_NAME = "Detection Result  [ O ] 開啟新圖片  [ Q / ESC ] 離開"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

current_path = r"C:\Users\homec\Downloads\side-view-cat-walking-field.jpg"
frame = run_inference(current_path)

if frame is None:
    print("⚠️ 預設圖片讀取失敗，請手動選擇要開啟的圖片。")
    new_path = ask_image_path(_tk_root)
    if new_path:
        current_path = new_path
        frame = run_inference(current_path)

while True:
    if frame is not None:
        cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(50) & 0xFF  # 50 ms 輪詢一次

    if key in (ord('q'), ord('Q'), 27):   # Q 或 ESC 離開
        break
    elif key in (ord('o'), ord('O')):     # O 開啟檔案選擇器
        new_path = ask_image_path(_tk_root)
        if new_path:
            current_path = new_path
            frame = run_inference(current_path)

cv2.destroyAllWindows()
_tk_root.destroy()
