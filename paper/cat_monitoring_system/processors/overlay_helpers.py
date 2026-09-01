"""bbox / overlay 繪圖共用小工具。

刻意只依賴 cv2 + numpy、**不 import config 或 utils**——這樣 tools/ 底下
只把 `cat_monitoring_system/` 加進 sys.path（沒有加專案根目錄）的診斷腳本
（1_skeleton_visualizer / 1_measure_ear_distance_single_video 等）也能
`from processors.overlay_helpers import ...` 共用同一套外觀，不用各自
複製一份「裸文字」版本、也不會因為連帶 import 到 config 而爆 ModuleNotFoundError。

`processors/visualizer.py` 會 re-export 這兩個函式，維持既有 import 路徑相容。
"""

import cv2
import numpy as np


def compute_overlay_scale(width, height, base_width=1920.0, base_height=1080.0):
    """依影像對角線估算 overlay 縮放倍率，讓 bbox 標籤在不同解析度下視覺一致
    （跟 tools/1_run_video_inference.py 的 compute_ui_scale 同一套公式）。
    1080p ≈ 1.0；2560×1440 ≈ 1.33；夾在 [0.65, 2.4]。"""
    diag = float(np.hypot(max(1.0, float(width)), max(1.0, float(height))))
    base_diag = float(np.hypot(base_width, base_height))
    return float(np.clip(diag / max(base_diag, 1.0), 0.65, 2.4))


def draw_bbox_conf_label(frame, x1, y1, box_color, text, ui_scale):
    """仿 Ultralytics Annotator.box_label：在 bbox 左上角畫一塊填色標籤底 +
    深色文字，字級/內距/框線隨 ui_scale 縮放；標籤放不下框上緣時翻到框內側，
    避免被畫面裁掉。取代原本「一行裸黑字、無底、固定字級」的作法（深色貓 /
    雜亂背景上幾乎看不到）。即時系統（visualizer.draw）與 tools/ 診斷腳本共用。"""
    h_frame, w_frame = frame.shape[:2]
    fs = max(0.5, 0.62 * ui_scale)   # 低於 ~750p 的顯示畫面不讓字級縮到看不清
    th = max(1, int(round(1.5 * ui_scale)))
    (tw, tht), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    pad = max(3, int(round(5 * ui_scale)))
    box_h = tht + baseline + pad * 2
    box_w = tw + pad * 2

    fits_above = (y1 - box_h) >= 0
    ry1 = (y1 - box_h) if fits_above else y1
    ry2 = y1 if fits_above else (y1 + box_h)
    rx1 = max(0, x1)
    rx2 = min(rx1 + box_w, w_frame - 1)

    # 黑外框 + 填色底：亮框上緣也能定義出標籤邊界
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 0), max(1, int(round(1.5 * ui_scale))), cv2.LINE_AA)
    cv2.rectangle(frame, (rx1 + 1, ry1 + 1), (rx2 - 1, ry2 - 1), box_color, -1, cv2.LINE_AA)
    cv2.putText(
        frame, text, (rx1 + pad, ry2 - pad - baseline),
        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th, cv2.LINE_AA,
    )
