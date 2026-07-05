"""
YOLO Pose 資料增強（含關鍵點/BBox 座標同步變換）
=======================================================
與 augmentation.py 的差異：augmentation.py 只增強影像本身，不會處理標註檔，
幾何類增強（翻轉/旋轉/裁剪/縮放）套用後，YOLO label 的 bbox 與關鍵點座標會
跟原圖對不上。這支腳本讀取 images/ + labels/ 成對資料，幾何類增強會同步
正確變換 bbox 與關鍵點座標；純色彩類增強（模糊/亮度/對比/噪點...）不影響
座標，直接複製原始 label。

座標格式偵測：自動讀取每個 label 檔的關鍵點總數（YOLO pose 格式：
class cx cy w h [x y v]*N），並自動找出「真正有標註（v>0）」的那一段連續
區間視為實際的 17 點貓骨架，其餘視為 padding（本專案 11.yolov8 匯出檔
kpt_shape 錯誤寫成 125，只有最後 17 個 slot 是真實資料，其餘皆為 0 padding）。
輸出的 label 只會寫入真實的 17 個關鍵點（1+4+17*3=56 欄），不會照抄輸入檔的
125-slot padding——Ultralytics 依 data.yaml 的 kpt_shape 決定每行欄數，欄數
對不上會整批被跳過（"labels require 56 columns each"）。

水平翻轉時，貓骨架的左右關節（耳朵/前肢/後肢）會依 FLIP_SWAP_REAL 對應交換，
避免翻轉後「左耳」座標其實變成貼在右耳位置卻沒交換標籤。
"""
import cv2
import numpy as np
import glob
import os
import random
import uuid
from pathlib import Path

# ==================== 輸入 / 輸出路徑 ====================
INPUT_IMAGES_DIR = r"C:\ai_project\paper\11.yolov8\train\images"
INPUT_LABELS_DIR = r"C:\ai_project\paper\11.yolov8\train\labels"
OUTPUT_IMAGES_DIR = r"C:\ai_project\paper\11.yolov8\train_augmented\images"
OUTPUT_LABELS_DIR = r"C:\ai_project\paper\11.yolov8\train_augmented\labels"

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# ==================== 17 點貓骨架設定 ====================
# 0 Nose, 1 L_Ear, 2 R_Ear, 3 Chest, 4 Mid_Back, 5 Hip,
# 6 LF_Elbow, 7 LF_Paw, 8 RF_Elbow, 9 RF_Paw,
# 10 LH_Knee, 11 LH_Paw, 12 RH_Knee, 13 RH_Paw,
# 14 Tail_Root, 15 Tail_Mid, 16 Tail_Tip
# 水平翻轉時需要交換的左右關節對（索引皆相對「真實 17 點區塊」內）
FLIP_SWAP_REAL = [(1, 2), (6, 8), (7, 9), (10, 12), (11, 13)]

# ============================================================
# ✔ 增強功能開關 (True=啟用, False=關閉)
# ============================================================
# 座標類（幾何）增強全部關閉：不動座標，label 直接沿用原始值，不需要走任何變換數學
USE_HORIZONTAL_FLIP = False     # 水平翻轉（座標會交換左右關節）
USE_VERTICAL_FLIP = False       # 垂直翻轉
USE_ROTATION = False            # 隨機旋轉（座標同步旋轉）
USE_CROP = False                # 隨機裁剪（座標同步平移+縮放）
USE_SCALE = False               # 隨機縮放（座標同步縮放+平移）

# 色彩類增強：完全不動座標，label 100% 等於原始 label
USE_BLUR = True                 # 高斯模糊
USE_BRIGHTNESS = True           # 亮度調整
USE_CONTRAST = True             # 對比度調整
USE_NOISE = True                # 添加高斯噪點
USE_SATURATION = True           # 飽和度調整
USE_HUE = True                  # 色調偏移
USE_SHARPENING = True           # 銳化處理
USE_HISTOGRAM_EQ = True         # 直方圖均衡化

# ============================================================
# ✔ 參數範圍設定（已加大，效果更明顯）
# ============================================================
BLUR_KSIZE_MIN, BLUR_KSIZE_MAX = 3, 13
BRIGHTNESS_FACTOR_MIN, BRIGHTNESS_FACTOR_MAX = 0.5, 1.8
CONTRAST_FACTOR_MIN, CONTRAST_FACTOR_MAX = 0.5, 1.8
ROTATION_ANGLE_MIN, ROTATION_ANGLE_MAX = -35, 35
NOISE_MEAN, NOISE_STD_MIN, NOISE_STD_MAX = 0, 10, 40
SATURATION_FACTOR_MIN, SATURATION_FACTOR_MAX = 0.3, 1.8
HUE_SHIFT_MIN, HUE_SHIFT_MAX = -30, 30
CROP_RATIO_MIN, CROP_RATIO_MAX = 0.6, 0.85
SCALE_FACTOR_MIN, SCALE_FACTOR_MAX = 0.7, 1.3
SHARPENING_STRENGTH_MIN, SHARPENING_STRENGTH_MAX = 0.5, 2.5

# 幾何增強後，若 bbox 寬或高小於原圖的這個比例（代表貓幾乎被裁出畫面），視為無效樣本並跳過
MIN_BBOX_FRAC = 0.02

# ============================================================
# ✔ 輸出數量控制
# ============================================================
# 不再「每種增強各出一張」，改成每張原圖固定產生 SAMPLES_PER_IMAGE 張，
# 每張都是「隨機挑一種幾何變換 + 疊 COLOR_AUGS_PER_SAMPLE 種色彩變換」組合而成，
# 效果比單一增強更明顯（強度加強），同時總輸出量可直接用這兩個數字控制。
SAMPLES_PER_IMAGE = 4        # 每張原圖輸出幾張增強圖（TOTAL_OUTPUT_IMAGES 設定時會被覆蓋）
COLOR_AUGS_PER_SAMPLE = 2    # 每張增強圖疊幾種色彩類效果（隨機挑選、依序疊加）
MAX_GEO_RETRY = 3            # 幾何變換出框時，最多重試幾次其他隨機參數/類型

# 若設定總量（例如 200），會取代 SAMPLES_PER_IMAGE，自動平均分配到每張原圖，
# 除不盡的餘數平均分給前幾張圖，確保輸出總數精準等於這個數字。設 None 則用 SAMPLES_PER_IMAGE。
TOTAL_OUTPUT_IMAGES = 200


# ============================================================
# ✔ YOLO pose label 讀寫（自動偵測真實 17 點所在區塊）
# ============================================================
def read_label(path):
    """回傳 (cls, bbox[4] 正規化, kpts[N,3] 正規化, real_start, real_count)。"""
    nums = [float(x) for x in open(path, encoding="utf-8").read().split()]
    cls = int(nums[0])
    bbox = np.array(nums[1:5], dtype=np.float64)
    kpt_nums = nums[5:]
    n_slots = len(kpt_nums) // 3
    kpts = np.array(kpt_nums, dtype=np.float64).reshape(n_slots, 3)

    visible = np.where(kpts[:, 2] > 0)[0]
    if len(visible) == 0:
        real_start, real_count = 0, n_slots
    else:
        real_start, real_count = int(visible[0]), int(visible[-1] - visible[0] + 1)
        if real_count != len(visible):
            # 真實點不是連續區塊（非本專案已知的 padding 型態），退回整段都當真實點處理
            real_start, real_count = 0, n_slots
    return cls, bbox, kpts, real_start, real_count


def write_label(path, cls, bbox, kpts):
    parts = [str(cls)] + [f"{v:.6f}" for v in bbox]
    for x, y, v in kpts:
        v_str = str(int(v)) if float(v).is_integer() else f"{v:.6f}"
        parts += [f"{x:.6f}", f"{y:.6f}", v_str]
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(parts) + "\n")


# ============================================================
# ✔ 座標轉換輔助
# ============================================================
def bbox_to_pixel_corners(bbox, W, H):
    cx, cy, w, h = bbox
    x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
    x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def corners_to_bbox(corners, W, H):
    x1, x2 = np.clip([corners[:, 0].min(), corners[:, 0].max()], 0, W)
    y1, y2 = np.clip([corners[:, 1].min(), corners[:, 1].max()], 0, H)
    return np.array([(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H])


def flip_swap_real_block(real_kpts):
    out = real_kpts.copy()
    for a, b in FLIP_SWAP_REAL:
        if a < len(out) and b < len(out):
            out[[a, b]] = out[[b, a]]
    return out


def _clip_and_mask(pix_xy, vis, W, H):
    inb = (pix_xy[:, 0] >= 0) & (pix_xy[:, 0] < W) & (pix_xy[:, 1] >= 0) & (pix_xy[:, 1] < H)
    vis = vis.copy()
    vis[~inb] = 0
    pix_xy = pix_xy.copy()
    pix_xy[~inb] = 0
    return pix_xy, vis


# ============================================================
# ✔ 幾何類增強（影像 + 標註同步變換）
# ============================================================
def horizontal_flip_with_label(img, bbox, kpts, real_start, real_count):
    W = img.shape[1]
    img2 = cv2.flip(img, 1)
    bbox2 = bbox.copy()
    bbox2[0] = 1.0 - bbox[0]

    kpts2 = kpts.copy()
    real = kpts2[real_start:real_start + real_count].copy()
    mask = real[:, 2] > 0
    real[mask, 0] = 1.0 - real[mask, 0]
    real = flip_swap_real_block(real)
    kpts2[real_start:real_start + real_count] = real
    return img2, bbox2, kpts2


def vertical_flip_with_label(img, bbox, kpts, real_start, real_count):
    img2 = cv2.flip(img, 0)
    bbox2 = bbox.copy()
    bbox2[1] = 1.0 - bbox[1]

    kpts2 = kpts.copy()
    real = kpts2[real_start:real_start + real_count].copy()
    mask = real[:, 2] > 0
    real[mask, 1] = 1.0 - real[mask, 1]
    kpts2[real_start:real_start + real_count] = real
    return img2, bbox2, kpts2


def rotate_with_label(img, bbox, kpts, real_start, real_count, angle=None):
    if angle is None:
        angle = random.uniform(ROTATION_ANGLE_MIN, ROTATION_ANGLE_MAX)
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    img2 = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)

    corners = bbox_to_pixel_corners(bbox, W, H)
    corners_t = cv2.transform(corners.reshape(-1, 1, 2), M).reshape(-1, 2)
    bbox2 = corners_to_bbox(corners_t, W, H)
    if bbox2[2] < MIN_BBOX_FRAC or bbox2[3] < MIN_BBOX_FRAC:
        return None

    kpts2 = kpts.copy()
    real = kpts2[real_start:real_start + real_count].copy()
    pix = real[:, :2] * [W, H]
    pix_t = cv2.transform(pix.reshape(-1, 1, 2), M).reshape(-1, 2)
    pix_t, vis = _clip_and_mask(pix_t, real[:, 2], W, H)
    real[:, :2] = pix_t / [W, H]
    real[:, 2] = vis
    kpts2[real_start:real_start + real_count] = real
    return img2, bbox2, kpts2


def random_crop_with_label(img, bbox, kpts, real_start, real_count, ratio=None):
    if ratio is None:
        ratio = random.uniform(CROP_RATIO_MIN, CROP_RATIO_MAX)
    H, W = img.shape[:2]
    new_h, new_w = int(H * ratio), int(W * ratio)
    top = random.randint(0, H - new_h)
    left = random.randint(0, W - new_w)
    img2 = cv2.resize(img[top:top + new_h, left:left + new_w], (W, H))
    sx, sy = W / new_w, H / new_h

    corners = bbox_to_pixel_corners(bbox, W, H)
    corners2 = (corners - [left, top]) * [sx, sy]
    bbox2 = corners_to_bbox(corners2, W, H)
    if bbox2[2] < MIN_BBOX_FRAC or bbox2[3] < MIN_BBOX_FRAC:
        return None

    kpts2 = kpts.copy()
    real = kpts2[real_start:real_start + real_count].copy()
    pix = real[:, :2] * [W, H]
    pix2 = (pix - [left, top]) * [sx, sy]
    pix2, vis = _clip_and_mask(pix2, real[:, 2], W, H)
    real[:, :2] = pix2 / [W, H]
    real[:, 2] = vis
    kpts2[real_start:real_start + real_count] = real
    return img2, bbox2, kpts2


def random_scale_with_label(img, bbox, kpts, real_start, real_count, factor=None):
    if factor is None:
        factor = random.uniform(SCALE_FACTOR_MIN, SCALE_FACTOR_MAX)
    H, W = img.shape[:2]
    new_h, new_w = int(H * factor), int(W * factor)
    scaled = cv2.resize(img, (new_w, new_h))
    if factor < 1.0:
        pad_h, pad_w = (H - new_h) // 2, (W - new_w) // 2
        img2 = np.zeros_like(img)
        img2[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = scaled
        offset = np.array([pad_w, pad_h], dtype=np.float64)
    else:
        start_h, start_w = (new_h - H) // 2, (new_w - W) // 2
        img2 = scaled[start_h:start_h + H, start_w:start_w + W]
        offset = np.array([-start_w, -start_h], dtype=np.float64)

    corners = bbox_to_pixel_corners(bbox, W, H)
    corners2 = corners * factor + offset
    bbox2 = corners_to_bbox(corners2, W, H)
    if bbox2[2] < MIN_BBOX_FRAC or bbox2[3] < MIN_BBOX_FRAC:
        return None

    kpts2 = kpts.copy()
    real = kpts2[real_start:real_start + real_count].copy()
    pix = real[:, :2] * [W, H]
    pix2 = pix * factor + offset
    pix2, vis = _clip_and_mask(pix2, real[:, 2], W, H)
    real[:, :2] = pix2 / [W, H]
    real[:, 2] = vis
    kpts2[real_start:real_start + real_count] = real
    return img2, bbox2, kpts2


# ============================================================
# ✔ 色彩類增強（不影響座標，label 原樣複製）
# ============================================================
def apply_blur(img, ksize=None):
    if ksize is None:
        ksize = random.randrange(BLUR_KSIZE_MIN, BLUR_KSIZE_MAX + 1, 2)
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def adjust_brightness(img, factor=None):
    if factor is None:
        factor = random.uniform(BRIGHTNESS_FACTOR_MIN, BRIGHTNESS_FACTOR_MAX)
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)


def adjust_contrast(img, factor=None):
    if factor is None:
        factor = random.uniform(CONTRAST_FACTOR_MIN, CONTRAST_FACTOR_MAX)
    mean = np.mean(img)
    return cv2.convertScaleAbs(img, alpha=factor, beta=mean * (1 - factor))


def add_noise(img, std=None):
    if std is None:
        std = random.uniform(NOISE_STD_MIN, NOISE_STD_MAX)
    noise = np.random.normal(NOISE_MEAN, std, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def adjust_saturation(img, factor=None):
    if factor is None:
        factor = random.uniform(SATURATION_FACTOR_MIN, SATURATION_FACTOR_MAX)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_hue(img, shift=None):
    if shift is None:
        shift = random.randint(HUE_SHIFT_MIN, HUE_SHIFT_MAX)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_sharpening(img, strength=None):
    if strength is None:
        strength = random.uniform(SHARPENING_STRENGTH_MIN, SHARPENING_STRENGTH_MAX)
    kernel = np.array([[-1, -1, -1], [-1, 8 + strength, -1], [-1, -1, -1]]) / (strength + 1)
    return cv2.filter2D(img, -1, kernel)


def histogram_equalization(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


# ============================================================
# ✔ 主程式
# ============================================================
def save_pair(tag, filename, img, cls, bbox, kpts):
    cv2.imwrite(str(Path(OUTPUT_IMAGES_DIR) / f"{filename}_{tag}.jpg"), img)
    write_label(str(Path(OUTPUT_LABELS_DIR) / f"{filename}_{tag}.txt"), cls, bbox, kpts)


def compute_samples_per_image(n_images):
    """回傳長度 n_images 的 list，每張圖要輸出幾張增強圖，總和精準等於 TOTAL_OUTPUT_IMAGES
    （除不盡的餘數分給前幾張）；TOTAL_OUTPUT_IMAGES 為 None 時，每張都用 SAMPLES_PER_IMAGE。"""
    if TOTAL_OUTPUT_IMAGES is None or n_images == 0:
        return [SAMPLES_PER_IMAGE] * n_images
    base, extra = divmod(TOTAL_OUTPUT_IMAGES, n_images)
    return [base + 1 if i < extra else base for i in range(n_images)]


def main():
    image_paths = sorted(glob.glob(os.path.join(INPUT_IMAGES_DIR, "*.jpg")) +
                          glob.glob(os.path.join(INPUT_IMAGES_DIR, "*.png")))
    print(f"輸入圖片數量: {len(image_paths)} 張")
    samples_per_image = compute_samples_per_image(len(image_paths))

    geo_jobs_all = []
    if USE_HORIZONTAL_FLIP:
        geo_jobs_all.append(("hflip", horizontal_flip_with_label))
    if USE_VERTICAL_FLIP:
        geo_jobs_all.append(("vflip", vertical_flip_with_label))
    if USE_ROTATION:
        geo_jobs_all.append(("rotate", rotate_with_label))
    if USE_CROP:
        geo_jobs_all.append(("crop", random_crop_with_label))
    if USE_SCALE:
        geo_jobs_all.append(("scale", random_scale_with_label))

    color_jobs_all = []
    if USE_BLUR:
        color_jobs_all.append(("blur", apply_blur))
    if USE_BRIGHTNESS:
        color_jobs_all.append(("bright", adjust_brightness))
    if USE_CONTRAST:
        color_jobs_all.append(("contrast", adjust_contrast))
    if USE_NOISE:
        color_jobs_all.append(("noise", add_noise))
    if USE_SATURATION:
        color_jobs_all.append(("saturation", adjust_saturation))
    if USE_HUE:
        color_jobs_all.append(("hue", adjust_hue))
    if USE_SHARPENING:
        color_jobs_all.append(("sharp", apply_sharpening))
    if USE_HISTOGRAM_EQ:
        color_jobs_all.append(("histeq", histogram_equalization))

    print(f"幾何類增強候選: {len(geo_jobs_all)} 種  色彩類增強候選: {len(color_jobs_all)} 種")
    if TOTAL_OUTPUT_IMAGES is not None:
        print(f"總輸出目標: {TOTAL_OUTPUT_IMAGES} 張（自動平均分配到每張原圖）")
    else:
        print(f"每張原圖輸出 {SAMPLES_PER_IMAGE} 張組合樣本")
    print(f"每張組合樣本疊 {'1 種幾何 + ' if geo_jobs_all else ''}{COLOR_AUGS_PER_SAMPLE} 種色彩")

    output_count, skipped_count = 0, 0

    for img_path, n_samples in zip(image_paths, samples_per_image):
        filename = Path(img_path).stem
        label_path = Path(INPUT_LABELS_DIR) / f"{filename}.txt"
        if not label_path.exists():
            print(f"  ⚠ 找不到對應 label，略過: {filename}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠ 無法讀取圖像: {img_path}")
            continue
        cls, bbox, kpts, real_start, real_count = read_label(str(label_path))

        for i in range(n_samples):
            img_cur, bbox_cur, kpts_cur = img, bbox, kpts
            tags = []

            # 幾何變換：隨機挑一種，出框就換下一種，全部失敗則這張不套幾何、只做色彩
            if geo_jobs_all:
                candidates = random.sample(geo_jobs_all, k=min(MAX_GEO_RETRY, len(geo_jobs_all)))
                for geo_tag, geo_fn in candidates:
                    result = geo_fn(img_cur, bbox_cur, kpts_cur, real_start, real_count)
                    if result is not None:
                        img_cur, bbox_cur, kpts_cur = result
                        tags.append(geo_tag)
                        break
                else:
                    skipped_count += 1

            # 色彩變換：隨機挑幾種疊加
            if color_jobs_all:
                k = min(COLOR_AUGS_PER_SAMPLE, len(color_jobs_all))
                for color_tag, color_fn in random.sample(color_jobs_all, k=k):
                    img_cur = color_fn(img_cur)
                    tags.append(color_tag)

            if not tags:
                continue  # 沒有任何增強候選（全部開關關閉），跳過
            # 加短亂數尾碼：避免重跑腳本時，同一張圖剛好又選中同一組效果組合而覆蓋掉先前的輸出
            uid = uuid.uuid4().hex[:6]
            tag = f"aug{i}_" + "_".join(tags) + f"_{uid}"
            # 只輸出真正的 17 個關鍵點（不含 padding），Ultralytics 需要 1+4+17*3=56 欄，
            # 不能照抄輸入檔的 125-slot padding 格式，否則欄數對不上會被整批跳過
            real_kpts = kpts_cur[real_start:real_start + real_count]
            save_pair(tag, filename, img_cur, cls, bbox_cur, real_kpts)
            output_count += 1

    print(f"\n{'='*50}")
    print("YOLO Pose 資料增強完成")
    print(f"{'='*50}")
    print(f"輸入圖片數量: {len(image_paths)} 張")
    print(f"輸出圖片數量: {output_count} 張（幾何變換全數出框略過 {skipped_count} 次）")
    print(f"輸出位置: {OUTPUT_IMAGES_DIR}")


if __name__ == "__main__":
    main()
