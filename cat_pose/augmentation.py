import cv2
import numpy as np
import glob
import os
import random

input_folder = r"C:\cat_pose\cat_images"      # 輸入資料夾路徑
output_folder = r"C:\cat_pose\cat_output"     # 輸出資料夾路徑

os.makedirs(output_folder, exist_ok=True)

# ============================================================
# ✔ 增強功能開關 (True=啟用, False=關閉)
# ============================================================
USE_HORIZONTAL_FLIP = True      # 水平翻轉
USE_VERTICAL_FLIP = False       # 垂直翻轉
USE_BLUR = False                # 高斯模糊
USE_BRIGHTNESS = False          # 亮度調整
USE_CONTRAST = False            # 對比度調整
USE_ROTATION = False            # 隨機旋轉
USE_NOISE = False               # 添加高斯噪點
USE_SATURATION = False          # 飽和度調整
USE_HUE = False                 # 色調偏移
USE_CROP = False                # 隨機裁剪
USE_SCALE = False               # 隨機縮放
USE_SHARPENING = False          # 銳化處理
USE_HISTOGRAM_EQ = False        # 直方圖均衡化

# ============================================================
# ✔ 參數範圍設定 (可調整最小值與最大值)
# ============================================================
# 模糊參數
BLUR_KSIZE_MIN = 3              # 模糊核大小最小值 (必須為奇數)
BLUR_KSIZE_MAX = 9              # 模糊核大小最大值 (必須為奇數)

# 亮度參數
BRIGHTNESS_FACTOR_MIN = 0.7     # 亮度因子最小值 (<1變暗, >1變亮)
BRIGHTNESS_FACTOR_MAX = 1.5     # 亮度因子最大值

# 對比度參數
CONTRAST_FACTOR_MIN = 0.7       # 對比度因子最小值 (<1降低對比, >1增加對比)
CONTRAST_FACTOR_MAX = 1.5       # 對比度因子最大值

# 旋轉參數
ROTATION_ANGLE_MIN = -30        # 旋轉角度最小值 (度)
ROTATION_ANGLE_MAX = 30         # 旋轉角度最大值 (度)

# 噪點參數
NOISE_MEAN = 0                  # 高斯噪點均值
NOISE_STD_MIN = 5               # 高斯噪點標準差最小值
NOISE_STD_MAX = 25              # 高斯噪點標準差最大值

# 飽和度參數
SATURATION_FACTOR_MIN = 0.5     # 飽和度因子最小值 (<1降低飽和度, >1增加飽和度)
SATURATION_FACTOR_MAX = 1.5     # 飽和度因子最大值

# 色調參數
HUE_SHIFT_MIN = -20             # 色調偏移最小值
HUE_SHIFT_MAX = 20              # 色調偏移最大值

# 裁剪參數
CROP_RATIO_MIN = 0.7            # 裁剪比例最小值 (相對於原圖)
CROP_RATIO_MAX = 0.9            # 裁剪比例最大值

# 縮放參數
SCALE_FACTOR_MIN = 0.8          # 縮放因子最小值
SCALE_FACTOR_MAX = 1.2          # 縮放因子最大值

# 銳化參數
SHARPENING_STRENGTH_MIN = 0.5   # 銳化強度最小值
SHARPENING_STRENGTH_MAX = 2.0   # 銳化強度最大值

# ============================================================
# ✔ 增強函數定義
# ============================================================

def horizontal_flip(img):
    """水平翻轉 - 左右鏡像"""
    return cv2.flip(img, 1)

def vertical_flip(img):
    """垂直翻轉 - 上下鏡像"""
    return cv2.flip(img, 0)

def apply_blur(img, ksize=None):
    """高斯模糊 - 平滑圖像，減少噪點"""
    if ksize is None:
        ksize = random.randrange(BLUR_KSIZE_MIN, BLUR_KSIZE_MAX + 1, 2)  # 確保為奇數
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def adjust_brightness(img, factor=None):
    """亮度調整 - 改變圖像明暗程度"""
    if factor is None:
        factor = random.uniform(BRIGHTNESS_FACTOR_MIN, BRIGHTNESS_FACTOR_MAX)
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

def adjust_contrast(img, factor=None):
    """對比度調整 - 改變明暗差異程度"""
    if factor is None:
        factor = random.uniform(CONTRAST_FACTOR_MIN, CONTRAST_FACTOR_MAX)
    mean = np.mean(img)
    return cv2.convertScaleAbs(img, alpha=factor, beta=mean * (1 - factor))

def rotate_image(img, angle=None):
    """隨機旋轉 - 在指定角度範圍內旋轉圖像"""
    if angle is None:
        angle = random.uniform(ROTATION_ANGLE_MIN, ROTATION_ANGLE_MAX)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

def add_noise(img, std=None):
    """添加高斯噪點 - 模擬真實拍攝的噪點"""
    if std is None:
        std = random.uniform(NOISE_STD_MIN, NOISE_STD_MAX)
    noise = np.random.normal(NOISE_MEAN, std, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def adjust_saturation(img, factor=None):
    """飽和度調整 - 改變色彩鮮豔程度"""
    if factor is None:
        factor = random.uniform(SATURATION_FACTOR_MIN, SATURATION_FACTOR_MAX)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def adjust_hue(img, shift=None):
    """色調偏移 - 改變整體色彩傾向"""
    if shift is None:
        shift = random.randint(HUE_SHIFT_MIN, HUE_SHIFT_MAX)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180  # H通道範圍為0-179
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def random_crop(img, ratio=None):
    """隨機裁剪 - 裁剪圖像的一部分並調整大小"""
    if ratio is None:
        ratio = random.uniform(CROP_RATIO_MIN, CROP_RATIO_MAX)
    h, w = img.shape[:2]
    new_h, new_w = int(h * ratio), int(w * ratio)
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    cropped = img[top:top + new_h, left:left + new_w]
    return cv2.resize(cropped, (w, h))  # 調整回原始大小

def random_scale(img, factor=None):
    """隨機縮放 - 放大或縮小圖像"""
    if factor is None:
        factor = random.uniform(SCALE_FACTOR_MIN, SCALE_FACTOR_MAX)
    h, w = img.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    scaled = cv2.resize(img, (new_w, new_h))
    # 如果縮放後較小，則在中心填充；如果較大，則裁剪中心
    if factor < 1.0:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        result = np.zeros_like(img)
        result[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = scaled
    else:
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        result = scaled[start_h:start_h + h, start_w:start_w + w]
    return result

def apply_sharpening(img, strength=None):
    """銳化處理 - 增強圖像邊緣細節"""
    if strength is None:
        strength = random.uniform(SHARPENING_STRENGTH_MIN, SHARPENING_STRENGTH_MAX)
    kernel = np.array([[-1, -1, -1],
                       [-1, 8 + strength, -1],
                       [-1, -1, -1]]) / (strength + 1)
    return cv2.filter2D(img, -1, kernel)

def histogram_equalization(img):
    """直方圖均衡化 - 增強圖像對比度和細節"""
    if len(img.shape) == 3:
        # 彩色圖像：轉換到YUV空間，只對Y通道均衡化
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        # 灰度圖像
        return cv2.equalizeHist(img)

# ============================================================
# ✔ 主程式：遍歷圖像並應用增強
# ============================================================

# 支援 jpg 和 png 格式
image_paths = glob.glob(input_folder + "/*.jpg") + glob.glob(input_folder + "/*.png")

# 統計輸入圖片數量
input_count = len(image_paths)
print(f"輸入圖片數量: {input_count} 張")

# 計算啟用的增強功能數量
enabled_augmentations = sum([
    USE_HORIZONTAL_FLIP, USE_VERTICAL_FLIP, USE_BLUR, USE_BRIGHTNESS,
    USE_CONTRAST, USE_ROTATION, USE_NOISE, USE_SATURATION,
    USE_HUE, USE_CROP, USE_SCALE, USE_SHARPENING, USE_HISTOGRAM_EQ
])
print(f"啟用的增強功能數量: {enabled_augmentations} 種")

# 輸出計數器
output_count = 0

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"警告：無法讀取圖像 {path}")
        continue
    filename = os.path.basename(path).split(".")[0]

    # 1. 水平翻轉
    if USE_HORIZONTAL_FLIP:
        img_aug = horizontal_flip(img)
        cv2.imwrite(f"{output_folder}/{filename}_hflip.jpg", img_aug)
        output_count += 1

    # 2. 垂直翻轉
    if USE_VERTICAL_FLIP:
        img_aug = vertical_flip(img)
        cv2.imwrite(f"{output_folder}/{filename}_vflip.jpg", img_aug)
        output_count += 1

    # 3. 高斯模糊
    if USE_BLUR:
        img_aug = apply_blur(img)
        cv2.imwrite(f"{output_folder}/{filename}_blur.jpg", img_aug)
        output_count += 1

    # 4. 亮度調整
    if USE_BRIGHTNESS:
        img_aug = adjust_brightness(img)
        cv2.imwrite(f"{output_folder}/{filename}_bright.jpg", img_aug)
        output_count += 1

    # 5. 對比度調整
    if USE_CONTRAST:
        img_aug = adjust_contrast(img)
        cv2.imwrite(f"{output_folder}/{filename}_contrast.jpg", img_aug)
        output_count += 1

    # 6. 隨機旋轉
    if USE_ROTATION:
        img_aug = rotate_image(img)
        cv2.imwrite(f"{output_folder}/{filename}_rotate.jpg", img_aug)
        output_count += 1

    # 7. 添加噪點
    if USE_NOISE:
        img_aug = add_noise(img)
        cv2.imwrite(f"{output_folder}/{filename}_noise.jpg", img_aug)
        output_count += 1

    # 8. 飽和度調整
    if USE_SATURATION:
        img_aug = adjust_saturation(img)
        cv2.imwrite(f"{output_folder}/{filename}_saturation.jpg", img_aug)
        output_count += 1

    # 9. 色調偏移
    if USE_HUE:
        img_aug = adjust_hue(img)
        cv2.imwrite(f"{output_folder}/{filename}_hue.jpg", img_aug)
        output_count += 1

    # 10. 隨機裁剪
    if USE_CROP:
        img_aug = random_crop(img)
        cv2.imwrite(f"{output_folder}/{filename}_crop.jpg", img_aug)
        output_count += 1

    # 11. 隨機縮放
    if USE_SCALE:
        img_aug = random_scale(img)
        cv2.imwrite(f"{output_folder}/{filename}_scale.jpg", img_aug)
        output_count += 1

    # 12. 銳化處理
    if USE_SHARPENING:
        img_aug = apply_sharpening(img)
        cv2.imwrite(f"{output_folder}/{filename}_sharp.jpg", img_aug)
        output_count += 1

    # 13. 直方圖均衡化
    if USE_HISTOGRAM_EQ:
        img_aug = histogram_equalization(img)
        cv2.imwrite(f"{output_folder}/{filename}_histeq.jpg", img_aug)
        output_count += 1

# 輸出統計結果
print(f"\n{'='*50}")
print(f"資料增強完成！Data augmentation completed!")
print(f"{'='*50}")
print(f"輸入圖片數量: {input_count} 張")
print(f"輸出圖片數量: {output_count} 張")
print(f"增強倍率: {output_count / input_count if input_count > 0 else 0:.1f}x")
