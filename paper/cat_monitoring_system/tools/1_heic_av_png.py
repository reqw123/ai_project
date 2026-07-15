"""
=====================================================
MODE = 1
HEIC / HEIF -> PNG
轉換後刪除原始 HEIC。

MODE = 2
PNG 右旋轉 90 度，
輸出至：
ROOT_FOLDER_右旋轉

MODE = 3
YOLO Pose Augmentation，
輸出至：
ROOT_FOLDER/augmention
=====================================================
"""

MODE = 2

ROOT_FOLDER = r"C:\Users\homec\OneDrive\貓咪圖\側躺圖片"
OUTPUT_FOLDER = r"C:\Users\homec\OneDrive\貓咪圖\augmention"
# ==========================
# MODE 3 參數
# ==========================

NUM_AUG_PER_IMAGE = 3

FLIP_PROB = 0.5

ROTATE_DEGREE = 25
TRANSLATE_RATIO = 0.05
SCALE_RATIO = 0.50
SHEAR_DEGREE = 5

HSV_H = 0.2
HSV_S = 0.7
HSV_V = 0.7

CONTRAST_MIN = 0.7
CONTRAST_MAX = 1.3

BRIGHTNESS_MIN = -20
BRIGHTNESS_MAX = 20

BLUR_PROB = 0.2
BLUR_KERNEL = 5

NOISE_PROB = 0.2
NOISE_STD = 10

SHARPEN_PROB = 0.2

COPY_ORIGINAL = True

# ==========================

import os
import cv2
import random
import numpy as np
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()


# ===================================================
# 解決中文路徑問題
# ===================================================

def cv_imread(file_path):
    try:
        return cv2.imdecode(
            np.fromfile(
                file_path,
                dtype=np.uint8
            ),
            cv2.IMREAD_COLOR
        )
    except:
        return None


def cv_imwrite(file_path, img):
    ext = os.path.splitext(file_path)[1]

    success, encoded = cv2.imencode(
        ext,
        img
    )

    if success:
        encoded.tofile(file_path)


# ===================================================
# MODE 1
# ===================================================

def mode1():

    for root, dirs, files in os.walk(ROOT_FOLDER):

        for file in files:

            if not file.lower().endswith(
                    (".heic", ".heif")):
                continue

            input_path = os.path.join(
                root,
                file
            )

            output_path = os.path.join(
                root,
                os.path.splitext(file)[0] + ".png"
            )

            try:
                img = Image.open(input_path)

                img.save(
                    output_path,
                    "PNG"
                )

                os.remove(input_path)

                print("✓", input_path)

            except Exception as e:
                print(e)


# ===================================================
# MODE 2
# ===================================================

def mode2():

    output_folder = ROOT_FOLDER + "_右旋轉"

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    for root, dirs, files in os.walk(ROOT_FOLDER):

        for file in files:

            if not file.lower().endswith(".png"):
                continue

            path = os.path.join(
                root,
                file
            )

            relative = os.path.relpath(
                root,
                ROOT_FOLDER
            )

            save_dir = os.path.join(
                output_folder,
                relative
            )

            os.makedirs(
                save_dir,
                exist_ok=True
            )

            output_path = os.path.join(
                save_dir,
                file
            )

            try:
                img = Image.open(path)

                img = img.transpose(
                    Image.ROTATE_270
                )

                img.save(output_path)

                print("✓", path)

            except Exception as e:
                print(e)


# ===================================================
# augmentation
# ===================================================

def augment(img):

    h, w = img.shape[:2]

    out = img.copy()

    # 左右翻轉
    if random.random() < FLIP_PROB:
        out = cv2.flip(out, 1)

    # rotation + scale
    angle = random.uniform(
        -ROTATE_DEGREE,
        ROTATE_DEGREE
    )

    scale = random.uniform(
        1 - SCALE_RATIO,
        1 + SCALE_RATIO
    )

    M = cv2.getRotationMatrix2D(
        (w / 2, h / 2),
        angle,
        scale
    )

    # shear
    shear = np.tan(
        np.radians(
            random.uniform(
                -SHEAR_DEGREE,
                SHEAR_DEGREE
            )
        )
    )

    M[0, 1] += shear

    # translate
    tx = random.randint(
        int(-w * TRANSLATE_RATIO),
        int(w * TRANSLATE_RATIO)
    )

    ty = random.randint(
        int(-h * TRANSLATE_RATIO),
        int(h * TRANSLATE_RATIO)
    )

    M[0, 2] += tx
    M[1, 2] += ty

    out = cv2.warpAffine(
        out,
        M,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )

    # hsv
    hsv = cv2.cvtColor(
        out,
        cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    hsv[:, :, 0] *= random.uniform(
        1 - HSV_H,
        1 + HSV_H
    )

    hsv[:, :, 1] *= random.uniform(
        1 - HSV_S,
        1 + HSV_S
    )

    hsv[:, :, 2] *= random.uniform(
        1 - HSV_V,
        1 + HSV_V
    )

    hsv = np.clip(
        hsv,
        0,
        255
    ).astype(np.uint8)

    out = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    # contrast
    alpha = random.uniform(
        CONTRAST_MIN,
        CONTRAST_MAX
    )

    beta = random.randint(
        BRIGHTNESS_MIN,
        BRIGHTNESS_MAX
    )

    out = cv2.convertScaleAbs(
        out,
        alpha=alpha,
        beta=beta
    )

    # blur
    if random.random() < BLUR_PROB:
        out = cv2.GaussianBlur(
            out,
            (BLUR_KERNEL, BLUR_KERNEL),
            0
        )

    # noise
    if random.random() < NOISE_PROB:

        noise = np.random.normal(
            0,
            NOISE_STD,
            out.shape
        )

        out = np.clip(
            out.astype(np.float32) + noise,
            0,
            255
        ).astype(np.uint8)

    # sharpen
    if random.random() < SHARPEN_PROB:

        kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ])

        out = cv2.filter2D(
            out,
            -1,
            kernel
        )

    return out


# ===================================================
# MODE 3
# ===================================================

def mode3():

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    total = 0

    for root, dirs, files in os.walk(ROOT_FOLDER):

        if "augmention" in root.lower():
            continue

        for file in files:

            if not file.lower().endswith(".png"):
                continue

            path = os.path.join(root, file)

            img = cv_imread(path)

            if img is None:
                print("讀取失敗：", path)
                continue

            name = os.path.splitext(file)[0]

            # 儲存原圖
            if COPY_ORIGINAL:

                save_path = os.path.join(
                    OUTPUT_FOLDER,
                    f"{name}_original.png"
                )

                cv_imwrite(
                    save_path,
                    img
                )

            # 儲存增強圖
            for i in range(NUM_AUG_PER_IMAGE):

                aug = augment(img)

                save_path = os.path.join(
                    OUTPUT_FOLDER,
                    f"{name}_aug_{i+1}.png"
                )

                cv_imwrite(
                    save_path,
                    aug
                )

                total += 1

            print("✓", file)

    print()
    print("完成")
    print("共產生", total, "張圖片")
    print("輸出資料夾：")
    print(OUTPUT_FOLDER)


# ===================================================
# 執行
# ===================================================

if MODE == 1:
    mode1()

elif MODE == 2:
    mode2()

elif MODE == 3:
    mode3()

else:
    print("MODE 設定錯誤")