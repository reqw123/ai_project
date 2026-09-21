"""
=====================================================
MODE = 1
HEIC / HEIF -> PNG
轉換後刪除原始 HEIC。

MODE = 2
PNG 右旋轉 90 度，
輸出至：
ROOT_FOLDER_右旋轉
=====================================================
"""

MODE = 2

ROOT_FOLDER = r"C:\Users\homec\OneDrive\貓咪圖\側躺圖片"

# ==========================

import os
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()


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
# 執行
# ===================================================

if MODE == 1:
    mode1()

elif MODE == 2:
    mode2()

else:
    print("MODE 設定錯誤")
