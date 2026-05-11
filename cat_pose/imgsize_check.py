import os
import shutil
from PIL import Image

# ====== 參數設定 ======
input_dir = r"C:\cat_pose\test\images"      # 原始圖片資料夾
output_dir = "filtered_images"  # 篩選後輸出資料夾
min_width = 1920
min_height = 1080

# 建立輸出資料夾
os.makedirs(output_dir, exist_ok=True)

# 支援的圖片副檔名
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

count_total = 0
count_keep = 0

for filename in os.listdir(input_dir):
    if filename.lower().endswith(valid_ext):
        count_total += 1
        img_path = os.path.join(input_dir, filename)

        try:
            with Image.open(img_path) as img:
                width, height = img.size

                if width >= min_width and height >= min_height:
                    shutil.copy(img_path, os.path.join(output_dir, filename))
                    count_keep += 1

        except Exception as e:
            print(f"⚠️ 無法讀取圖片 {filename}：{e}")

print("========== 篩選完成 ==========")
print(f"總圖片數：{count_total}")
print(f"符合條件圖片數（≥{min_width}×{min_height}）：{count_keep}")
print(f"已輸出至資料夾：{output_dir}")
