import os
import shutil
from PIL import Image
import imagehash

# 原圖資料夾
src_folder = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\check"
# 重複圖輸出資料夾
dup_folder = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\duplicates"

# 若資料夾不存在就自動建立
os.makedirs(dup_folder, exist_ok=True)

hash_dict = {}

for filename in os.listdir(src_folder):
    file_path = os.path.join(src_folder, filename)

    # 跳過資料夾或非圖片檔
    if not os.path.isfile(file_path):
        continue
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        continue

    try:
        img_hash = imagehash.phash(Image.open(file_path))
    except:
        print(f"⚠️ 無法讀取: {filename}")
        continue

    # 若之前已出現過 → 判定為重複，移動到 duplicates/
    if img_hash in hash_dict:
        print(f"🔁 找到重複圖: {filename}  <--->  {hash_dict[img_hash]}")
        shutil.move(file_path, os.path.join(dup_folder, filename))
    else:
        hash_dict[img_hash] = filename

print("\n🎉 完成！所有重複圖已移動到 duplicates/ 資料夾！")
