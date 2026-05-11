import os
from PIL import Image
import pillow_avif  # 啟用 AVIF

SRC_DIR    = r"C:\Users\homec\Downloads\21"   # 原始資料夾
DST_DIR    = "jpg_images"    # 輸出資料夾
OUTPUT_FMT = "jpg"           # 輸出格式：'jpg' 或 'png'
QUALITY    = 98              # JPG 品質 90~95（PNG 忽略此設定）

assert OUTPUT_FMT in ("jpg", "png"), "OUTPUT_FMT 只能是 'jpg' 或 'png'"

for root, _, files in os.walk(SRC_DIR):
    rel = os.path.relpath(root, SRC_DIR)
    out_dir = os.path.join(DST_DIR, rel)
    os.makedirs(out_dir, exist_ok=True)

    for f in files:
        if f.lower().endswith(".avif"):
            src_path = os.path.join(root, f)
            dst_name = os.path.splitext(f)[0] + "." + OUTPUT_FMT
            dst_path = os.path.join(out_dir, dst_name)

            img = Image.open(src_path).convert("RGB")
            if OUTPUT_FMT == "jpg":
                img.save(dst_path, format="JPEG", quality=QUALITY, subsampling=0)
            else:
                img.save(dst_path, format="PNG")

print("✅ 轉換完成")
