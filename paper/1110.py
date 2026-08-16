from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError

# ==========================================================
# 固定路徑設定：請只修改這三個值
# ==========================================================
INPUT_PATH = Path(r"C:\VFX\source")               # 單張圖片或圖片資料夾
OUTPUT_DIR = Path(r"C:\VFX\output_transparent")  # 去背後 PNG 的輸出資料夾
RECURSIVE = True                                  # True = 連子資料夾也處理
# ==========================================================

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp",
    ".bmp", ".tif", ".tiff"
}


def remove_checkerboard(input_path: Path, output_path: Path):
    """
    將圖片內的灰黑棋盤格背景轉為 Alpha 透明背景。
    適合藍色發光特效、魔法特效、劍氣等素材。
    """

    with Image.open(input_path) as source:
        source_rgba = source.convert("RGBA")
        rgba = np.asarray(source_rgba).copy()

    # 轉為 0~1 範圍方便計算
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    original_alpha = rgba[..., 3].astype(np.float32) / 255.0

    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)

    # 灰色／黑色棋盤格飽和度低，藍色劍氣飽和度高
    saturation = np.where(
        maximum > 0,
        (maximum - minimum) / maximum,
        0
    )

    # 保留藍色、青色等有顏色的發光特效
    color_alpha = np.clip((saturation - 0.08) / 0.22, 0, 1)

    # 保留劍氣中心偏白、亮度高的區域
    brightness_alpha = np.clip((maximum - 0.55) / 0.25, 0, 1)

    # 取色彩與亮度保留結果中較強的一個
    alpha = np.maximum(color_alpha, brightness_alpha)

    # 若原圖原本具有透明度，保留它
    alpha = alpha * original_alpha
    alpha = (alpha * 255).astype(np.uint8)

    # 稍微柔化透明邊緣，減少鋸齒
    alpha_image = Image.fromarray(alpha, "L").filter(
        ImageFilter.GaussianBlur(radius=0.6)
    )

    # 輸出真正具有 Alpha 通道的 PNG
    result = Image.fromarray(rgba, "RGBA")
    result.putalpha(alpha_image)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, "PNG")

    print(f"完成：{input_path.name}")
    print(f"輸出：{output_path}")


def collect_images(input_path: Path, recursive: bool):
    """取得單張圖片，或資料夾中的所有支援圖片。"""

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支援的檔案格式：{input_path.suffix}\n"
                f"支援格式：{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(
            f"找不到輸入路徑：\n{input_path}\n\n"
            "請確認 INPUT_PATH 是否設定正確。"
        )

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")

    return sorted(
        file_path
        for file_path in iterator
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main():
    try:
        images = collect_images(INPUT_PATH, RECURSIVE)

        if not images:
            print("找不到任何支援的圖片。")
            print(f"目前輸入路徑：{INPUT_PATH}")
            return

        print(f"找到 {len(images)} 張圖片，開始處理...\n")

        success_count = 0

        for image_path in images:
            try:
                # 全部輸出為 PNG，避免覆蓋原始圖片
                output_path = OUTPUT_DIR / f"{image_path.stem}_alpha.png"

                remove_checkerboard(image_path, output_path)
                success_count += 1

            except UnidentifiedImageError:
                print(f"略過：無法讀取圖片：{image_path}")

            except Exception as error:
                print(f"處理失敗：{image_path}")
                print(f"原因：{error}")

        print("\n==============================")
        print(f"處理完成：{success_count} / {len(images)} 張")
        print(f"輸出資料夾：{OUTPUT_DIR}")
        print("==============================")

    except Exception as error:
        print("\n程式無法執行：")
        print(error)


if __name__ == "__main__":
    main()