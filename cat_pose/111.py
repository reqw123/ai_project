import os
from pathlib import Path
import subprocess

INPUT_FOLDER = Path(r"C:\Users\homec\Downloads\24")

# 若設定 TEST_VIDEO_PATH 環境變數，優先使用該路徑（覆蓋上面寫死的 INPUT_FOLDER，
# 對應 settings_window.py 的「🎬 影片路徑」欄位）。可為單一影片檔案，或跟
# INPUT_FOLDER 一樣的資料夾。
_env_test_video = os.getenv("TEST_VIDEO_PATH", "").strip()
if _env_test_video:
    INPUT_FOLDER = Path(_env_test_video)

OUTPUT_FOLDER = (INPUT_FOLDER if INPUT_FOLDER.is_dir() else INPUT_FOLDER.parent) / "output_1920x1080"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def get_video_size(video_path: Path):
    """透過 ffprobe 取得影片寬高。"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    width, height = result.stdout.strip().split("x")
    return int(width), int(height)


def convert_video(video_path: Path, output_path: Path):
    """
    等比例縮放並補黑邊，避免畫面變形。
    最終尺寸固定為 1920x1080。
    """
    vf = (
        "scale=1920:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,"
        "setsar=1"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    subprocess.run(cmd, check=True)


def main():
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    if INPUT_FOLDER.is_file():
        videos = [INPUT_FOLDER] if INPUT_FOLDER.suffix.lower() in VIDEO_EXTENSIONS else []
    else:
        videos = [
            file for file in INPUT_FOLDER.iterdir()
            if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS
        ]

    if not videos:
        print("找不到影片檔。")
        return

    for video_path in videos:
        try:
            width, height = get_video_size(video_path)

            # 只要任一邊小於目標尺寸，就略過
            if width < 1920 or height < 1080:
                print(f"略過（解析度不足）：{video_path.name} [{width}x{height}]")
                continue

            output_path = OUTPUT_FOLDER / f"{video_path.stem}_1920x1080.mp4"

            if output_path.exists():
                print(f"略過（已存在）：{output_path.name}")
                continue

            print(f"處理中：{video_path.name} [{width}x{height}]")
            convert_video(video_path, output_path)
            print(f"完成：{output_path.name}")

        except Exception as error:
            print(f"失敗：{video_path.name}\n原因：{error}")

    print("\n全部處理完成。")


if __name__ == "__main__":
    main()