from pathlib import Path
import subprocess

INPUT_FOLDER = Path(r"C:\Users\homec\Downloads\24")
OUTPUT_FOLDER = INPUT_FOLDER / "output_1920x1080"

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