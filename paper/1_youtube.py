import os
import time
import yt_dlp

DOWNLOAD_DIR = r"C:\CatDataset\YouTube"
CHANNEL_URL = "https://www.youtube.com/@ImpressedCatVideo/videos"
FFMPEG = r"C:\ffmpeg\bin"

# 每次下載範圍
PLAYLIST_START = 1
PLAYLIST_END = 100

# 遇到限流等待秒數
RATE_LIMIT_WAIT = 3600

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def get_total_videos(url):
    print("=" * 60)
    print("讀取頻道資訊...")
    print("=" * 60)

    opts = {
        "extract_flat": True,
        "quiet": True,
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    entries = info.get("entries", []) if info else []
    total = len(entries)

    print(f"頻道：{info.get('title') if info else '未知'}")
    print(f"目前可讀取影片數：{total}")
    print("=" * 60)

    return total


def build_opts():
    return {
        # 最高畫質
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "ffmpeg_location": FFMPEG,

        # 儲存位置
        "paths": {
            "home": DOWNLOAD_DIR
        },
        "outtmpl": os.path.join(
            DOWNLOAD_DIR,
            "%(uploader)s",
            "%(upload_date>%Y-%m-%d)s - %(title)s [%(id)s].%(ext)s"
        ),

        # 每次只下載指定範圍
        "playliststart": PLAYLIST_START,
        "playlistend": PLAYLIST_END,

        # 已下載永久略過
        "download_archive": os.path.join(DOWNLOAD_DIR, "downloaded.txt"),

        # 續傳與不覆蓋
        "continuedl": True,
        "overwrites": False,
        "ignoreerrors": True,

        # 不下載縮圖
        "writethumbnail": False,

        # 限流保護
        "sleep_interval": 3,
        "max_sleep_interval": 8,
        "sleep_requests": 1.5,

        # 重試
        "retries": 20,
        "fragment_retries": 20,
        "socket_timeout": 30,

        # 模擬瀏覽器
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/137.0 Safari/537.36"
        ),

        # 若 Chrome 有登入 YouTube，可減少部分限制
       # "cookiesfrombrowser": ("chrome",),

        # Node.js JavaScript runtime
        "js_runtimes": {
            "node": None,
        },

        "progress_with_newline": True,
    }


def download_with_rate_limit_retry(url):
    while True:
        try:
            print()
            print("=" * 60)
            print(f"開始下載第 {PLAYLIST_START} 到 {PLAYLIST_END} 部影片")
            print("=" * 60)

            with yt_dlp.YoutubeDL(build_opts()) as ydl:
                ydl.download([url])

            print()
            print("本次下載完成")
            break

        except Exception as e:
            msg = str(e)

            if (
                "rate-limited" in msg.lower()
                or "try again later" in msg.lower()
                or "429" in msg
                or "This content isn't available" in msg
            ):
                print()
                print("=" * 60)
                print("偵測到 YouTube 限流")
                print(f"等待 {RATE_LIMIT_WAIT // 60} 分鐘後自動重試...")
                print("=" * 60)
                time.sleep(RATE_LIMIT_WAIT)
            else:
                print()
                print("發生非限流錯誤：")
                print(e)
                break


if __name__ == "__main__":
    total = get_total_videos(CHANNEL_URL)

    print()
    print(f"本次設定下載範圍：第 {PLAYLIST_START} 到第 {PLAYLIST_END} 部")
    print(f"頻道目前可讀取影片數：約 {total} 部")

    download_with_rate_limit_retry(CHANNEL_URL)