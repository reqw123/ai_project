#!/usr/bin/env python3
"""
合併 Tapo C200 監視器影片片段（依時間順序）— 固定路徑版本

使用方式：
    1. 修改下方 INPUT_FOLDER 和 OUTPUT_NAME 兩個變數
    2. 選擇 MODE：
         1 = 原始模式，直接依序合併（stream copy，速度快，片段間若有時間空隙不會補畫面）
         2 = 補黑畫面模式，偵測片段之間的時間空隙，超過 MIN_GAP_SECONDS 就用黑畫面+靜音補上
    3. 直接執行：python merge_tapo_clips.py

需求：
    - 已安裝 ffmpeg / ffprobe 並加入系統 PATH
    - Python 3.7+

注意：
    - 模式 2 因為需要混入產生的黑畫面片段，最終輸出會重新編碼（無法單純 stream copy），
      處理時間會比模式 1 長，且畫質/檔案大小取決於下方 ENCODE 參數。
"""

import os
import re
import json
import subprocess
import tempfile
from datetime import datetime, timedelta

# ========== 請在這裡修改成你自己的路徑 ==========
INPUT_FOLDER = r"C:\Users\homec\Downloads\0709_0600"
OUTPUT_NAME = "output\\merged_0709.mp4"
MODE = 1  # 1 = 直接合併（不補黑畫面）；2 = 補黑畫面銜接空隙
MIN_GAP_SECONDS = 2.0  # 只有空隙大於此秒數才會補黑畫面（避免因誤差幾秒就誤補）
# ================================================

# 模式 2 重新編碼參數（找不到來源影片參數時的預設值，以及最終輸出編碼設定）
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = "15/1"
DEFAULT_SAMPLE_RATE = "16000"
DEFAULT_CHANNELS = 1
ENCODE_CRF = "20"
ENCODE_PRESET = "veryfast"


FILENAME_PATTERN = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2})_(\d{2})_(\d{2})_(\d+)(?:\((\d+)\))?\.mp4$",
    re.IGNORECASE,
)


def parse_timestamp(filename):
    match = FILENAME_PATTERN.search(filename)
    if not match:
        return None
    year, month, day, hour, minute, second, seq, dup = match.groups()
    dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    seq = int(seq)
    dup = int(dup) if dup else 0
    return (dt, seq, dup)


def collect_clips(folder):
    clips = []
    skipped = []
    for name in os.listdir(folder):
        if not name.lower().endswith(".mp4"):
            continue
        key = parse_timestamp(name)
        full_path = os.path.join(folder, name)
        if key is None:
            skipped.append(name)
            continue
        clips.append((key, full_path, name))

    clips.sort(key=lambda x: x[0])
    return clips, skipped


def build_concat_list(entries, list_path):
    """entries: 路徑字串的 list，依序寫入 concat 清單"""
    with open(list_path, "w", encoding="utf-8") as f:
        for full_path in entries:
            escaped = full_path.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")


def run_ffmpeg_copy(list_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        output_path,
    ]
    print("執行指令：", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True)


def run_ffmpeg_reencode(list_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c:v", "libx264",
        "-preset", ENCODE_PRESET,
        "-crf", ENCODE_CRF,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        output_path,
    ]
    print("執行指令：", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True)


def probe_info(path):
    """回傳 duration / 影片與音訊參數，供模式2計算空隙與產生黑畫面使用"""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_entries",
        "format=duration:stream=codec_type,width,height,r_frame_rate,sample_rate,channels,channel_layout",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    duration = float(data["format"]["duration"])
    video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)

    info = {"duration": duration}
    if video_stream:
        info["width"] = video_stream.get("width", DEFAULT_WIDTH)
        info["height"] = video_stream.get("height", DEFAULT_HEIGHT)
        info["fps"] = video_stream.get("r_frame_rate", DEFAULT_FPS)
    if audio_stream:
        info["sample_rate"] = audio_stream.get("sample_rate", DEFAULT_SAMPLE_RATE)
        info["channels"] = audio_stream.get("channels", DEFAULT_CHANNELS)
        info["channel_layout"] = audio_stream.get("channel_layout")

    return info


def make_black_clip(duration_seconds, ref_info, out_path):
    """用 lavfi 產生指定秒數的黑畫面 + 靜音片段，參數比照參考片段"""
    width = ref_info.get("width", DEFAULT_WIDTH)
    height = ref_info.get("height", DEFAULT_HEIGHT)
    fps = ref_info.get("fps", DEFAULT_FPS)
    sample_rate = ref_info.get("sample_rate", DEFAULT_SAMPLE_RATE)
    channels = ref_info.get("channels", DEFAULT_CHANNELS)
    channel_layout = ref_info.get("channel_layout") or ("stereo" if channels == 2 else "mono")

    # 秒數用小數點表示，避免整數化造成銜接誤差
    duration_str = f"{duration_seconds:.3f}"

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}:d={duration_str}",
        "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl={channel_layout}",
        "-t", duration_str,
        "-c:v", "libx264", "-preset", ENCODE_PRESET, "-crf", ENCODE_CRF, "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        out_path,
    ]
    subprocess.run(cmd, check=True)


def merge_mode1(clips, output_path):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        list_path = tmp.name

    try:
        build_concat_list([full_path for _, full_path, _ in clips], list_path)
        run_ffmpeg_copy(list_path, output_path)
    finally:
        os.remove(list_path)


def merge_mode2(clips, output_path):
    print("\n[模式2] 讀取每個片段的長度資訊（ffprobe）...")
    infos = []
    for (dt, seq, dup), full_path, name in clips:
        info = probe_info(full_path)
        infos.append(info)
        print(f"  {name}  時長 {info['duration']:.2f} 秒")

    ref_info = infos[0]  # 用第一個片段的解析度/fps/音訊參數作為黑畫面的參考規格

    tmp_dir = tempfile.mkdtemp(prefix="tapo_merge_")
    entries = []
    gap_count = 0

    try:
        for i, ((dt, seq, dup), full_path, name) in enumerate(clips):
            entries.append(full_path)

            if i == len(clips) - 1:
                break  # 最後一段之後不用補空隙

            this_end = dt + timedelta(seconds=infos[i]["duration"])
            next_start = clips[i + 1][0][0]
            gap_seconds = (next_start - this_end).total_seconds()

            if gap_seconds > MIN_GAP_SECONDS:
                gap_count += 1
                black_path = os.path.join(tmp_dir, f"black_{gap_count:03d}.mp4")
                print(
                    f"  偵測到空隙 {gap_seconds:.1f} 秒："
                    f"{this_end.strftime('%H:%M:%S')} -> {next_start.strftime('%H:%M:%S')}，補黑畫面"
                )
                make_black_clip(gap_seconds, ref_info, black_path)
                entries.append(black_path)
            elif gap_seconds < 0:
                print(
                    f"  警告：{name} 與下一段時間重疊 {-gap_seconds:.1f} 秒，"
                    f"已略過補畫面（可能是檔名時間戳記或片段長度有誤差）"
                )

        if gap_count == 0:
            print("  沒有偵測到需要補畫面的空隙。")
        else:
            print(f"  共補了 {gap_count} 段黑畫面。")

        list_path = os.path.join(tmp_dir, "concat_list.txt")
        build_concat_list(entries, list_path)

        print("\n[模式2] 開始重新編碼輸出（因混有黑畫面片段，無法使用 stream copy）...")
        run_ffmpeg_reencode(list_path, output_path)

    finally:
        # 清理暫存的黑畫面片段與清單檔案
        for fname in os.listdir(tmp_dir):
            try:
                os.remove(os.path.join(tmp_dir, fname))
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


def main():
    folder = INPUT_FOLDER
    output_name = OUTPUT_NAME

    if not os.path.isdir(folder):
        print(f"錯誤：找不到資料夾 {folder}")
        print("請確認腳本開頭的 INPUT_FOLDER 是否設定正確。")
        return

    clips, skipped = collect_clips(folder)

    if not clips:
        print("錯誤：資料夾中沒有找到符合命名格式的 mp4 檔案。")
        return

    print(f"找到 {len(clips)} 個影片片段，將依以下時間順序合併：\n")
    for (dt, seq, dup), _, name in clips:
        print(f"  {dt.strftime('%Y-%m-%d %H:%M:%S')}  ->  {name}")

    if skipped:
        print(f"\n以下 {len(skipped)} 個檔案因命名格式不符而被略過：")
        for name in skipped:
            print(f"  - {name}")

    output_path = os.path.join(folder, output_name)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n目前模式：MODE = {MODE}")

    if MODE == 1:
        merge_mode1(clips, output_path)
    elif MODE == 2:
        merge_mode2(clips, output_path)
    else:
        print(f"錯誤：不支援的 MODE 值 {MODE}，請設為 1 或 2。")
        return

    print(f"\n完成！輸出檔案：{output_path}")


if __name__ == "__main__":
    main()