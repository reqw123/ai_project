# -*- coding: utf-8 -*-
"""
merge_tapo_timeline.py
=======================

目的
----
將 TP-Link Tapo C425 事件錄影所產生的多支 MP4 片段，依照檔名中的時間資訊，
還原成「與真實時間長度一致」的完整時間軸影片。

由於 C425 為事件觸發錄影，影片與影片之間會存在沒有錄影的空白時間，
本程式會自動偵測這些空白區間，並補上顯示「No Event Recording」與
即時時間字幕的黑畫面，讓輸出影片的總長度等於真實經過的時間。

執行環境
--------
- Python 3.11
- Windows 11
- FFmpeg / ffprobe 已加入系統 PATH

僅使用下列套件（皆為標準函式庫或 tqdm）：
    os, pathlib, subprocess, datetime, re, csv, json, typing, tqdm

未使用 MoviePy，所有影片處理皆透過呼叫 ffmpeg / ffprobe 子行程完成。

使用方式
--------
1. 依需求修改本檔案最上方「可設定參數」區塊。
2. 直接執行：
       python merge_tapo_timeline.py
3. 完成後於 OUTPUT_FOLDER 內可找到：
       - timeline.mp4  （合併後的完整時間軸影片）
       - timeline.csv  （每段影片 / 黑畫面的詳細資訊，若 ENABLE_CSV=True）
       - error.log     （處理過程中發生的錯誤紀錄，若有錯誤才會產生）
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

from tqdm import tqdm

# =====================================================================
# 可設定參數（請依實際需求修改此區塊）
# =====================================================================

INPUT_FOLDER: str = r"C:\Users\homec\Downloads\0709_0600"
OUTPUT_FOLDER: str = r"C:\Users\homec\Downloads\0709_0600\output"

# 輸出檔名
OUTPUT_FILENAME: str = "timeline.mp4"

# 字型檔路徑，若留空字串則自動嘗試搜尋 arial.ttf
FONT_PATH: str = ""

# 重新編碼時使用的畫質參數（僅在無法直接 -c copy 時套用）
CRF: int = 18
PRESET: str = "medium"

# 功能開關
ENABLE_TIMESTAMP: bool = True        # 黑畫面右上角即時時間 + 下方靜態起始時間
ENABLE_NO_EVENT_TEXT: bool = True    # 黑畫面中央「No Event Recording」文字
ENABLE_CSV: bool = True              # 是否輸出 timeline.csv
ENABLE_LOG: bool = True              # 是否輸出 error.log

# -----------------------------------------------------------------
# 選用功能：指定要產生的時間區間（不填則處理資料夾內全部影片）
# 格式："HH:MM:SS"，日期會自動採用資料夾內第一支影片的日期。
# 例如只想要 06:00 ~ 08:00 的完整時間軸，設定：
#     FILTER_START_TIME = "06:00:00"
#     FILTER_END_TIME   = "08:00:00"
# -----------------------------------------------------------------
FILTER_START_TIME: Optional[str] = None
FILTER_END_TIME: Optional[str] = None

# 允許的黑畫面觸發最小空白秒數（小於此值不補黑畫面，避免產生極短片段）
MIN_GAP_SECONDS: float = 0.5

# -----------------------------------------------------------------
# ffmpeg / ffprobe 執行檔路徑。
# 預設為 "ffmpeg" / "ffprobe"，表示依賴系統 PATH 尋找。
# 若你的環境 PATH 沒有正確設定（例如出現 WinError 2 找不到檔案），
# 請直接改成完整絕對路徑，例如：
#     FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"
#     FFPROBE_PATH = r"C:\ffmpeg\bin\ffprobe.exe"
# -----------------------------------------------------------------
FFMPEG_PATH: str = "ffmpeg"
FFPROBE_PATH: str = "ffprobe"

# =====================================================================
# 內部常數
# =====================================================================

# 檔名格式範例：2026-07-09 06_07_54_0.mp4
#              2026-07-09 05_55_17_0(1).mp4
#              2026-07-09 06_07_54_0 副本.mp4
FILENAME_PATTERN = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})_0"
)

ERROR_LOG_LINES: List[str] = []  # 收集錯誤訊息，最後統一寫檔


# =====================================================================
# 資料結構定義
# =====================================================================

class VideoMetadata(NamedTuple):
    """單支影片的技術資訊（由 ffprobe 取得）。"""

    duration: float
    fps: float
    width: int
    height: int
    video_codec: str
    pix_fmt: str
    has_audio: bool
    audio_codec: Optional[str]


class TimelineEntry(NamedTuple):
    """時間軸上每一筆紀錄，用於輸出 timeline.csv。"""

    index: int
    original_filename: str
    start_time: datetime
    end_time: datetime
    duration: float
    gap_before: float
    gap_after: float
    black_screen_duration: float


# =====================================================================
# 工具函式：Log / 錯誤處理
# =====================================================================

def log_error(message: str) -> None:
    """
    記錄一筆錯誤訊息。

    此函式只將訊息收集到記憶體中的列表，並於程式結束前統一寫入
    error.log，避免因為單一影片處理失敗而中斷整個程式。
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(f"[錯誤] {message}")
    ERROR_LOG_LINES.append(line)


def flush_error_log(output_folder: Path) -> None:
    """將收集到的錯誤訊息寫入 error.log（若有內容且 ENABLE_LOG 為 True）。"""
    if not ENABLE_LOG or not ERROR_LOG_LINES:
        return
    log_path = output_folder / "error.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ERROR_LOG_LINES))
        f.write("\n")
    print(f"已將 {len(ERROR_LOG_LINES)} 筆錯誤訊息寫入：{log_path}")


# =====================================================================
# 工具函式：字型搜尋
# =====================================================================

def find_font_path() -> Optional[str]:
    """
    尋找可用的字型檔路徑。

    優先順序：
        1. FONT_PATH 設定值（若有填寫且檔案存在）
        2. 系統 PATH 中常見的 arial.ttf
        3. Windows 預設路徑 C:\\Windows\\Fonts\\arial.ttf

    若都找不到，回傳 None 並印出提醒；drawtext 濾鏡將不指定 fontfile，
    改用 ffmpeg 內建的預設字型行為（可能因系統而異）。
    """
    candidates = []
    if FONT_PATH:
        candidates.append(FONT_PATH)
    candidates.append("arial.ttf")
    candidates.append(r"C:\Windows\Fonts\arial.ttf")

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())

    print(
        "[警告] 找不到 arial.ttf 字型檔，字幕可能無法正常顯示。"
        "請於 FONT_PATH 參數手動指定字型檔路徑。"
    )
    return None


def escape_path_for_filter(path: str) -> str:
    """
    將 Windows 路徑轉換成可安全放入 ffmpeg filter 參數字串內的格式。

    做法：
        - 反斜線改為斜線（ffmpeg filter 對反斜線的處理容易出錯）
        - 冒號（磁碟機代號後的冒號）加上反斜線跳脫
    """
    normalized = path.replace("\\", "/")
    normalized = normalized.replace(":", r"\:")
    return normalized


# =====================================================================
# 工具函式：檔名時間解析
# =====================================================================

def parse_filename_datetime(filename: str) -> Optional[datetime]:
    """
    從檔名解析出拍攝起始時間。

    檔名格式固定為：YYYY-MM-DD HH_MM_SS_0.mp4
    可能夾帶額外後綴，例如 (1)、(2)、副本，皆以 Regex 忽略後綴的方式解析。

    回傳：
        解析成功回傳 datetime 物件；解析失敗回傳 None。
    """
    match = FILENAME_PATTERN.search(filename)
    if not match:
        return None
    try:
        date_str = match.group("date")
        hour = int(match.group("hour"))
        minute = int(match.group("minute"))
        second = int(match.group("second"))
        base_date = datetime.strptime(date_str, "%Y-%m-%d")
        return base_date.replace(hour=hour, minute=minute, second=second)
    except ValueError:
        return None


# =====================================================================
# 工具函式：ffprobe 取得影片資訊
# =====================================================================

def _parse_frame_rate(rate_str: str) -> float:
    """
    將 ffprobe 回傳的 r_frame_rate（例如 "30000/1001" 或 "25/1"）轉為浮點數 FPS。
    """
    if not rate_str:
        return 0.0
    if "/" in rate_str:
        numerator_str, denominator_str = rate_str.split("/", 1)
        try:
            numerator = float(numerator_str)
            denominator = float(denominator_str)
            if denominator == 0:
                return 0.0
            return numerator / denominator
        except ValueError:
            return 0.0
    try:
        return float(rate_str)
    except ValueError:
        return 0.0


def get_video_metadata(filepath: Path) -> Optional[VideoMetadata]:
    """
    使用 ffprobe 取得單支影片的時長、FPS、解析度、Codec、是否有音軌等資訊。

    若 ffprobe 執行失敗或影片損毀導致無法解析，回傳 None，
    呼叫端應記錄錯誤並跳過該影片。
    """
    command = [
        FFPROBE_PATH,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(filepath),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        log_error(f"ffprobe 執行失敗（{filepath.name}）：{exc}")
        return None

    if result.returncode != 0:
        log_error(f"ffprobe 回傳錯誤碼（{filepath.name}）：{result.stderr.strip()}")
        return None

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        log_error(f"ffprobe 輸出無法解析為 JSON（{filepath.name}）：{exc}")
        return None

    video_stream = None
    audio_stream = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    if video_stream is None:
        log_error(f"找不到影像串流（{filepath.name}），影片可能損毀。")
        return None

    try:
        duration_str = info.get("format", {}).get("duration") or video_stream.get("duration")
        duration = float(duration_str) if duration_str else 0.0
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        video_codec = str(video_stream.get("codec_name", "unknown"))
        pix_fmt = str(video_stream.get("pix_fmt", "yuv420p"))
        fps = _parse_frame_rate(video_stream.get("r_frame_rate", ""))
    except (TypeError, ValueError) as exc:
        log_error(f"解析影片資訊時發生錯誤（{filepath.name}）：{exc}")
        return None

    has_audio = audio_stream is not None
    audio_codec = str(audio_stream.get("codec_name")) if has_audio else None

    if duration <= 0 or width <= 0 or height <= 0 or fps <= 0:
        log_error(f"影片資訊不完整或異常（{filepath.name}），可能已損毀，將跳過。")
        return None

    return VideoMetadata(
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        video_codec=video_codec,
        pix_fmt=pix_fmt,
        has_audio=has_audio,
        audio_codec=audio_codec,
    )


# =====================================================================
# 時間區間篩選（選用功能：指定起始 / 結束時間）
# =====================================================================

def apply_time_range_filter(
    videos: List[Tuple[Path, datetime]],
    filter_start_str: Optional[str],
    filter_end_str: Optional[str],
) -> Tuple[List[Tuple[Path, datetime]], Optional[datetime], Optional[datetime]]:
    """
    依照 FILTER_START_TIME / FILTER_END_TIME 篩選影片清單。

    日期會採用清單中第一支影片的日期。只保留起始時間落在指定區間內的影片。

    回傳：
        (篩選後的影片清單, 區間起始 datetime 或 None, 區間結束 datetime 或 None)
    """
    if not filter_start_str and not filter_end_str:
        return videos, None, None

    if not videos:
        return videos, None, None

    base_date = videos[0][1].date()
    range_start: Optional[datetime] = None
    range_end: Optional[datetime] = None

    if filter_start_str:
        hour, minute, second = (int(part) for part in filter_start_str.split(":"))
        range_start = datetime.combine(base_date, datetime.min.time()).replace(
            hour=hour, minute=minute, second=second
        )
    if filter_end_str:
        hour, minute, second = (int(part) for part in filter_end_str.split(":"))
        range_end = datetime.combine(base_date, datetime.min.time()).replace(
            hour=hour, minute=minute, second=second
        )

    filtered = []
    for filepath, start_dt in videos:
        if range_start and start_dt < range_start:
            continue
        if range_end and start_dt >= range_end:
            continue
        filtered.append((filepath, start_dt))

    return filtered, range_start, range_end


# =====================================================================
# 黑畫面（No Event Recording）片段產生
# =====================================================================

def build_drawtext_filters(
    duration: float,
    segment_start_time: datetime,
    font_path: Optional[str],
) -> str:
    """
    組合黑畫面上要疊加的所有 drawtext 濾鏡字串。

    包含：
        - 中央：No Event Recording（ENABLE_NO_EVENT_TEXT 控制）
        - 中央下方：日期 / 起始時間（靜態文字，ENABLE_NO_EVENT_TEXT 控制）
        - 右上角：即時時間，每秒更新（ENABLE_TIMESTAMP 控制）
    """
    filters: List[str] = []
    font_option = f"fontfile='{escape_path_for_filter(font_path)}':" if font_path else ""

    if ENABLE_NO_EVENT_TEXT:
        filters.append(
            f"drawtext={font_option}"
            "text='No Event Recording':fontcolor=white:fontsize=48:"
            "box=1:boxcolor=black@0.5:boxborderw=12:"
            "x=(w-text_w)/2:y=(h-text_h)/2-70"
        )
        date_text = segment_start_time.strftime("%Y-%m-%d")
        filters.append(
            f"drawtext={font_option}"
            f"text='{date_text}':fontcolor=white:fontsize=32:"
            "box=1:boxcolor=black@0.4:boxborderw=8:"
            "x=(w-text_w)/2:y=(h-text_h)/2+10"
        )
        time_text = segment_start_time.strftime(r"%H\:%M\:%S")
        filters.append(
            f"drawtext={font_option}"
            f"text='{time_text}':fontcolor=white:fontsize=32:"
            "box=1:boxcolor=black@0.4:boxborderw=8:"
            "x=(w-text_w)/2:y=(h-text_h)/2+55"
        )

    if ENABLE_TIMESTAMP:
        epoch_seconds = int(segment_start_time.timestamp())
        # ffmpeg drawtext 的 pts:localtime 展開語法：
        #   %{pts\:localtime\:<起始epoch秒數>\:<strftime格式>}
        # 會依照畫面的 pts（播放進度）從指定的 epoch 秒數往後累加並格式化顯示，
        # 因此可以達到「每秒更新一次即時時間」的效果，且時間會等於
        # segment_start_time + 已播放秒數。
        live_clock_expr = (
            r"%{pts\:localtime\:" + str(epoch_seconds) + r"\:%H\\\:%M\\\:%S}"
        )
        filters.append(
            f"drawtext={font_option}"
            f"text='{live_clock_expr}':fontcolor=yellow:fontsize=28:"
            "box=1:boxcolor=black@0.4:boxborderw=6:"
            "x=w-text_w-20:y=20"
        )

    return ",".join(filters)


def generate_black_segment(
    output_path: Path,
    duration: float,
    reference: VideoMetadata,
    segment_start_time: datetime,
    font_path: Optional[str],
) -> bool:
    """
    產生一段指定長度的黑畫面影片，畫面規格（解析度 / FPS / 編碼）與原始影片一致。

    使用 ffmpeg 的 color lavfi 濾鏡直接產生黑畫面來源，不建立任何圖片檔案。
    若原始影片有音軌，會另外產生一軌靜音音軌，避免最終合併時音軌數量不一致。

    回傳：
        True 表示產生成功；False 表示 ffmpeg 執行失敗。
    """
    fps_str = f"{reference.fps:.6f}"
    color_source = f"color=c=black:s={reference.width}x{reference.height}:r={fps_str}:d={duration:.3f}"

    drawtext_chain = build_drawtext_filters(duration, segment_start_time, font_path)

    command: List[str] = [FFMPEG_PATH, "-y", "-f", "lavfi", "-i", color_source]

    if reference.has_audio:
        command += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]

    if drawtext_chain:
        filter_complex = f"[0:v]{drawtext_chain}[v]"
        command += ["-filter_complex", filter_complex, "-map", "[v]"]
    else:
        command += ["-map", "0:v"]

    if reference.has_audio:
        command += ["-map", "1:a", "-shortest"]

    command += [
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-crf", str(CRF),
        "-preset", PRESET,
        "-pix_fmt", reference.pix_fmt if reference.pix_fmt else "yuv420p",
    ]

    if reference.has_audio:
        command += ["-c:a", "aac", "-b:a", "128k"]

    command += [str(output_path)]

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    except (subprocess.SubprocessError, OSError) as exc:
        log_error(f"產生黑畫面失敗（{output_path.name}）：{exc}")
        return False

    if result.returncode != 0:
        log_error(f"ffmpeg 產生黑畫面回傳錯誤（{output_path.name}）：{result.stderr.strip()[-500:]}")
        return False

    return True


# =====================================================================
# 建立時間軸與片段清單
# =====================================================================

def build_timeline_and_segments(
    videos_with_meta: List[Tuple[Path, datetime, VideoMetadata]],
    output_folder: Path,
    font_path: Optional[str],
    range_start: Optional[datetime],
    range_end: Optional[datetime],
) -> Tuple[List[TimelineEntry], List[Path]]:
    """
    走訪已排序的影片清單，計算每支影片的起訖時間與彼此間的空白秒數，
    並在有空白時呼叫 generate_black_segment 產生對應的黑畫面片段。

    若有指定 range_start / range_end，也會在時間軸最前 / 最後方視需要補上
    黑畫面，讓輸出影片精確涵蓋整個指定區間。

    回傳：
        (timeline_entries, segment_files)
        - timeline_entries：用於輸出 CSV 的紀錄清單
        - segment_files：依播放順序排列的影片檔案清單（原始影片 + 黑畫面片段）
    """
    black_dir = output_folder / "black_segments"
    black_dir.mkdir(parents=True, exist_ok=True)

    timeline_entries: List[TimelineEntry] = []
    segment_files: List[Path] = []

    total = len(videos_with_meta)
    black_index = 0

    # 若有指定區間起始時間，且第一支影片開始時間晚於區間起始，於最前方補黑畫面
    if range_start is not None:
        first_start = videos_with_meta[0][1]
        leading_gap = (first_start - range_start).total_seconds()
        if leading_gap > MIN_GAP_SECONDS:
            black_index += 1
            black_path = black_dir / f"black_{black_index:04d}.mp4"
            reference_meta = videos_with_meta[0][2]
            if generate_black_segment(black_path, leading_gap, reference_meta, range_start, font_path):
                segment_files.append(black_path)
            else:
                log_error(f"開頭黑畫面產生失敗，時間軸開頭可能不完整（預期空白 {leading_gap:.1f} 秒）。")

    progress_bar = tqdm(enumerate(videos_with_meta, start=1), total=total, desc="建立時間軸")
    for index, (filepath, start_time, meta) in progress_bar:
        end_time = start_time + timedelta(seconds=meta.duration)

        gap_before = 0.0
        if index > 1:
            previous_end = videos_with_meta[index - 2][1] + timedelta(
                seconds=videos_with_meta[index - 2][2].duration
            )
            gap_before = max(0.0, (start_time - previous_end).total_seconds())

        gap_after = 0.0
        black_duration_after = 0.0
        if index < total:
            next_start = videos_with_meta[index][1]
            gap_after = max(0.0, (next_start - end_time).total_seconds())

        progress_bar.set_postfix_str(
            f"{filepath.name} | 長度 {meta.duration:.1f}s | 前方空白 {gap_before:.1f}s"
        )

        # 加入原始影片片段
        segment_files.append(filepath)

        # 若與下一支影片之間有空白，產生黑畫面
        if index < total and gap_after > MIN_GAP_SECONDS:
            black_index += 1
            black_path = black_dir / f"black_{black_index:04d}.mp4"
            if generate_black_segment(black_path, gap_after, meta, end_time, font_path):
                segment_files.append(black_path)
                black_duration_after = gap_after
            else:
                log_error(
                    f"影片 {filepath.name} 之後的黑畫面產生失敗"
                    f"（預期空白 {gap_after:.1f} 秒），時間軸可能不連續。"
                )

        timeline_entries.append(
            TimelineEntry(
                index=index,
                original_filename=filepath.name,
                start_time=start_time,
                end_time=end_time,
                duration=meta.duration,
                gap_before=gap_before,
                gap_after=gap_after,
                black_screen_duration=black_duration_after,
            )
        )

    # 若有指定區間結束時間，且最後一支影片結束時間早於區間結束，於最後方補黑畫面
    if range_end is not None:
        last_end = videos_with_meta[-1][1] + timedelta(seconds=videos_with_meta[-1][2].duration)
        trailing_gap = (range_end - last_end).total_seconds()
        if trailing_gap > MIN_GAP_SECONDS:
            black_index += 1
            black_path = black_dir / f"black_{black_index:04d}.mp4"
            reference_meta = videos_with_meta[-1][2]
            if generate_black_segment(black_path, trailing_gap, reference_meta, last_end, font_path):
                segment_files.append(black_path)
            else:
                log_error(f"結尾黑畫面產生失敗，時間軸結尾可能不完整（預期空白 {trailing_gap:.1f} 秒）。")

    return timeline_entries, segment_files


# =====================================================================
# 合併影片
# =====================================================================

def write_concat_file(segment_files: List[Path], concat_path: Path) -> None:
    """
    依播放順序寫出 ffmpeg concat demuxer 所需的清單檔（concat.txt）。

    ffmpeg concat 語法要求每行格式為：file '絕對路徑'
    路徑中的單引號需要另外跳脫，這裡以絕對路徑並轉換反斜線的方式降低出錯機會。
    """
    with open(concat_path, "w", encoding="utf-8") as f:
        for segment in segment_files:
            absolute_path = str(segment.resolve()).replace("\\", "/")
            escaped_path = absolute_path.replace("'", r"'\''")
            f.write(f"file '{escaped_path}'\n")


def concat_videos(concat_path: Path, output_path: Path) -> bool:
    """
    使用 ffmpeg concat demuxer 合併所有片段。

    策略：
        1. 優先嘗試 -c copy（不重新編碼，保留原始畫質、速度最快）。
        2. 若因為片段間編碼參數不一致導致失敗，改用 libx264 重新編碼
           （CRF / PRESET 依可設定參數區塊的設定），確保能成功輸出。
    """
    copy_command = [
        FFMPEG_PATH, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_path),
        "-c", "copy",
        str(output_path),
    ]

    print("正在嘗試以 -c copy（不重新編碼）合併影片...")
    try:
        result = subprocess.run(copy_command, capture_output=True, text=True, timeout=1800)
    except (subprocess.SubprocessError, OSError) as exc:
        log_error(f"合併影片時發生例外（-c copy）：{exc}")
        result = None

    if result is not None and result.returncode == 0:
        print("已成功以 -c copy 合併影片（未重新編碼，保留原始畫質）。")
        return True

    if result is not None:
        log_error(f"-c copy 合併失敗，改用重新編碼：{result.stderr.strip()[-500:]}")
    print("-c copy 合併失敗，改以 libx264 重新編碼合併（此步驟可能耗時較久）...")

    reencode_command = [
        FFMPEG_PATH, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_path),
        "-c:v", "libx264",
        "-crf", str(CRF),
        "-preset", PRESET,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_path),
    ]

    try:
        result = subprocess.run(reencode_command, capture_output=True, text=True, timeout=3600)
    except (subprocess.SubprocessError, OSError) as exc:
        log_error(f"合併影片時發生例外（重新編碼）：{exc}")
        return False

    if result.returncode != 0:
        log_error(f"重新編碼合併仍然失敗：{result.stderr.strip()[-800:]}")
        return False

    print("已成功以重新編碼方式合併影片。")
    return True


# =====================================================================
# CSV 輸出
# =====================================================================

def write_csv_report(timeline_entries: List[TimelineEntry], csv_path: Path) -> None:
    """將時間軸紀錄輸出成 timeline.csv，方便後續整理分析（例如貓咪行為資料）。"""
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Index",
                "Original Filename",
                "Start Time",
                "End Time",
                "Duration",
                "Gap Before",
                "Gap After",
                "Black Screen Duration",
            ]
        )
        for entry in timeline_entries:
            writer.writerow(
                [
                    entry.index,
                    entry.original_filename,
                    entry.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    entry.end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    f"{entry.duration:.3f}",
                    f"{entry.gap_before:.3f}",
                    f"{entry.gap_after:.3f}",
                    f"{entry.black_screen_duration:.3f}",
                ]
            )
    print(f"已輸出 CSV 報表：{csv_path}")


# =====================================================================
# 前置檢查：確認 ffmpeg / ffprobe 可以被正確呼叫
# =====================================================================

def check_ffmpeg_dependencies() -> bool:
    """
    在正式開始處理前，先確認 FFMPEG_PATH / FFPROBE_PATH 是否可以成功執行。

    這可以避免像「跑完全部 74 支影片才發現 ffprobe 根本找不到」這種
    浪費時間又不容易一眼看出原因的情況（Windows 上常見的錯誤訊息是
    WinError 2：系統找不到指定的檔案，代表 PATH 裡沒有這個執行檔，
    或是只安裝了 ffmpeg.exe 卻沒有 ffprobe.exe）。

    回傳：
        True 表示兩者皆可正常執行；False 表示至少一個有問題，
        此時 main() 應該直接中止，不要再繼續處理影片。
    """
    all_ok = True
    for label, exe_path in (("ffmpeg", FFMPEG_PATH), ("ffprobe", FFPROBE_PATH)):
        try:
            result = subprocess.run(
                [exe_path, "-version"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                print(f"[錯誤] 執行 {label} -version 回傳非 0 狀態碼，請確認安裝是否完整。")
                all_ok = False
        except FileNotFoundError:
            print(
                f"[錯誤] 找不到 {label} 執行檔（目前設定路徑：{exe_path}）。\n"
                f"       請確認：\n"
                f"       1. 是否真的有安裝 {label}.exe（部分精簡版 FFmpeg 只有 ffmpeg.exe，缺 ffprobe.exe）\n"
                f"       2. 在同一個終端機視窗手動輸入「{exe_path} -version」是否能執行\n"
                f"       3. 若 PATH 設定不穩定，可直接把本檔案開頭的 FFMPEG_PATH / FFPROBE_PATH\n"
                f"          改成完整絕對路徑，例如 r\"C:\\ffmpeg\\bin\\{label}.exe\""
            )
            all_ok = False
        except (subprocess.SubprocessError, OSError) as exc:
            print(f"[錯誤] 檢查 {label} 時發生例外：{exc}")
            all_ok = False
    return all_ok


# =====================================================================
# 主流程
# =====================================================================

def collect_and_sort_videos(input_folder: Path) -> List[Tuple[Path, datetime]]:
    """
    掃描輸入資料夾內所有 MP4，解析檔名時間並依時間排序。

    無法解析檔名的檔案會記錄錯誤並跳過，不影響其餘檔案處理。
    """
    mp4_files = sorted(input_folder.glob("*.mp4"))
    parsed: List[Tuple[Path, datetime]] = []

    for filepath in mp4_files:
        parsed_time = parse_filename_datetime(filepath.name)
        if parsed_time is None:
            log_error(f"檔名時間解析失敗，已跳過：{filepath.name}")
            continue
        parsed.append((filepath, parsed_time))

    parsed.sort(key=lambda item: item[1])
    return parsed


def probe_all_videos(
    videos: List[Tuple[Path, datetime]]
) -> List[Tuple[Path, datetime, VideoMetadata]]:
    """對已排序的影片清單逐一呼叫 ffprobe，取得技術資訊；失敗者記錄錯誤並跳過。"""
    results: List[Tuple[Path, datetime, VideoMetadata]] = []
    for filepath, start_time in tqdm(videos, desc="讀取影片資訊 (ffprobe)"):
        meta = get_video_metadata(filepath)
        if meta is None:
            continue
        results.append((filepath, start_time, meta))
    return results


def main() -> None:
    """程式進入點：整合所有步驟，完成影片時間軸還原與合併。"""
    input_folder = Path(INPUT_FOLDER)
    output_folder = Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)

    if not input_folder.is_dir():
        print(f"[錯誤] 輸入資料夾不存在：{input_folder}")
        return

    print("檢查 ffmpeg / ffprobe 是否可正常執行...")
    if not check_ffmpeg_dependencies():
        print("[錯誤] ffmpeg / ffprobe 檢查未通過，程式中止。請參考上方訊息排除問題後再重新執行。")
        return
    print("ffmpeg / ffprobe 檢查通過。")

    print(f"掃描輸入資料夾：{input_folder}")
    sorted_videos = collect_and_sort_videos(input_folder)
    if not sorted_videos:
        print("[錯誤] 資料夾內沒有找到可解析檔名的 MP4 檔案。")
        flush_error_log(output_folder)
        return

    print(f"共找到 {len(sorted_videos)} 支可解析的影片，開始套用時間區間篩選（若有設定）...")
    filtered_videos, range_start, range_end = apply_time_range_filter(
        sorted_videos, FILTER_START_TIME, FILTER_END_TIME
    )
    if not filtered_videos:
        print("[錯誤] 指定的時間區間內沒有任何影片，請確認 FILTER_START_TIME / FILTER_END_TIME 設定。")
        flush_error_log(output_folder)
        return

    print(f"篩選後共 {len(filtered_videos)} 支影片，開始讀取影片技術資訊...")
    videos_with_meta = probe_all_videos(filtered_videos)
    if not videos_with_meta:
        print("[錯誤] 所有影片皆讀取失敗，無法繼續。")
        flush_error_log(output_folder)
        return

    font_path = find_font_path()

    print("開始建立時間軸並產生所需的黑畫面片段...")
    timeline_entries, segment_files = build_timeline_and_segments(
        videos_with_meta, output_folder, font_path, range_start, range_end
    )

    if not segment_files:
        print("[錯誤] 沒有可合併的片段。")
        flush_error_log(output_folder)
        return

    concat_path = output_folder / "concat.txt"
    write_concat_file(segment_files, concat_path)

    output_video_path = output_folder / OUTPUT_FILENAME
    print(f"開始合併影片，共 {len(segment_files)} 個片段，輸出至：{output_video_path}")
    success = concat_videos(concat_path, output_video_path)

    if ENABLE_CSV:
        csv_path = output_folder / "timeline.csv"
        write_csv_report(timeline_entries, csv_path)

    flush_error_log(output_folder)

    if success:
        total_duration = sum(
            entry.duration + entry.black_screen_duration for entry in timeline_entries
        )
        print("=" * 60)
        print("處理完成！")
        print(f"輸出影片：{output_video_path}")
        print(f"時間軸總長度（約）：{total_duration:.1f} 秒")
        print("=" * 60)
    else:
        print("[錯誤] 影片合併失敗，請檢查 error.log 內容。")


if __name__ == "__main__":
    main()