#!/usr/bin/env python3
"""
把資料夾底下所有解析度超過 1080p 的影片降到 1080p，直接取代原始檔案（檔名不變）。

判斷「超過 1080p」的方式：短邊（min(寬, 高)）> 1080 才算超過，不管影片是橫式還是
直式（手機直拍常見寬高對調），縮放時短邊固定縮到 1080、長邊依原始比例等比縮放，
不會變形。短邊本來就 <=1080 的影片直接跳過，不會重新編碼（避免無意義的畫質耗損）。

使用方式：
    1. 修改下方 INPUT_FOLDER（要處理的資料夾）
    2. 需要的話調整 RECURSIVE（是否連子資料夾一起處理）、DRY_RUN（先預覽不動手）
    3. 直接執行：python 影片降1080p.py

需求：
    - 已安裝 ffmpeg / ffprobe 並加入系統 PATH（跟 影片拼接.py 同一個需求）
    - Python 3.7+（只用標準函式庫：os/sys/subprocess/json/tempfile/time/threading，
      不需要另外 pip install 任何東西）

編碼器選擇（啟動時自動偵測，不用手動設定）：
    這台機器若偵測到 NVIDIA NVENC（RTX 3060 Laptop GPU 這類機器通常都有），優先用
    GPU 編碼，比純 CPU 的 libx264 快非常多；偵測不到 NVENC（沒有 NVIDIA 顯卡、驅動
    異常、這份 ffmpeg build 沒編譯 NVENC 支援等情況）會自動退回 CPU 的 libx264，
    腳本不會因此中止或需要手動介入。詳細的分層 fallback 邏輯見
    `build_pipeline_candidates()` 跟 `downscale_video()` 的說明。

注意（會直接覆蓋原始檔案，沒有備份機制）：
    - 縮放一定要重新編碼影片（畫面尺寸變了，沒辦法用 stream copy），音軌預設用
      `-c:a copy` 原樣保留；只有在 copy 失敗時才會重新編碼成 AAC（見下方
      「音訊處理」說明），不會平白無故損失音質。
    - 覆蓋方式是先輸出到同資料夾的暫存檔，ffmpeg 成功、且通過輸出驗證
      （`verify_output()`：檔案存在/大小>0/有影像串流/解析度正確/時長合理）
      之後，才用 os.replace() 換掉原始檔案（同一顆磁碟機上是原子操作）。任何
      一步失敗（ffmpeg crash、GPU 錯誤、驗證不過、甚至使用者按 Ctrl+C 中止）都
      只會留下沒清乾淨的暫存檔，原始檔案完全不會被動到。
    - 建議第一次執行先把 DRY_RUN 設成 True 看一下會處理哪些檔案，確認沒問題
      （尤其是資料夾路徑、RECURSIVE 範圍）再設回 False 正式執行——這個操作沒有
      「復原」按鈕，原始高解析度檔案處理完就沒了。

旋轉 metadata（手機直拍影片常見）：
    這裡刻意不強制用 GPU 解碼（`-hwaccel cuda`）處理有旋轉 metadata 的來源
    影片——GPU 解碼路徑（`-hwaccel_output_format cuda`）畫面全程留在顯示卡記憶體
    裡，ffmpeg 沒辦法像一般軟體解碼那樣自動插入修正方向用的 transpose 濾鏡，
    直接用 GPU 解碼可能會讓輸出影片方向跑掉（尤其手機直拍最常見）。腳本會先用
    ffprobe 檢查每支影片有沒有旋轉 metadata（`probe_rotation()`），有的話這支
    影片一律跳過 GPU 解碼管線、直接用 CPU 軟體解碼（軟體解碼路徑的自動旋轉修正
    行為是 ffmpeg 內建、可靠的，不需要額外處理），只有畫面縮放／編碼那一段還是
    會盡量用 GPU（NVENC）加速，不是整支影片都退回純 CPU。
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time

# ========== 請在這裡修改成你自己的路徑 / 選項 ==========
INPUT_FOLDER = r"C:\CatDataset\YouTube\Impressed cat video"
RECURSIVE = True  # True＝連子資料夾一起掃；False＝只掃 INPUT_FOLDER 這一層
DRY_RUN = False  # True＝只列出會處理哪些檔案，不會真的動任何檔案
# ========================================================

TARGET_SHORT_SIDE = 1080  # 短邊縮到這個高度/寬度（依影片是橫式還是直式決定對到哪一邊）
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".wmv", ".flv", ".ts"}

# CPU 編碼（libx264）參數——只有偵測不到 NVENC 時，或 NVENC 這條路徑本身失敗時
# 才會用到，跟 影片拼接.py 用同一組畫質參數（數字越小畫質越好、檔案越大）。
ENCODE_CRF = "20"
ENCODE_PRESET = "veryfast"

# NVENC（NVIDIA GPU 編碼）參數。RTX 3060 這類 Ampere 世代顯卡的 NVENC 支援
# p1～p7 新一代 preset（數字越大畫質越好、速度越慢），p4 是官方建議的中間點；
# 這份腳本主要用途是整理 AI/YOLO 資料集（不是剪輯成品），優先速度、其次才是
# 壓縮率，cq 20 在「肉眼看不太出來壓縮痕跡」跟「檔案不會過度肥大」之間算是
# 常見的折衷值，跟 libx264 的 crf 20 是同一套「數字越小畫質越好」的概念，方便
# 直接類比。這三個常數之後想自行調整畫質/速度平衡，改這裡就好，不用去改
# build_pipeline_candidates() 裡的組裝邏輯。
NVENC_PRESET = "p4"
NVENC_TUNE = "hq"
NVENC_CQ = "20"

# 目前只支援循序處理（一支影片處理完才換下一支）——筆電型 GPU 有功耗/散熱/VRAM/
# NVENC session 數量限制，跑太多平行工作反而不穩定，這裡先保守設計。之後若要
# 實驗平行處理，把這個值改成 >1，但目前的 main() 還沒有真的讀這個值去平行跑，
# 只是先把常數留在這裡、之後要加平行處理時不用另外設計參數命名。
MAX_PARALLEL_JOBS = 1

FAILED_LOG_FILENAME = "failed_videos.txt"


# ── 硬體能力偵測 ─────────────────────────────────────────────────────

def _capture_ffmpeg_help(args, timeout=15):
    """執行 `ffmpeg -hide_banner <args>` 並回傳完整輸出文字（stdout+stderr 合併，
    -encoders/-filters/-hwaccels 這類查詢指令有時候把內容印到 stderr）；ffmpeg
    根本不存在（沒裝、沒加進 PATH）或逾時都直接回傳空字串，呼叫端據此判斷「這個
    能力偵測不到」，不會讓整支腳本因為偵測階段就掛掉。"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner"] + args,
            capture_output=True, text=True, timeout=timeout,
        )
        return (result.stdout or "") + (result.stderr or "")
    except (OSError, subprocess.TimeoutExpired):
        return ""


def detect_capabilities():
    """回傳這台機器目前的 ffmpeg 實際具備哪些硬體加速能力——不是問使用者、不是
    假設「應該有」，是真的執行 `ffmpeg -encoders`/`-filters`/`-hwaccels` 讀輸出
    判斷。回傳的每一項都會拿去決定 build_pipeline_candidates() 要不要把對應的
    管線排進候選清單。"""
    encoders_out = _capture_ffmpeg_help(["-encoders"])
    filters_out = _capture_ffmpeg_help(["-filters"])
    hwaccels_out = _capture_ffmpeg_help(["-hwaccels"])
    ffmpeg_found = bool(encoders_out or filters_out or hwaccels_out)
    return {
        "ffmpeg_found": ffmpeg_found,
        "h264_nvenc": "h264_nvenc" in encoders_out,
        "hevc_nvenc": "hevc_nvenc" in encoders_out,
        "cuda_hwaccel": any(line.strip() == "cuda" for line in hwaccels_out.splitlines()),
        "scale_cuda": "scale_cuda" in filters_out,
    }


def print_capability_report(caps):
    def mark(flag):
        return "✓" if flag else "✗"

    print("=== 硬體能力檢測 ===")
    print(f"NVIDIA NVENC（h264_nvenc）：{mark(caps['h264_nvenc'])}")
    print(f"NVIDIA NVENC（hevc_nvenc）：{mark(caps['hevc_nvenc'])}")
    print(f"CUDA 硬體解碼（-hwaccel cuda）：{mark(caps['cuda_hwaccel'])}")
    print(f"GPU 縮放濾鏡（scale_cuda）：{mark(caps['scale_cuda'])}")

    if caps["h264_nvenc"] and caps["cuda_hwaccel"] and caps["scale_cuda"]:
        mode = "NVENC GPU 加速（GPU 解碼＋GPU 縮放＋NVENC 編碼，失敗時自動退回較保守的管線）"
    elif caps["h264_nvenc"]:
        mode = "NVENC GPU 加速（CPU 解碼/縮放＋NVENC 編碼，失敗時自動退回 CPU 編碼）"
    else:
        mode = "CPU 編碼（libx264）—— 這份 ffmpeg 沒偵測到可用的 NVENC"
    print(f"目前模式：{mode}\n")


# ── 編碼管線組裝 ─────────────────────────────────────────────────────

def build_pipeline_candidates(caps, allow_gpu_decode):
    """依偵測到的硬體能力，排出一串「依序嘗試」的編碼管線設定；downscale_video()
    會照順序試到第一個成功（ffmpeg 執行成功 + 輸出驗證通過）為止：

        1. GPU 解碼 + GPU 縮放（scale_cuda）+ NVENC —— 全程留在顯示卡記憶體，
           CPU 負擔最小、速度最快，但只有 h264_nvenc／cuda hwaccel／scale_cuda
           三者都偵測得到、而且 allow_gpu_decode 為 True（來源沒有旋轉 metadata，
           見模組說明）時才會排進候選清單。
        2. CPU 解碼 + CPU 縮放 + NVENC —— 只要偵測到 h264_nvenc 就會排進來，
           兼容性比全 GPU 管線好（不依賴 GPU 解碼支援該來源的編碼格式），編碼
           本身還是吃 GPU，仍然比純 CPU 快很多。
        3. CPU 解碼 + CPU 縮放 + libx264 —— 最後手段，任何機器都能跑，NVENC
           完全不可用或前兩個管線都失敗時才會用到。

    這個清單的目的是「不能因為 NVENC/GPU 不存在或某個來源格式跟 GPU 解碼不相容
    就讓整支影片處理失敗」——只要清單裡還有沒試過的管線，就繼續往下試，全部試
    完才真的算失敗。"""
    candidates = []

    nvenc_video_args = [
        "-c:v", "h264_nvenc",
        "-preset", NVENC_PRESET,
        "-tune", NVENC_TUNE,
        "-rc", "vbr",
        "-cq", NVENC_CQ,
        "-b:v", "0",
    ]
    libx264_video_args = [
        "-c:v", "libx264",
        "-preset", ENCODE_PRESET,
        "-crf", ENCODE_CRF,
        "-pix_fmt", "yuv420p",
    ]

    if allow_gpu_decode and caps["h264_nvenc"] and caps["cuda_hwaccel"] and caps["scale_cuda"]:
        candidates.append({
            "label": "GPU 解碼＋GPU 縮放＋NVENC",
            "use_hwaccel_cuda": True,
            "use_scale_cuda": True,
            "video_args": nvenc_video_args,
            "is_nvenc": True,
        })
    if caps["h264_nvenc"]:
        candidates.append({
            "label": "CPU 解碼＋CPU 縮放＋NVENC",
            "use_hwaccel_cuda": False,
            "use_scale_cuda": False,
            "video_args": nvenc_video_args,
            "is_nvenc": True,
        })
    candidates.append({
        "label": "CPU 解碼＋CPU 縮放＋libx264（CPU fallback）",
        "use_hwaccel_cuda": False,
        "use_scale_cuda": False,
        "video_args": libx264_video_args,
        "is_nvenc": False,
    })
    return candidates


def build_ffmpeg_cmd(pipeline, input_path, output_path, new_width, new_height, audio_mode):
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    if pipeline["use_hwaccel_cuda"]:
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    cmd += ["-i", input_path]

    if pipeline["use_scale_cuda"]:
        cmd += ["-vf", f"scale_cuda=w={new_width}:h={new_height}"]
    else:
        cmd += ["-vf", f"scale={new_width}:{new_height}"]

    cmd += pipeline["video_args"]

    if audio_mode == "copy":
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "192k"]

    # -map_metadata 0：盡量保留原始檔案的 metadata（拍攝時間等）。旋轉方向的
    # 處理見模組說明——有旋轉 metadata 的來源已經被排除在 GPU 解碼管線之外，
    # 走 CPU 解碼時 ffmpeg 對這類來源的自動旋轉修正是內建、可靠的行為。
    cmd += ["-map_metadata", "0"]
    cmd += ["-progress", "pipe:1", "-nostats", "-loglevel", "warning"]
    cmd += [output_path]
    return cmd


# ── 探測影片資訊 ─────────────────────────────────────────────────────

def collect_videos(folder, recursive):
    """回傳資料夾底下（依 recursive 決定要不要含子資料夾）所有副檔名符合
    VIDEO_EXTENSIONS 的檔案完整路徑，依路徑排序，方便輸出結果穩定可預期。"""
    paths = []
    if recursive:
        for root, _dirs, names in os.walk(folder):
            for name in names:
                if os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS:
                    paths.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder):
            full = os.path.join(folder, name)
            if os.path.isfile(full) and os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS:
                paths.append(full)
    paths.sort()
    return paths


def probe_video_info(path):
    """回傳 {"width":, "height":, "duration":}；讀取失敗（檔案損毀、不是有效
    影片等）回傳 None。duration 讀不到時是 None（不影響 width/height 的判斷，
    只會讓進度顯示跟輸出驗證的時長比對退化成「無法估算」，不會整個失敗）。"""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            return None
        width = streams[0].get("width")
        height = streams[0].get("height")
        if not width or not height:
            return None
        duration = None
        fmt_duration = data.get("format", {}).get("duration")
        if fmt_duration is not None:
            try:
                duration = float(fmt_duration)
            except (TypeError, ValueError):
                duration = None
        return {"width": int(width), "height": int(height), "duration": duration}
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError, subprocess.TimeoutExpired):
        return None


def probe_rotation(path):
    """回傳這支影片是不是帶有旋轉 metadata（手機直拍常見）。True 的話
    build_pipeline_candidates() 會把 GPU 解碼管線排除掉，理由見模組開頭的
    「旋轉 metadata」說明。讀取失敗一律當成「沒有旋轉」（回傳 False）——這只是
    一個「要不要放心用 GPU 解碼」的保守判斷，讀不到就保守地不假設有旋轉問題，
    仍然會嘗試 GPU 解碼管線；真正影響輸出正確性的是後面的 verify_output()。"""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream_side_data=rotation:stream_tags=rotate",
        "-of", "json", path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            return False
        stream = streams[0]
        tags = stream.get("tags", {})
        if str(tags.get("rotate", "0")).strip() not in ("0", ""):
            return True
        for side_data in stream.get("side_data_list", []):
            rotation = side_data.get("rotation")
            if rotation not in (None, 0):
                return True
        return False
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError, subprocess.TimeoutExpired):
        return False


def compute_target_size(width, height):
    """短邊縮到 TARGET_SHORT_SIDE、長邊依原始比例等比縮放，回傳 (new_width,
    new_height)（都無條件捨去到最接近的偶數——H.264 要求寬高都是偶數，奇數會
    編碼失敗）。呼叫前已經確認過短邊 > TARGET_SHORT_SIDE，這裡不重複判斷。"""
    short_side = min(width, height)
    scale = TARGET_SHORT_SIDE / short_side
    new_width = int(width * scale) // 2 * 2
    new_height = int(height * scale) // 2 * 2
    return (max(new_width, 2), max(new_height, 2))


# ── 輸出驗證 ─────────────────────────────────────────────────────────

def verify_output(tmp_path, expected_width, expected_height, original_duration):
    """在 os.replace() 覆蓋原始檔案之前的最後一道檢查：輸出檔真的存在、大小
    不是 0、讀得到影像串流、解析度符合預期、時長跟原始影片差不多（容許一點
    誤差，避免因為容器格式的時長計算方式差異就誤判）。任何一項不符都回傳
    (False, 原因字串)，呼叫端看到 False 就不會覆蓋，會把暫存檔刪掉、留原始
    檔案不動。"""
    if not os.path.isfile(tmp_path):
        return (False, "輸出檔不存在")
    if os.path.getsize(tmp_path) <= 0:
        return (False, "輸出檔大小為 0")

    info = probe_video_info(tmp_path)
    if info is None:
        return (False, "ffprobe 讀不到輸出檔的影像串流資訊")
    if info["width"] != expected_width or info["height"] != expected_height:
        return (
            False,
            f"輸出解析度 {info['width']}x{info['height']} 跟預期的"
            f" {expected_width}x{expected_height} 不符",
        )

    if original_duration and original_duration > 0:
        if info["duration"] is None:
            return (False, "輸出檔讀不到時長，無法比對是否完整")
        diff_seconds = abs(info["duration"] - original_duration)
        diff_ratio = diff_seconds / original_duration
        # 誤差同時超過 5% 跟 2 秒才算異常——短片段 5% 可能只是零點幾秒的容器
        # 時長計算誤差，長片段 2 秒以內的差異也是常見的正常誤差範圍。
        if diff_ratio > 0.05 and diff_seconds > 2.0:
            return (
                False,
                f"輸出時長 {info['duration']:.1f}s 跟原始 {original_duration:.1f}s"
                f" 差異過大（可能是編碼中途被截斷）",
            )
    return (True, "")


# ── 進度顯示 ─────────────────────────────────────────────────────────

def format_duration_short(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _parse_ffmpeg_timestamp(text):
    """把 -progress 輸出的 `out_time=00:00:12.345678` 這種字串轉成秒數（float）；
    格式不符就回傳 0.0，不拋例外——這只是拿來算進度百分比跟 ETA 用的，算錯也
    不該讓整個 ffmpeg 執行流程中斷。"""
    try:
        hours, minutes, secs = text.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(secs)
    except (ValueError, AttributeError):
        return 0.0


def _print_progress_line(fields, duration_seconds, prefix):
    out_time = fields.get("out_time", "00:00:00.000000")
    speed_text = fields.get("speed", "0x").strip()
    processed_seconds = _parse_ffmpeg_timestamp(out_time)

    percent_text = "  ?%"
    eta_text = "未知"
    if duration_seconds and duration_seconds > 0:
        percent = min(100.0, processed_seconds / duration_seconds * 100)
        percent_text = f"{percent:5.1f}%"
        try:
            speed_val = float(speed_text.rstrip("x"))
        except ValueError:
            speed_val = 0.0
        if speed_val > 0.01:
            remaining_seconds = max(0.0, duration_seconds - processed_seconds)
            eta_text = "約" + format_duration_short(remaining_seconds / speed_val)

    line = (
        f"\r{prefix}進度 {percent_text}  已處理 {format_duration_short(processed_seconds)}"
        f"  速度 {speed_text or '?'}  預估剩餘 {eta_text}    "
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def run_ffmpeg_with_progress(cmd, duration_seconds, prefix="    "):
    """執行 ffmpeg，即時解析 `-progress pipe:1` 寫到 stdout 的結構化進度輸出
    （out_time/speed/progress 這幾個 key=value 欄位），比直接 regex 硬解 stderr
    的人類可讀統計行穩定——那一行的格式在不同 ffmpeg 版本之間變過，`-progress`
    這個機制本來就是官方提供給程式化解析用的，不會這樣變動。

    stderr 另外開一個背景執行緒讀走並保留下來（不即時印出，避免跟進度行互相
    干擾），只有失敗時才回傳給呼叫端當診斷訊息用；不這樣做的話 stderr 管線的
    緩衝區塞滿會讓 ffmpeg 卡死（stdout/stderr 各自的 OS 管線緩衝區是獨立且
    容量有限的，只讀 stdout 不讀 stderr，長時間跑下來 stderr 那邊會被塞滿）。

    使用者按 Ctrl+C 時：盡快 terminate() 這個 ffmpeg 行程（給 5 秒優雅關閉，
    超過就 kill()），再把 KeyboardInterrupt 往外拋——呼叫端（downscale_video）
    接手清暫存檔，main() 再接手印出「使用者中止」的訊息並結束整支腳本，全程
    不會覆蓋原始檔案。"""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    stderr_lines = []

    def _drain_stderr():
        try:
            for line in proc.stderr:
                stderr_lines.append(line)
        except (OSError, ValueError):
            pass

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    fields = {}
    try:
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            fields[key] = value
            if key == "progress":
                _print_progress_line(fields, duration_seconds, prefix)
                if value == "end":
                    break
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        stderr_thread.join(timeout=2)
        sys.stdout.write("\n")
        raise

    proc.wait()
    stderr_thread.join(timeout=5)
    sys.stdout.write("\n")
    return (proc.returncode == 0, "".join(stderr_lines))


# ── 核心處理 ─────────────────────────────────────────────────────────

def _safe_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def downscale_video(path, new_width, new_height, pipelines, original_duration):
    """依序嘗試 pipelines 清單裡的每一種編碼管線，每種管線先試 `-c:a copy`
    （音軌不重新編碼，省時間也不損音質），失敗才試 `-c:a aac`（某些容器/編碼
    組合沒辦法直接 stream copy 音軌）。第一個「ffmpeg 執行成功 + 輸出驗證通過」
    的組合就是最終結果，之後的候選都不會再嘗試。

    全程輸出到同資料夾的暫存檔，只有最後真的要採用時才 os.replace() 覆蓋原始
    檔案；任何一種組合失敗，或使用者中途按 Ctrl+C，暫存檔都會被清掉、原始檔案
    完全不會被動到。

    回傳 (success, used_pipeline_label, error_reason)。"""
    folder = os.path.dirname(path)
    ext = os.path.splitext(path)[1]
    fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix=".downscale_tmp_", dir=folder)
    os.close(fd)

    last_error = ""
    used_label = None
    try:
        for pipeline in pipelines:
            for audio_mode in ("copy", "aac"):
                suffix = "" if audio_mode == "copy" else "（音軌改重新編碼為 AAC）"
                print(f"    嘗試：{pipeline['label']}{suffix}")
                cmd = build_ffmpeg_cmd(pipeline, path, tmp_path, new_width, new_height, audio_mode)
                success, stderr_text = run_ffmpeg_with_progress(cmd, original_duration)

                if not success:
                    last_error = f"[{pipeline['label']} / audio={audio_mode}] {stderr_text.strip()[-500:]}"
                    _safe_remove(tmp_path)
                    continue

                ok, reason = verify_output(tmp_path, new_width, new_height, original_duration)
                if not ok:
                    last_error = f"[{pipeline['label']} / audio={audio_mode}] 輸出驗證失敗：{reason}"
                    _safe_remove(tmp_path)
                    continue

                used_label = pipeline["label"]
                break
            if used_label:
                break

        if used_label is None:
            return (False, None, last_error or "所有編碼管線都嘗試失敗")

        try:
            os.replace(tmp_path, path)
        except OSError as e:
            return (False, None, f"換檔失敗（暫存檔已產生，但無法覆蓋原始檔案）：{e}")

        return (True, used_label, None)
    except KeyboardInterrupt:
        _safe_remove(tmp_path)
        raise
    finally:
        # 正常成功時 tmp_path 已經被 os.replace() 用掉、檔案不存在了，這裡再
        # 刪一次是安全的 no-op；只有在真的還留著暫存檔的失敗路徑上才有作用。
        _safe_remove(tmp_path)


# ── 主流程 ───────────────────────────────────────────────────────────

def write_failed_log(folder, failed_entries):
    if not failed_entries:
        return None
    log_path = os.path.join(folder, FAILED_LOG_FILENAME)
    with open(log_path, "w", encoding="utf-8") as f:
        for full_path, reason, attempted in failed_entries:
            f.write(f"{full_path}\n原因：{reason}\n嘗試過的編碼器：{attempted}\n\n")
    return log_path


def main():
    # Windows 上的終端機（尤其舊版 cmd.exe，或 IDE/排程工具用非 UTF-8 codepage
    # 啟動 python 的情況）預設輸出編碼可能不是 UTF-8，直接印這裡用到的 ✓／✗／⚠
    # 這類符號會整支腳本因為 UnicodeEncodeError 中斷。這裡強制把 stdout 轉成
    # UTF-8、遇到印不出來的字元用替代符號頂替（不拋例外），跟哪個 codepage
    # 無關，一律不會因為印字元就讓腳本停在半路——尤其不能讓這件事發生在正在
    # 處理一支影片、還沒 os.replace() 完成的當下。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    folder = INPUT_FOLDER
    if not os.path.isdir(folder):
        print(f"錯誤：找不到資料夾 {folder}")
        print("請確認腳本開頭的 INPUT_FOLDER 是否設定正確。")
        sys.exit(1)

    caps = detect_capabilities()
    if not caps["ffmpeg_found"]:
        print("錯誤：找不到 ffmpeg（或執行逾時）。請確認已安裝 ffmpeg / ffprobe 並加入系統 PATH。")
        sys.exit(1)
    print_capability_report(caps)

    if MAX_PARALLEL_JOBS != 1:
        print(
            f"⚠ MAX_PARALLEL_JOBS 目前設為 {MAX_PARALLEL_JOBS}，但這一版還沒實作平行處理，"
            "仍會照一支接一支的順序執行。\n"
        )

    videos = collect_videos(folder, RECURSIVE)
    if not videos:
        print(f"資料夾底下沒有找到符合副檔名的影片（{', '.join(sorted(VIDEO_EXTENSIONS))}）。")
        return

    print(f"掃到 {len(videos)} 支影片，開始逐一檢查解析度...")
    if DRY_RUN:
        print("【DRY_RUN 模式，不會真的處理任何檔案】")
    print()

    to_process_paths = []
    skipped_ok = 0
    failed_entries = []  # (full_path, reason, attempted_encoders)

    for path in videos:
        info = probe_video_info(path)
        if info is None:
            rel = os.path.relpath(path, folder)
            print(f"  ⚠ 跳過（讀不到解析度，可能檔案損毀）：{rel}")
            failed_entries.append((path, "讀不到解析度／可能檔案損毀", "（尚未嘗試編碼）"))
            continue
        short_side = min(info["width"], info["height"])
        if short_side <= TARGET_SHORT_SIDE:
            skipped_ok += 1
            continue
        to_process_paths.append((path, info))

    total_videos = len(videos)
    total_to_process = len(to_process_paths)
    success_count = 0
    nvenc_count = 0
    cpu_fallback_count = 0

    batch_start = time.time()
    processed_wall_seconds = 0.0

    try:
        for index, (path, info) in enumerate(to_process_paths, start=1):
            rel = os.path.relpath(path, folder)
            width, height = info["width"], info["height"]
            new_width, new_height = compute_target_size(width, height)
            duration = info["duration"]

            batch_percent = (index - 1) / total_to_process * 100 if total_to_process else 0.0
            eta_note = ""
            if index > 1 and processed_wall_seconds > 0:
                avg_seconds = processed_wall_seconds / (index - 1)
                remaining = avg_seconds * (total_to_process - index + 1)
                eta_note = f"，整批預估剩餘 約{format_duration_short(remaining)}（粗估，僅供參考）"

            print(f"[{index}/{total_to_process}]（整批進度 {batch_percent:5.1f}%{eta_note}）")
            print(f"  目前影片：{rel}")
            print(f"  原始解析度：{width}x{height}  →  目標解析度：{new_width}x{new_height}")
            print(f"  影片長度：{format_duration_short(duration) if duration else '未知'}")

            if DRY_RUN:
                print()
                continue

            has_rotation = probe_rotation(path)
            if has_rotation:
                print("  偵測到旋轉 metadata，這支影片跳過 GPU 解碼管線（改用 CPU 解碼）")
            pipelines = build_pipeline_candidates(caps, allow_gpu_decode=not has_rotation)

            video_start = time.time()
            ok, used_label, error_reason = downscale_video(path, new_width, new_height, pipelines, duration)
            processed_wall_seconds += time.time() - video_start

            if ok:
                success_count += 1
                if "NVENC" in used_label:
                    nvenc_count += 1
                else:
                    cpu_fallback_count += 1
                print(f"    ✓ 完成（{used_label}）\n")
            else:
                print(f"    ✗ 失敗：{error_reason}\n")
                failed_entries.append((path, error_reason or "未知錯誤", "／".join(p["label"] for p in pipelines)))
    except KeyboardInterrupt:
        print("\n\n使用者中止處理（Ctrl+C）")
        print("已停止目前的 ffmpeg 行程並清除暫存檔，原始影片未被覆蓋。")
        log_path = write_failed_log(folder, failed_entries)
        print(
            f"\n中止前已成功處理 {success_count}/{total_to_process} 支需要降解析度的影片"
            f"（{skipped_ok} 支本來就 <=1080p 不受影響）。"
        )
        if log_path:
            print(f"目前已知的失敗/跳過清單寫在：{log_path}")
        sys.exit(130)

    total_elapsed = time.time() - batch_start
    log_path = write_failed_log(folder, failed_entries)

    print("=" * 26)
    print("處理完成\n")
    print(f"總影片：{total_videos}")
    print(f"需要降解析度：{total_to_process}\n")
    print(f"成功：{success_count}")
    print(f"跳過（本來就 <=1080p）：{skipped_ok}")
    print(f"失敗：{len(failed_entries)}\n")
    print(f"NVENC：{nvenc_count}")
    print(f"CPU fallback（libx264）：{cpu_fallback_count}\n")
    print(f"總耗時：{format_duration_short(total_elapsed)}")
    print("=" * 26)

    if log_path:
        print(f"\n失敗清單已寫入：{log_path}")
    if DRY_RUN and total_to_process:
        print(
            "\n目前是 DRY_RUN 模式，以上檔案都還沒被實際處理；確認沒問題後把"
            " DRY_RUN 改成 False 再執行一次。"
        )


if __name__ == "__main__":
    main()
