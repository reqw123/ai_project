"""
YouTube 貓咪影片資料集下載工具（增量更新 + 歷史回填模式）

目標是大量收集 YouTube 貓咪影片作為深度學習資料集，設計以穩定性、可長時間
無人值守執行為優先，而非下載速度：逐支影片個別下載（單支失敗不影響整批）、
失敗自動重試、限流自動等待、每批次自動休息、全程記錄 log/CSV/失敗清單。

每次執行做兩件事：

  ① 增量掃描（追新影片，全自動）
     從頻道最新影片開始，逐步擴大搜尋視窗尋找上次記錄的 latest_video_id，
     找到就停止——只回傳比它更新的影片，不重新掃描/讀取整個頻道（可能上萬支）。
     進度記在 last_checkpoint.json，全自動、不用手動管理。

  ② 歷史回填（backfill，處理現有舊影片，手動指定區間）
     直接用 BACKFILL_START / BACKFILL_END 兩個變數明確指定這次要處理第幾支
     到第幾支（1-based，新→舊排序，跟原本 PLAYLIST_START/END 用法一樣）。
     沒有隱藏狀態、不會自動變動——要換下一批（例如 101~200）就自己改這兩個
     數字重新執行，一目了然。重複處理同一個區間也安全：download_archive
     會自動略過已經下載過的，不會重複下載。

download_archive 是「是否真的下載過」的唯一依據；checkpoint 只負責加速①的
掃描（找不到對應影片時自動退回完整掃描一次，不影響下載正確性），跟②完全無關。
"""
import os
import csv
import json
import time
import logging
from datetime import datetime

import yt_dlp

# ==================== 基本設定 ====================
DOWNLOAD_DIR = r"C:\CatDataset\YouTube"
CHANNEL_URL = "https://www.youtube.com/@ImpressedCatVideo/videos"
FFMPEG = r"C:\ffmpeg\bin"

# 遇到限流（429 / rate-limited / try again later）時等待秒數，等完會無限重試直到解除
RATE_LIMIT_WAIT = 3600

# 單支影片下載失敗時的外層重試上限（不含限流等待；限流等待不計入這個上限）
MAX_RETRIES_PER_VIDEO = 10

# 每下載這麼多支影片，自動休息一段時間，降低被 YouTube 封鎖的風險
BATCH_SIZE = 50
BATCH_REST_SECONDS = 300  # 5 分鐘

# ── 增量掃描視窗設定（①追新影片用，全自動不用管） ──────────────────────────
# 第一次嘗試只讀取「最新 INITIAL_SCAN_WINDOW 支」影片，在裡面找上次的
# checkpoint；找不到就把視窗放大 SCAN_WINDOW_GROWTH 倍再試一次，直到找到、
# 或視窗達到 MAX_SCAN_WINDOW 仍找不到（此時退回完整掃描一次）。
INITIAL_SCAN_WINDOW = 30
SCAN_WINDOW_GROWTH  = 5
MAX_SCAN_WINDOW      = 2000

# ── 歷史回填區間（②處理頻道現有舊影片用，手動指定，明確清楚） ──────────────
# 1-based、含頭尾，依頻道新→舊排序算第幾支。例如這次 1~100，下次要換下一批
# 就自己改成 101~200，以此類推——不需要另外設「最大處理數量」，區間大小
# （BACKFILL_END - BACKFILL_START + 1）本身就決定了這次會處理幾支。
BACKFILL_START = 80
BACKFILL_END   = 150

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

ARCHIVE_PATH    = os.path.join(DOWNLOAD_DIR, "downloaded.txt")
CSV_PATH        = os.path.join(DOWNLOAD_DIR, "video_list.csv")
FAILED_PATH     = os.path.join(DOWNLOAD_DIR, "failed.txt")
LOG_PATH        = os.path.join(DOWNLOAD_DIR, "download.log")
CHECKPOINT_PATH = os.path.join(DOWNLOAD_DIR, "last_checkpoint.json")

RATE_LIMIT_SIGNS = ("rate-limited", "try again later", "429", "this content isn't available")


# ==================== Logger（console + download.log 雙輸出） ====================
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("youtube_dataset_downloader")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


logger = setup_logger()


def is_rate_limited(error_msg: str) -> bool:
    m = error_msg.lower()
    return any(sign in m for sign in RATE_LIMIT_SIGNS)


# ==================== Checkpoint（只給①增量掃描加速用，跟②歷史回填無關） ====================
def load_checkpoint():
    """讀取 last_checkpoint.json；不存在、格式錯誤或缺 latest_video_id 都視為
    「遺失」，回傳 None，呼叫端會自動退回完整掃描一次。"""
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("latest_video_id"):
            return data
    except Exception as e:
        logger.warning(f"checkpoint 讀取失敗，視為遺失：{e}")
    return None


def save_checkpoint(latest_video_id: str):
    """寫入/更新 checkpoint。只在「整批新影片都處理完（不論成功/失敗）」之後
    呼叫一次——否則中途中斷會讓還沒處理到的影片被下次的增量掃描誤判成
    『比 checkpoint 舊』而永久漏掉。"""
    data = {
        "latest_video_id": latest_video_id,
        "last_scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"checkpoint 寫入失敗：{e}")


# ==================== 頻道 / 播放清單資訊 ====================
def get_channel_entries(url: str):
    """完整掃描：用 extract_flat 取得整個頻道/播放清單的影片清單（不觸發下載），
    回傳 (頻道名稱, entries)。只在第一次執行、或增量掃描找不到 checkpoint 對應
    影片時呼叫，頻道影片很多時這一步會比增量掃描慢很多。"""
    logger.info("=" * 60)
    logger.info("完整掃描：讀取整個頻道/播放清單資訊...")
    logger.info("=" * 60)

    opts = {"extract_flat": True, "quiet": True, "ignoreerrors": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    entries = [e for e in (info.get("entries", []) if info else []) if e]
    channel_name = info.get("title") if info else "未知"

    logger.info(f"頻道／播放清單：{channel_name}")
    logger.info(f"目前可讀取影片數：{len(entries)}")
    logger.info("=" * 60)
    return channel_name, entries


def find_new_entries_incremental(url: str, checkpoint_id: str):
    """
    增量掃描：只從頻道最新影片開始，逐步擴大視窗尋找 checkpoint_id
    （上次記錄的最新影片），找到就立刻停止，回傳「比它更新」的那些 entries
    （newest-first），不繼續往更舊的影片讀取，因此正常情況下不需要掃過
    整個頻道（可能上萬支影片）。

    前提：頻道 /videos 分頁預設是「新→舊」排序（YouTube 標準行為，
    CHANNEL_URL 本身指向的就是這個分頁）。

    找不到（可能該影片已被下架，或新增數量超過 MAX_SCAN_WINDOW）時回傳
    None，交由呼叫端退回完整掃描一次。
    """
    window = INITIAL_SCAN_WINDOW
    while True:
        logger.info(f"增量掃描：讀取最新 {window} 支影片，尋找上次記錄的位置...")
        opts = {"extract_flat": True, "quiet": True, "ignoreerrors": True,
                "playlistend": window}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        entries = [e for e in (info.get("entries", []) if info else []) if e]

        ids = [e.get("id") for e in entries]
        if checkpoint_id in ids:
            idx = ids.index(checkpoint_id)
            return entries[:idx]

        if len(entries) < window or window >= MAX_SCAN_WINDOW:
            # 已經掃到清單尾端，或視窗已達上限仍找不到 → 交給呼叫端退回完整掃描
            return None

        window = min(window * SCAN_WINDOW_GROWTH, MAX_SCAN_WINDOW)


def get_backfill_range(url: str, start: int, end: int):
    """
    歷史回填：抓第 start ~ end 支影片（1-based，含頭尾，依頻道新→舊排序，
    對應 BACKFILL_START/BACKFILL_END）。

    做法是 extract_flat + playlistend=end 再切片 [start-1:end]——end 越大，
    需要讀取的清單就越長，這是頻道分頁機制本身的限制（跟原本用
    yt-dlp 原生 playliststart/playlistend 的成本一樣，不是這裡多出來的負擔）。

    回傳 (chunk_entries, total_scanned)；total_scanned 是這次實際讀到的清單
    長度，可用來判斷 end 是否已經超過頻道現有影片數（代表這已經是最後一批）。
    """
    opts = {"extract_flat": True, "quiet": True, "ignoreerrors": True,
            "playlistend": end}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = [e for e in (info.get("entries", []) if info else []) if e]
    return entries[start - 1:end], len(entries)


def entry_to_url(entry: dict) -> str:
    url = entry.get("url") or entry.get("webpage_url")
    if url and not url.startswith("http"):
        # extract_flat 有時只給 video id 當 url
        url = f"https://www.youtube.com/watch?v={url}"
    if not url and entry.get("id"):
        url = f"https://www.youtube.com/watch?v={entry['id']}"
    return url or ""


def load_archived_ids() -> set:
    """讀取 download_archive，回傳已下載過的 video id 集合，用於下載前主動略過
    （比讓 yt-dlp 事後才判斷已下載更省一次網路請求）。這是「是否真的下載過」
    的唯一依據，跟 checkpoint（只負責加速①的掃描）職責分開。"""
    ids = set()
    if os.path.exists(ARCHIVE_PATH):
        with open(ARCHIVE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    ids.add(parts[1])
    return ids


# ==================== yt-dlp 下載選項 ====================
def build_opts_for_video() -> dict:
    """單支影片下載用的 opts。Cookies 為可選，不在此啟用；若之後要開啟
    cookiesfrombrowser，因為整個下載呼叫都包在 try/except 裡，
    取不到 cookies 也只會被視為單支影片失敗、重試，不會中止整批程式。"""
    return {
        # 最高畫質：YouTube 提供到哪個解析度（1080p/1440p/2160p/4320p）就抓到哪
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "ffmpeg_location": FFMPEG,

        "paths": {"home": DOWNLOAD_DIR},
        "outtmpl": os.path.join(
            DOWNLOAD_DIR,
            "%(uploader)s",
            "%(upload_date>%Y-%m-%d)s - %(title)s [%(id)s].%(ext)s",
        ),

        "download_archive": ARCHIVE_PATH,
        "continuedl": True,
        "overwrites": False,
        # False（而非 True）是刻意的：我們已經自己逐支下載＋外層重試/failed.txt 記錄，
        # 若這裡設 True，yt-dlp 遇到內部錯誤會自己吞掉、只印 "ERROR:" 不往外拋例外，
        # download_single_video() 的 try/except 就永遠抓不到，會把失敗誤判成功。
        "ignoreerrors": False,
        "writethumbnail": False,

        "sleep_interval": 3,
        "max_sleep_interval": 8,
        "sleep_requests": 1.5,

        "retries": 20,
        "fragment_retries": 20,
        "socket_timeout": 30,

        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/137.0 Safari/537.36"
        ),

        # "cookiesfrombrowser": ("chrome",),  # 可選：若 Chrome 有登入 YouTube 可減少部分限制

        "js_runtimes": {"node": {}},  # 值必須是設定 dict（可為空 {}），傳 None 會在
                                       # YoutubeDL._js_runtimes 內部呼叫 config.get('path')
                                       # 時炸掉（'NoneType' object has no attribute 'get'）
        "progress_with_newline": True,
    }


# ==================== CSV / 失敗清單 ====================
CSV_FIELDS = ["video_id", "title", "url", "upload_date", "channel",
              "duration_sec", "resolution", "download_time"]


def ensure_csv_header():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv_row(info: dict):
    row = {
        "video_id":     info.get("id", ""),
        "title":        info.get("title", ""),
        "url":          info.get("webpage_url") or info.get("original_url", ""),
        "upload_date":  info.get("upload_date", ""),
        "channel":      info.get("uploader", ""),
        "duration_sec": info.get("duration", ""),
        "resolution":   f"{info.get('width', '')}x{info.get('height', '')}",
        "download_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def append_failed(video_url: str, error_msg: str):
    with open(FAILED_PATH, "a", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts}\t{video_url}\t{error_msg}\n")


# ==================== 逐支下載（單支失敗不影響整批） ====================
def download_single_video(video_url: str) -> bool:
    """下載單支影片，最多重試 MAX_RETRIES_PER_VIDEO 次。
    遇到限流會等待 RATE_LIMIT_WAIT 秒後重試，限流等待不計入重試次數上限。
    成功（含已下載過、被 archive 略過的情況）回傳 True，重試用盡仍失敗回傳 False。
    """
    attempt = 0
    while attempt < MAX_RETRIES_PER_VIDEO:
        attempt += 1
        try:
            with yt_dlp.YoutubeDL(build_opts_for_video()) as ydl:
                info = ydl.extract_info(video_url, download=True)
            if info:
                append_csv_row(info)
            return True

        except Exception as e:
            msg = str(e)
            if is_rate_limited(msg):
                logger.warning(f"偵測到限流，等待 {RATE_LIMIT_WAIT // 60} 分鐘後重試：{video_url}")
                time.sleep(RATE_LIMIT_WAIT)
                attempt -= 1  # 限流等待不算一次失敗重試
                continue

            logger.warning(f"[第 {attempt}/{MAX_RETRIES_PER_VIDEO} 次嘗試失敗] {video_url}\n  錯誤：{msg}")
            if attempt >= MAX_RETRIES_PER_VIDEO:
                append_failed(video_url, msg)
                return False
            time.sleep(5)

    return False


def process_entries(entries: list, archived_ids: set, tag: str) -> tuple:
    """對一批 entries 逐支下載：套用 archive 主動略過、失敗重試、批次休息，
    跟①②共用同一套邏輯。tag 只用來讓 log 標明是哪個階段（新影片/歷史回填）。
    回傳 (success_count, fail_count, skip_count, urls_with_id)。"""
    urls_with_id = []
    for e in entries:
        vid = e.get("id", "")
        url = entry_to_url(e)
        if url:
            urls_with_id.append((vid, url))

    success_count = fail_count = skip_count = 0
    for i, (vid, video_url) in enumerate(urls_with_id, 1):
        logger.info("-" * 60)

        if vid and vid in archived_ids:
            logger.info(f"[{tag} {i}/{len(urls_with_id)}] 已下載過，略過：{video_url}")
            skip_count += 1
            continue

        logger.info(f"[{tag} {i}/{len(urls_with_id)}] 開始下載：{video_url}")
        ok = download_single_video(video_url)
        if ok:
            success_count += 1
        else:
            fail_count += 1
            logger.error(f"放棄此影片（已重試 {MAX_RETRIES_PER_VIDEO} 次仍失敗）：{video_url}")

        if i % BATCH_SIZE == 0 and i < len(urls_with_id):
            logger.info(f"[{tag}] 已處理 {i} 支，休息 {BATCH_REST_SECONDS // 60} 分鐘後繼續...")
            time.sleep(BATCH_REST_SECONDS)

    return success_count, fail_count, skip_count, urls_with_id


# ==================== 主流程（① 增量更新 + ② 歷史回填） ====================
def run():
    ensure_csv_header()
    archived_ids = load_archived_ids()
    checkpoint = load_checkpoint()

    # ── ① 增量掃描：抓「比上次記錄還新」的新影片（全自動） ──────────────────
    incremental = False
    channel_name = None
    new_entries = []
    latest_id_seen = checkpoint.get("latest_video_id") if checkpoint else None

    if checkpoint:
        logger.info(f"讀取到 checkpoint：latest_video_id={checkpoint['latest_video_id']}  "
                    f"上次掃描時間={checkpoint.get('last_scan_time', '未知')}")
        found = find_new_entries_incremental(CHANNEL_URL, checkpoint["latest_video_id"])
        if found is not None:
            incremental = True
            new_entries = found
            if new_entries:
                latest_id_seen = new_entries[0].get("id")
            logger.info(f"增量掃描完成，發現 {len(new_entries)} 支新影片（未重新讀取整個頻道）")
        else:
            logger.warning("checkpoint 對應的影片在掃描視窗內找不到（可能已被下架，或新增數量"
                           "超過搜尋上限），改為完整掃描一次")

    if not incremental:
        is_first_run = checkpoint is None
        logger.info("執行完整掃描以取得頻道現況" +
                    ("（第一次執行）" if is_first_run else "（checkpoint 失效，保守起見退回完整掃描）") + "...")
        channel_name, full_entries = get_channel_entries(CHANNEL_URL)
        if full_entries:
            latest_id_seen = full_entries[0].get("id")
        if is_first_run:
            # 真正的第一次執行：這裡只記錄「目前最新影片」的位置，不下載整個
            # 頻道——現有的舊影片交給下面②的歷史回填處理（由 BACKFILL_START/
            # BACKFILL_END 決定範圍），避免第一次就是一個下載上萬支影片、
            # 跑很久又容易中斷的巨大單次執行。
            new_entries = []
        else:
            # checkpoint 存在但增量掃描找不到對應影片：無法安全判斷新舊分界，
            # 保守起見整份清單都當待處理，archive 會自動略過真正已下載過的，
            # 不會真的重複下載，只是這一輪會比平常慢。
            new_entries = full_entries

    print()
    s1 = f1 = k1 = 0
    if new_entries:
        mode_str = "增量模式" if incremental else "完整掃描退回"
        logger.info(f"[① 新影片] 本次待處理：{len(new_entries)} 支（{mode_str}）")
        s1, f1, k1, _ = process_entries(new_entries, archived_ids, tag="新影片")
    else:
        logger.info("[① 新影片] 沒有發現需要處理的新影片。")

    if latest_id_seen:
        save_checkpoint(latest_id_seen)
        logger.info(f"Checkpoint（最新影片位置）已更新：latest_video_id={latest_id_seen}")

    # ── ② 歷史回填：處理 BACKFILL_START ~ BACKFILL_END 這個明確區間 ──────────
    # 完全由這兩個變數決定，沒有任何自動狀態；要換下一批（例如 101~200）
    # 直接改上面的 BACKFILL_START/BACKFILL_END 重新執行即可。
    logger.info(f"[② 歷史回填] 處理第 {BACKFILL_START} ~ {BACKFILL_END} 支影片...")
    chunk, total_scanned = get_backfill_range(CHANNEL_URL, BACKFILL_START, BACKFILL_END)

    s2 = f2 = k2 = 0
    if chunk:
        s2, f2, k2, _ = process_entries(chunk, archived_ids, tag="歷史回填")
    else:
        logger.info(f"[② 歷史回填] 第 {BACKFILL_START}~{BACKFILL_END} 支超出頻道範圍"
                    f"（頻道目前共 {total_scanned} 支），沒有影片需要處理。")

    if total_scanned <= BACKFILL_END:
        logger.info(f"[② 歷史回填] BACKFILL_END={BACKFILL_END} 已達或超過頻道目前總影片數"
                    f"（{total_scanned} 支），這已經是最後一批。")
    else:
        logger.info(f"[② 歷史回填] 下一批可以把 BACKFILL_START/BACKFILL_END 改成 "
                    f"{BACKFILL_END + 1}~{BACKFILL_END + (BACKFILL_END - BACKFILL_START + 1)}。")

    logger.info("=" * 60)
    logger.info(f"「{channel_name or CHANNEL_URL}」本次下載作業結束")
    logger.info(f"[① 新影片]   成功：{s1}　略過：{k1}　失敗：{f1}")
    logger.info(f"[② 歷史回填] 成功：{s2}　略過：{k2}　失敗：{f2}")
    logger.info(f"CSV 紀錄：{CSV_PATH}")
    logger.info(f"失敗清單：{FAILED_PATH}")
    logger.info(f"Checkpoint：{CHECKPOINT_PATH}")
    logger.info(f"完整 log：{LOG_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
