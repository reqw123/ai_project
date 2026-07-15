#!/usr/bin/env python3
"""
fetch_cat_behavior_videos.py

從 Pixabay API 與 Pexels API 依「貓咪行為」關鍵字批次搜尋免費影片素材，
整理成一份 CSV，方便後續人工篩選、下載、標註。

=== 使用前準備 ===
1. 申請 Pixabay API key（免費）：
   https://pixabay.com/api/docs/  -> 註冊帳號 -> 頁面上會直接顯示你的 key
2. 申請 Pexels API key（免費）：
   https://www.pexels.com/api/  -> 註冊帳號 -> 立即取得 key
3. 設定環境變數（或直接改下面 API_KEYS 區塊）：
   export PIXABAY_API_KEY="your_pixabay_key"
   export PEXELS_API_KEY="your_pexels_key"
4. 安裝需求套件：
   pip install requests --break-system-packages

=== 使用方式 ===
   python fetch_cat_behavior_videos.py
   # 預設會搜尋 BEHAVIOR_KEYWORDS 中列出的所有關鍵字
   # 結果輸出到 cat_behavior_videos.csv
   # 若 DOWNLOAD_VIDEOS = True（預設開啟），會自動把影片下載到
   #   downloaded_videos/<執行時間戳記>/ 資料夾（每次執行獨立子資料夾）
   # 可調整程式開頭的 MAX_DOWNLOADS 控制最多下載幾支，避免一次佔滿硬碟/流量

=== 執行紀錄與去重複 ===
- 每次執行都會用當下時間建立獨立子資料夾，例如：
    downloaded_videos/20260701_153045/
  所以不同次執行的結果不會互相覆蓋或混在一起。
- 每次執行的摘要（時間、找到筆數、下載成功數、重複數、失敗數）會
  append 到 downloaded_videos/run_log.csv，方便回顧歷次執行紀錄。
- 去重複分兩層：
  1) 同一次搜尋若同一支影片被不同關鍵字搜到兩次（同 source + 同 id），
     只保留一筆，避免重複下載。
  2) 下載時會即時計算每支影片的 MD5 內容雜湊值，若跟已下載過的影片內容
     完全相同，會自動刪除剛下載的重複檔案。
     去重複範圍由 GLOBAL_DEDUP 控制：
       GLOBAL_DEDUP = False（預設）：只比對「這次執行的資料夾內」。
       GLOBAL_DEDUP = True：比對 downloaded_videos/ 底下「所有歷次執行」
         下載過的影片，執行前會先掃描歷史檔案建立雜湊索引，檔案數量多
         時會比較耗時。

=== 授權注意事項 ===
- Pixabay：CC0-like，商用/非商用皆可免費使用，不強制標示來源（但建議標示）。
- Pexels：免費使用，官方建議在展示時標明「Photos/Videos provided by Pexels」。
- 兩者皆禁止把下載連結直接做成競品服務;僅供研究/個人專案下載使用沒有問題。
- 學術論文使用建議仍在附錄註明素材來源與授權條款，以求嚴謹。
"""

import os
import csv
import time
import hashlib
import requests
from datetime import datetime

# ------------------------------------------------------------------
# 設定區
# ------------------------------------------------------------------

# 直接寫在這裡的 key 優先使用；若留空字串，則退回讀取環境變數。
# 注意：這個檔案若要上傳 GitHub 或分享給別人，記得先把 key 移除或換新的，
# 避免被別人拿去用掉你的免費額度。
PIXABAY_API_KEY = "53629479-5c062b9cea883c17b394d0d2f" or os.environ.get("PIXABAY_API_KEY", "")
PEXELS_API_KEY = "" or os.environ.get("PEXELS_API_KEY", "")

# 針對貓咪行為辨識研究，鎖定的關鍵字（可自行增減）
BEHAVIOR_KEYWORDS = [
    "cat walking",
    "cat scratching",
    "cat grooming",
    "cat licking",
    "cat licking paw",
    "cat itching",
    "cat scratching post",
    "cat self grooming",
    "cat lying down",
    "cat lying on floor",
    "cat shaking head",
    "cat head shake",
]

RESULTS_PER_KEYWORD = 50   # 每個關鍵字各平台最多抓幾筆
OUTPUT_CSV = "cat_behavior_videos.csv"

DOWNLOAD_VIDEOS = True      # 是否自動下載影片檔案
DOWNLOAD_BASE_DIR = "downloaded_videos"   # 影片存放的根資料夾
MAX_DOWNLOADS = 500         # 保護用：最多下載幾支影片（避免一次抓太多佔滿硬碟/流量）

# False（預設）：只在「這次執行的子資料夾內」去重複。
# True：去重複範圍擴大到 downloaded_videos/ 底下所有歷史執行的資料夾，
#       執行前會先掃描所有舊影片算出內容雜湊值，影片數量多時會比較耗時。
GLOBAL_DEDUP = False

RUN_LOG_FILE = os.path.join(DOWNLOAD_BASE_DIR, "run_log.csv")  # 記錄每次執行時間與結果的總表

# 每次執行都用時間戳記建立獨立子資料夾，例如 downloaded_videos/20260701_153045/
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(DOWNLOAD_BASE_DIR, RUN_TIMESTAMP)


# ------------------------------------------------------------------
# Pixabay
# ------------------------------------------------------------------

def search_pixabay(keyword: str, per_page: int = 20):
    """搜尋 Pixabay 影片，回傳整理過的 list[dict]"""
    if not PIXABAY_API_KEY:
        return []

    url = "https://pixabay.com/api/videos/"
    params = {
        "key": PIXABAY_API_KEY,
        "q": keyword,
        "per_page": max(3, min(per_page, 200)),  # Pixabay 規定 per_page 需介於 3~200
        "category": "animals",
        "safesearch": "true",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"  [Pixabay 錯誤] 關鍵字「{keyword}」查詢失敗：{e}")
        return []

    rows = []
    for hit in data.get("hits", []):
        # 取最大解析度的影片檔連結（large > medium > small > tiny）
        video_files = hit.get("videos", {})
        best = video_files.get("large") or video_files.get("medium") or video_files.get("small") or {}
        rows.append({
            "source": "Pixabay",
            "keyword": keyword,
            "id": hit.get("id"),
            "page_url": hit.get("pageURL"),
            "download_url": best.get("url", ""),
            "width": best.get("width", ""),
            "height": best.get("height", ""),
            "duration_sec": hit.get("duration", ""),
            "tags": hit.get("tags", ""),
            "views": hit.get("views", ""),
            "downloads": hit.get("downloads", ""),
            "license": "Pixabay Content License (CC0-like, free for commercial/non-commercial use)",
        })
    return rows


# ------------------------------------------------------------------
# Pexels
# ------------------------------------------------------------------

def search_pexels(keyword: str, per_page: int = 20):
    """搜尋 Pexels 影片，回傳整理過的 list[dict]"""
    if not PEXELS_API_KEY:
        return []

    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": keyword, "per_page": min(per_page, 80)}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"  [Pexels 錯誤] 關鍵字「{keyword}」查詢失敗：{e}")
        return []

    rows = []
    for video in data.get("videos", []):
        # video_files 依畫質排序不一定，取檔案面積最大的當作最高解析度
        files = video.get("video_files", [])
        best = max(files, key=lambda f: (f.get("width") or 0) * (f.get("height") or 0), default={})
        rows.append({
            "source": "Pexels",
            "keyword": keyword,
            "id": video.get("id"),
            "page_url": video.get("url"),
            "download_url": best.get("link", ""),
            "width": best.get("width", ""),
            "height": best.get("height", ""),
            "duration_sec": video.get("duration", ""),
            "tags": "",  # Pexels API 不回傳 tags
            "views": "",
            "downloads": "",
            "license": "Pexels License (free to use, attribution appreciated)",
        })
    return rows


# ------------------------------------------------------------------
# 下載影片
# ------------------------------------------------------------------

def compute_file_hash(filepath: str, chunk_size: int = 1024 * 256) -> str:
    """計算檔案內容的 MD5 雜湊值"""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def build_existing_hash_index(base_dir: str) -> set:
    """
    掃描 base_dir（downloaded_videos/）底下所有子資料夾中的 .mp4 檔案，
    計算內容雜湊值並建立索引，供 GLOBAL_DEDUP 全域去重複使用。
    """
    hashes = set()
    if not os.path.isdir(base_dir):
        return hashes

    for root, _, files in os.walk(base_dir):
        for fname in files:
            if not fname.lower().endswith(".mp4"):
                continue
            fpath = os.path.join(root, fname)
            try:
                hashes.add(compute_file_hash(fpath))
            except OSError as e:
                print(f"    [警告] 無法讀取 {fpath}：{e}")
    return hashes


def sanitize_filename(text: str) -> str:
    """把關鍵字轉成安全的檔名片段（去掉空白與特殊字元）"""
    return "".join(c if c.isalnum() else "_" for c in text).strip("_")


def download_video(row: dict, dest_dir: str, seen_hashes: set) -> str:
    """
    依 row 裡的 download_url 下載影片到 dest_dir。
    下載完成後計算 MD5，若與同一次執行中已下載過的檔案內容重複，則刪除該檔案。

    回傳狀態字串："success" / "duplicate" / "failed" / "skipped"
    """
    url = row.get("download_url", "")
    if not url:
        return "skipped"

    filename = f"{row['source']}_{sanitize_filename(row['keyword'])}_{row['id']}.mp4"
    filepath = os.path.join(dest_dir, filename)

    if os.path.exists(filepath):
        print(f"    檔名已存在，略過：{filename}")
        return "skipped"

    md5 = hashlib.md5()
    try:
        with requests.get(url, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
                        md5.update(chunk)
    except requests.RequestException as e:
        print(f"    [下載失敗] {filename}：{e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return "failed"

    file_hash = md5.hexdigest()
    if file_hash in seen_hashes:
        os.remove(filepath)
        print(f"    內容重複（與本次已下載影片相同），已刪除：{filename}")
        return "duplicate"

    seen_hashes.add(file_hash)
    print(f"    下載完成：{filename}")
    return "success"


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

def append_run_log(log_path: str, record: dict, fieldnames: list):
    """把這次執行的摘要 append 進總表 run_log.csv（若檔案不存在則先寫入表頭）"""
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def main():
    run_start = datetime.now()
    print(f"=== 執行時間：{run_start.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    if not PIXABAY_API_KEY and not PEXELS_API_KEY:
        print("錯誤：請先設定 PIXABAY_API_KEY 與/或 PEXELS_API_KEY 環境變數。")
        print("參考檔案開頭的『使用前準備』說明申請免費 API key。")
        return

    all_rows = []
    for kw in BEHAVIOR_KEYWORDS:
        print(f"搜尋關鍵字：{kw}")

        if PIXABAY_API_KEY:
            pix_rows = search_pixabay(kw, RESULTS_PER_KEYWORD)
            print(f"  Pixabay：找到 {len(pix_rows)} 筆")
            all_rows.extend(pix_rows)
            time.sleep(0.5)  # 避免超過 rate limit（100 req / 60s）

        if PEXELS_API_KEY:
            pex_rows = search_pexels(kw, RESULTS_PER_KEYWORD)
            print(f"  Pexels：找到 {len(pex_rows)} 筆")
            all_rows.extend(pex_rows)
            time.sleep(0.5)

    if not all_rows:
        print("沒有抓到任何結果，請確認 API key 是否正確。")
        return

    # 同一支影片可能因不同關鍵字被搜到兩次（同 source + 同 id），
    # 這裡先依 (source, id) 去重複，避免重複下載同一支影片。
    unique_rows = []
    seen_ids = set()
    for row in all_rows:
        key = (row["source"], row["id"])
        if key in seen_ids:
            continue
        seen_ids.add(key)
        unique_rows.append(row)

    dup_by_id = len(all_rows) - len(unique_rows)
    if dup_by_id:
        print(f"\n（依影片 ID 去重複：{len(all_rows)} 筆 -> {len(unique_rows)} 筆，"
              f"移除 {dup_by_id} 筆跨關鍵字重複的搜尋結果）")

    fieldnames = [
        "source", "keyword", "id", "page_url", "download_url",
        "width", "height", "duration_sec", "tags", "views", "downloads", "license",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)

    print(f"\n完成！共整理 {len(unique_rows)} 筆不重複影片資訊，已輸出至 {OUTPUT_CSV}")

    log_record = {
        "run_timestamp": run_start.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": RUN_DIR,
        "keywords_count": len(BEHAVIOR_KEYWORDS),
        "total_found": len(all_rows),
        "unique_found": len(unique_rows),
        "global_dedup": GLOBAL_DEDUP,
        "downloaded": 0,
        "duplicate_content_skipped": 0,
        "failed": 0,
    }
    log_fieldnames = list(log_record.keys())

    if not DOWNLOAD_VIDEOS:
        print("（DOWNLOAD_VIDEOS = False，未自動下載影片。可自行依 CSV 的 download_url 下載。）")
        os.makedirs(DOWNLOAD_BASE_DIR, exist_ok=True)
        append_run_log(RUN_LOG_FILE, log_record, log_fieldnames)
        return

    os.makedirs(RUN_DIR, exist_ok=True)
    to_download = unique_rows[:MAX_DOWNLOADS]
    if len(unique_rows) > MAX_DOWNLOADS:
        print(f"\n共 {len(unique_rows)} 筆不重複結果，超過 MAX_DOWNLOADS 上限 "
              f"({MAX_DOWNLOADS})，僅下載前 {MAX_DOWNLOADS} 支。"
              "如需全部下載，請調整程式開頭的 MAX_DOWNLOADS。")

    print(f"\n本次影片將存放於：{RUN_DIR}/")

    seen_hashes = set()
    if GLOBAL_DEDUP:
        print("GLOBAL_DEDUP = True，正在掃描 downloaded_videos/ 底下所有歷史影片建立去重複索引...")
        seen_hashes = build_existing_hash_index(DOWNLOAD_BASE_DIR)
        print(f"  已建立 {len(seen_hashes)} 筆歷史影片內容雜湊值索引"
              "（範圍：所有歷次執行的資料夾）")
    else:
        print("GLOBAL_DEDUP = False，去重複範圍僅限本次執行的資料夾。")
    print("下載時會同步計算內容雜湊值（MD5），偵測到重複內容會自動刪除。\n")

    seen_hashes = set()   # 這次執行中已下載過的影片內容 hash（用來偵測「內容重複」而非只是檔名重複）
    success_count = 0
    duplicate_count = 0
    failed_count = 0

    for i, row in enumerate(to_download, 1):
        print(f"  [{i}/{len(to_download)}] {row['source']} - {row['keyword']} (id={row['id']})")
        status = download_video(row, RUN_DIR, seen_hashes)
        if status == "success":
            success_count += 1
        elif status == "duplicate":
            duplicate_count += 1
        elif status == "failed":
            failed_count += 1
        time.sleep(0.3)  # 避免請求過快

    print(f"\n下載完成：成功 {success_count} 支，內容重複刪除 {duplicate_count} 支，"
          f"失敗 {failed_count} 支，存放於 {RUN_DIR}/")

    log_record["downloaded"] = success_count
    log_record["duplicate_content_skipped"] = duplicate_count
    log_record["failed"] = failed_count
    append_run_log(RUN_LOG_FILE, log_record, log_fieldnames)
    print(f"（本次執行摘要已記錄到 {RUN_LOG_FILE}）")


if __name__ == "__main__":
    main()