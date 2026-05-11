import os
import shutil
import hashlib
# =========================
# 你要修改的路徑設定
# =========================
FOLDER_1 = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\總暫存"
FOLDER_2 = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\walk"
OUTPUT_FOLDER = r"C:\cat_pose\unique_videos"

OLD_FOLDER = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\摳圖影片集"
NEW_FOLDER = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\摳圖影像集2"

# 產生的 txt 檔案
UNIQUE_TXT = os.path.join(OUTPUT_FOLDER, "unique_videos.txt")
EXCLUDED_TXT = os.path.join(OUTPUT_FOLDER, "excluded_duplicates.txt")
DELETED_TXT = os.path.join(NEW_FOLDER, "deleted_duplicates.txt")
KEPT_TXT = os.path.join(NEW_FOLDER, "kept_new_videos.txt")

# 影片副檔名（可自行增加）
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}

def sha256_file(filepath, chunk_size=1024 * 1024):
    """計算檔案 SHA256（適合大檔案）"""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()

def get_all_videos(folder):
    """取得資料夾內所有影片（包含子資料夾）"""
    videos = []
    for root, _, files in os.walk(folder):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in VIDEO_EXTS:
                videos.append(os.path.join(root, file))
    return videos

def safe_copy(src, dst_folder):
    """避免同名檔案覆蓋：如果檔名重複就自動加 _1 _2"""
    base = os.path.basename(src)
    name, ext = os.path.splitext(base)

    dst_path = os.path.join(dst_folder, base)
    count = 1

    while os.path.exists(dst_path):
        dst_path = os.path.join(dst_folder, f"{name}_{count}{ext}")
        count += 1

    shutil.copy2(src, dst_path)
    return dst_path

def deduplicate_mode():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    videos_1 = get_all_videos(FOLDER_1)
    videos_2 = get_all_videos(FOLDER_2)
    all_videos = videos_1 + videos_2

    print(f"📂 Folder1影片數: {len(videos_1)}")
    print(f"📂 Folder2影片數: {len(videos_2)}")
    print(f"🎬 總影片數: {len(all_videos)}")

    hash_map = {}  # sha256 -> kept file path
    unique_files = []
    excluded_files = []

    for idx, video_path in enumerate(all_videos, 1):
        print(f"[{idx}/{len(all_videos)}] 🔍 檢查: {video_path}")

        try:
            file_hash = sha256_file(video_path)

            if file_hash not in hash_map:
                # 第一次出現 -> 保留
                copied_path = safe_copy(video_path, OUTPUT_FOLDER)
                hash_map[file_hash] = copied_path
                unique_files.append(copied_path)  # 修正：記錄實際複製後的路徑
                print(f"   ✅ 保留 -> {copied_path}")
            else:
                # hash 已存在 -> 排除
                excluded_files.append(video_path)
                print(f"   ❌ 重複 -> 已排除 (同於 {hash_map[file_hash]})")

        except Exception as e:
            print(f"   ⚠️ 讀取失敗: {video_path}, error={e}")
            excluded_files.append(video_path)

    # 輸出 txt
    with open(UNIQUE_TXT, "w", encoding="utf-8") as f:
        for p in unique_files:
            f.write(p + "\n")

    with open(EXCLUDED_TXT, "w", encoding="utf-8") as f:
        for p in excluded_files:
            f.write(p + "\n")

    print("\n============================")
    print("✅ 去重完成")
    print(f"保留影片數: {len(unique_files)}")
    print(f"排除影片數: {len(excluded_files)}")
    print(f"輸出資料夾: {OUTPUT_FOLDER}")
    print(f"TXT(保留): {UNIQUE_TXT}")
    print(f"TXT(排除): {EXCLUDED_TXT}")
    print("============================")

def delete_new_duplicates_mode():
    print("📌 開始建立舊影片資料庫 hash...")

    old_videos = get_all_videos(OLD_FOLDER)
    old_hash_set = set()

    for idx, path in enumerate(old_videos, 1):
        print(f"[OLD {idx}/{len(old_videos)}] Hashing: {path}")
        try:
            h = sha256_file(path)
            old_hash_set.add(h)
        except Exception as e:
            print(f"⚠️ 無法讀取舊影片: {path}, error={e}")

    print(f"\n✅ 舊資料庫影片數: {len(old_videos)}")
    print(f"✅ 舊資料庫 unique hash 數: {len(old_hash_set)}")

    print("\n📌 開始檢查新影片...")

    new_videos = get_all_videos(NEW_FOLDER)

    deleted_list = []
    kept_list = []

    for idx, path in enumerate(new_videos, 1):
        print(f"[NEW {idx}/{len(new_videos)}] Checking: {path}")

        try:
            h = sha256_file(path)

            if h in old_hash_set:
                # 重複 -> 刪除
                print(f"   🔥 Duplicate of existing hash: {h}")
                os.remove(path)
                deleted_list.append(path)
                print(f"   ❌ 重複已刪除: {path}")
            else:
                kept_list.append(path)
                print(f"   ✅ 保留: {path}")

        except Exception as e:
            print(f"⚠️ 無法處理新影片: {path}, error={e}")

    # 輸出 txt 報告
    with open(DELETED_TXT, "w", encoding="utf-8") as f:
        for p in deleted_list:
            f.write(p + "\n")

    with open(KEPT_TXT, "w", encoding="utf-8") as f:
        for p in kept_list:
            f.write(p + "\n")

    print("\n============================")
    print("🎬 新影片檢查完成")
    print(f"新影片總數: {len(new_videos)}")
    print(f"刪除重複: {len(deleted_list)}")
    print(f"保留新影片: {len(kept_list)}")
    print(f"📄 刪除清單: {DELETED_TXT}")
    print(f"📄 保留清單: {KEPT_TXT}")
    print("============================")

if __name__ == "__main__":
    print("請選擇模式：")
    print("1. deduplicate (合併兩資料夾並去重)")
    print("2. delete_new_duplicates (刪除新資料夾重複影片)")
    mode = input("請輸入模式 (1 或 2): ").strip()
    if mode == "1":
        deduplicate_mode()
    elif mode == "2":
        delete_new_duplicates_mode()
    else:
        print("無效的模式選擇。請輸入 1 或 2。")