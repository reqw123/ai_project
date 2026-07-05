"""
影片去重腳本：掃描資料夾（含子資料夾）下的所有影片，以 MD5 雜湊判斷「完全相同檔案」。

判定標準：整個檔案位元組完全一致才算重複（跟 dedup_images.py 同樣邏輯）。
編碼/解析度/容器不同的相同內容影片不會被抓到，只抓「同一支影片重複下載/複製」的情況。

流程：
  1. 掃描所有來源資料夾，蒐集影片檔案。
  2. 先依檔案大小分組（大小不同就不可能是同一支影片，避免對明顯不同的大檔案
     做昂貴的全檔 MD5 雜湊——影片檔案通常很大，這一步能省下大量時間）。
  3. 只對「大小相同」的分組計算 MD5，找出真正逐位元組相同的重複檔案。
  4. 在終端列出所有重複影片名單（同一組中最早掃描到的視為正本），
     停下來詢問是否授權刪除；輸入 y 才會實際刪除，其餘任何輸入都會取消，
     不會刪除任何檔案。正本永遠不會被刪除。

用法：
  python dedup_videos.py <資料夾1> [資料夾2] ...
  未帶命令列參數時，改用下方 SOURCE_FOLDERS 的預設值。
"""

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

# ===== 設定區 =====
SOURCE_FOLDERS = [
    # 未從命令列帶入資料夾路徑時，預設掃描這裡列出的資料夾，例如：
    r"C:\Users\homec\Downloads\istock",
]

SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".ts", ".webm", ".3gp"}

HASH_CHUNK_SIZE = 1024 * 1024  # 1MB；影片檔案大，用大一點的區塊減少 read() 次數
# =================


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_folder(folder: Path) -> list[Path]:
    return sorted(
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    )


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"【{title}】")
    print("=" * 60)


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


def scan_sources(folders: list) -> tuple[list[Path], int]:
    print_header("掃描來源資料夾")
    all_files: list[Path] = []
    for src in folders:
        p = Path(src)
        if not p.exists():
            print(f"  ⚠ 資料夾不存在，略過: {src}")
            continue
        files = scan_folder(p)
        print(f"  {p}  {len(files):5d} 支")
        all_files.extend(files)
    print(f"\n  合計來源影片：{len(all_files)} 支")
    return all_files, len(all_files)


def find_duplicates(files: list[Path]) -> list[tuple[Path, Path]]:
    """回傳 [(重複檔, 正本檔), ...]；正本 = 同一組中最早掃描到的檔案。"""
    print_header("比對重複（先依檔案大小分組，再對同大小的檔案算 MD5）")

    by_size: dict[int, list[Path]] = defaultdict(list)
    for f in files:
        try:
            by_size[f.stat().st_size].append(f)
        except OSError as e:
            print(f"  ⚠ 無法讀取檔案大小 {f}：{e}")

    candidate_groups = {size: fs for size, fs in by_size.items() if len(fs) > 1}
    candidate_count = sum(len(fs) for fs in candidate_groups.values())
    print(
        f"  檔案大小相同、需要進一步雜湊比對的候選：{candidate_count} 支"
        f"（其餘 {len(files) - candidate_count} 支檔案大小獨一無二，直接排除）"
    )

    seen_hash: dict[tuple, Path] = {}
    duplicates: list[tuple[Path, Path]] = []
    for size, group in candidate_groups.items():
        for f in group:
            try:
                h = md5_of_file(f)
            except OSError as e:
                print(f"  ⚠ 無法讀取 {f}：{e}")
                continue
            key = (size, h)
            if key in seen_hash:
                duplicates.append((f, seen_hash[key]))
            else:
                seen_hash[key] = f

    return duplicates


def confirm_and_delete(duplicates: list[tuple[Path, Path]]) -> None:
    if not duplicates:
        print_header("結果")
        print("  沒有發現重複影片。")
        return

    print_header(f"發現 {len(duplicates)} 支重複影片")
    for dup, orig in duplicates:
        try:
            size_str = human_size(dup.stat().st_size)
        except OSError:
            size_str = "?"
        print(f"  重複: {dup}  ({size_str})")
        print(f"    ←→ 正本: {orig}")

    print("\n以上檔案將被刪除，正本不受影響。")
    answer = input("確定要刪除以上重複影片嗎？輸入 y 確認，其他任意輸入取消：").strip().lower()
    if answer != "y":
        print("已取消，未刪除任何檔案。")
        return

    print_header("刪除中")
    deleted = 0
    for dup, _ in duplicates:
        try:
            dup.unlink()
            deleted += 1
            print(f"  已刪除: {dup}")
        except OSError as e:
            print(f"  ⚠ 刪除失敗 {dup}：{e}")

    print_header("完成")
    print(f"  已刪除 {deleted}/{len(duplicates)} 支重複影片。")


def main():
    folders = sys.argv[1:] if len(sys.argv) > 1 else SOURCE_FOLDERS
    if not folders:
        print("❌ 未指定資料夾。用法：python dedup_videos.py <資料夾1> [資料夾2] ...")
        print("   或直接編輯本檔案頂部的 SOURCE_FOLDERS 設定預設資料夾。")
        return

    files, total = scan_sources(folders)
    if total == 0:
        print("\n❌ 無可處理的影片，請確認路徑與副檔名。")
        return

    duplicates = find_duplicates(files)
    confirm_and_delete(duplicates)


if __name__ == "__main__":
    main()
