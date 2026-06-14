"""
圖片去重腳本：掃描多個來源資料夾，依 MD5 雜湊判斷重複影像。

模式 1（MODE=1）：複製不重複圖片到新資料夾，來源資料夾不異動。
模式 2（MODE=2）：就地刪除重複圖片（以先出現的為正本，後出現的刪除），不建立新資料夾。
"""

import hashlib
import shutil
from pathlib import Path
from collections import defaultdict, Counter

# ===== 設定區 =====
MODE = 2          # 1 = 複製去重到新資料夾；2 = 就地刪除重複檔案
DRY_RUN = False   # True = 僅預覽，不實際執行複製 / 刪除

_BASE = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集"
SOURCE_FOLDERS = [
    rf"{_BASE}\freepik",
    *[rf"{_BASE}\freepik{i}" for i in range(2, 42)],
]

# 僅 MODE=1 使用
OUTPUT_FOLDER = rf"{_BASE}\freepik_dedup"

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
# =================


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
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


# ── 掃描各資料夾 ──────────────────────────────────────────────

def scan_sources() -> tuple[dict[str, list[Path]], int]:
    print_header("掃描來源資料夾")
    folder_files: dict[str, list[Path]] = {}
    total = 0
    for src in SOURCE_FOLDERS:
        p = Path(src)
        if not p.exists():
            print(f"  ⚠ 資料夾不存在，略過: {src}")
            continue
        files = scan_folder(p)
        folder_files[src] = files
        print(f"  {p.name:30s}  {len(files):5d} 張  ({src})")
        total += len(files)
    print(f"\n  合計來源圖片：{total} 張")
    return folder_files, total


# ── MD5 去重（共用）────────────────────────────────────────────

def build_dedup_sets(
    folder_files: dict[str, list[Path]],
) -> tuple[list[Path], list[tuple[Path, Path]]]:
    """
    回傳 (unique_files, duplicate_files)
    duplicate_files 中每項為 (重複檔路徑, 正本路徑)
    """
    print_header("執行去重（MD5）")
    seen_md5: dict[str, Path] = {}
    unique_files: list[Path] = []
    duplicate_files: list[tuple[Path, Path]] = []

    for _, files in folder_files.items():
        for f in files:
            try:
                h = md5_of_file(f)
            except Exception as e:
                print(f"  ⚠ 無法讀取 {f}：{e}")
                continue
            if h in seen_md5:
                duplicate_files.append((f, seen_md5[h]))
            else:
                seen_md5[h] = f
                unique_files.append(f)

    print(f"  唯一圖片：{len(unique_files)} 張")
    print(f"  重複圖片：{len(duplicate_files)} 張")

    if duplicate_files:
        print("\n  重複清單（最多顯示 20 筆）：")
        for dup, orig in duplicate_files[:20]:
            print(f"    重複: {dup.name}  ←→  正本: {orig.name}")
        if len(duplicate_files) > 20:
            print(f"    ... 另有 {len(duplicate_files) - 20} 筆省略")

    return unique_files, duplicate_files


# ── MODE 1：複製到新資料夾 ──────────────────────────────────────

def run_mode1(unique_files: list[Path], total_input: int, duplicate_count: int) -> None:
    if not DRY_RUN:
        Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    prefix = "[DRY RUN] " if DRY_RUN else ""
    print_header(f"{prefix}複製到輸出資料夾")
    print(f"  目標：{OUTPUT_FOLDER}")

    filename_count: defaultdict[str, int] = defaultdict(int)
    copied = 0

    for f in unique_files:
        stem = f.stem
        suffix = f.suffix.lower()
        count = filename_count[f.name]
        filename_count[f.name] += 1
        dest_name = f.name if count == 0 else f"{stem}_{count}{suffix}"
        dest = Path(OUTPUT_FOLDER) / dest_name

        if DRY_RUN:
            copied += 1
        else:
            try:
                shutil.copy2(f, dest)
                copied += 1
            except Exception as e:
                print(f"  ⚠ 複製失敗 {f}：{e}")

    print_header("結果摘要（MODE 1）")
    print(f"  來源總圖片數：{total_input:6d} 張")
    print(f"  重複圖片數  ：{duplicate_count:6d} 張")
    print(f"  {prefix}輸出圖片數  ：{copied:6d} 張  →  {OUTPUT_FOLDER}")
    print("=" * 60)


# ── MODE 2：就地刪除重複檔案 ────────────────────────────────────

def run_mode2(
    duplicate_files: list[tuple[Path, Path]],
    total_input: int,
) -> None:
    prefix = "[DRY RUN] " if DRY_RUN else ""
    print_header(f"{prefix}就地刪除重複檔案（MODE 2）")

    folder_del_count: Counter = Counter()
    deleted = 0

    for dup, orig in duplicate_files:
        folder_del_count[str(dup.parent)] += 1
        if DRY_RUN:
            print(f"  [預覽] 刪除 {dup}")
            print(f"          正本: {orig}")
            deleted += 1
        else:
            try:
                dup.unlink()
                deleted += 1
            except Exception as e:
                print(f"  ⚠ 刪除失敗 {dup}：{e}")

    print_header("結果摘要（MODE 2）")
    print(f"  來源總圖片數  ：{total_input:6d} 張")
    print(f"  {prefix}已刪除重複數：{deleted:6d} 張")
    print(f"  保留唯一數    ：{total_input - deleted:6d} 張")
    if folder_del_count:
        print("\n  各資料夾刪除數：")
        for folder, cnt in sorted(folder_del_count.items(), key=lambda x: -x[1]):
            print(f"    {Path(folder).name:30s}  {cnt:5d} 張")
    print("=" * 60)


# ── 主流程 ─────────────────────────────────────────────────────

def main():
    if MODE not in (1, 2):
        print(f"❌ 不支援的 MODE={MODE}，請設為 1 或 2。")
        return

    print(f"執行模式：MODE {MODE}  |  DRY_RUN={'是（僅預覽）' if DRY_RUN else '否（實際執行）'}")

    folder_files, total_input = scan_sources()
    if total_input == 0:
        print("\n❌ 無可處理的圖片，請確認路徑。")
        return

    unique_files, duplicate_files = build_dedup_sets(folder_files)

    if MODE == 1:
        run_mode1(unique_files, total_input, len(duplicate_files))
    else:
        run_mode2(duplicate_files, total_input)


if __name__ == "__main__":
    main()
