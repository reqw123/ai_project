"""
ST-GCN 骨架資料集的資料夾結構（2026-09-23 起）：先分 split，再分類別。

    skeletons/
      train/   訓練用                             walk/ lick/ scratch/ shake/ stop/
      val/     只用來早停／挑 checkpoint           walk/ lick/ scratch/ shake/ stop/
      test/    完全不參與訓練與任何決策，只在最後評估  walk/ lick/ scratch/ shake/ stop/

切分就是「檔案放在哪個 split 資料夾」，類別子資料夾只是方便在檔案總管裡一眼看出
每一類有哪些影片（類別本身仍以檔名開頭判斷，例如 lick_32 → lick、stop539 → stop）。
0_dataset_collect.py 新抽的骨架放到影片所在的 split（影片在 模型專用/val/… 就放 val/），
影片不在 split 資料夾裡才放 train/<類別>/；之後要調整切分用
tools/gcn_dataset_manager.py（模式 2）。

相容：直接放在 split 資料夾底下（沒有類別子資料夾）的 JSON 一樣算那個 split；
直接放在 skeletons/ 根目錄的算「尚未分配」，訓練時暫時當 train，並提醒用模式 2 歸位
（歸位時會一併整理進類別子資料夾）。

原始影片資料夾（VIDEO_ROOT，模型專用/）也用同一套 <split>/<類別>/ 排法，影片的 split
永遠跟它的骨架一致，見下方「原始影片」一節。

這個模組只用標準函式庫，0_dataset_collect.py / 0_train_gcn.py / gcn_dataset_manager.py /
eval_*.py 共用，避免各自寫一套找檔案的邏輯而失去同步。
"""
import re
from pathlib import Path

SPLITS = ("train", "val", "test")
NEW_DATA_SPLIT = "train"          # 新抽出來的骨架預設放哪一個
UNASSIGNED = "unassigned"         # 直接放在根目錄、尚未分配的檔案


def class_folder_of(stem) -> str:
    """類別子資料夾名稱＝檔名開頭的英文字母（lick_32 → lick、stop539 → stop、walk1 → walk）。"""
    m = re.match(r"[A-Za-z]+", stem)
    return m.group(0).lower() if m else "_other"


def has_split_layout(root) -> bool:
    """根目錄底下是否已經有 train/val/test 任一子資料夾。"""
    root = Path(root)
    return any((root / s).is_dir() for s in SPLITS)


def _is_skeleton(p):
    return p.suffix == ".json" and not p.name.startswith("_")   # 底線開頭的是報表／名單檔


def iter_skeleton_files(root):
    """回傳所有骨架 JSON（依檔名排序）：根目錄、split 資料夾、split/類別 資料夾。"""
    root = Path(root)
    files = [p for p in root.glob("*.json") if _is_skeleton(p)]
    for s in SPLITS:
        d = root / s
        if d.is_dir():
            files += [p for p in d.glob("*.json") if _is_skeleton(p)]
            files += [p for p in d.glob("*/*.json") if _is_skeleton(p)]
    return sorted(files, key=lambda p: p.stem)


def split_of(path) -> str:
    """骨架檔屬於哪個 split：在 <split>/ 或 <split>/<類別>/ 底下就是那個 split，
    其餘（根目錄）回傳 UNASSIGNED。"""
    p = Path(path)
    if p.parent.name in SPLITS:
        return p.parent.name
    if p.parent.parent.name in SPLITS:
        return p.parent.parent.name
    return UNASSIGNED


def canonical_path(root, split, stem):
    """某筆骨架在某個 split 裡應該放的位置：<root>/<split>/<類別>/<stem>.json。"""
    return Path(root) / split / class_folder_of(stem) / f"{stem}.json"


def find_skeleton(root, stem):
    """找 <stem>.json（split/類別、split、根目錄都找），找不到回傳 None。"""
    root = Path(root)
    cands = []
    for s in SPLITS:
        cands += [canonical_path(root, s, stem), root / s / f"{stem}.json"]
    cands.append(root / f"{stem}.json")
    found = next((p for p in cands if p.exists()), None)
    if found is None:
        # 被拖進別的類別資料夾（例如 lick_10.json 放在 train/walk/）也要找得到，否則重抽骨架時
        # 會當成新檔另外建一份，同一個檔名變兩份、原本的標記也不會被還原
        found = next((p for s in SPLITS for p in sorted((root / s).glob(f"*/{stem}.json"))), None)
    return found


def skeleton_path_for(root, stem, split=None):
    """寫入骨架時用的路徑：已存在就覆寫原位置（保留它原本的 split），
    新檔放 <split>/<類別>/；split 沒給（例如影片不在 VIDEO_ROOT 的 split 資料夾裡）
    就放 NEW_DATA_SPLIT。"""
    existing = find_skeleton(root, stem)
    if existing is not None:
        return existing
    p = canonical_path(root, split if split in SPLITS else NEW_DATA_SPLIT, stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ── 原始影片（2026-09-24 起跟骨架同一套切分）────────────────────────────────
# 骨架對應的原始影片放在 VIDEO_ROOT/<split>/<類別>/，跟 skeletons/ 同樣先分 split
# 再分類別；影片所在的 split 永遠跟它的骨架一致（gcn_dataset_manager 搬骨架時
# 影片跟著搬，見 move_video_with_skeleton）。骨架 JSON 的 video_metadata.video_path
# 永遠指向影片目前實際的位置。新影片放進哪個 VIDEO_ROOT/<split>/<類別>/，抽出的骨架就在
# 同一個 split（見 split_of_video）。
# 相容：舊排法 VIDEO_ROOT/<類別>/ 還存在的話一樣會被掃到。
VIDEO_ROOT = Path(r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\1_貓咪姿勢影片分類\模型專用")
VIDEO_CLASSES = ("walk", "lick", "scratch", "shake", "stop")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg", ".webm"}


def video_class_folders(root=None, classes=VIDEO_CLASSES):
    """所有存在的影片類別資料夾（依類別、再依 train/val/test 排序），每個資料夾的
    上一層名稱就是類別。給「逐一走訪行為資料夾」的工具取代原本寫死的
    模型專用\\walk、模型專用\\lick…清單。傳單一類別名稱（字串）只回傳該類別的資料夾。"""
    root = Path(root) if root else VIDEO_ROOT
    if isinstance(classes, str):
        classes = (classes,)
    folders = []
    for c in classes:
        folders += [str(d) for d in (root / s / c for s in SPLITS) if d.is_dir()]
        legacy = root / c   # 舊排法：裡面還有影片才列入（搬完後留下的空殼／筆記檔不算）
        if legacy.is_dir() and any(p.suffix.lower() in VIDEO_EXTS for p in legacy.iterdir()):
            folders.append(str(legacy))
    return folders


def split_of_video(video_path, root=None):
    """影片放在 <root>/<split>/... 底下就回傳那個 split；舊排法（<root>/<類別>/）或
    不在 root 底下的影片回傳 None。0_dataset_collect.py 抽新骨架時用它決定骨架放哪個
    split，讓「把新影片丟進 val/ 或 test/」就等於決定了它的切分。"""
    root = Path(root) if root else VIDEO_ROOT
    try:
        rel = Path(video_path).relative_to(root).parts
    except ValueError:
        return None
    return rel[0] if len(rel) >= 2 and rel[0] in SPLITS else None


def find_video(name, root=None):
    """依檔名在 <root>/<split>/<類別>/（或舊排法 <root>/<類別>/）找影片，找不到或同名不只一支
    回傳 None。影片會跟著骨架換 split，工具要指定某支影片時用它，不要寫死 split 路徑。"""
    root = Path(root) if root else VIDEO_ROOT
    hits = list(root.glob(f"*/*/{name}")) + list(root.glob(f"*/{name}"))
    return hits[0] if len(hits) == 1 else None


def video_path_in_split(video_path, split, root=None):
    """影片在某個 split 裡應該放的位置：<root>/<split>/<類別>/<檔名>，類別取影片目前
    所在的資料夾名稱（跟 0_dataset_collect.py 判斷類別的方式一致）。未分配的骨架
    （根目錄）對應 train。"""
    p = Path(video_path)
    split = NEW_DATA_SPLIT if split == UNASSIGNED else split
    return (Path(root) if root else VIDEO_ROOT) / split / p.parent.name / p.name


def read_video_path(json_path):
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data.get("video_metadata") or {}).get("video_path") or data.get("video_path") or ""


def rewrite_video_path(json_path, old, new):
    """只替換 JSON 文字裡的 video_path 字串，其餘內容一個位元組都不動（不重新序列化
    整份骨架，避免數字格式、縮排等被改寫）。"""
    import json
    text = Path(json_path).read_text(encoding="utf-8")
    old_s = json.dumps(str(old), ensure_ascii=False)
    new_s = json.dumps(str(new), ensure_ascii=False)
    if text.count(old_s) != 1:
        raise RuntimeError(f"{Path(json_path).name}：找不到唯一的 video_path 字串，未改寫")
    tmp = Path(json_path).with_suffix(".json.tmp")
    tmp.write_text(text.replace(old_s, new_s), encoding="utf-8")
    tmp.replace(json_path)


def move_video_with_skeleton(json_path, split, root=None):
    """把骨架對應的影片搬到 <root>/<split>/<類別>/，並把骨架 JSON 的 video_path 改成新
    位置。回傳 (舊路徑, 新路徑)；不需要搬時回傳 None。

    不搬（回傳 None）的情況：JSON 沒記錄影片、影片不存在、影片不在 root 底下
    （例如來自別的資料夾的骨架）、已經在正確位置。目的地已有同名檔案時丟出例外，
    不覆蓋。先搬影片、再改 JSON；改 JSON 失敗會把影片搬回原位，確保 video_path
    永遠指向影片實際所在的位置。"""
    import shutil
    old = read_video_path(json_path)
    if not old:
        return None
    src = Path(old)
    root = Path(root) if root else VIDEO_ROOT
    if not src.exists():
        return None
    try:
        src.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    dst = video_path_in_split(src, split, root)
    if src.resolve() == dst.resolve():
        return None
    if dst.exists():
        raise FileExistsError(f"目的地已有同名影片，未搬移：{dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    try:
        rewrite_video_path(json_path, old, dst)
    except Exception:
        shutil.move(str(dst), str(src))
        raise
    return str(src), str(dst)


# ── 標記狀態（2026-09-24 起每筆訓練骨架都必須有人工標記）──────────────────────
# 0_dataset_collect.py 模式 2 標記時寫入 annotation_review；舊檔只要有
# action_intervals 就視為已標記。人工清空（manual_empty）是刻意決定「這支沒有
# 有效片段」，算已處理；除此之外沒有區間的檔案一律算尚未標記。
REVIEW_METHODS = ("full_clip", "manual_intervals", "manual_empty")


def annotation_state(data) -> str:
    """回傳 full_clip / manual_intervals / manual_empty / legacy_intervals / pending。"""
    review = data.get("annotation_review") or {}
    if isinstance(review, dict) and review.get("status") == "reviewed":
        method = review.get("method")
        if method in ("full_clip", "manual_intervals") and data.get("action_intervals"):
            return method
        if method == "manual_empty":
            return method
    return "legacy_intervals" if data.get("action_intervals") else "pending"


def find_unmarked_skeletons(root):
    """找出還沒有標記區段的骨架檔，回傳 [(path, 原因), ...]；讀不到的檔案也列入。"""
    import json
    unmarked = []
    for path in iter_skeleton_files(root):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            unmarked.append((path, f"無法讀取：{exc}"))
            continue
        if annotation_state(data) == "pending":
            unmarked.append((path, "沒有標記區段"))
    return unmarked


def format_unmarked(unmarked, limit=20) -> str:
    """把 find_unmarked_skeletons 的結果整理成依類別分組的多行文字。"""
    from collections import Counter
    counts = Counter(class_folder_of(p.stem) for p, _ in unmarked)
    lines = ["  各類別：" + "  ".join(f"{c} {n}" for c, n in sorted(counts.items()))]
    for p, reason in unmarked[:limit]:
        lines.append(f"    {split_of(p)}/{p.name}：{reason}")
    if len(unmarked) > limit:
        lines.append(f"    ……其餘 {len(unmarked) - limit} 筆省略")
    return "\n".join(lines)
