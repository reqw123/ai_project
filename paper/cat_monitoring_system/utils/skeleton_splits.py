"""
ST-GCN 骨架資料集的資料夾結構（2026-09-23 起）：先分 split，再分類別。

    skeletons/
      train/   訓練用                             walk/ lick/ scratch/ shake/ stop/
      val/     只用來早停／挑 checkpoint           walk/ lick/ scratch/ shake/ stop/
      test/    完全不參與訓練與任何決策，只在最後評估  walk/ lick/ scratch/ shake/ stop/

切分就是「檔案放在哪個 split 資料夾」，類別子資料夾只是方便在檔案總管裡一眼看出
每一類有哪些影片（類別本身仍以檔名開頭判斷，例如 lick_32 → lick、stop539 → stop）。
0_dataset_collect.py 新抽的骨架預設放 train/<類別>/；要調整切分用
tools/gcn_dataset_manager.py（模式 2）。

相容：直接放在 split 資料夾底下（沒有類別子資料夾）的 JSON 一樣算那個 split；
直接放在 skeletons/ 根目錄的算「尚未分配」，訓練時暫時當 train，並提醒用模式 2 歸位
（歸位時會一併整理進類別子資料夾）。

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
    return next((p for p in cands if p.exists()), None)


def skeleton_path_for(root, stem):
    """寫入骨架時用的路徑：已存在就覆寫原位置（保留它原本的 split），
    新檔一律放 NEW_DATA_SPLIT/<類別>/。"""
    existing = find_skeleton(root, stem)
    if existing is not None:
        return existing
    p = canonical_path(root, NEW_DATA_SPLIT, stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
