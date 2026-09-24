"""
ST-GCN 骨架資料集管理（合併原 eval_window_counts.py 與 make_splits.py）。

  模式 1：統計訓練視窗數（純讀取，含各類別佔比）
      直接重用 0_train_gcn.py 的 CatSkeletonDataset——訓練當下真正會跑到的同一份
      切窗／過濾邏輯（STRICT_WINDOW_FILTER／MAX_NO_DETECT_FRAMES／SEQUENCE_LENGTH／
      WINDOW_STRIDE 都讀同一份 stgcn_config.yaml），列出各類別在 train/val/test
      各有幾支影片、能切出幾個視窗，以及佔同一切分五類總和的百分比。不修改任何檔案。

  模式 2：train / val / test 切分管理視窗（skeletons/train、val、test）
      切分就是「檔案放在哪個子資料夾」，0_train_gcn.py 直接依資料夾切。
      0_dataset_collect.py 新抽的骨架放到影片所在的 split（影片在 模型專用/val/… 就進 val/）。
      視窗分成 train / val / test 三欄（可捲動、可依類別篩選或搜尋、可排序），
      選取限同一個 split：Ctrl＋左鍵加選、Shift＋左鍵選一段、按住左鍵拖曳選一段（Ctrl＋拖＝追加），
      選好後批次移到另一個切分，可「復原上一步」。另有按鈕：
        ・套用規則／歸位新檔：把根目錄未分配的檔案歸位、把固定的影片放回固定位置、讓重複檔同邊
        ・整份重新切分：依類別分層重抽（固定的不動；會大量搬檔，先顯示清單再確認）

      規則：
        1. 以「影片群組」為單位：同一段內容以不同檔名重複存在的骨架（關鍵點序列幾乎相同），
           以及 X 與 X_hq（同一支影片的不同畫質版本），一律視為同一組、放在同一邊，避免洩漏。
        2. 固定：把某筆資料強制鎖在一個 split（train／val／test 各一份固定名單），
           它就只會待在那裡——手動移動會被擋下、「套用規則」會把它放回去、整份重切也不會動它。
           要移動請先「解除固定」。名單存在 skeletons/_split_rules.json；在視窗裡多選後按
           「固定在目前位置」「固定到 train/val/test」「解除固定」批次設定（右鍵選單也有）。
        3. 整份重切時依類別分層，照 RATIOS 分配（固定的先放好，其餘補足比例）。
        4. 平常「只增不改」：已在子資料夾的檔案不換邊（包括手動拖過的）。
      每次實際搬動都附加記錄到 skeletons/_split_moves_log.csv。
      原始影片（模型專用/<split>/<類別>/）跟著骨架搬到同一個 split，骨架 JSON 的
      video_path 同步更新；影片搬動記錄在 模型專用/_video_moves_log.csv。

  模式 3：影片資料夾同步
      讓 模型專用/ 的影片跟骨架切分一致：影片搬到骨架所在的 split（模型專用/<split>/<類別>/），
      骨架 JSON 的 video_path 改成新位置；JSON 記的路徑找不到影片時，依檔名在 模型專用/
      底下找回來（唯一一支才採用）。舊排法（模型專用/<類別>/）裡沒有骨架的新影片搬到 train/。
      先印預覽，輸入 ok 才搬。平常不用跑——模式 2 搬骨架時影片已經跟著搬；只有在檔案
      總管手動拖過骨架或影片、或模式 2 提示影片沒跟著搬時才需要。

  模式 4：還原最近一次模式 3 同步（影片搬回原位、JSON 路徑改回去）。

  ══════════════════════════════════════════════════════════════════════
  在檔案總管手動拖檔的錯誤示範
  ══════════════════════════════════════════════════════════════════════
  0_train_gcn.py 訓練前會檢查（check_split_consistency），①～⑦ 都會擋下訓練；本工具一啟動
  也會先檢查，用同一個警告框一次列出全部問題。
  切分請一律在模式 2 視窗裡調整：骨架和影片會一起搬、JSON 的影片路徑會更新，
  也會遵守固定名單與重複組。

  ── ① 只拖骨架 ──────────────────────────────────────── 擋下訓練
      做法：skeletons/train/walk/walk5.json 拖到 skeletons/val/walk/，影片留在 train
      訊息：[骨架和影片放在不同的 split] walk5：骨架在 val，影片在 train
      修正：以骨架為準 → 模式 3（影片搬過去）
            以影片為準 → 模式 2 把骨架移回 train

  ── ② 只拖影片 ──────────────────────────────────────── 擋下訓練
      做法：模型專用/train/walk/walk5.mp4 拖到 模型專用/test/walk/
      訊息：[骨架和影片放在不同的 split] walk5：骨架在 train，影片在 test
      說明：切分看的是骨架，只拖影片等於沒換 split；跑模式 3 會把影片搬回 train

  ── ③ 拖了被固定的資料 ──────────────────────────────── 擋下訓練
      做法：lick_8 固定在 train，骨架＋影片一起拖到 test
      訊息：[固定的資料不在它被固定的 split] lick_8：現在在 test，固定在 train
      修正：模式 2 按「套用規則／歸位新檔」搬回去；真的要換就先「解除固定」

  ── ④ 同一支影片存了兩份、放在不同 split ──────────────── 擋下訓練
      做法：shake_32（val）的影片又以 shake_val1 的名字存了一份，抽成骨架後在 train
      訊息：[檔名不同、骨架內容卻完全相同] shake_32（val） = shake_val1（train）
      說明：比對的是骨架座標（開頭 60 幀平均差不到 1 像素），跟檔名流水號無關——
            shake_1、shake_2 只要是不同影片就不會被判成相同。2026-09-23 刪過 9 組、
            09-24 又抓到 shake_32／shake_val1，實際上發生過
      修正：留一份就好，刪掉另一支；兩份都要留就用模式 2「套用規則」放到同一邊

  ── ④' 原版和 _hq 高畫質版放在不同 split ──────────────── 擋下訓練
      做法：lick_8 在 train，lick_8_hq 在 test
      訊息：[原版和 _hq 高畫質版放在不同 split] lick_8（train） / lick_8_hq（test）
      說明：唯一看檔名的規則；兩個版本都保留沒關係，但要在同一個 split
      修正：模式 2「套用規則／歸位新檔」

  ── ⑤ 同一個檔案被複製成多份 ─────────────────────────── 擋下訓練
      做法：Ctrl+C／Ctrl+V 或按住 Ctrl 拖曳，train 和 val 都有 shake_32.json
      訊息：[同一個檔案被複製成多份（檔名相同）] shake_32：train/shake / val/shake
      修正：刪掉多出來的那份。模式 2 視窗在修好之前也會拒絕開啟

  ── ⑥ 放錯類別資料夾 ──────────────────────────────────── 擋下訓練
      做法：lick_10.json 拖到 skeletons/train/walk/（或 lick_10.mp4 拖到 模型專用/train/walk/）
      訊息：[放在別的類別資料夾] lick_10：骨架放在 walk/，應該在 lick/
      說明：檔名開頭就是類別，放進別的類別資料夾就是錯位。影片放錯最危險：抽骨架時用
            影片所在的資料夾當初始標籤，lick 會被標成 walk
      修正：骨架 → 模式 2「套用規則／歸位新檔」整理回 train/lick/
            影片 → 在檔案總管搬回 模型專用/train/lick/，再跑模式 3 更新路徑

  ── ⑦ 骨架記錄的影片檔名跟骨架檔名不同 ─────────────────── 擋下訓練
      做法：把 walk33.json 複製一份改名成 walk777.json（平常很少發生，測試防呆時出現過）
      訊息：[骨架裡記錄的影片檔名跟骨架檔名不同] walk777.json 記錄的影片是 walk33.mp4
      說明：內容跟 walk33 一樣，所以也會同時出現 ④；模式 3 遇到這種骨架不會搬影片
      修正：刪掉 walk777.json

  ── 合法的手動調整 ─────────────────────────────────────── 不擋
      做法：骨架和影片一起拖到同一個 split（資料夾就是切分）
      之後：跑一次模式 3 更新 JSON 裡的影片路徑，否則標記工具打不開影片

      為什麼要固定切分：原本每次訓練都用 train_test_split 重抽 val，資料一增減整份 val
      就重新洗牌，不同 run 無法比較；同一份 val 又拿來選 checkpoint 又拿來報成績會高估
      表現。val 只用來早停／挑 checkpoint，test 只在最後評估（不要依 test 誤判回頭改資料）。

用法：
    python gcn_dataset_manager.py                      # 互動選單選模式
    python gcn_dataset_manager.py --mode 1             # 統計視窗數
    python gcn_dataset_manager.py --mode 1 --skeleton_dir <其他骨架資料夾>
    python gcn_dataset_manager.py --mode 2             # 開啟切分管理視窗
    python gcn_dataset_manager.py --mode 2 --dry_run   # （命令列）只印出套用規則會怎麼搬
    python gcn_dataset_manager.py --mode 2 --rebuild   # （命令列）整份重新切分（會搬動檔案）
    python gcn_dataset_manager.py --mode 3 --dry_run   # 預覽影片資料夾同步
    python gcn_dataset_manager.py --mode 3             # 影片資料夾同步（預覽後輸入 ok 才搬）
    python gcn_dataset_manager.py --mode 4             # 還原最近一次影片資料夾同步
"""
import argparse
import contextlib
import csv
import importlib.util
import io
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parent
_TRAIN_SCRIPT = _TOOLS / "0_train_gcn.py"
CMS_DIR = _TOOLS.parent / "cat_monitoring_system"
CONFIG_PATH = CMS_DIR / "stgcn_config.yaml"
sys.path.insert(0, str(CMS_DIR))
from utils.skeleton_splits import (  # noqa: E402
    SPLITS, UNASSIGNED, iter_skeleton_files, split_of, canonical_path, class_folder_of,
    VIDEO_ROOT, VIDEO_CLASSES, VIDEO_EXTS, move_video_with_skeleton, video_path_in_split,
    read_video_path, rewrite_video_path, find_video, split_of_video,
)
from utils.console_alert import alert_box  # noqa: E402

# 原始影片的搬動記錄（模式 2 搬骨架時影片跟著搬、模式 3 同步），放在影片資料夾根目錄
VIDEO_MOVES_LOG = VIDEO_ROOT / "_video_moves_log.csv"


RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}
SEED = 42

# ↓ 下面兩份只是「固定名單檔不存在時」的初始值；實際名單存在 skeletons/_split_rules.json，
#   在模式 2 視窗裡選取影片後按「固定在目前位置／固定到…／解除固定」設定，不用改程式碼。
FORCE_TRAIN = {"lick_8", "lick_123", "lick_25", "lick_32"}   # 固定在 train（使用者指定）
HARD_TEST = {"stop_53", "stop_153"}                          # 固定在 test（run_140～144 反覆被判錯的 stop）

# 重複偵測：兩支同類骨架在開頭重疊段的平均座標差低於此值（像素）即視為同一段內容
DUP_MAX_MEAN_DIFF_PX = 1.0
DUP_COMPARE_FRAMES = 60


# ═══════════════════════════ 模式 1：統計訓練視窗數 ═══════════════════════════
def _load_train_module():
    """把 0_train_gcn.py 當一般模組載入（不執行它的 __main__ 區塊），拿到跟訓練當下
    完全一致的 CatSkeletonDataset 與設定。"""
    spec = importlib.util.spec_from_file_location("_gcn_train_ref", str(_TRAIN_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_window_counts(skeleton_dir=None, feature_mode="xy_conf_v_bone"):
    print("載入 0_train_gcn.py 的 CatSkeletonDataset（跟訓練當下同一份邏輯）...")
    with contextlib.redirect_stdout(io.StringIO()):
        gcn = _load_train_module()

    skeleton_dir = skeleton_dir or gcn.SKELETON_DATA_FOLDER
    if not Path(skeleton_dir).is_dir():
        print(f"❌ 資料夾不存在：{skeleton_dir}")
        return

    print(f"資料夾：{skeleton_dir}")
    print(f"切窗設定：T={gcn.SEQUENCE_LENGTH}  stride={gcn.WINDOW_STRIDE}  "
          f"num_joints={gcn.NUM_JOINTS}  strict_filter={gcn.STRICT_WINDOW_FILTER}  "
          f"max_no_detect={gcn.MAX_NO_DETECT_FRAMES}")

    # CatSkeletonDataset 內部用 tqdm 畫進度條；設定視窗的輸出面板不處理 \r，會擠成
    # 一長行——只換掉這個模組實例裡的 tqdm，不影響 0_train_gcn.py 自己執行時的進度條
    gcn.tqdm = lambda iterable, **_kw: iterable

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ds = gcn.CatSkeletonDataset(
            skeleton_folder=skeleton_dir,
            sequence_length=gcn.SEQUENCE_LENGTH,
            num_joints=gcn.NUM_JOINTS,
            window_stride=gcn.WINDOW_STRIDE,
            feature_mode=feature_mode,
        )
    # 只轉印資料載入診斷（被略過的影片等）；類別分布改用下面分切分的表格
    for line in buf.getvalue().splitlines():
        if line.strip().startswith(("[資料載入診斷]", "⚠")):
            print(line)

    names = [n for n, _ in sorted(gcn.BEHAVIOR_PREFIXES.items(), key=lambda kv: kv[1])]
    cols = list(SPLITS)
    if any(q["split"] == UNASSIGNED for q in ds.sequences):
        cols.append(UNASSIGNED)
    win = Counter((q["label"], q["split"]) for q in ds.sequences)
    vids = defaultdict(set)
    for q in ds.sequences:
        vids[(q["label"], q["split"])].add(q["video_id"])

    # 每格 = 數量（佔該欄五個類別總和的 %）；「合計」欄是 train+val+test 全部加總
    counts = {
        "影片數": {(c, s): len(vids[(c, s)]) for c in range(len(names)) for s in cols},
        "視窗數": {(c, s): win[(c, s)] for c in range(len(names)) for s in cols},
    }
    for title, cnt in counts.items():
        col_tot = {s: sum(cnt[(c, s)] for c in range(len(names))) for s in cols}
        all_tot = sum(col_tot.values())
        print(f"\n【{title}】  數量（佔同一欄五類總和的 %）")
        print(f"  {'類別':<9}" + "".join(f"{s:>16}" for s in cols) + f"{'合計':>16}")
        for c, n in enumerate(names):
            row_tot = sum(cnt[(c, s)] for s in cols)
            cells = [f"{cnt[(c, s)]} ({cnt[(c, s)] / col_tot[s]:.1%})" if col_tot[s] else "0" for s in cols]
            cells.append(f"{row_tot} ({row_tot / all_tot:.1%})" if all_tot else "0")
            print(f"  {n:<9}" + "".join(f"{x:>16}" for x in cells))
        print(f"  {'合計':<9}" + "".join(f"{col_tot[s]:>16}" for s in cols) + f"{all_tot:>16}")

    print("\n說明：")
    print("  ・影片數 %：這一類有幾支影片，佔該切分五類影片總數的比例。")
    print("  ・視窗數 %：這一類切出的訓練視窗（16 幀一段），佔該切分五類視窗總數的比例。")
    print("    模型實際是用「視窗」訓練與評分，所以類別平衡要看視窗數 %，不是影片數 %。")
    print("  ・同一類的影片 % 明顯高於視窗 %（例如 shake）：每支影片標註段短，切出的視窗少；")
    print("    反過來視窗 % 高於影片 %（例如 lick、stop）：每支影片貢獻的視窗多。")
    print("  ・train / val / test 三欄的 % 若差很多，代表三個切分的類別組成不一致，")
    print("    val / test 的分數會偏向占比高的類別（這時看 macro-F1 比 accuracy 公平）。")
    if UNASSIGNED in cols:
        print(f"\n⚠ 有骨架直接放在 {skeleton_dir} 根目錄、尚未分配（訓練時暫當 train），可用模式 2 歸位。")


# ═════════════════ 模式 2：train / val / test 切分（規則與搬檔） ═════════════════
def _load_config():
    try:
        import yaml
    except ImportError:
        raise SystemExit("需要 PyYAML（請用 yolo_new 環境執行）")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _class_of(video_id, prefixes):
    for name in prefixes:
        if video_id.lower().startswith(name):
            return name
    return None


# ── 切分規則（存在 skeletons/_split_rules.json；底線開頭，不會被當成骨架讀取） ──
RULES_FILENAME = "_split_rules.json"
_RULES = {
    # 固定名單：列在哪個 split，那筆資料就只會待在那個 split（要先解除固定才能移走）
    "fixed": {"train": sorted(FORCE_TRAIN), "val": [], "test": sorted(HARD_TEST)},
}
# 舊版規則檔裡的「檔名樣式」（例如 ^lick_.*_hq$）：讀檔時先暫存，load_state() 掃完檔案後
# 展開成一筆一筆的固定名單，之後就不再使用樣式
_LEGACY_PATTERNS = []


def load_rules(root):
    """讀取固定名單檔；不存在時用上面的初始值建立一份。回傳並更新模組層級的 _RULES。
    舊格式（force_train／hard_test、檔名樣式）會自動轉換，樣式交給 _expand_legacy_patterns() 展開。"""
    path = Path(root) / RULES_FILENAME
    _LEGACY_PATTERNS.clear()
    if not path.exists():
        save_rules(root)
        return _RULES
    data = json.loads(path.read_text(encoding="utf-8"))
    if "fixed" in data:
        _RULES["fixed"] = {sp: list(data["fixed"].get(sp, [])) for sp in SPLITS}
    else:
        _RULES["fixed"] = {"train": list(data.get("force_train", [])), "val": [],
                           "test": list(data.get("hard_test", []))}
    _LEGACY_PATTERNS.extend(data.get("train_patterns", []) or data.get("force_train_patterns", []))
    if "fixed" not in data and not _LEGACY_PATTERNS:
        save_rules(root)
    return _RULES


def _expand_legacy_patterns(root, video_ids):
    """把舊版的檔名樣式展開成一筆一筆的「固定 train」，存回新格式（只會發生一次）。"""
    if not _LEGACY_PATTERNS:
        return []
    added = sorted(v for v in video_ids
                   if any(re.match(p, v) for p in _LEGACY_PATTERNS) and v not in _RULES["fixed"]["train"])
    for v in added:
        for sp in SPLITS:
            if v in _RULES["fixed"][sp]:
                _RULES["fixed"][sp].remove(v)
        _RULES["fixed"]["train"].append(v)
    _LEGACY_PATTERNS.clear()
    save_rules(root)
    return added


def save_rules(root):
    path = Path(root) / RULES_FILENAME
    data = {
        "_說明": "ST-GCN 骨架固定名單，由 tools/gcn_dataset_manager.py 模式 2 視窗維護。"
                 "列在 fixed 的哪個 split，那筆資料就只會待在那個 split（要先解除固定才能移走）。",
        "updated": datetime.now().isoformat(timespec="seconds"),
        "fixed": {sp: sorted(set(_RULES["fixed"][sp])) for sp in SPLITS},
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fixed_split_of(video_id):
    """這筆資料被固定在哪個 split；沒有固定回傳 None。"""
    for sp in SPLITS:
        if video_id in _RULES["fixed"][sp]:
            return sp
    return None


def group_fixed_split(state, video_id):
    """整組（重複檔／X_hq）的固定切分：組內任何一支被固定，整組就跟著它。"""
    for m in group_members(state, video_id):
        fs = fixed_split_of(m)
        if fs:
            return fs
    return None


def _is_force_train(video_id):
    return fixed_split_of(video_id) == "train"


def _is_hard_test(video_id):
    return fixed_split_of(video_id) == "test"


def _load_head(path):
    """讀骨架開頭 DUP_COMPARE_FRAMES 幀的 (T,17,2) 座標，沒偵測到的幀填 NaN。"""
    with open(path, encoding="utf-8") as f:
        frames = json.load(f)["frames"][:DUP_COMPARE_FRAMES]
    out = np.full((len(frames), 17, 2), np.nan)
    for t, fr in enumerate(frames):
        kps = fr.get("keypoints", [])
        if len(kps) == 17:
            out[t] = [[k["x"], k["y"]] for k in kps]
    return out


def build_groups(paths, prefixes):
    """paths: {video_id: Path}。回傳 ({video_id: group_id}, 重複配對清單)。
    同組 = 重複內容或 X/X_hq 配對。"""
    ids = sorted(paths)
    parent = {v: v for v in ids}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for v in ids:
        if v.endswith("_hq") and v[:-3] in paths:
            union(v, v[:-3])

    dup_pairs = []
    by_cls = defaultdict(list)
    for v in ids:
        by_cls[_class_of(v, prefixes)].append(v)
    for vids in by_cls.values():
        heads = {v: _load_head(paths[v]) for v in vids}
        for i, a in enumerate(vids):
            for b in vids[i + 1:]:
                A, B = heads[a], heads[b]
                n = min(len(A), len(B))
                if n < 5:
                    continue
                diff = np.abs(A[:n] - B[:n])
                if np.isnan(diff).all():
                    continue
                if np.nanmean(diff) < DUP_MAX_MEAN_DIFF_PX:
                    union(a, b)
                    dup_pairs.append((a, b))
    return {v: find(v) for v in ids}, dup_pairs


def fresh_split(video_ids, groups, prefixes, rng):
    """整份重新切分：依類別分層，以群組為單位分配；固定名單裡的群組先放到各自的切分，
    其餘再依 RATIOS 補足各切分的目標數量。"""
    assign = {}
    members = defaultdict(list)
    for v in video_ids:
        members[groups[v]].append(v)

    def gfix(g):
        return next((fixed_split_of(v) for v in members[g] if fixed_split_of(v)), None)

    by_cls = defaultdict(list)
    for gid in members:
        by_cls[_class_of(gid, prefixes)].append(gid)

    for _, gids in sorted(by_cls.items()):
        gids = sorted(gids)
        forced = {sp: [g for g in gids if gfix(g) == sp] for sp in SPLITS}
        free = [g for g in gids if gfix(g) is None]
        rng.shuffle(free)

        n = len(gids)
        n_test = max(1, round(n * RATIOS["test"])) if n >= 3 else 0
        n_val = max(1, round(n * RATIOS["val"])) if n >= 3 else 0
        need_test = max(0, n_test - len(forced["test"]))
        need_val = max(0, n_val - len(forced["val"]))
        test_g = forced["test"] + free[:need_test]
        val_g = forced["val"] + free[need_test:need_test + need_val]
        train_g = forced["train"] + free[need_test + need_val:]

        for split, gs in (("train", train_g), ("val", val_g), ("test", test_g)):
            for g in gs:
                for v in members[g]:
                    assign[v] = split
    return assign


def load_state(verbose=True):
    """讀取骨架資料夾目前的狀態（檔案位置、類別、重複檔群組），CLI 與 GUI 共用。"""
    cfg = _load_config()
    root = Path(cfg["SKELETON_DATA_FOLDER"])
    prefixes = list(cfg["BEHAVIOR_PREFIXES"].keys())

    load_rules(root)
    files = iter_skeleton_files(root)
    stems = Counter(p.stem for p in files)
    clash = sorted(s for s, n in stems.items() if n > 1)
    if clash:
        raise RuntimeError(f"同一個檔名出現在多個子資料夾，請先手動處理：{clash}")
    paths = {p.stem: p for p in files}
    unknown = sorted(v for v in paths if _class_of(v, prefixes) is None)
    if unknown and verbose:
        print(f"⚠ 以下檔名對不到任何類別前綴，略過：{unknown}")
    for v in unknown:
        paths.pop(v)
    expanded = _expand_legacy_patterns(root, sorted(paths))
    if expanded and verbose:
        print(f"  已把舊的檔名樣式展開成 {len(expanded)} 筆「固定 train」：{expanded}")
    groups, dup_pairs = build_groups(paths, prefixes)
    return {
        "root": root, "prefixes": prefixes, "paths": paths,
        "current": {v: split_of(p) for v, p in paths.items()},
        "groups": groups, "dup_pairs": dup_pairs,
    }


def group_members(state, video_id):
    g = state["groups"][video_id]
    return sorted(v for v, gg in state["groups"].items() if gg == g)


def plan_split(state, rebuild=False):
    """依規則算出每支影片該在哪個切分，回傳 (assign, moves)；moves = [(id, 從, 到)]。"""
    video_ids = sorted(state["paths"])
    current, groups, prefixes = state["current"], state["groups"], state["prefixes"]
    prev = {v: s for v, s in current.items() if s != UNASSIGNED}
    if rebuild:
        assign = fresh_split(video_ids, groups, prefixes, random.Random(SEED))
    else:
        assign = {v: prev.get(v, "train") for v in video_ids}   # 未分配的新檔一律進 train
        # 同組一定同邊：以組內已分配成員為準（test > val > train，保守避免洩漏）
        members = defaultdict(list)
        for v in video_ids:
            members[groups[v]].append(v)
        rank = {"test": 2, "val": 1, "train": 0}
        for vids in members.values():
            known = [prev[v] for v in vids if v in prev]
            if known:
                target = max(known, key=lambda s: rank[s])
                for v in vids:
                    assign[v] = target
    # 固定名單最後套用（優先於一切）：被固定的影片連同同組一起放到固定的切分
    for v in video_ids:
        fs = fixed_split_of(v)
        if fs and assign[v] != fs:
            for m in group_members(state, v):
                assign[m] = fs
    # 換 split 的，以及 split 沒變但還沒放進類別子資料夾的（例如手動丟進 train/ 的）都要搬
    moves = [(v, current[v], assign[v]) for v in video_ids
             if current[v] != assign[v] or state["paths"][v] != canonical_path(state["root"], assign[v], v)]
    return assign, moves


def _log_video_moves(rows):
    """rows = [(mode, video_id, 舊路徑, 新路徑, 骨架 json)]，附加到 VIDEO_MOVES_LOG。"""
    if not rows:
        return
    new_log = not VIDEO_MOVES_LOG.exists()
    now = datetime.now().isoformat(timespec="seconds")
    with open(VIDEO_MOVES_LOG, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["time", "mode", "video_id", "from", "to", "json"])
        for r in rows:
            w.writerow([now, *r])


def apply_moves(state, moves, mode):
    """實際搬檔，並附加記錄到 skeletons/_split_moves_log.csv；搬完同步更新 state。
    對應的原始影片跟著搬到同一個 split（見 skeleton_splits.move_video_with_skeleton），
    影片搬不動（例如正被播放器開著）時骨架照搬、影片留在原位並記進
    state["video_warnings"]，之後可用模式 3 補同步。"""
    root = state["root"]
    video_rows, video_warnings = [], []
    for s in SPLITS:
        (root / s).mkdir(exist_ok=True)
    log_path = root / "_split_moves_log.csv"
    new_log = not log_path.exists()
    now = datetime.now().isoformat(timespec="seconds")
    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["time", "mode", "video_id", "from", "to"])
        for v, a, b in moves:
            src = state["paths"][v]
            dst = canonical_path(root, b, v)          # <split>/<類別>/<檔名>.json
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            state["paths"][v] = dst
            state["current"][v] = b
            w.writerow([now, mode, v, a, b])
            try:
                moved = move_video_with_skeleton(dst, b)
            except Exception as e:
                video_warnings.append(f"{v}：影片沒有跟著搬（{e}）")
            else:
                if moved:
                    video_rows.append((mode, v, *moved, str(dst)))
    _log_video_moves(video_rows)
    state["video_warnings"] = video_warnings
    for msg in video_warnings:
        print(f"⚠ {msg}")
    return log_path


def _print_table(state, assign):
    prefixes = state["prefixes"]
    table = {s: Counter(_class_of(v, prefixes) for v, sp in assign.items() if sp == s) for s in SPLITS}
    print(f"  {'類別':<8}{'train':>7}{'val':>6}{'test':>6}{'合計':>6}")
    for c in prefixes:
        tot = sum(table[s][c] for s in SPLITS)
        print(f"  {c:<8}{table['train'][c]:>7}{table['val'][c]:>6}{table['test'][c]:>6}{tot:>6}")
    tot = {s: sum(table[s].values()) for s in SPLITS}
    print(f"  {'合計':<8}{tot['train']:>7}{tot['val']:>6}{tot['test']:>6}{sum(tot.values()):>6}")


def run_split(rebuild=False, dry_run=False, confirm=False):
    """命令列版的模式 2：rebuild 整份依類別分層重切；dry_run 只印不搬；confirm 搬動前先問一次。"""
    try:
        state = load_state()
    except RuntimeError as e:
        print(f"❌ {e}")
        return
    print(f"骨架資料夾：{state['root']}（{len(state['paths'])} 支）")
    if state["dup_pairs"]:
        print(f"⚠ 偵測到 {len(state['dup_pairs'])} 組內容重複的骨架（綁成同一組放同一邊；建議刪掉其中一支）：")
        for a, b in state["dup_pairs"]:
            print(f"    {a}  ≈  {b}")
    assign, moves = plan_split(state, rebuild)
    mode = "rebuild" if rebuild else "update"
    ratio_note = "（依類別分層 " + "/".join(f"{RATIOS[s]:.0%}" for s in SPLITS) + "）" if rebuild else ""
    print(f"\n切分：{mode}{ratio_note}")
    _print_table(state, assign)
    for sp in SPLITS:
        fixed = sorted(v for v in assign if fixed_split_of(v) == sp)
        print(f"  固定 {sp}（{len(fixed)} 支）：{fixed}")

    if not moves:
        print("\n✓ 不需要搬動任何檔案。")
        return
    shown = moves if len(moves) <= 30 else moves[:30]
    print(f"\n需要搬動 {len(moves)} 支：")
    for v, a, b in shown:
        print(f"    {v:<16} {a:>10} → {b}" + ("（整理進類別資料夾）" if a == b else ""))
    if len(moves) > len(shown):
        print(f"    …（其餘 {len(moves) - len(shown)} 支略）")
    if dry_run:
        print("\n(dry run，未搬動任何檔案)")
        return
    if confirm and input('\n確認搬動請輸入 "ok"（其他任意鍵取消）：').strip().lower() != "ok":
        print("✗ 已取消，未搬動任何檔案。")
        return
    log_path = apply_moves(state, moves, mode)
    print(f"\n✓ 已搬動 {len(moves)} 支，記錄在 {log_path}")


# ═══════════════════════ 模式 3：影片資料夾同步 ═══════════════════════
def check_split_consistency(skeleton_root=None):
    """訓練前的切分檢查（0_train_gcn.py 呼叫）：抓出在檔案總管手動拖檔造成的問題。回傳 dict：
      clash    [(檔名, [路徑…])]         同一個檔名出現在多個位置（通常是「複製」而不是「搬移」）
      foreign  [(檔名, 記錄的影片檔名)]  骨架記錄的影片不是自己的（複製別人的 JSON 再改檔名）
      video    [(檔名, 骨架 split, 影片 split)]  骨架和原始影片放在不同的 split
      fixed    [(檔名, 目前 split, 固定 split)]  固定名單裡的資料不在它被固定的 split
      same_content [[(檔名, 位置)…]]     檔名不同、骨架內容完全相同（同一支影片存了兩份），放在不同 split
      hq_pair  [[(檔名, 位置)…]]         原版 X 和高畫質版 X_hq 放在不同 split
      misplaced [(檔名, 骨架/影片, 所在資料夾)]  放在別的類別資料夾（例如 lick_10 放進 walk/）
      missing  [檔名]                    找不到原始影片（不影響訓練，只提醒）
    除了 missing 都會讓 0_train_gcn.py 拒絕訓練。所有問題一次列出：同名多份（clash）的每一份
    也會各自檢查類別資料夾，並參與內容重複比對（例如複製 walk33 改名成 walk777 放到別的 split）。"""
    root = Path(skeleton_root) if skeleton_root else Path(_load_config()["SKELETON_DATA_FOLDER"])
    prefixes = list(_load_config()["BEHAVIOR_PREFIXES"].keys())
    issues = {"clash": [], "foreign": [], "video": [], "fixed": [], "same_content": [], "hq_pair": [],
              "misplaced": [], "missing": []}
    by_stem = defaultdict(list)
    for f in iter_skeleton_files(root):
        if _class_of(f.stem, prefixes):
            by_stem[f.stem].append(f)
    issues["clash"] = [(v, fs) for v, fs in sorted(by_stem.items()) if len(fs) > 1]
    paths = {v: fs[0] for v, fs in by_stem.items() if len(fs) == 1}

    def split_eff(f):   # 根目錄的「未分配」檔案訓練時當 train，這裡也用 train 比對
        return "train" if split_of(f) == UNASSIGNED else split_of(f)

    # 檔名開頭就是類別（lick_10 → lick）：放進別的類別資料夾就是放錯（同名多份的每一份都查）
    for v, fs in sorted(by_stem.items()):
        for f in fs:
            if f.parent.parent.name in SPLITS and f.parent.name != class_folder_of(v):
                issues["misplaced"].append((v, "骨架", f.parent.relative_to(root).as_posix()))
    current = {v: split_eff(f) for v, f in paths.items()}

    # 同名多份的每一份也要查「記錄的影片是不是自己的」（例如複製 walk33.json 改名成 walk777）
    for v, fs in issues["clash"]:
        names = set()
        for f in fs:
            try:
                recorded = read_video_path(f)
            except Exception:
                recorded = ""
            if recorded and Path(recorded).stem.lower() != v.lower():
                names.add(Path(recorded).name)
        issues["foreign"] += [(v, n) for n in sorted(names)]

    for v, f in sorted(paths.items()):
        try:
            recorded = read_video_path(f)
        except Exception:
            recorded = ""
        if recorded and Path(recorded).stem.lower() != v.lower():
            # 記錄的是別支影片：影片位置跟這份骨架無關，不做 split／資料夾比對（模式 3 也不會搬它）
            issues["foreign"].append((v, Path(recorded).name))
            continue
        actual = Path(recorded) if recorded else None
        if actual is None or not actual.exists():
            actual = find_video(actual.name if actual else f"{v}.mp4")
        if actual is None:
            issues["missing"].append(v)
            continue
        v_split = split_of_video(actual)
        # 影片放錯類別資料夾更危險：0_dataset_collect.py 抽骨架時用影片所在資料夾當初始標籤
        if v_split and actual.parent.name.lower() != class_folder_of(v):
            issues["misplaced"].append((v, "影片", f"{v_split}/{actual.parent.name}"))
        if v_split and v_split != current[v]:
            issues["video"].append((v, current[v], v_split))

    load_rules(root)
    for v in sorted(paths):
        fs = fixed_split_of(v)
        if fs and current[v] != fs:
            issues["fixed"].append((v, current[v], fs))

    # 內容重複比對：同名多份的每一份都用「檔名@位置」參加，才抓得到「複製 walk33 改名 walk777」
    cand = dict(paths)
    where = {v: current[v] for v in paths}
    for v, fs in issues["clash"]:
        for f in fs:
            key = f"{v}@{f.parent.relative_to(root).as_posix()}"
            cand[key], where[key] = f, split_eff(f)
    # 判斷方式跟檔名流水號無關：內容相同＝骨架座標比對（build_groups），只有 X／X_hq 看檔名
    groups, dup_pairs = build_groups(cand, prefixes)
    members = defaultdict(list)
    for k, g in groups.items():
        members[g].append(k)
    for keys in members.values():
        bases = {k.split("@")[0] for k in keys}
        if len(bases) > 1 and len({where[k] for k in keys}) > 1:   # 只有同名多份彼此重複的，clash 已經報了
            entry = [(k.split("@")[0], k.split("@")[1] if "@" in k else where[k]) for k in sorted(keys)]
            ks = set(keys)
            same = any(a in ks and b in ks and a.split("@")[0] != b.split("@")[0] for a, b in dup_pairs)
            issues["same_content" if same else "hq_pair"].append(entry)
    return issues


def issue_listing(items, limit=15):
    rows = [f"- {x}" for x in items[:limit]]
    return rows + ([f"- …其餘 {len(items) - limit} 筆省略"] if len(items) > limit else [])


def split_issue_report(issues, skeleton_root):
    """把 check_split_consistency() 的結果整理成警告框內容，0_train_gcn.py 與本工具共用。
    回傳 (sections, notes)：sections 給 alert_box（擋下訓練的問題），notes = [(標題, [項目])] 只提醒。"""
    root = Path(skeleton_root)
    problems = []   # (標題, [每筆一行], [修正方法], 單位)
    if issues["clash"]:
        problems.append(("同一個檔案被複製成多份（檔名相同）",
                         [f"{v}：{' / '.join(f.parent.relative_to(root).as_posix() for f in fs)}"
                          for v, fs in issues["clash"]],
                         ["多半是在檔案總管按了 Ctrl+C／Ctrl+V，或按住 Ctrl 拖曳",
                          "修正：刪掉多出來的那份，只留一份"], "筆"))
    if issues["foreign"]:
        problems.append(("骨架裡記錄的影片檔名跟骨架檔名不同",
                         [f"{v}.json 記錄的影片是 {name}" for v, name in issues["foreign"]],
                         ["多半是複製別的 JSON 再改檔名。修正：刪掉這份骨架",
                          "真的要用這支影片：放好影片後用 0_dataset_collect.py 模式 1 抽骨架"],
                         "筆"))
    if issues["video"]:
        problems.append(("骨架和影片放在不同的 split",
                         [f"{v}：骨架在 {a}，影片在 {b}" for v, a, b in issues["video"]],
                         ["修正（以骨架為準）：gcn_dataset_manager.py 模式 3，把影片搬過去",
                          "修正（以影片為準）：gcn_dataset_manager.py 模式 2，把骨架移過去"], "筆"))
    if issues["fixed"]:
        problems.append(("固定的資料不在它被固定的 split",
                         [f"{v}：現在在 {a}，固定在 {b}" for v, a, b in issues["fixed"]],
                         ["修正：gcn_dataset_manager.py 模式 2 按「套用規則／歸位新檔」搬回去",
                          "真的要換 split：先在模式 2「解除固定」再移動"], "筆"))
    def group_line(grp, sep):
        # 同名多份合併成一項：walk777（4 份：test/stop、test/walk…）
        by_name = defaultdict(list)
        for v, where in grp:
            by_name[v].append(where)
        return sep.join(f"{v}（{ws[0]}）" if len(ws) == 1 else f"{v}（{len(ws)} 份：{'、'.join(ws)}）"
                        for v, ws in by_name.items())

    if issues["same_content"]:
        problems.append(("檔名不同、骨架內容卻完全相同，放在不同 split",
                         [group_line(g, " = ") for g in issues["same_content"]],
                         ["判斷方式：比對骨架座標，跟檔名流水號無關",
                          "通常是同一支影片存了兩份，不同 split 看到同一段畫面＝資料洩漏",
                          "修正：留一份就好，刪掉另一支的骨架和影片",
                          "兩份都要留：gcn_dataset_manager.py 模式 2 按「套用規則」放到同一邊"], "組"))
    if issues["hq_pair"]:
        problems.append(("原版和 _hq 高畫質版放在不同 split",
                         [group_line(g, " / ") for g in issues["hq_pair"]],
                         ["X 和 X_hq 是同一支影片的兩種畫質，兩份都保留沒關係，但要放在同一個 split",
                          "修正：gcn_dataset_manager.py 模式 2 按「套用規則／歸位新檔」"], "組"))
    if issues["misplaced"]:
        problems.append(("放在別的類別資料夾（檔名開頭就是類別）",
                         [f"{v}：{kind}放在 {folder}/，應該在 {class_folder_of(v)}/ 資料夾"
                          for v, kind, folder in issues["misplaced"]],
                         ["修正（骨架）：gcn_dataset_manager.py 模式 2 按「套用規則／歸位新檔」",
                          "修正（影片）：在檔案總管搬回正確的類別資料夾，再跑模式 3 更新路徑"], "筆"))
    sections = [(f"[{i}] {title}：{len(items)} {unit}", issue_listing(items) + [""] + fixes)
                for i, (title, items, fixes, unit) in enumerate(problems, 1)]
    notes = [("找不到原始影片", list(issues["missing"]))] if issues["missing"] else []
    return sections, notes


def show_split_issues(pause=True):
    """啟動時先檢查切分，一次列出所有問題（跟 0_train_gcn.py 訓練前擋下的是同一份清單）。"""
    root = Path(_load_config()["SKELETON_DATA_FOLDER"])
    print("檢查 train/val/test 切分（手動搬檔、固定名單、重複組、影片位置、類別資料夾）...")
    sections, notes = split_issue_report(check_split_consistency(root), root)
    for title, items in notes:
        print(f"  ⚠ 提醒（不影響訓練）：{title}，{len(items)} 筆")
        for line in issue_listing(items, 10):
            print(f"      {line}")
    if not sections:
        print("  ✓ 切分一致\n")
        return
    alert_box("警告：train/val/test 切分有問題（訓練會被擋下）", sections,
              footer="下面的模式可以用來修正：模式 2 切分管理視窗（套用規則／移動）、模式 3 影片資料夾同步。"
                     "有「同一個檔名出現在多個 split」時，要先刪掉多的那份，模式 2 視窗才能開啟。",
              pause=pause)


def plan_video_sync(skeleton_root):
    """比對每支骨架目前所在的 split 跟它的原始影片所在位置，回傳同步計畫 dict：
      moves     [(video_id, json, 影片實際位置, 目的地)]：影片要搬到 VIDEO_ROOT/<骨架 split>/<類別>/
      relinks   [(video_id, json, JSON 記錄的舊路徑, 影片實際位置)]：影片已在正確位置，
                只是 JSON 記的路徑過時（例如在檔案總管手動搬過影片），只改 JSON
      orphans   [(影片, 目的地)]：舊排法（VIDEO_ROOT/<類別>/）裡沒有骨架的影片，搬到 train/<類別>/
      problems  [文字]：找不到影片、同名影片不只一支、目的地已被佔用等，不處理只列出
    JSON 記錄的路徑找不到影片時，改用檔名在 VIDEO_ROOT 底下找（唯一一支才採用），
    依 JSON 裡的影片檔名把對應關係還原回來。"""
    by_name = defaultdict(list)
    for p in VIDEO_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            by_name[p.name.lower()].append(p)
    plan = {"moves": [], "relinks": [], "orphans": [], "problems": []}
    claimed = set()
    skeletons = iter_skeleton_files(skeleton_root)
    stem_count = Counter(js.stem for js in skeletons)
    for js in skeletons:
        vid = js.stem
        if stem_count[vid] > 1:
            # 同一個檔名有多份骨架（多半是複製而不是搬移）：不知道哪一份才算數，影片不動
            msg = f"{vid}：有 {stem_count[vid]} 份骨架，先刪掉多出來的那份再同步"
            if msg not in plan["problems"]:
                plan["problems"].append(msg)
            continue
        recorded = read_video_path(js)
        if not recorded:
            plan["problems"].append(f"{vid}：JSON 沒有記錄影片路徑")
            continue
        if Path(recorded).stem.lower() != vid.lower():
            plan["problems"].append(f"{vid}：記錄的影片是 {Path(recorded).name}（不是自己的），不處理")
            continue
        actual = Path(recorded)
        if not actual.exists():
            cands = by_name.get(actual.name.lower(), [])
            if len(cands) != 1:
                plan["problems"].append(
                    f"{vid}：找不到影片 {actual.name}" if not cands else
                    f"{vid}：{actual.name} 有 {len(cands)} 支同名影片，無法判斷是哪一支")
                continue
            actual = cands[0]
        try:
            actual.resolve().relative_to(VIDEO_ROOT.resolve())
        except ValueError:
            continue   # 影片不在 模型專用/ 底下（別的來源），不動
        claimed.add(actual.resolve())
        dst = video_path_in_split(actual, split_of(js))
        if actual.resolve() != dst.resolve():
            if dst.exists():
                plan["problems"].append(f"{vid}：目的地已有同名影片 {dst}")
            else:
                plan["moves"].append((vid, js, actual, dst))
        elif str(actual) != recorded:
            plan["relinks"].append((vid, js, recorded, actual))
    for c in VIDEO_CLASSES:
        legacy = VIDEO_ROOT / c
        if legacy.is_dir():
            for p in sorted(legacy.iterdir()):
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS and p.resolve() not in claimed:
                    plan["orphans"].append((p, VIDEO_ROOT / "train" / c / p.name))
    return plan


def run_video_sync(dry_run=False, confirm=True):
    """模式 3：讓 模型專用/ 的影片跟骨架切分一致（影片放到骨架所在的 split），並把骨架
    JSON 的 video_path 改成影片的新位置。第一次執行就是把舊排法（模型專用/<類別>/）
    整理成 模型專用/<split>/<類別>/；之後只有手動拖過骨架或影片時才需要再跑。"""
    cfg = _load_config()
    root = Path(cfg["SKELETON_DATA_FOLDER"])
    plan = plan_video_sync(root)
    moves, relinks, orphans, problems = plan["moves"], plan["relinks"], plan["orphans"], plan["problems"]
    print(f"影片資料夾：{VIDEO_ROOT}")
    print(f"骨架資料夾：{root}")
    if moves:
        counts = Counter((split_of(js), d.parent.name) for _, js, _, d in moves)
        print(f"\n要搬到骨架所在切分的影片：{len(moves)} 支")
        for sp in SPLITS:
            row = "  ".join(f"{c} {counts[(sp, c)]}" for c in VIDEO_CLASSES if counts[(sp, c)])
            if row:
                print(f"    {sp:<5}：{row}")
        for vid, _, a, b in moves[:8]:
            print(f"    例：{vid:<14} {a.relative_to(VIDEO_ROOT)} → {b.relative_to(VIDEO_ROOT)}")
    if relinks:
        print(f"\n影片位置正確、只需更新骨架 JSON 記錄的路徑：{len(relinks)} 支")
    if orphans:
        print(f"\n舊資料夾裡沒有骨架的影片（新影片），搬到 train/：{len(orphans)} 支")
        for a, b in orphans[:8]:
            print(f"    {a.relative_to(VIDEO_ROOT)} → {b.relative_to(VIDEO_ROOT)}")
    if problems:
        print(f"\n⚠ 無法處理、需要手動確認：{len(problems)} 筆")
        for msg in problems[:20]:
            print(f"    {msg}")
    if not (moves or relinks or orphans):
        print("\n✓ 影片資料夾已經跟骨架切分一致，不需要搬動。")
        return
    if dry_run:
        print("\n(dry run，未搬動任何檔案)")
        return
    if confirm and input('\n確認執行請輸入 "ok"（其他任意鍵取消）：').strip().lower() != "ok":
        print("✗ 已取消，未搬動任何檔案。")
        return

    rows, failed = [], []
    for vid, js, recorded, actual in relinks:
        rewrite_video_path(js, recorded, actual)
        rows.append(("sync_relink", vid, recorded, str(actual), str(js)))
    for vid, js, actual, dst in moves:
        recorded = read_video_path(js)
        try:
            if str(actual) != recorded:           # JSON 記的是過時路徑：先指到影片實際位置
                rewrite_video_path(js, recorded, actual)
            moved = move_video_with_skeleton(js, split_of(js))
        except Exception as e:
            failed.append(f"{vid}：{e}")
            continue
        if moved:
            rows.append(("sync", vid, *moved, str(js)))
    for src, dst in orphans:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        except Exception as e:
            failed.append(f"{src.name}：{e}")
            continue
        rows.append(("sync_orphan", dst.stem, str(src), str(dst), ""))
    _log_video_moves(rows)

    # 每個 split 都建好五個類別資料夾（新影片放 train/<類別>/）；舊排法的類別資料夾清空了就移除
    for sp in SPLITS:
        for c in VIDEO_CLASSES:
            (VIDEO_ROOT / sp / c).mkdir(parents=True, exist_ok=True)
    removed, kept = [], []
    for c in VIDEO_CLASSES:
        legacy = VIDEO_ROOT / c
        if not legacy.is_dir():
            continue
        leftovers = [p.name for p in legacy.iterdir()]
        if leftovers:   # 還有非影片檔（筆記、desktop.ini…）→ 不動，交給使用者決定
            kept.append(f"{c}/（還有 {', '.join(leftovers[:3])}）")
            continue
        try:
            legacy.rmdir()
            removed.append(c)
        except OSError:  # OneDrive 同步中常會暫時佔住資料夾，不影響結果
            kept.append(f"{c}/（空資料夾，被其他程式佔用，可手動刪除）")
    print(f"\n✓ 已處理 {len(rows)} 筆，記錄在 {VIDEO_MOVES_LOG}")
    if removed:
        print(f"  已移除清空的舊資料夾：{', '.join(removed)}")
    if kept:
        print("  以下舊資料夾沒有移除：" + "；".join(kept))
    if failed:
        print(f"⚠ {len(failed)} 筆失敗（檔案可能正被其他程式開著），關掉後重跑模式 3 即可：")
        for msg in failed[:20]:
            print(f"    {msg}")


def undo_last_video_sync(confirm=True):
    """還原最近一次模式 3 同步：依 _video_moves_log.csv 把那一批影片搬回原位，骨架 JSON
    的 video_path 也改回去。只處理模式 3 的紀錄（模式 2 搬骨架連帶搬的影片要從模式 2
    「復原上一步」撤回，才不會讓影片跟骨架切分對不上）。"""
    if not VIDEO_MOVES_LOG.exists():
        print("✗ 沒有影片搬動紀錄。")
        return
    with open(VIDEO_MOVES_LOG, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["mode"].startswith("sync")]
    if not rows:
        print("✗ 紀錄裡沒有模式 3 的同步。")
        return
    last = rows[-1]["time"]
    batch = [r for r in rows if r["time"] == last]
    print(f"最近一次同步：{last}，共 {len(batch)} 筆")
    if confirm and input('確認還原請輸入 "ok"（其他任意鍵取消）：').strip().lower() != "ok":
        print("✗ 已取消。")
        return
    done, failed = [], []
    for r in reversed(batch):
        src, dst, js = Path(r["from"]), Path(r["to"]), r["json"]
        try:
            if r["mode"] != "sync_relink":
                if not dst.exists() or src.exists():
                    raise RuntimeError("影片已不在同步後的位置，或原位置已有檔案")
                src.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dst), str(src))
            if js and Path(js).exists():
                rewrite_video_path(js, str(dst), str(src))
        except Exception as e:
            failed.append(f"{r['video_id']}：{e}")
            continue
        done.append(("undo_sync", r["video_id"], str(dst), str(src), js))
    _log_video_moves(done)
    print(f"✓ 已還原 {len(done)} 筆")
    if failed:
        print(f"⚠ {len(failed)} 筆沒有還原：")
        for msg in failed[:20]:
            print(f"    {msg}")


# ═══════════════════════ 模式 2 GUI：三欄 train / val / test ═══════════════════════
class SplitManagerGUI:
    """三個可捲動清單（train / val / test），可多選（Ctrl／Shift）後批次移到另一個切分。

    移動時的規則跟命令列版一致：
      ・FORCE_TRAIN 的影片不能移出 train（會略過並提示）
      ・重複檔／X 與 X_hq 同組的影片會整組一起移動（避免同一段內容跨切分洩漏）
      ・HARD_TEST 的影片移出 test 前會先確認
    每次移動都是真的搬檔，並記錄到 skeletons/_split_moves_log.csv；「復原上一步」可撤回最近一批。
    """

    def __init__(self, state):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.state = state
        self.undo_stack = []          # 每一批移動：[(video_id, 從, 到), ...]
        self.frames_cache = {}        # video_id -> (總幀數, 已標註幀數)

        self.root = tk.Tk()
        self.root.title("ST-GCN 骨架資料集切分管理（train / val / test）")
        self.root.geometry("1280x760")
        self.root.minsize(900, 500)

        self._build_toolbar()
        self._build_panels()
        self._build_statusbar()
        self.refresh()

    # ── 版面 ────────────────────────────────────────────────────────────────
    def _build_toolbar(self):
        tk, ttk = self.tk, self.ttk
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(fill="x")

        ttk.Label(bar, text="類別：").pack(side="left")
        self.class_var = tk.StringVar(value="全部")
        cb = ttk.Combobox(bar, textvariable=self.class_var, width=9, state="readonly",
                          values=["全部"] + self.state["prefixes"])
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self.refresh())

        ttk.Label(bar, text="  搜尋：").pack(side="left")
        self.search_var = tk.StringVar()
        ent = ttk.Entry(bar, textvariable=self.search_var, width=18)
        ent.pack(side="left")
        ent.bind("<KeyRelease>", lambda e: self.refresh())

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(bar, text="選取項目移到：").pack(side="left")
        for s in SPLITS:
            ttk.Button(bar, text=f"→ {s}", width=8,
                       command=lambda t=s: self.move_selected(t)).pack(side="left", padx=2)

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        self.undo_btn = ttk.Button(bar, text="↶ 復原上一步", command=self.undo)
        self.undo_btn.pack(side="left", padx=2)

        ttk.Button(bar, text="整份重新切分…", command=self.rebuild).pack(side="right", padx=2)
        ttk.Button(bar, text="套用規則／歸位新檔", command=self.apply_rules).pack(side="right", padx=2)
        ttk.Button(bar, text="重新整理", command=self.reload).pack(side="right", padx=2)

        # 第二排：批次固定／解除固定選取的影片（寫進 skeletons/_split_rules.json）
        rbar = ttk.Frame(self.root, padding=(8, 0, 8, 6))
        rbar.pack(fill="x")
        ttk.Label(rbar, text="選取項目：").pack(side="left")
        ttk.Button(rbar, text="固定在目前位置", command=lambda: self.pin(None)).pack(side="left", padx=2)
        fix_to = ttk.Menubutton(rbar, text="固定到…")
        fix_menu = tk.Menu(fix_to, tearoff=0)
        for sp in SPLITS:
            fix_menu.add_command(label=f"固定到 {sp}", command=lambda t=sp: self.pin(t))
        fix_to["menu"] = fix_menu
        fix_to.pack(side="left", padx=2)
        ttk.Button(rbar, text="解除固定", command=self.unpin).pack(side="left", padx=2)
        ttk.Separator(rbar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(rbar, text="查看固定名單…", command=self.edit_rules_dialog).pack(side="left", padx=2)
        self.rules_summary = ttk.Label(rbar, foreground="#555")
        self.rules_summary.pack(side="left", padx=10)

    def _build_panels(self):
        tk, ttk = self.tk, self.ttk
        body = ttk.Frame(self.root, padding=(8, 0, 8, 0))
        body.pack(fill="both", expand=True)
        self.trees, self.headers = {}, {}
        for i, s in enumerate(SPLITS):
            body.columnconfigure(i, weight=1, uniform="col")
            body.rowconfigure(1, weight=1)
            hdr = ttk.Label(body, text=s, font=("Microsoft JhengHei", 10, "bold"), anchor="w")
            hdr.grid(row=0, column=i, sticky="we", padx=4, pady=(4, 2))
            self.headers[s] = hdr

            box = ttk.Frame(body)
            box.grid(row=1, column=i, sticky="nsew", padx=4)
            tree = ttk.Treeview(box, columns=("cls", "frames", "note"), selectmode="extended")
            tree.heading("#0", text="檔名", command=lambda sp=s: self._sort(sp, "#0"))
            tree.heading("cls", text="類別", command=lambda sp=s: self._sort(sp, "cls"))
            tree.heading("frames", text="標註/總幀", command=lambda sp=s: self._sort(sp, "frames"))
            tree.heading("note", text="備註")
            tree.column("#0", width=130, stretch=True)
            tree.column("cls", width=60, stretch=False, anchor="center")
            tree.column("frames", width=78, stretch=False, anchor="e")
            tree.column("note", width=90, stretch=True)
            ysb = ttk.Scrollbar(box, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=ysb.set)
            tree.pack(side="left", fill="both", expand=True)
            ysb.pack(side="right", fill="y")
            tree.tag_configure("fix_train", foreground="#1f6feb")
            tree.tag_configure("fix_val", foreground="#8e44ad")
            tree.tag_configure("fix_test", foreground="#c0392b")
            tree.tag_configure("dup", background="#fff4d6")
            tree.bind("<<TreeviewSelect>>", lambda e: self._update_status())
            tree.bind("<Button-3>", lambda e, t=tree, sp=s: self._context_menu(e, t, sp))
            for seq in ("<Control-a>", "<Control-A>"):   # 大小寫（Caps Lock）都支援
                tree.bind(seq, lambda e, sp=s: (self._select_all(sp), "break")[1])
            # 選取只限單一 split：在這一欄按下滑鼠就清掉另外兩欄的選取；
            # 按住左鍵拖曳＝一次選一段（拖到清單邊緣會自動捲動），按住 Ctrl 再拖＝追加一段
            tree.bind("<ButtonPress-1>", lambda e, sp=s: self._drag_start(e, sp))
            tree.bind("<B1-Motion>", lambda e, sp=s: self._drag_motion(e, sp))
            tree.bind("<ButtonRelease-1>", lambda e: self._drag_end())
            self.trees[s] = tree

            btns = ttk.Frame(body)
            btns.grid(row=2, column=i, sticky="we", padx=4, pady=(3, 6))
            ttk.Button(btns, text="全選", width=6,
                       command=lambda sp=s: self._select_all(sp)).pack(side="left")
            ttk.Button(btns, text="取消選取", width=8,
                       command=lambda t=tree: t.selection_remove(t.selection())).pack(side="left", padx=4)
            for tgt in SPLITS:
                if tgt != s:
                    ttk.Button(btns, text=f"選取 → {tgt}", width=11,
                               command=lambda src=s, t=tgt: self.move_selected(t, only=src)).pack(side="right", padx=2)
        self.sort_key = {s: ("#0", False) for s in SPLITS}
        self._drag = None           # 拖曳多選的狀態：{split, anchor, base, active}
        self._drag_after = None     # 拖到邊緣時自動捲動的計時器

    # ── 選取：限單一 split、按住拖曳多選 ────────────────────────────────────
    def _clear_other_splits(self, keep):
        for sp, t in self.trees.items():
            if sp != keep and t.selection():
                t.selection_remove(t.selection())

    def _select_all(self, split):
        self._clear_other_splits(split)
        self.trees[split].selection_set(self.trees[split].get_children())

    def _drag_start(self, event, split):
        """按下左鍵：清掉其他欄的選取，記下拖曳起點。點選本身交給 Treeview 預設行為
        （單擊＝單選、Ctrl+單擊＝加選／取消、Shift+單擊＝選一段），這裡不攔截。"""
        self._clear_other_splits(split)
        tree = self.trees[split]
        if tree.identify_region(event.x, event.y) not in ("tree", "cell"):
            self._drag = None           # 點在欄位標題／空白處：不啟動拖曳
            return
        ctrl = bool(event.state & 0x0004)
        self._drag = {
            "split": split,
            "anchor": tree.identify_row(event.y),
            "base": set(tree.selection()) if ctrl else set(),   # Ctrl+拖＝在原本的選取上追加
            "active": False,
        }

    def _drag_select_to(self, split, row):
        d = self._drag
        tree = self.trees[split]
        items = list(tree.get_children())
        if not d or not d["anchor"] or row not in items or d["anchor"] not in items:
            return
        a, b = sorted((items.index(d["anchor"]), items.index(row)))
        tree.selection_set(list(d["base"] | set(items[a:b + 1])))

    def _drag_motion(self, event, split):
        d = self._drag
        if not d or d["split"] != split:
            return
        tree = self.trees[split]
        row = tree.identify_row(event.y)
        if row and (row != d["anchor"] or d["active"]):
            d["active"] = True
            self._drag_select_to(split, row)
        # 拖到清單上緣／下緣之外：持續往那個方向捲動並延伸選取
        h = tree.winfo_height()
        direction = -1 if event.y < 0 else (1 if event.y > h else 0)
        d["scroll"] = direction
        if direction and self._drag_after is None:
            self._drag_autoscroll(split)
        return "break"                  # 不讓 Treeview 預設的拖曳行為干擾

    def _drag_autoscroll(self, split):
        d = self._drag
        if not d or not d.get("scroll"):
            self._drag_after = None
            return
        tree = self.trees[split]
        tree.yview_scroll(d["scroll"], "units")
        edge_y = 2 if d["scroll"] < 0 else tree.winfo_height() - 2
        row = tree.identify_row(edge_y)
        if row:
            d["active"] = True
            self._drag_select_to(split, row)
        self._drag_after = self.root.after(40, lambda: self._drag_autoscroll(split))

    def _drag_end(self):
        if self._drag_after is not None:
            self.root.after_cancel(self._drag_after)
            self._drag_after = None
        self._drag = None
        self._update_status()

    def _build_statusbar(self):
        ttk = self.ttk
        self.status = ttk.Label(self.root, anchor="w", padding=(10, 4), relief="sunken")
        self.status.pack(fill="x", side="bottom")
        legend = ttk.Label(self.root, anchor="w", padding=(10, 0, 10, 2), foreground="#555",
                           text="文字顏色＝已固定（藍＝固定在 train、紫＝固定在 val、紅＝固定在 test），"
                                "固定的資料只會待在那個 split，要移動請先「解除固定」　"
                                "黃底＝內容重複的同一組資料（會一起移動）\n"
                                "選取（限同一個 split）：Ctrl＋左鍵＝加選／取消、Shift＋左鍵＝選一段、"
                                "按住左鍵拖曳＝選一段（Ctrl＋拖＝追加）、Ctrl＋A＝全選該欄、右鍵＝移動／固定")
        legend.pack(fill="x", side="bottom")

    # ── 資料 ────────────────────────────────────────────────────────────────
    def _frames(self, v):
        if v not in self.frames_cache:
            try:
                d = json.loads(self.state["paths"][v].read_text(encoding="utf-8"))
                fr = d.get("frames", [])
                lab = sum(1 for f in fr if f.get("label", "unannotated") != "unannotated")
                self.frames_cache[v] = (lab, len(fr))
            except Exception:
                self.frames_cache[v] = (0, 0)
        return self.frames_cache[v]

    def _note(self, v):
        notes = []
        fs = fixed_split_of(v)
        if fs:
            notes.append(f"固定在 {fs}")
        mates = [m for m in group_members(self.state, v) if m != v]
        if mates:
            notes.append("重複:" + ",".join(mates))
        return "  ".join(notes)

    def _natural(self, s):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

    def refresh(self):
        warnings = self.state.pop("video_warnings", None)
        if warnings:
            from tkinter import messagebox
            messagebox.showwarning(
                "影片沒有跟著搬",
                "骨架已搬到新的切分，但以下影片沒有跟著搬（骨架 JSON 仍指向影片目前的位置，"
                "不影響訓練）；關掉佔用影片的程式後，用模式 3「影片資料夾同步」補搬：\n\n"
                + "\n".join(warnings[:20]) + ("\n…" if len(warnings) > 20 else ""))
        cls = self.class_var.get()
        q = self.search_var.get().strip().lower()
        prefixes = self.state["prefixes"]
        for s in SPLITS:
            tree = self.trees[s]
            keep = set(tree.selection())
            tree.delete(*tree.get_children())
            rows = []
            for v, sp in self.state["current"].items():
                sp = "train" if sp == UNASSIGNED else sp
                if sp != s:
                    continue
                c = _class_of(v, prefixes)
                if cls != "全部" and c != cls:
                    continue
                if q and q not in v.lower():
                    continue
                lab, tot = self._frames(v)
                rows.append((v, c, lab, tot))
            col, rev = self.sort_key[s]
            key = {"#0": lambda r: self._natural(r[0]), "cls": lambda r: (prefixes.index(r[1]), self._natural(r[0])),
                   "frames": lambda r: r[2]}[col]
            rows.sort(key=key, reverse=rev)
            for v, c, lab, tot in rows:
                tags = []
                fs = fixed_split_of(v)
                if fs:
                    tags.append(f"fix_{fs}")
                if len(group_members(self.state, v)) > 1:
                    tags.append("dup")
                unassigned = self.state["current"][v] == UNASSIGNED
                tree.insert("", "end", iid=v, text=v + ("（未歸位）" if unassigned else ""),
                            values=(c, f"{lab}/{tot}", self._note(v)), tags=tags)
            tree.selection_set([v for v in keep if tree.exists(v)])
            self._update_header(s)
        self._update_status()

    def _update_header(self, s):
        prefixes = self.state["prefixes"]
        vids = [v for v, sp in self.state["current"].items() if (("train" if sp == UNASSIGNED else sp) == s)]
        per = Counter(_class_of(v, prefixes) for v in vids)
        shown = len(self.trees[s].get_children())
        filt = f"（顯示 {shown}）" if shown != len(vids) else ""
        self.headers[s].configure(
            text=f"{s.upper()}　{len(vids)} 支{filt}　" + " · ".join(f"{c} {per[c]}" for c in prefixes))

    def _update_status(self):
        sel = sum(len(t.selection()) for t in self.trees.values())
        unassigned = sum(1 for sp in self.state["current"].values() if sp == UNASSIGNED)
        extra = f"　⚠ 有 {unassigned} 支還在根目錄未歸位（按「套用規則／歸位新檔」）" if unassigned else ""
        self.status.configure(text=f"已選取 {sel} 支　骨架資料夾：{self.state['root']}{extra}")
        self.undo_btn.configure(state="normal" if self.undo_stack else "disabled")
        n = Counter(fixed_split_of(v) for v in self.state["paths"])
        self.rules_summary.configure(
            text=f"已固定：train {n['train']} 筆、val {n['val']} 筆、test {n['test']} 筆")

    def _sort(self, s, col):
        cur, rev = self.sort_key[s]
        self.sort_key[s] = (col, not rev if cur == col else False)
        self.refresh()

    # ── 動作 ────────────────────────────────────────────────────────────────
    def move_selected(self, target, only=None):
        from tkinter import messagebox
        selected = []
        for s, tree in self.trees.items():
            if only and s != only:
                continue
            selected += list(tree.selection())
        if not selected:
            messagebox.showinfo("移動", "請先在清單中選取要移動的影片（可用 Ctrl／Shift 多選）。")
            return

        blocked, extra = [], []
        todo = set()
        for v in selected:
            gfs = group_fixed_split(self.state, v)
            if gfs and gfs != target:
                blocked.append(f"{v}（固定在 {gfs}）")
                continue
            for m in group_members(self.state, v):
                if m not in selected:
                    extra.append(m)
                todo.add(m)

        moves = [(v, self.state["current"][v], target) for v in sorted(todo)
                 if self.state["current"][v] != target]
        msg = []
        if blocked:
            msg.append(f"以下 {len(blocked)} 支已固定（或與固定的影片同組），不移動；要移動請先「解除固定」：\n  "
                       + ", ".join(sorted(set(blocked))))
        if extra:
            msg.append(f"以下與選取項目同組（重複檔／X_hq），會一起移到 {target}：\n  " + ", ".join(sorted(set(extra))))
        if not moves:
            messagebox.showinfo("移動", "\n\n".join(msg) if msg else f"選取的影片已經都在 {target}。")
            return
        from_panels = {a if a != UNASSIGNED else "train" for _, a, _ in moves}
        if msg or len(moves) > 1 or len(from_panels) > 1:
            # 選取可能分散在不同欄（工具列按鈕會一起處理三欄的選取），逐支標出來源欄
            head = f"將 {len(moves)} 支影片移到 {target}：\n  " + \
                   ", ".join(f"{v}（{a}）" for v, a, _ in moves[:40]) + ("…" if len(moves) > 40 else "")
            if len(from_panels) > 1:
                head += f"\n\n⚠ 要搬動的項目來自 {len(from_panels)} 個欄位：{'、'.join(sorted(from_panels))}"
            if not messagebox.askyesno("確認移動", head + ("\n\n" + "\n\n".join(msg) if msg else "")):
                return
        apply_moves(self.state, moves, "gui")
        self.undo_stack.append(moves)
        # 移動後清掉所有選取（不保留選取狀態，免得下一次按工具列按鈕時被一起搬走），
        # 改成把目標欄捲到剛搬進去的第一支
        for t in self.trees.values():
            t.selection_remove(t.selection())
        self.refresh()
        first = next((v for v, _, _ in moves if self.trees[target].exists(v)), None)
        if first:
            self.trees[target].see(first)
        self.status.configure(text=f"✓ 已將 {len(moves)} 支移到 {target}（可按「復原上一步」撤回）")

    def undo(self):
        if not self.undo_stack:
            return
        moves = self.undo_stack.pop()
        # 只撤回「目前還在當時搬去的位置」的檔案（之後又被搬過的就不動）
        back = [(v, b, a) for v, a, b in moves if self.state["current"].get(v) == b]
        apply_moves(self.state, [m for m in back if m[2] != UNASSIGNED], "gui_undo")
        # 原本在根目錄（未歸位）的放回根目錄
        root = self.state["root"]
        for v, _, _ in [m for m in back if m[2] == UNASSIGNED]:
            src = self.state["paths"][v]
            shutil.move(str(src), str(root / src.name))
            self.state["paths"][v], self.state["current"][v] = root / src.name, UNASSIGNED
            try:
                moved = move_video_with_skeleton(root / src.name, UNASSIGNED)   # 未分配＝train
            except Exception as e:
                self.state.setdefault("video_warnings", []).append(f"{v}：影片沒有跟著搬（{e}）")
            else:
                if moved:
                    _log_video_moves([("gui_undo", v, *moved, str(root / src.name))])
        self.refresh()
        self.status.configure(text=f"↶ 已復原 {len(back)} 支")

    def _run_plan(self, rebuild):
        from tkinter import messagebox
        assign, moves = plan_split(self.state, rebuild)
        if not moves:
            messagebox.showinfo("套用規則", "✓ 目前的切分已符合所有規則，不需要搬動任何檔案。")
            return
        title = "整份重新切分" if rebuild else "套用規則／歸位新檔"
        lines = "\n".join(f"  {v}：整理進 {b}/ 的類別資料夾" if a == b else
                          f"  {v}：{'根目錄（未歸位）' if a == UNASSIGNED else a} → {b}" for v, a, b in moves[:25])
        more = f"\n  …（其餘 {len(moves) - 25} 支）" if len(moves) > 25 else ""
        warn = ("\n\n⚠ 整份重切會重新抽 val / test，之後的 test 分數就不能跟之前的模型直接比較。"
                if rebuild else "")
        if messagebox.askyesno(title, f"將搬動 {len(moves)} 支：\n{lines}{more}{warn}\n\n確定嗎？",
                               icon="warning" if rebuild else "question"):
            apply_moves(self.state, moves, "rebuild" if rebuild else "update")
            self.undo_stack.append(moves)
            self.refresh()
            self.status.configure(text=f"✓ {title}：已搬動 {len(moves)} 支（可按「復原上一步」撤回）")

    # ── 規則：設定／清除／編輯 ─────────────────────────────────────────────
    def _selected_ids(self):
        return [v for t in self.trees.values() for v in t.selection()]

    def _context_menu(self, event, tree, split):
        tk = self.tk
        self._clear_other_splits(split)     # 選取只限單一 split
        row = tree.identify_row(event.y)
        if row and row not in tree.selection():
            tree.selection_set(row)          # 右鍵點在未選取的列上 → 改選這一列
        if not self._selected_ids():
            return
        menu = tk.Menu(self.root, tearoff=0)
        for s in SPLITS:
            if s != split:
                menu.add_command(label=f"移到 {s}", command=lambda t=s: self.move_selected(t))
        menu.add_separator()
        menu.add_command(label="固定在目前位置", command=lambda: self.pin(None))
        for s in SPLITS:
            menu.add_command(label=f"固定到 {s}", command=lambda t=s: self.pin(t))
        menu.add_command(label="解除固定", command=self.unpin)
        menu.tk_popup(event.x_root, event.y_root)

    def pin(self, target=None):
        """批次固定選取的影片。target=None：各自固定在目前所在的切分（根目錄未歸位的算 train）；
        target='train'/'val'/'test'：固定到指定切分，不在那裡的會詢問是否順便搬過去。"""
        from tkinter import messagebox
        sel = self._selected_ids()
        if not sel:
            messagebox.showinfo("固定", "請先選取影片（可用 Ctrl／Shift 多選）。")
            return
        applied, skipped = [], []
        for v in sel:
            cur = self.state["current"][v]
            dest = target or ("train" if cur == UNASSIGNED else cur)
            mate = next((m for m in group_members(self.state, v)
                         if m != v and fixed_split_of(m) and fixed_split_of(m) != dest), None)
            if mate:
                skipped.append(f"{v}（和它內容重複的 {mate} 已固定在 {fixed_split_of(mate)}）")
                continue
            for sp in SPLITS:
                if v in _RULES["fixed"][sp]:
                    _RULES["fixed"][sp].remove(v)
            _RULES["fixed"][dest].append(v)
            applied.append((v, dest))
        save_rules(self.state["root"])
        self.refresh()

        per = Counter(d for _, d in applied)
        info = ("已固定 " + "、".join(f"{per[sp]} 支在 {sp}" for sp in SPLITS if per[sp]) + "。") if applied else ""
        if skipped:
            info += f"\n\n以下 {len(skipped)} 支無法固定到指定位置：\n  " + "\n  ".join(skipped)

        # 固定的位置跟目前位置不同的（連同同組）→ 問要不要現在搬過去
        todo = {}
        for v, dest in applied:
            for m in group_members(self.state, v):
                if self.state["current"][m] != dest:
                    todo[m] = dest
        if todo:
            ask = (info + f"\n\n其中 {len(todo)} 支（含同組）目前不在固定的位置：\n  "
                   + ", ".join(f"{v}（{self.state['current'][v]} → {d}）" for v, d in list(todo.items())[:30])
                   + ("…" if len(todo) > 30 else "") + "\n\n要現在搬過去嗎？")
            if messagebox.askyesno("固定", ask):
                moves = [(v, self.state["current"][v], d) for v, d in sorted(todo.items())]
                apply_moves(self.state, moves, "pin")
                self.undo_stack.append(moves)
                for t in self.trees.values():
                    t.selection_remove(t.selection())
                self.refresh()
            else:
                messagebox.showinfo("固定", "固定名單已存檔，但影片沒有搬動。\n之後按「套用規則／歸位新檔」會把它們移到固定的位置。")
        elif info:
            messagebox.showinfo("固定", info)
        self.status.configure(text=f"固定名單已更新（{RULES_FILENAME}）")

    def unpin(self):
        """批次解除固定：把選取的資料從固定名單移除（不搬動檔案）。"""
        from tkinter import messagebox
        sel = self._selected_ids()
        if not sel:
            messagebox.showinfo("解除固定", "請先選取資料。")
            return
        cleared = []
        for v in sel:
            for sp in SPLITS:
                if v in _RULES["fixed"][sp]:
                    _RULES["fixed"][sp].remove(v)
                    cleared.append(v)
        save_rules(self.state["root"])
        self.refresh()
        messagebox.showinfo("解除固定", f"已解除 {len(cleared)} 筆的固定，現在可以自由移動。"
                            if cleared else "選取的資料本來就沒有固定。")

    def edit_rules_dialog(self):
        """固定名單總覽：列出固定在 train／val／test 的資料，可選取後解除固定。"""
        tk, ttk = self.tk, self.ttk
        win = tk.Toplevel(self.root)
        win.title("固定名單")
        win.geometry("760x480")
        win.transient(self.root)
        ttk.Label(win, padding=(10, 10, 10, 4), foreground="#555",
                  text="固定在哪個 split，那筆資料就只會待在那裡。要移動請先在這裡（或主視窗）解除固定。").pack(fill="x")

        body = ttk.Frame(win, padding=(10, 0, 10, 6))
        body.pack(fill="both", expand=True)
        boxes = {}
        for i, k in enumerate(SPLITS):
            body.columnconfigure(i, weight=1)
            body.rowconfigure(0, weight=1)
            lf = ttk.LabelFrame(body, text=f"固定在 {k}", padding=6)
            lf.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
            lb = tk.Listbox(lf, selectmode="extended")
            sb = ttk.Scrollbar(lf, orient="vertical", command=lb.yview)
            lb.configure(yscrollcommand=sb.set)
            lb.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")
            boxes[k] = lb

        def show_lists():
            for k, lb in boxes.items():
                lb.delete(0, "end")
                items = sorted(_RULES["fixed"][k], key=self._natural)
                lb.master.configure(text=f"固定在 {k}（{len(items)} 筆）")
                for v in items:
                    where = self.state["current"].get(v)
                    mark = "" if where == k else f"   ⚠ 目前在 {where or '已不在資料夾'}"
                    lb.insert("end", f"{v}{mark}")

        def remove_selected():
            n = 0
            for k, lb in boxes.items():
                items = sorted(_RULES["fixed"][k], key=self._natural)
                for i in reversed(lb.curselection()):
                    _RULES["fixed"][k].remove(items[i])
                    n += 1
            if n:
                save_rules(self.state["root"])
                show_lists(); self.refresh()

        bottom = ttk.Frame(win, padding=(10, 0, 10, 10))
        bottom.pack(fill="x")
        ttk.Button(bottom, text="解除選取項目的固定", command=remove_selected).pack(side="left")
        ttk.Button(bottom, text="關閉", command=win.destroy).pack(side="right")
        show_lists()

    def apply_rules(self):
        self._run_plan(rebuild=False)

    def rebuild(self):
        self._run_plan(rebuild=True)

    def reload(self):
        """重新掃描資料夾（例如在檔案總管手動拖過檔案、或剛抽了新骨架）。"""
        from tkinter import messagebox
        try:
            self.state = load_state(verbose=False)
        except RuntimeError as e:
            messagebox.showerror("重新整理", str(e))
            return
        self.frames_cache.clear()
        self.undo_stack.clear()
        self.refresh()

    def run(self):
        self.root.mainloop()


def run_split_gui():
    print("讀取骨架資料夾並比對重複檔（約需數秒）...")
    try:
        state = load_state()
    except RuntimeError as e:
        print(f"❌ {e}")
        return
    if state["dup_pairs"]:
        print(f"⚠ 偵測到 {len(state['dup_pairs'])} 組內容重複的骨架（GUI 內以黃底標示）")
    print("開啟切分管理視窗…（關閉視窗即結束）")
    SplitManagerGUI(state).run()


# ═══════════════════════════════════ 入口 ═══════════════════════════════════
def _menu():
    print("=" * 60)
    print("ST-GCN 骨架資料集管理")
    print("=" * 60)
    print("1. 統計訓練視窗數（各類別 × train/val/test，含百分比，純讀取）")
    print("2. 切分管理視窗（train/val/test 三欄，批次選取後移到另一個切分）")
    print("3. 影片資料夾同步（模型專用/ 的影片搬到跟骨架相同的 train/val/test，先預覽再確認）")
    print("4. 還原最近一次影片資料夾同步")
    mode = input("\n請選擇模式 (1/2/3/4)：").strip()
    if mode == "1":
        run_window_counts()
    elif mode == "2":
        run_split_gui()
    elif mode == "3":
        run_video_sync()
    elif mode == "4":
        undo_last_video_sync()
    else:
        print("✗ 未選擇，結束。")


def main():
    ap = argparse.ArgumentParser(
        description="ST-GCN 骨架資料集管理：模式 1=統計訓練視窗數，模式 2=train/val/test 切分管理，"
                    "模式 3=影片資料夾同步，模式 4=還原最近一次影片同步")
    ap.add_argument("--mode", choices=["1", "2", "3", "4"], default=None, help="不給則顯示互動選單")
    ap.add_argument("--skeleton_dir", default=None,
                    help="（模式 1）骨架資料夾；不指定則用 stgcn_config.yaml 的 SKELETON_DATA_FOLDER")
    ap.add_argument("--feature_mode", default="xy_conf_v_bone",
                    help="（模式 1）僅影響特徵張量組裝，不影響視窗數統計（預設 xy_conf_v_bone）")
    ap.add_argument("--rebuild", action="store_true", help="（模式 2，命令列）整份重新切分（會搬動檔案）")
    ap.add_argument("--dry_run", action="store_true", help="（模式 2、3）只印結果，不搬任何檔案")
    args = ap.parse_args()

    if args.mode in (None, "2", "3"):
        show_split_issues(pause=sys.stdin.isatty())

    if args.mode == "1":
        run_window_counts(args.skeleton_dir, args.feature_mode)
    elif args.mode == "2":
        if args.rebuild or args.dry_run:
            run_split(rebuild=args.rebuild, dry_run=args.dry_run)
        else:
            run_split_gui()
    elif args.mode == "3":
        run_video_sync(dry_run=args.dry_run)
    elif args.mode == "4":
        undo_last_video_sync()
    else:
        _menu()


if __name__ == "__main__":
    try:
        main()
    except EOFError:
        pass
    except KeyboardInterrupt:
        print("\n已中斷。")
