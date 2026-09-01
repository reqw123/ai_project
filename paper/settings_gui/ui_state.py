"""設定視窗的「使用者介面偏好」持久化——**跟系統設定完全分開**。

`runtime_settings.current.json` / `default_runtime_settings.json` 是
`settings_manager.FIELD_SCHEMA` 管的「會影響 main.py 執行的系統設定」；
這裡存的是純 GUI 便利性資料（例如最近用過的影片路徑），不進那份檔案、
也不進版控。

檔案：paper/local_state/gui_state.json（見 paper/local_state/README.md——
那個資料夾統一收「每台機器各自、換電腦就該重來」的本機狀態檔）。
**全程 fail-safe**：檔案不存在／損毀／無寫入權限一律當「沒有偏好」處理，
絕不 raise，不影響設定視窗開啟。

--------------------------------------------------------------------------
「最近用過的影片路徑」清單（`recent_video_paths`）
--------------------------------------------------------------------------
單一影片檔和資料夾**都收**、混在同一份清單裡，共保留最多 10 筆
（`_MAX_RECENT`，鎖定項也一起算）。每筆存成 `{"path": str, "locked": bool}`。

* 一般項：依「最後一次用到」排序（新的在前）；清單滿了、又用到新路徑時，
  從尾端往前砍**未鎖定**的最舊項來騰位子。
* 鎖定項（🔒）：位置由使用者在「管理最近影片路徑」視窗手動排定，不會被
  新路徑往下擠、不會因「路徑暫時不存在」（例如外接碟沒插）被自動略過、
  也不會被砍。要移除得先在管理視窗解鎖。
* **任何情況都不會自動從檔案裡刪掉項目**——`get_recent_video_paths()` 只是
  「下拉要顯示哪些」的讀取端過濾（未鎖定且目前找不到的不顯示），磁碟上那筆
  還在；真正移除只有管理視窗的「🗑 刪除」。

舊格式相容：`recent_video_paths` 的值若是 `list[str]`（2026-08-31 當天的舊格式）
或更早的 `recent_video_folders` key，載入時都會正規化成上面的 dict 結構。
"""

import json
import os
import tempfile
from pathlib import Path

_PAPER_DIR = Path(__file__).resolve().parent.parent
_STATE_PATH = _PAPER_DIR / "local_state" / "gui_state.json"
# 2026-08-31 之前的舊位置；新檔不存在時一次性讀回來，之後首次 _save 就會寫到新位置。
_LEGACY_STATE_PATH = _PAPER_DIR / ".gui_state.json"

RECENT_VIDEO_PATHS_KEY = "recent_video_paths"
_LEGACY_FOLDERS_KEY = "recent_video_folders"  # 更早的 key，只讀不寫
_MAX_RECENT = 10

# 「獨立腳本工具」下拉的顯示順序偏好——每筆 {"key": 相對 tools/ 的路徑, "locked": bool}。
# 這是「疊在掃描結果上的排序覆寫」：掃到、但偏好裡沒有的腳本排最後；偏好裡有、
# 但目前掃不到的（改名/刪檔）忽略。沒有筆數上限、也沒有「刪除」（腳本是掃出來的）。
SCRIPT_ORDER_KEY = "tool_script_order"


# ---------------------------------------------------------------- 底層讀寫

def _load() -> dict:
    for path in (_STATE_PATH, _LEGACY_STATE_PATH):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return {}


def _save(data: dict) -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(_STATE_PATH.parent), prefix=".gui_state_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, _STATE_PATH)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    except Exception:
        pass  # 純便利性資料，寫不進去就算了，不打斷 GUI


# ---------------------------------------------------------------- 路徑工具

def _key(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def _path_exists(path: str) -> bool:
    try:
        return os.path.isfile(path) or os.path.isdir(path)
    except Exception:
        return False


# ---------------------------------------------------------------- entry 正規化

def _normalize_entries(raw) -> list:
    """把任意來源（新 dict 格式／舊 list[str]／夾雜）整理成
    [{"path": str, "locked": bool}, ...]，去空字串、去重（忽略大小寫與寫法差異），
    保留出現順序。不做「檔案是否存在」的過濾。"""
    out, seen = [], set()
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, str):
            path, locked = item, False
        elif isinstance(item, dict) and isinstance(item.get("path"), str):
            path, locked = item["path"], bool(item.get("locked", False))
        else:
            continue
        path = path.strip()
        if not path:
            continue
        k = _key(path)
        if k in seen:
            continue
        seen.add(k)
        out.append({"path": path, "locked": locked})
    return out


def _load_entries() -> list:
    data = _load()
    raw = data.get(RECENT_VIDEO_PATHS_KEY)
    if raw is None:
        raw = data.get(_LEGACY_FOLDERS_KEY)
    return _normalize_entries(raw or [])


def _save_entries(entries: list) -> None:
    data = _load()
    data.pop(_LEGACY_FOLDERS_KEY, None)  # 順手清掉已淘汰的舊 key
    data[RECENT_VIDEO_PATHS_KEY] = [
        {"path": e["path"], "locked": bool(e["locked"])}
        for e in _normalize_entries(entries)
    ]
    _save(data)


def _enforce_cap(entries: list, protect_key: str = None) -> list:
    """清單超過 _MAX_RECENT 時，從尾端往前砍「未鎖定」的項目來騰位子。
    protect_key 指到的項（通常是這次剛加進來的）即使未鎖定也不砍。
    若除了鎖定項＋受保護項之外沒東西可砍、清單仍超標，就把受保護的新項
    本身丟掉（＝清單被鎖滿了，新路徑這次記不進來）。"""
    if len(entries) <= _MAX_RECENT:
        return entries
    over = len(entries) - _MAX_RECENT
    kept_rev = []
    for e in reversed(entries):
        if over > 0 and not e["locked"] and _key(e["path"]) != protect_key:
            over -= 1
            continue
        kept_rev.append(e)
    kept = list(reversed(kept_rev))
    if len(kept) > _MAX_RECENT and protect_key and _key(kept[0]["path"]) == protect_key:
        kept = kept[1:]
    return kept


# ---------------------------------------------------------------- 對外 API

def get_recent_video_paths() -> list:
    """給「🎬 影片路徑（選填）」下拉當候選值：純路徑字串清單，依顯示順序。

    過濾規則（只影響「顯示」，不動磁碟）：鎖定項一律顯示（即使目前找不到）；
    未鎖定項只在目前確實存在時才顯示。最多 _MAX_RECENT 筆。
    """
    out = []
    for e in _load_entries():
        if e["locked"] or _path_exists(e["path"]):
            out.append(e["path"])
        if len(out) >= _MAX_RECENT:
            break
    return out


def get_recent_video_entries() -> list:
    """給「管理最近影片路徑」視窗：[{"path", "locked", "exists"}]，**不過濾**
    （連目前找不到的也照列，讓使用者自己決定要不要刪）。"""
    return [
        {"path": e["path"], "locked": e["locked"], "exists": _path_exists(e["path"])}
        for e in _load_entries()
    ]


def add_recent_video_path(path: str) -> None:
    """把一個影片檔或資料夾路徑加到最近清單最前面。

    * 已鎖定的路徑再被用到：**保留它原本的手動位置**，不移到最前面（避免打亂
      使用者排好的釘選順序）。
    * 未鎖定的：移到最前面。
    * 清單滿了走 `_enforce_cap`（砍尾端未鎖定的最舊項）。
    * 空字串／目前不存在的路徑直接忽略。
    """
    if not path or not isinstance(path, str):
        return
    path = path.strip()
    if not path or not _path_exists(path):
        return

    k = _key(path)
    entries = _load_entries()
    existing = next((e for e in entries if _key(e["path"]) == k), None)

    if existing is not None and existing["locked"]:
        pass  # 鎖定項：位置不動
    else:
        entries = [e for e in entries if _key(e["path"]) != k]
        entries.insert(0, {"path": path, "locked": False})
        entries = _enforce_cap(entries, protect_key=k)

    _save_entries(entries)


def set_recent_video_entries(entries: list) -> None:
    """「管理最近影片路徑」視窗按下 上移／下移／鎖定／刪除 後，把整理好的
    清單整包寫回（順序 + 鎖定旗標）。接受 [{"path","locked"}, ...]。"""
    norm = _normalize_entries(entries)
    _save_entries(norm[:_MAX_RECENT])


# ---------------------------------------------------------------- 腳本顯示順序

def get_script_order() -> list:
    """回傳使用者排定的「獨立腳本工具」順序偏好：[{"key": str, "locked": bool}]，
    保留檔案裡的順序。壞資料一律略過。"""
    raw = _load().get(SCRIPT_ORDER_KEY, [])
    if not isinstance(raw, list):
        return []
    out, seen = [], set()
    for item in raw:
        if isinstance(item, str):
            key, locked = item, False
        elif isinstance(item, dict) and isinstance(item.get("key"), str):
            key, locked = item["key"], bool(item.get("locked", False))
        else:
            continue
        key = key.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"key": key, "locked": locked})
    return out


def set_script_order(entries: list) -> None:
    """「管理腳本顯示順序」視窗操作後整包寫回。接受 [{"key","locked"}, ...]。"""
    clean, seen = [], set()
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        key = str(e.get("key", "")).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        clean.append({"key": key, "locked": bool(e.get("locked", False))})
    data = _load()
    data[SCRIPT_ORDER_KEY] = clean
    _save(data)


def apply_script_order(discovered_keys: list) -> list:
    """把順序偏好套到目前掃描到的腳本 key 清單上，回傳排好序的 key 清單：
    偏好裡有且目前也掃到的照偏好順序在前，其餘（新腳本）依原順序接在後面。"""
    pref = [e["key"] for e in get_script_order()]
    present = list(discovered_keys)
    present_set = set(present)
    pref_set = set(pref)
    return [k for k in pref if k in present_set] + [k for k in present if k not in pref_set]
