"""分頁右欄「欄位說明」面板右上角的小按鈕：點一下彈出一個獨立浮動視窗
（Toplevel）顯示圖片，再點一下收合；視窗被使用者直接用右上角 X 關掉時，
按鈕文字要跟著恢復成「顯示圖片」，不能停在「關閉圖片」卻沒有視窗可關。

圖片來源是 `settings_gui/assets/` 底下打包進模組的樣本圖（PIL 現場產生的色塊/
幾何圖案測試圖，不是真的網路下載——這台機器跑這支程式的環境不保證有對外網路，
而且下載來路不明的圖片有版權/內容風險，用程式產生的測試圖案可以達到一樣「每個
分頁都能看到一張像樣的圖」的效果，又不用依賴外部來源）。

**每個分頁對應自己專屬的一張圖，不共用**（`_TAB_IMAGE_FILE`）：檔名維持「先分
類別、類別內再編號」的規則（`category_<類別>_<序號>.png`），一看檔名前綴還是
看得出這個分頁屬於哪一類，但每個分頁實際看到的圖是各自獨立的一張（圖片本身也
印上分頁的中文名稱、圖案數量依序號變化，不是同類別裡貼相同的圖）：

    category_model_1.png        模型與輸入來源
    category_model_2.png        YOLO 推論
    category_model_3.png        ST-GCN 推論
    category_schedule_1.png     執行模式與排程
    category_schedule_2.png     進階設定
    category_integration_1.png  Flask 與 Node-RED
    category_detection_1.png    視覺化與串流顯示
    category_detection_2.png    異常偵測與骨架品質
    category_detection_3.png    行為追蹤與警報門檻
    category_data_1.png         貓咪身份驗證
    category_data_2.png         日誌、CSV、資料庫與輸出路徑
    category_misc.png           保底用：分頁不在上面對照表時的預設圖（例如日後
                                 新增分頁忘記加進 `_TAB_IMAGE_FILE`）

之後要換成每個分頁專屬的真實圖片時，直接覆蓋掉對應檔名的 png 即可，不用改程式
碼；新增分頁時記得在 `_TAB_IMAGE_FILE` 補一筆，`attach_popup_button()` 的介面跟
按鈕/視窗開關邏輯都不用動。
"""

import tkinter as tk
from pathlib import Path

from PIL import Image, ImageOps, ImageTk

from settings_gui.style import BTN_SECONDARY_ACTIVE, BTN_SECONDARY_BG, _styled_button

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_MIN_POPUP_SIZE = (520, 420)  # 容器還沒被畫出來、量到的寬高是 0/1 時的保底大小
_FALLBACK_IMAGE = "category_misc.png"

# 分頁名稱 -> 專屬圖檔檔名。新增分頁時記得補一筆，沒補到的分頁會自動退回
# category_misc.png（見 _pick_image_path），不會壞掉、只是看得出來「忘了設定」
# （圖片內容跟分頁主題對不上），不是靜默出錯。
_TAB_IMAGE_FILE = {
    "模型與輸入來源": "category_model_1.png",
    "YOLO 推論": "category_model_2.png",
    "ST-GCN 推論": "category_model_3.png",
    "執行模式與排程": "category_schedule_1.png",
    "進階設定": "category_schedule_2.png",
    "Flask 與 Node-RED": "category_integration_1.png",
    "視覺化與串流顯示": "category_detection_1.png",
    "異常偵測與骨架品質": "category_detection_2.png",
    "行為追蹤與警報門檻": "category_detection_3.png",
    "貓咪身份驗證": "category_data_1.png",
    "日誌、CSV、資料庫與輸出路徑": "category_data_2.png",
}

# tab_name -> ImageTk.PhotoImage。PhotoImage 沒有其他地方保留參照就會被 GC 回收
# 導致視窗顯示空白；同一個分頁、同一個視窗尺寸的圖也不用每次開關都重新縮放。
_image_cache = {}


def _pick_image_path(tab_name):
    filename = _TAB_IMAGE_FILE.get(tab_name, _FALLBACK_IMAGE)
    path = _ASSETS_DIR / filename
    if path.is_file():
        return path
    # 對照表寫的檔案本身不存在（例如手動刪錯檔）時，最後再退回 misc 一次；
    # 兩次都找不到就是真的沒有圖可用，回傳 None 交給呼叫端顯示警告文字。
    fallback = _ASSETS_DIR / _FALLBACK_IMAGE
    return fallback if fallback.is_file() else None


def _load_fitted_image(tab_name, box_size):
    cache_key = (tab_name, box_size)
    if cache_key in _image_cache:
        return _image_cache[cache_key]

    path = _pick_image_path(tab_name)
    if path is None:
        return None

    with Image.open(path) as src:
        img = src.convert("RGB")
        # ImageOps.contain：等比縮放到剛好塞進 box_size，不會變形拉伸；圖片本身
        # 比例跟彈跳視窗比例不一致時，寧可留一點邊也不要讓圖片被拉扁/拉長變形。
        fitted = ImageOps.contain(img, box_size)

    photo = ImageTk.PhotoImage(fitted)
    _image_cache[cache_key] = photo
    return photo


def attach_popup_button(parent, window, tab_name):
    """在 parent（分頁右欄最上方的一列）放一個靠右對齊的小按鈕，點擊切換顯示/
    收合 tab_name 對應的圖片浮動視窗。window 是 SettingsWindow 本體，Toplevel
    要掛在它底下（母視窗關閉時子視窗才會一起被清掉）。彈跳視窗大小至少對齊
    整個分頁右欄（parent 的上層容器，也就是 tab_docs_panel.render() 收到的
    right_col）目前的實際寬高，不是一個寫死、可能比右欄小很多的固定尺寸。"""
    state = {"toplevel": None}
    right_col = parent.master  # header_row 的上層就是 SettingsWindow._tab_right_columns[tab_name]

    def _on_close():
        top = state["toplevel"]
        state["toplevel"] = None
        if top is not None:
            top.destroy()
        btn.config(text="🖼 顯示圖片")

    def _toggle():
        if state["toplevel"] is not None:
            _on_close()
            return

        popup_w = max(right_col.winfo_width(), _MIN_POPUP_SIZE[0])
        popup_h = max(right_col.winfo_height(), _MIN_POPUP_SIZE[1])

        top = tk.Toplevel(window)
        top.title(f"{tab_name} － 預覽圖")
        top.configure(bg="#1e1e1e")
        top.geometry(f"{popup_w}x{popup_h}")
        # 使用者直接按浮動視窗右上角的 X 關掉時，也要跑跟按鈕一樣的收尾流程
        # （按鈕文字恢復、state 清空），不然按鈕會停在「關閉圖片」卻沒有視窗
        # 可以真的關，下次點擊會誤判成「還開著」而什麼事都不做。
        top.protocol("WM_DELETE_WINDOW", _on_close)

        # 圖片依比例縮放塞進整個彈跳視窗（扣一點邊界），置中顯示；視窗留白處
        # 用跟圖片背景接近的深色，不會突兀。
        photo = _load_fitted_image(tab_name, (popup_w - 24, popup_h - 24))
        holder = tk.Frame(top, bg="#1e1e1e")
        holder.pack(fill="both", expand=True)
        if photo is not None:
            tk.Label(holder, image=photo, bg="#1e1e1e").place(relx=0.5, rely=0.5, anchor="center")
        else:
            tk.Label(
                holder,
                text="⚠️ 找不到樣本圖片（settings_gui/assets/ 底下缺少對應的"
                " category_*.png，含保底用的 category_misc.png）",
                bg="#1e1e1e", fg="#ffffff",
            ).place(relx=0.5, rely=0.5, anchor="center")

        state["toplevel"] = top
        btn.config(text="🖼 關閉圖片")

    btn = _styled_button(
        parent, "🖼 顯示圖片", _toggle, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
        font=window._font_hint, compact=True,
    )
    btn.pack(side="right", anchor="ne")
    return btn
