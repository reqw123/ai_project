"""分頁右欄的「說明文件」面板：即時解析
`paper/docs/設定分頁模組與核心函式對照表.md`，把目前分頁對應的表格內容顯示在
右欄，不另外手寫一份會失去同步的文字。

解析方式比照 settings_window.py 既有的 `_load_tool_script_descriptions()`（解析
`docs/獨立運行腳本索引.md`）：fail-safe（讀檔/解析失敗一律回傳空字典，不拋例外、
不影響 GUI 啟動），用章節標題邊界切段落，表格列用「特定欄位一定有反引號」這個
格式特徵當判斷依據，天然跳過表頭與分隔線列。

`設定分頁模組與核心函式對照表.md` 的表格格式（每個分頁一個 `### <分頁名稱>`
標題，底下緊接一張表——這份文件回答的是「這個分頁的參數實際上是被哪個模組、
哪個函式讀走用掉的」，不是欄位層級的 JSON 路徑對照，那份是另一份
`設定視窗欄位對照表.md`）：

    | 模組檔案 | 核心函式／類別 | 說明 |
    |---|---|---|
    | `cat_monitoring_system/detectors/keypoint_detector.py` | `KeypointDetector.detect()` | 對單一影格跑 YOLO-Pose 推論... |

第 1/2 欄（模組檔案／核心函式）永遠是反引號包住，第 3 欄（說明）不是——用這個
當 row 判斷條件，表頭列跟分隔線列自然對不上而被跳過，不用另外判斷「這是不是
表頭」。

**橫向捲動**：每一行文字改用「依內容實際寬度」而不是固定寬度顯示（見
`_selectable_line` 的 `width=` 計算），部分欄位（尤其 JSON key／環境變數／
config.py 屬性那一行，三個字串接在一起常常很長）常常會比右欄本身還寬。整個卡片
包在一個 Canvas 裡，太寬的內容捲動看，而不是被截斷看不到或每一行各自要點進去用
方向鍵才能看完。

**橫向捲軸的位置是浮動的，不是排在卡片下面**：分頁的垂直可視範圍常常比終端機
輸出面板底下剩的空間還小（終端機面板預設蓋掉視窗下方大部分區域，見
settings_window.py 的 `_build_middle_area`），捲軸如果乖乖排在卡片內容流裡，
位置會被終端機蓋住、要先把終端機拖小/收合才找得到。所以 `render()` 回傳的
`hscroll` 元件其實不屬於這個可垂直捲動的分頁版面，呼叫端（`_select_tab`／
`_reposition_active_docs_hscroll`，都在 settings_window.py）改用 `place()` 把它
固定貼在「終端機面板正上方、跟右欄同寬」的位置，不管分頁垂直捲到哪裡都看得到、
拉得到。
"""

import re
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import ttk

from settings_gui import image_popup
from settings_gui.style import (
    COLOR_TOOL_DESC_ACCENT,
    COLOR_TOOL_DESC_BG,
    COLOR_TOOL_DESC_BORDER,
    COLOR_TOOL_DESC_FG,
    CONSOLE_FONT_FAMILY,
)

_ROW_RE = re.compile(
    r"^\|\s*`(?P<module>[^`]+)`\s*\|\s*`(?P<func>[^`]+)`\s*\|\s*(?P<desc>.+?)\s*\|\s*$",
    re.M,
)
_HEADING_RE = re.compile(r"^### (.+?)\s*$", re.M)

_WRAPLENGTH = 700
_WIDTH_PADDING = 2  # 字元寬度估算值再加一點餘裕，避免估算誤差導致文字被裁掉一角

# 說明文件文字區最下方要留的空白高度。這條橫向捲軸現在是浮動貼在終端機面板
# 正上方（見 render() 說明），畫面上的位置是固定的，但文字內容還是跟著分頁一起
# 垂直捲動——分頁捲到最底時，如果文字剛好延伸到跟拉桿同一個畫面高度，最後幾行
# 就會被疊在上層的拉桿蓋住看不到。留一段空白墊在文字區最後，捲到底時墊的是這段
# 空白而不是真正的文字，拉桿才不會蓋到有內容的地方。
_BOTTOM_SPACER_HEIGHT = 48


def _block_edit_keys(event):
    """`_selectable_line` 的按鍵過濾器：放行方向鍵／Ctrl 組合鍵（Ctrl+C 複製、
    Ctrl+A 全選...），其餘一律吃掉（回傳 "break" 讓 Tk 不執行預設行為），達到
    「能選取/複製，但打字打不進去」的效果。滑鼠拖曳選取、雙擊選字是滑鼠事件，
    不經過這個按鍵過濾器，完全不受影響。"""
    if event.state & 0x0004:  # Control 鍵有按著
        return None
    if event.keysym in ("Left", "Right", "Up", "Down", "Home", "End",
                         "Prior", "Next", "Shift_L", "Shift_R", "Tab"):
        return None
    return "break"


def _selectable_line(parent, text, bg, fg, font):
    """唯讀但可反白／Ctrl+C 複製的一行文字，寬度依內容實際需要（而不是固定寬度
    或撐滿容器）——右欄這些內容常常是使用者想直接複製貼到別處用的字串（json
    key／環境變數名稱），用 tk.Label 完全沒辦法選取複製。

    這裡刻意用 tk.Text（height=1，單行）而不是 tk.Entry(state="readonly")：
    後者在 Windows 上即使設了 relief="flat"／bd=0／highlightthickness=0，readonly
    狀態仍會被系統視覺樣式強制畫出一圈很細的內縮邊框/底線（Tk 在 Windows 上對
    Entry 的 readonly 外觀有這個已知限制，Tk 層級的樣式參數蓋不掉），看起來像
    每行文字周圍多了一條不該有的線。tk.Text 不吃這套系統樣式，relief="flat"／
    bd=0／highlightthickness=0 才會真的完全沒有邊框，視覺上才會像單純的文字。
    可編輯性用 `_block_edit_keys` 攔截按鍵輸入達成「唯讀但能選取複製」，不是靠
    `state="disabled"`——Text 在 disabled 狀態下連滑鼠選取都會被一併擋掉，反而
    達不到「可以複製」這個真正的需求。

    `width=`（Text 的字元格數）過去用 `_display_width()`（東亞寬度分類：中文字元
    算 2 個半形寬）估算，但這只是啟發式猜測，對「Ambiguous」寬度分類的標點符號
    （例如這裡會用到的全形直線「｜」分隔符）估不準，而且字元格數本來就只是用
    平均字元寬度換算，跟實際字型渲染出來的像素寬度必然有落差。估算一旦偏低，
    Text widget 自己請求的像素寬度就會比文字實際需要的窄，超出的字元會被這個
    widget 自己內部裁掉（wrap="none" 又沒有掛專屬捲軸），而且外層 Canvas 的
    scrollregion 是依這個（偏窄的）card 寬度算出來的，於是外層水平拉桿也會誤判
    成「內容剛好塞得下」（xview 顯示 0.0～1.0，拉桿滿版、根本沒有可以往右拉的
    空間）——結果就是文字被截斷，使用者卻連拉桿都拉不動去看被截斷的部分。
    改成直接用這一行實際會用到的字型物件量測像素寬度（`font.measure(text)`），
    換算成字元格數時無條件進位並加上安全餘裕（`_WIDTH_PADDING`），保證這個
    widget 請求的寬度一定大於等於文字實際需要的寬度，不會再內部裁字；card／
    canvas 的 scrollregion 也才會如實反映內容真正的寬度，太寬時外層拉桿才會
    正確顯示成「有空間可以往右拉」。插入文字後 Tk 預設會把游標留在「文字結尾」，
    若渲染時仍有極小的四捨五入誤差，Text 自己的內部視圖會自動捲到跟著游標、也
    就是捲到文字尾端，畫面上第一眼看到的反而是每行的後半段（視覺上像是「從
    中間/右側開始」）——插入完之後明確把游標跟內部視圖都撥回開頭
    （`mark_set`/`xview_moveto`），確保每一行固定都是從最左邊（文字開頭）開始
    顯示。"""
    font_obj = tkfont.Font(parent, font=font)
    px_width = font_obj.measure(text)
    unit_width = max(font_obj.measure("0"), 1)
    width_chars = -(-px_width // unit_width) + _WIDTH_PADDING  # 無條件進位 + 安全餘裕
    widget = tk.Text(
        parent, height=1, width=width_chars, wrap="none",
        font=font, bg=bg, fg=fg, relief="flat", bd=0, highlightthickness=0,
        padx=0, pady=0, cursor="xterm", insertwidth=0,
    )
    widget.insert("1.0", text)
    widget.mark_set("insert", "1.0")
    widget.xview_moveto(0.0)
    widget.bind("<Key>", _block_edit_keys)
    widget.pack(anchor="w")
    return widget


def parse(md_path) -> dict:
    """回傳 {分頁名稱: [{"module":..., "func":..., "desc":...}, ...]}；讀檔失敗或
    某分頁在文件裡找不到對應章節，該分頁就不會出現在回傳的字典裡（呼叫端用
    .get(tab_name, []) 取用，天然處理缺漏）。"""
    try:
        text = Path(md_path).read_text(encoding="utf-8")
    except OSError:
        return {}

    headings = list(_HEADING_RE.finditer(text))
    result = {}
    for i, heading in enumerate(headings):
        tab_name = heading.group(1).strip()
        start = heading.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        section_text = text[start:end]
        rows = [m.groupdict() for m in _ROW_RE.finditer(section_text)]
        if rows:
            result[tab_name] = rows
    return result


def render(parent, rows, window, tab_name, hscroll_master):
    """把某個分頁的 rows（parse() 回傳字典裡的一個 value）畫成說明卡片，塞進
    parent（即 SettingsWindow._tab_right_columns[tab_name]）。rows 為空（該分頁在
    文件裡找不到對應章節，或整份文件解析失敗）時顯示一行警告文字，不讓右欄空白到
    看起來像壞掉。

    回傳一個 dict：`{"resync": pin_left_and_resync, "hscroll": hscroll}`。

    `resync`——呼叫端（settings_window.py 的 `_select_tab`）要在這個分頁「即將被
    選取顯示」的當下呼叫它一次，讓橫向捲軸重新歸零並用當時（元件保證已映射到
    畫面上）的真實版面重新同步一次，畫面上的滑塊位置才會可靠地跟文字內容的最
    左邊對上（原因見函式內部 `pin_left_and_resync` 的說明）。

    `hscroll`——這個分頁專屬的橫向捲軸元件本身。刻意不 `.pack()`／不放進會垂直
    捲動的分頁內容裡：分頁常常比終端機輸出面板底下的可視範圍還高，捲軸如果跟著
    卡片內容排版，位置會落在哪裡完全看內容多寡，終端機面板預設又蓋掉視窗下方
    大部分區域（見 settings_window.py 的 `_build_middle_area` 說明），於是這條
    捲軸很容易被蓋住、使用者得先把終端機拖小/收合才找得到。改成用 `hscroll_master`
    （呼叫端傳進來的 content_area）當它的母元件，呼叫端用 `place()` 把它「浮動
    貼齊在終端機面板正上方、跟右欄同寬」，永遠可見、不會被蓋住；分頁內容真的
    太長看不完時，改用分頁自己既有的垂直捲軸捲動，這條橫向捲軸只負責左右平移
    說明卡片，兩者責任分開。"""
    # 圖片彈跳按鈕要放「右欄右上角」，而且不能跟著下面可橫向捲動的卡片一起被
    # 捲走——所以獨立用一列 header_row，包在會捲動的 outer/canvas 之外、疊在
    # 它上面，按鈕永遠固定在右上角，跟卡片內容捲到哪裡無關。
    header_row = tk.Frame(parent, bg=COLOR_TOOL_DESC_BG)
    header_row.pack(fill="x", padx=(0, 14))
    image_popup.attach_popup_button(header_row, window, tab_name)

    outer = tk.Frame(parent, bg=COLOR_TOOL_DESC_BG)
    outer.pack(fill="x", padx=(0, 14), pady=(0, 10))

    # height=1：Canvas 沒指定尺寸時預設請求較大的高度，這裡先蓋掉，實際高度由
    # 下面 _sync_scrollregion() 依卡片內容算出來的真實高度回填，不會又寬又扁。
    canvas = tk.Canvas(outer, bg=COLOR_TOOL_DESC_BG, highlightthickness=0, height=1)
    # hscroll 的母元件是 hscroll_master（content_area），不是 outer——這樣呼叫端
    # 才能用 place() 把它放到「跟這張卡片對應的分頁完全無關」的固定畫面位置
    # （終端機面板正上方），不受這裡的 pack 版面排列約束。用獨立具名樣式（見
    # settings_window.py 的 _apply_ttk_style），跟其他捲動軸的顏色分開，一眼
    # 就能認出「就是這一條」。
    hscroll = ttk.Scrollbar(
        hscroll_master, orient="horizontal", command=canvas.xview,
        style="DocsPanel.Horizontal.TScrollbar",
    )
    canvas.configure(xscrollcommand=hscroll.set)
    canvas.pack(side="top", fill="x")

    card = tk.Frame(
        canvas, bg=COLOR_TOOL_DESC_BG,
        highlightbackground=COLOR_TOOL_DESC_BORDER, highlightthickness=1,
    )
    canvas.create_window((0, 0), window=card, anchor="nw")

    def _sync_scrollregion(_e=None):
        """只負責「量測」跟「回報」，絕不動使用者當下的捲動位置。之前的版本在
        這裡無條件呼叫過 `canvas.xview_moveto(0.0)`，結果這個 callback 自己呼叫
        `canvas.configure(height=...)` 就會再觸發一次 canvas 的 <Configure>，
        形成迴圈；日後只要視窗被縮放、或任何造成 outer/card/canvas 尺寸變動的
        事件（甚至跟這個面板完全無關的視窗最大化/還原），都會再讓這個 callback
        跑一次、把捲動位置強制彈回最左邊——等於拉桿被「黏死」在最左邊，使用者
        不管怎麼把拉桿往右拉，下一個事件一來就被彈回去，往右拉形同無效。
        「面板剛顯示時要停在最左邊」這件事改由 `pin_left_and_resync()`（見下）
        在分頁真正被選取、即將顯示的當下才主動呼叫，跟這裡的例行量測結構分開。
        """
        canvas.configure(scrollregion=canvas.bbox("all"))
        # 只有水平方向要能捲動（內容太寬），垂直方向不用——這個 Canvas 本身的
        # 高度直接回填成卡片的實際高度，垂直捲動交給外層整個分頁共用的那個
        # Canvas（_ScrollableTab）處理，這裡不重複做一層垂直捲動。
        canvas.configure(height=card.winfo_reqheight())
        # xview 沒變，但 scrollregion／canvas 尺寸可能變了，拉桿滑塊該佔的比例
        # 也要跟著變——xscrollcommand 理論上會自動通知 hscroll，但這裡不信任
        # 這個自動通知一定準時／一定在元件已經被畫到畫面上之後才發生，直接明講
        # 呼叫 hscroll.set() 同步一次。
        hscroll.set(*canvas.xview())

    def pin_left_and_resync():
        """把捲動位置強制撥回最左邊，再重新量測一次——只在分頁「即將被使用者
        看到」的那一刻呼叫（見 settings_window.py 的 `_select_tab`），不是每次
        <Configure> 都呼叫。這裡還有一個實測踩到的坑：分頁在還沒被選取顯示之前
        是 `pack_forget` 狀態（沒有被映射到畫面上），這段時間對 `hscroll` 呼叫
        `.set()` 更新的只是內部狀態，ttk（尤其 'clam' 佈景）不保證未映射的元件
        之後被顯示出來時會自動用最新狀態重新繪製滑塊——實測會出現拉桿滑塊卡在
        建構當下（那時版面還沒定案）算出來的錯誤比例／位置，一直沒有跟著之後
        真正的版面更新過來，使用者看到的滑塊視覺位置因此跟 `canvas.xview()`
        內部真正的值兜不起來。解法：分頁被選取、即將顯示的當下（此時元件保證
        會被映射到畫面上），重新做一次完整的量測＋歸零＋同步，確保這一輪一定
        是在元件會被畫出來的狀態下呼叫 `hscroll.set()`，視覺滑塊才會跟內部狀態
        對上。

        座標系統方向已經用注入的長填充文字＋`canvas.canvasx(0)` 實際量測驗證過
        （見 tab_docs_panel 診斷紀錄）：撥到 0.0 時畫面最左邊對到的 canvas 座標
        確實是 0.0（文字最開頭），撥到 0.25/0.5/0.75 依序線性往右移動，方向完全
        正常、不是反的——先前「反過來試試」只是撥到 1.0（文字結尾）當對照組，
        確認不是座標系統本身的問題，現在撥回 0.0（最左邊／文字開頭）。

        另外還踩到一個同類型的坑：11 個分頁在 `_build_tabs()` 建構時是一次全部
        建好、只留一個顯示（其餘 `pack_forget`），這條 `hscroll` 是在分頁還是
        隱藏狀態時建立、套上 `DocsPanel.Horizontal.TScrollbar` 樣式的。ttk 對
        「還沒被映射到畫面上的元件」不保證樣式一定有真的套色完成，實測會出現
        分頁真正顯示出來後，粉紅色配色沒有生效、看起來還是預設灰階（跟「滑塊
        位置沒同步」是同一種「隱藏時設定的狀態，顯示時沒有被重新套用」的成因，
        只是這次影響的是外觀顏色而不是位置）。

        解法一樣：分頁真正要顯示的這一刻（這裡，元件保證已映射），重新指定一次
        style，強制 ttk 重新套色。但單純「設回原本已經是的值」不夠可靠——Tk 的
        `configure()` 常見實作是「新值跟目前值一樣就不做事」，直接
        `hscroll.configure(style="DocsPanel.Horizontal.TScrollbar")`（本來就是
        這個值）很可能被判定成沒有變化而整個跳過，等於沒修到。改成先切到另一個
        樣式（"TScrollbar"，全域基礎樣式，一定跟目前值不同）、再切回來——確保
        兩次呼叫都是「值真的變了」，一定會觸發 Tk 的樣式引擎重新計算＋重繪，不會
        被短路跳過。"""
        canvas.xview_moveto(0.0)
        hscroll.configure(style="TScrollbar")
        hscroll.configure(style="DocsPanel.Horizontal.TScrollbar")
        _sync_scrollregion()

    # 三層都要綁：card 尺寸變（內容增減）、canvas 尺寸變（例如視窗縮放）都會觸發，
    # 但真正決定「右欄可視寬度」的其實是 outer——它從 right_col 拿到實際分配到
    # 的寬度，是最終、最準的來源。只綁 card/canvas 沒綁 outer 時，實測會出現
    # 分頁在還沒被使用者切換顯示過之前，canvas 量到的寬度是建構當下（可能還沒
    # 定案）的暫時值，等分頁真正顯示、outer 拿到最終寬度時卻沒有重新同步，
    # 拉桿比例就跟實際內容對不上，滑塊看起來偏移到中間。這三個綁定現在只做
    # 「量測＋回報」，不會再動到使用者的捲動位置。
    outer.bind("<Configure>", _sync_scrollregion)
    card.bind("<Configure>", _sync_scrollregion)
    canvas.bind("<Configure>", _sync_scrollregion)

    tk.Frame(card, bg=COLOR_TOOL_DESC_ACCENT, width=5).pack(side="left", fill="y")

    body = tk.Frame(card, bg=COLOR_TOOL_DESC_BG)
    body.pack(side="left", fill="both", expand=True, padx=(10, 10), pady=10)

    tk.Label(
        body, text="🧩 模組與核心函式（來源：設定分頁模組與核心函式對照表.md）",
        bg=COLOR_TOOL_DESC_BG, fg=COLOR_TOOL_DESC_ACCENT, font=window._font_label_bold,
        anchor="w", justify="left", wraplength=_WRAPLENGTH,
    ).pack(fill="x", pady=(0, 8))

    if not rows:
        tk.Label(
            body,
            text="⚠️ 找不到這個分頁的說明文件內容（設定分頁模組與核心函式對照表.md"
            " 可能缺少對應章節，或格式跟解析規則對不上，以程式碼為準）。",
            bg=COLOR_TOOL_DESC_BG, fg=COLOR_TOOL_DESC_FG, font=window._font_hint,
            anchor="w", justify="left", wraplength=_WRAPLENGTH,
        ).pack(fill="x")
        tk.Frame(body, bg=COLOR_TOOL_DESC_BG, height=_BOTTOM_SPACER_HEIGHT).pack(fill="x")
        return {"resync": pin_left_and_resync, "hscroll": hscroll}

    mono_font = (CONSOLE_FONT_FAMILY, 9)
    for row in rows:
        row_frame = tk.Frame(body, bg=COLOR_TOOL_DESC_BG)
        row_frame.pack(fill="x", pady=(0, 10))
        # 顯示順序跟 .md 表格欄位順序一致（模組檔案 → 核心函式 → 說明），兩邊
        # 對照時不用在腦中換順序。模組路徑放第一行（粗體），核心函式放第二行
        # （等寬字、跟 style.py 其他說明卡片一致的 accent 色），説明放第三行。
        _selectable_line(
            row_frame, row["module"], COLOR_TOOL_DESC_BG, COLOR_TOOL_DESC_FG,
            window._font_label_bold,
        )
        _selectable_line(
            row_frame, row["func"], COLOR_TOOL_DESC_BG, COLOR_TOOL_DESC_ACCENT, mono_font,
        )
        _selectable_line(
            row_frame, row["desc"], COLOR_TOOL_DESC_BG, COLOR_TOOL_DESC_FG,
            window._font_hint,
        )

    tk.Frame(body, bg=COLOR_TOOL_DESC_BG, height=_BOTTOM_SPACER_HEIGHT).pack(fill="x")
    return {"resync": pin_left_and_resync, "hscroll": hscroll}
