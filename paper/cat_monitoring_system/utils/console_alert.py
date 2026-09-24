"""
終端機醒目警告框：0_dataset_collect.py 與 0_train_gcn.py 共用，避免重要警告被後續輸出洗掉。

只在真正的終端上色；設定視窗的輸出面板（pipe）不認得 ANSI 碼，只畫框線。
框內寬度以「中文算兩格」計算，右邊框線才對得齊；內容請避免 ≠、→、● 這類
East Asian Ambiguous 字元，它們在中文主控台可能佔兩格而讓框線歪掉。
"""
import os
import sys
import textwrap
import unicodedata


def _disp_width(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _wrap(s, limit):
    if _disp_width(s) <= limit:
        return [s]
    if s.isascii():   # 檔名清單：只在逗號後斷行，不把檔名切一半
        return textwrap.wrap(s, limit, break_on_hyphens=False, break_long_words=False)
    out, cur = [], ""
    for c in s:
        if _disp_width(cur + c) > limit:
            out.append(cur)
            cur = ""
        cur += c
    return out + [cur]


def alert_box(title, sections, footer="", width=78, pause=True):
    """印出獨立大框。sections = [(小標題, [內容行, ...]), ...]，區塊之間以虛線分隔；
    內容行傳空字串＝空一行。
    pause=True 時停下來等使用者按 Enter；會直接結束的程式（例如拒絕訓練）傳 False。"""
    color = sys.stdout.isatty()
    if color and os.name == "nt":
        os.system("")   # 讓 Windows 主控台開始解讀 ANSI 顏色碼
    edge = "\x1b[1;93m" if color else ""
    hot = "\x1b[1;97;41m" if color else ""
    reset = "\x1b[0m" if color else ""
    inner = width - 6

    def row(text="", style=""):
        pad = inner - _disp_width(text)
        print(f"{edge}##{reset} {style}{text}{' ' * pad}{reset} {edge}##{reset}")

    bar = f"{edge}{'#' * width}{reset}"
    print("\n" + bar)
    print(bar)
    row()
    for line in _wrap(f"!!  {title}  !!", inner):
        row(line, hot)
    row()
    for i, (heading, lines) in enumerate(sections):
        if i:
            row("-" * inner)
            row()
        row(f"* {heading}")
        for line in lines:
            for piece in _wrap(line, inner - 4):
                row("    " + piece)
        row()
    if footer:
        if sections:
            row("=" * inner)
            row()
        for piece in _wrap(footer, inner):
            row(piece)
        row()
    print(bar)
    print(bar + "\n", flush=True)
    if pause:
        input("  >>> 看完請按 Enter 繼續 <<< ")
