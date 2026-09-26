"""Ensure `paper/`（config.py 所在目錄）在 sys.path 上。"""

import sys
from pathlib import Path

_paper_dir = Path(__file__).resolve().parents[1]
if str(_paper_dir) not in sys.path:
    sys.path.insert(0, str(_paper_dir))



import pytest  # noqa: E402


def make_tk_root():
    """建立一個隱藏的 Tk 根視窗給測試用；沒有顯示環境時 pytest.skip。

    Windows 上同一個 pytest 行程裡「前一個測試檔銷毀 Tk 後，下一個測試檔再建立」偶爾會
    失敗（invalid command name "tcl_findLibrary"／Tcl wasn't installed properly），
    整檔被當成沒有顯示環境而略過。實測失敗是暫時性的：回收殘留物件後重試一次即可。
    刻意不做成 session 共用的根視窗——那會變成 tkinter 的預設 root，其他測試檔沒指定
    master 的 tk.StringVar() 會綁到它身上而壞掉。"""
    import gc
    import tkinter as tk
    last_error = None
    for _ in range(3):
        try:
            root = tk.Tk()
            root.withdraw()
            return root
        except tk.TclError as e:
            last_error = e
            gc.collect()
    pytest.skip(f"沒有可用的顯示環境：{last_error}")
