"""paper/tests 共用的 Tk 測試輔助函式。

原本放在 conftest.py、由測試檔 `from conftest import make_tk_root` 取用；但專案裡有好幾個
tests 資料夾各有自己的 conftest.py，一次跑整個 paper/ 時 Python 只會留一個叫 `conftest`
的模組，另一個資料夾的 conftest 先載入就會讓這裡 import 失敗（收集錯誤）。改放在名稱不重複
的模組，避開這個衝突。
"""

import pytest


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
