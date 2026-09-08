"""`python -m iot` 進入點。

從 ``paper/cat_monitoring_system/`` 執行：

    cd paper/cat_monitoring_system
    python -m iot

跟 ``main.py`` 是兩個完全獨立的行程，互不啟動對方。
"""

from iot.runner import run

if __name__ == "__main__":
    run()
