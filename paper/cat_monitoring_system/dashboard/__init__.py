"""
個體化基線儀表板（Python 端唯讀展示頁）。

只在 config.BaselineDashboardConfig.ENABLED 為 True 時才會被匯入——見
server/flask_app.py 的條件式 import。這個套件本身不對 analytics/ 的內部
結構做任何假設之外的耦合：cache.py 只搬移 dict，views.py 只讀 cache 跟
呼叫 analytics 的公開函式，refresher.py 負責定期觸發計算並寫回 cache
（跟 Node-RED 的 POST /api/deviation 是兩條獨立的觸發路徑，見
refresher.py 開頭說明），彼此都可以獨立測試。
"""
