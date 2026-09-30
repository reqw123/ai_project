"""獨立工具腳本的報告輸出位置（CSV／圖表／摘要）統一定義在這裡。

所有報告都放在 paper/reports/ 底下，按用途分成兩層：

    paper/reports/
    ├── eval/        模型評估與比較（ST-GCN、YOLO-Pose、EMA、光照分布…）
    ├── analysis/    姿態／關鍵點／推論行為的量測分析
    ├── identity/    個體辨識的驗證與推論
    ├── review/      人工複審的工作紀錄
    └── verify/      外掛模組在真實影片上的驗證產物

第三層是各工具自己的資料夾（例如 eval/gcn_compare/），工具在裡面再用
流水號或時間戳記分出每次執行的子資料夾。

不屬於「報告」、所以不放這裡的：跟著資料走的紀錄（skeletons/ 的
_split_moves_log.csv、影片搬移紀錄）、訓練產物（stgcn_models/run_*/）、
主系統執行時的紀錄（paper/logs、cat_monitoring_log.csv）、
下載／蒐集影片時一併產生的清單。

用法（腳本放在 tools/ 底下，執行時 tools/ 自然在 sys.path 第一位）：

    from _report_paths import report_dir
    OUTPUT_DIR = report_dir("eval", "gcn_compare")
"""

from pathlib import Path

REPORTS_ROOT = Path(__file__).resolve().parents[1] / "reports"

CATEGORIES = ("eval", "analysis", "identity", "review", "verify")


def report_dir(category: str, tool: str) -> Path:
    """回傳 reports/<category>/<tool>/ 的路徑（不會建立資料夾，由呼叫端需要時再 mkdir）。"""
    if category not in CATEGORIES:
        raise ValueError(f"未知的報告分類 {category!r}，可用：{', '.join(CATEGORIES)}")
    return REPORTS_ROOT / category / tool
