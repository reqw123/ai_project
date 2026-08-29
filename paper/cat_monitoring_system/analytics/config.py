"""個體化基線新引擎（``analytics/baseline.py``、``deviation.py``）內部可調參數
集中管理。

**跟主專案 ``paper/config.py`` 的關係**：這裡採用跟 ``plugins/lick_stage/config.py``
一樣的模式——獨立的、模組自己的 config，不搬進主 ``config.py``。理由：
主 ``config.py`` 管的是跨子系統的操作性設定（模型路徑、Flask/Node-RED
連線、CSV 輸出位置……），這裡管的是統計方法論本身的參數，每一個都跟一段
文獻依據或工程取捨綁在一起（見下方個別常數的註解），搬到別的檔案會把
數值跟它的依據拆開，對論文寫作跟日後校準都不利。完整文獻清單見
``paper/docs/個體化基線與異常偵測文獻來源.md``（下方簡稱「文獻清單」）。

**每個常數都標了下面四種標籤之一，誠實區分「這個數字為什麼是這樣」**：

- 【文獻支持】：有明確引用文獻直接支持這個數值本身（不只是方法方向）。
- 【與 Node-RED 原版數值一致】：純粹是移植時為了「基線不因搬遷而突然改變」
  刻意對齊 ``cat_health_v3_flow.json`` 舊版 JS 的既有數值，不是獨立校準過。
- 【工程慣例／專家設定，無直接文獻依據】：合理但沒有文獻直接校準這個具體
  數字，日後有真實診斷標記資料時應該用資料驅動方式重新校準（見文獻清單
  第三節 NEWS2／VAScat 的討論）。
- 【數值穩定用途，非統計方法論參數】：純粹是浮點數值計算的安全邊界（避免
  除以零、避免 log(0)、避免 float64 精度問題），不代表任何統計假設，不
  應該被當成「可以調整來改變判斷結果」的參數。

全部常數都開放環境變數覆寫（跟主專案 ``config.py``、``plugins/lick_stage/
config.py`` 同一套慣例），但**「數值穩定用途」那幾個不建議調整**——改了
不會讓判斷更準，只會讓數值計算在邊界情況下更容易出錯。
"""

# 2026-08-11：_env_float/_env_int 原本在這裡各自重複實作一份，跟
# plugins/lick_stage/config.py、plugins/lick_stage/ext_body_zones/config.py
# 幾乎一模一樣，改成共用 utils/env_parsing.py，見該檔案開頭說明。
from utils.env_parsing import env_float as _env_float, env_int as _env_int


class BaselineConfig:
    """``analytics/baseline.py``（個體化基線計算）用到的參數。"""

    # ── EWMA 平滑係數 ────────────────────────────────────────────────────
    # 【與 Node-RED 原版數值一致】對應 cat_health_v3_flow.json「個體化基線
    # 計算器」node 原本就用的 α=0.15，移植時刻意保持數值一致，讓既有基線
    # 不會因為搬遷到 Python 就突然改變（見 baseline.py 模組 docstring）。
    # 沒有獨立文獻校準過這個 0.15 本身；α 越小平滑越強、對近期變化反應越慢。
    EWMA_ALPHA = _env_float("CAT_MONITORING_ANALYTICS_EWMA_ALPHA", 0.15)

    # ── 基線天數門檻（納入計算所需的最少/最多天數） ─────────────────────
    # 【與 Node-RED 原版數值一致】MIN=7 對應 Node-RED 端 v2_user_settings.
    # baseline_day 的既有預設值（見 analytics_deviation_bridge.json「組成
    # /api/deviation 請求」節點的 fallback）。MAX=30【工程慣例，無直接文獻
    # 依據】——沒有找到特定文獻校準這個上限，只是「約一個月」的常識性選擇，
    # 避免基線視窗無限往回看、稀釋掉近期真正相關的行為模式。
    MIN_BASELINE_DAYS_DEFAULT = _env_int(
        "CAT_MONITORING_ANALYTICS_MIN_BASELINE_DAYS", 7, min_value=1
    )
    MAX_BASELINE_DAYS_DEFAULT = _env_int(
        "CAT_MONITORING_ANALYTICS_MAX_BASELINE_DAYS", 30, min_value=1
    )

    # ── 讀取歷史時的上限筆數（效能用途，非統計方法論參數） ───────────────
    # 2026-08-11：daily_history 這張表只會累積、從不刪除，system 跑越久
    # 資料越多，但 compute_baseline() 最終只會用到最近 MAX_BASELINE_DAYS_
    # DEFAULT 天——`analytics/daily_store.py.load_history()` 支援
    # limit_days 參數限制 SQL 讀取筆數，這裡設一個安全上限給呼叫端用（見
    # server/routes.py、dashboard/refresher.py、
    # analytics/manage_baseline_history.py），避免每次計算（背景排程每
    # ~2 秒一次）都不設限地整表掃描+建物件。
    # 刻意設成 MAX_BASELINE_DAYS_DEFAULT 的 4 倍（120 天）而不是剛好等於
    # MAX_BASELINE_DAYS_DEFAULT：compute_baseline() 的 max_days 截斷是在
    # 排除 excluded_dates／監測時數不足的天數「之後」才套用的，如果這裡
    # 限制得太緊，剛好卡在視窗邊界附近的排除天數可能導致算出來的有效天數
    # 比讀取完整歷史時更少——4 倍的緩衝在絕大多數實際使用情境下都足夠
    # （除非最近 120 天裡有超過 90 天被排除，那種極端情況本來就該考慮
    # 直接呼叫時不傳 limit_days 走完整歷史）。
    HISTORY_LOAD_LIMIT_DAYS = _env_int(
        "CAT_MONITORING_ANALYTICS_HISTORY_LOAD_LIMIT_DAYS",
        MAX_BASELINE_DAYS_DEFAULT * 4,
        min_value=1,
    )

    # ── 單日資料有效門檻（納入/排除標準：監測時數太短的天數視為不可靠） ──
    # 【工程慣例，無直接文獻依據】1 小時是「至少要看到有意義的一段行為樣本」
    # 的常識性下限，不是從某篇文獻推導出來的數字。日後若監測排程模式改變
    # （例如改成每天只排程監測 2 小時），這個門檻可能需要跟著調整，不然
    # 「有效天數」會系統性偏少。
    MIN_DAILY_MONITORING_SEC_DEFAULT = _env_float(
        "CAT_MONITORING_ANALYTICS_MIN_DAILY_MONITORING_SEC", 3600.0, min_value=0.0
    )

    # ── Sanity Check 群體參考值 ──────────────────────────────────────────
    # 【文獻支持：Eckstein & Hart, 2000, Applied Animal Behaviour Science
    # 68(2), 131-140】11 隻無體外寄生蟲家貓的錄影時間預算研究：健康貓每日
    # 舔舐約佔非睡眠時間 8%（約 60 min/day），抓撓約佔 0.2%（約 1 min/day），
    # 兩者比例約 50:1。用來檢查個體基線是否離群體常態太遠（可能是監控時間
    # 不足、模型辨識偏誤，而不是貓真的異常）。下限（0.5 倍）跟上限（6 倍）
    # 本身是圍繞這篇文獻參考值訂出的寬鬆容許帶，不是文獻直接給的門檻值。
    LICK_SANITY_UPPER_SEC = _env_float(
        "CAT_MONITORING_ANALYTICS_LICK_SANITY_UPPER_SEC", 6 * 3600.0
    )
    LICK_SANITY_LOWER_SEC = _env_float(
        "CAT_MONITORING_ANALYTICS_LICK_SANITY_LOWER_SEC", 0.5 * 3600.0
    )
    SCRATCH_SANITY_UPPER_SEC = _env_float(
        "CAT_MONITORING_ANALYTICS_SCRATCH_SANITY_UPPER_SEC", 10 * 60.0
    )


class DeviationConfig:
    """``analytics/deviation.py``（今日 vs 基線偏差評分）用到的參數。"""

    # ── MAD → 穩健 z-score 的常態一致性常數 ──────────────────────────────
    # 【文獻支持：Iglewicz & Hoaglin, 1993, "How to Detect and Handle
    # Outliers", ASQC Quality Press】modified z-score 的原始出處，1.4826 是
    # 讓 MAD 在常態分布假設下跟標準差同尺度的固定換算常數（= 1/Φ⁻¹(0.75)），
    # 不是可以自由調整的「門檻」——這是數學推導出來的定值，改了這個數字
    # sigma_equivalent 就不再對應真正的常態機率意義，不建議調整。
    MAD_SCALE = _env_float("CAT_MONITORING_ANALYTICS_MAD_SCALE", 1.4826)

    # 【數值穩定用途，非統計方法論參數】MAD 幾乎為零（連續多天數值完全相同）
    # 時，除以 scaled_mad 會爆成無限大或除以零錯誤；低於這個門檻直接判定
    # 「基線本身沒有波動、無法算穩健 z-score」，回傳 None 而非錯誤數值。
    MIN_VARIABILITY_MAD = _env_float(
        "CAT_MONITORING_ANALYTICS_MIN_VARIABILITY_MAD", 1e-9
    )

    # ── 稀疏計數指標：Poisson vs. 負二項（NB）模型選擇門檻 ────────────────
    # 【方向上有文獻支持，具體切點無直接校準】Lindén & Mäntyniemi (2011,
    # Ecology 92(7))：生態學計數資料過度離散時，NB 應作為預設起始分布；
    # Höhle & Paul (2008) 的 GLR-NB 監控管制圖也是「Poisson 不足時退回 NB」
    # 的直接方法論依據（deviation.py 的 _fit_count_model 正是這個邏輯的
    # 實作）。但「variance/mean 比值超過 1.5 才切換」這個具體切點是工程
    # 選擇，文獻沒有給出這個確切數字——概念上 variance/mean=1 是 Poisson
    # 的定義性質（equidispersion），1.5 是留一點容忍度避免抽樣雜訊讓正常
    # 的 Poisson 資料被誤判為過度離散。
    COUNT_OVERDISPERSION_RATIO = _env_float(
        "CAT_MONITORING_ANALYTICS_COUNT_OVERDISPERSION_RATIO", 1.5
    )

    # 【數值穩定用途，非統計方法論參數】對每日計數歷史的平均發生率做輕量的
    # 加法平滑（類似 Bayesian 的作法，但不是完整的共軛先驗推導），避免歷史
    # 視窗剛好全部是 0 次時，算出 λ=0，導致尾機率模型退化成除以零/無定義。
    RATE_SMOOTHING_PRIOR = _env_float(
        "CAT_MONITORING_ANALYTICS_RATE_SMOOTHING_PRIOR", 0.25
    )

    # 【數值穩定用途，非統計方法論參數】尾機率轉換成 sigma_equivalent 前的
    # 下限保護。這個值刻意選在比 float64 在 1.0 附近的機器精度（約 2.22e-16）
    # 更寬鬆的範圍：太小（例如過去踩過的 1e-300）會讓 `1.0 - p` 在浮點數
    # 運算下直接等於 1.0，導致反函數算出 inf，Flask 的 JSON 編碼器會把它
    # 序列化成不合法的 "Infinity" 字面值，讓 Node-RED 的 JSON parser 出錯
    # （見 deviation.py 對應註解，這是實際重現過的 bug，不是理論推測）。
    MIN_TAIL_P = _env_float("CAT_MONITORING_ANALYTICS_MIN_TAIL_P", 1e-15)
