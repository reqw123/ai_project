"""analytics/config.py 的單元測試。

跟 paper/tests/test_config_unit.py（主專案 config.py 的 _env_* 測試）同一個
策略：`BaselineConfig`/`DeviationConfig` 的屬性是模組載入當下就從環境變數
解析好的 class 屬性，事後用 monkeypatch.setenv() 改環境變數不會回頭影響
已經算好的屬性值——所以「env 覆寫真的有效」這件事，測的是 config.py 內部
的 `_env_float`/`_env_int` helper 函式本身，不是重新匯入整個模組後比對屬性。

刻意不用 importlib.reload(analytics.baseline)：reload 會讓
`analytics.baseline.DailyRecord`/`InsufficientDataError` 變成新的 class
物件，但 server/routes.py、analytics/daily_store.py 等模組在 pytest session
一開始就 import 過的是舊物件，`except InsufficientDataError` 之類的比對
會失敗——實際測過，這樣做會讓同一個 process 內其他測試檔案的
`/api/deviation`／`dashboard/refresher.py` 測試無故壞掉，是真的踩過的坑，
不是理論上的疑慮。
"""

from analytics.config import (
    BaselineConfig,
    DeviationConfig,
    _env_float,
    _env_int,
)

_ENV_NAME = "CAT_MONITORING_TEST_ANALYTICS_UNIT_VAR"  # 測試專用，不影響任何正式設定


def test_baseline_config_defaults_match_pre_migration_values():
    """鎖住搬遷前（原本寫在 baseline.py/deviation.py 模組層級）的原始數值，
    任何人不小心手滑改動都會被這個測試抓到。"""
    assert BaselineConfig.EWMA_ALPHA == 0.15
    assert BaselineConfig.MIN_BASELINE_DAYS_DEFAULT == 7
    assert BaselineConfig.MAX_BASELINE_DAYS_DEFAULT == 30
    assert BaselineConfig.MIN_DAILY_MONITORING_SEC_DEFAULT == 3600.0
    assert BaselineConfig.LICK_SANITY_UPPER_SEC == 6 * 3600.0
    assert BaselineConfig.LICK_SANITY_LOWER_SEC == 0.5 * 3600.0
    assert BaselineConfig.SCRATCH_SANITY_UPPER_SEC == 10 * 60.0


def test_deviation_config_defaults_match_pre_migration_values():
    assert DeviationConfig.MAD_SCALE == 1.4826
    assert DeviationConfig.MIN_VARIABILITY_MAD == 1e-9
    assert DeviationConfig.COUNT_OVERDISPERSION_RATIO == 1.5
    assert DeviationConfig.RATE_SMOOTHING_PRIOR == 0.25
    assert DeviationConfig.MIN_TAIL_P == 1e-15


def test_baseline_and_deviation_use_config_values_not_independent_copies():
    """baseline.py/deviation.py 向下相容用的模組層級別名（例如
    baseline.EWMA_ALPHA）必須真的引用 config.py 的同一個值，不是各自
    維護一份可能漂移的複本。"""
    from analytics import baseline, deviation

    assert baseline.EWMA_ALPHA == BaselineConfig.EWMA_ALPHA
    assert baseline.MIN_BASELINE_DAYS_DEFAULT == BaselineConfig.MIN_BASELINE_DAYS_DEFAULT
    assert baseline.MAX_BASELINE_DAYS_DEFAULT == BaselineConfig.MAX_BASELINE_DAYS_DEFAULT
    assert deviation.MAD_SCALE == DeviationConfig.MAD_SCALE
    assert deviation.COUNT_OVERDISPERSION_RATIO == DeviationConfig.COUNT_OVERDISPERSION_RATIO


class TestEnvFloat:
    def test_unset_variable_returns_default(self, monkeypatch):
        monkeypatch.delenv(_ENV_NAME, raising=False)
        assert _env_float(_ENV_NAME, 1.5) == 1.5

    def test_set_variable_returns_parsed_value(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, "0.3")
        assert _env_float(_ENV_NAME, 0.15) == 0.3

    def test_invalid_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, "not_a_number")
        assert _env_float(_ENV_NAME, 0.15) == 0.15


class TestEnvInt:
    def test_unset_variable_returns_default(self, monkeypatch):
        monkeypatch.delenv(_ENV_NAME, raising=False)
        assert _env_int(_ENV_NAME, 7) == 7

    def test_set_variable_returns_parsed_value(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, "14")
        assert _env_int(_ENV_NAME, 7) == 14

    def test_invalid_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, "not_a_number")
        assert _env_int(_ENV_NAME, 7) == 7

    def test_below_min_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, "0")
        assert _env_int(_ENV_NAME, 7, min_value=1) == 7
        monkeypatch.setenv(_ENV_NAME, "-3")
        assert _env_int(_ENV_NAME, 7, min_value=1) == 7

    def test_within_range_returns_parsed_value(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, "14")
        assert _env_int(_ENV_NAME, 7, min_value=1, max_value=90) == 14


class TestBaselineDaysGuard:
    def test_min_max_days_clamped_to_at_least_one(self):
        """compute_baseline 收到 0 / 負數的 min_days / max_days 時要夾成 1，
        而不是讓 valid[-0:] 靜默回傳整個歷史。"""
        from datetime import date, timedelta

        from analytics.baseline import DailyRecord, compute_baseline

        hist = [
            DailyRecord(day=date(2026, 1, 1) + timedelta(days=i), monitoring_seconds=7200.0)
            for i in range(10)
        ]
        bl = compute_baseline(hist, min_days=0, max_days=0)
        assert bl.days_count == 1
        assert bl.required_days == 1
