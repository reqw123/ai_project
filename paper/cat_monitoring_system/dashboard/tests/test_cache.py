"""dashboard/cache.py 的單元測試。

純記憶體邏輯，不依賴 Flask/cv2，任何環境都能跑。
"""

import threading

import pytest

from dashboard import cache


@pytest.fixture(autouse=True)
def _reset_cache():
    """每個測試前後都重置成初始狀態，避免測試互相汙染。"""
    cache.clear()
    yield
    cache.clear()


def test_get_latest_before_any_set_returns_not_yet_computed():
    result = cache.get_latest()
    assert result == {"status": "not_yet_computed"}


def test_set_then_get_returns_the_stored_payload_with_cached_at():
    cache.set_latest({"status": "ok", "fusion": {"score": 12.3, "level": "Normal"}})
    result = cache.get_latest()
    assert result["status"] == "ok"
    assert result["fusion"]["score"] == 12.3
    assert "cached_at" in result and result["cached_at"] is not None


def test_set_does_not_mutate_caller_dict_in_a_way_that_breaks_isolation():
    original = {"status": "ok", "fusion": {"score": 1}}
    cache.set_latest(original)
    result = cache.get_latest()
    # get_latest() 回傳的是帶了 cached_at 的複本，不應該連帶把 cached_at
    # 寫回呼叫端原本傳進來的那個 dict。
    assert "cached_at" not in original


def test_clear_resets_to_not_yet_computed():
    cache.set_latest({"status": "ok"})
    assert cache.get_latest()["status"] == "ok"
    cache.clear()
    assert cache.get_latest() == {"status": "not_yet_computed"}


def test_concurrent_set_and_get_does_not_raise():
    """基本的執行緒安全 smoke test：多執行緒同時讀寫不應該拋例外或 deadlock。"""
    errors = []

    def writer(i):
        try:
            for _ in range(50):
                cache.set_latest({"status": "ok", "fusion": {"score": i}})
        except Exception as e:  # pragma: no cover - 只是安全網
            errors.append(e)

    def reader():
        try:
            for _ in range(50):
                cache.get_latest()
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors
