"""`KeypointDetector.detect(single_cat=...)` 純單元測試——不需真模型 / GPU。

驗證兩件事：
  1. single_cat=True  → model.predict(max_det=1)；single_cat=False → max_det=300。
  2. single_cat=True 完全略過跨幀 IoU 追蹤延續：即使 _prev_bbox 已鎖定、
     這一幀有兩隻貓且其中一隻跟上一幀高度重疊，single_cat=True 仍然回傳
     「信心值最高」的那一隻；single_cat=False 才會延續追蹤到重疊的那一隻。
"""

import numpy as np
import pytest

import detectors.keypoint_detector as kd_mod
from detectors.keypoint_detector import KeypointDetector


class _Tensorish:
    """最小仿 torch tensor：支援 len()、索引、.cpu().numpy()。"""

    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float32)

    def __len__(self):
        return len(self._arr)

    def __getitem__(self, idx):
        return _Tensorish(self._arr[idx])

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _Keypoints:
    def __init__(self, xy, conf):
        self.xy = _Tensorish(xy)
        self.conf = _Tensorish(conf)


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = _Tensorish(xyxy)
        self.conf = _Tensorish(conf)

    def __len__(self):
        return len(self.xyxy)


class _Results:
    def __init__(self, xy, kconf, xyxy, bconf):
        self.keypoints = _Keypoints(xy, kconf)
        self.boxes = _Boxes(xyxy, bconf)


class _FakeModel:
    """記錄每次 predict 的 kwargs，回傳預先塞好的 _Results。"""

    def __init__(self, results):
        self._results = results
        self.predict_calls = []

    def to(self, *_a, **_kw):
        return self

    def predict(self, frame, **kwargs):
        self.predict_calls.append(kwargs)
        return [self._results]


def _two_cat_results():
    """兩隻貓：index 0 信心 0.95、index 1 信心 0.60。"""
    xy = np.zeros((2, 17, 2), dtype=np.float32)
    kconf = np.ones((2, 17), dtype=np.float32)
    xyxy = np.array(
        [[0.0, 0.0, 10.0, 10.0],      # 高信心，遠離 _prev_bbox
         [100.0, 100.0, 110.0, 110.0]],  # 低信心，緊貼 _prev_bbox
        dtype=np.float32,
    )
    bconf = np.array([0.95, 0.60], dtype=np.float32)
    return _Results(xy, kconf, xyxy, bconf)


@pytest.fixture
def make_detector(monkeypatch, tmp_path):
    def _make(results):
        fake = _FakeModel(results)
        monkeypatch.setattr(kd_mod, "YOLO", lambda *_a, **_kw: fake)
        model_file = tmp_path / "fake.pt"
        model_file.write_bytes(b"")
        det = KeypointDetector(str(model_file), device="cpu")
        return det, fake

    return _make


def test_single_cat_true_sets_max_det_1(make_detector):
    det, fake = make_detector(_two_cat_results())
    det.detect(np.zeros((16, 16, 3), dtype=np.uint8), single_cat=True)
    assert fake.predict_calls[-1]["max_det"] == 1


def test_single_cat_false_sets_max_det_300(make_detector):
    det, fake = make_detector(_two_cat_results())
    det.detect(np.zeros((16, 16, 3), dtype=np.uint8), single_cat=False)
    assert fake.predict_calls[-1]["max_det"] == 300


def test_single_cat_true_ignores_iou_tracking(make_detector):
    det, _ = make_detector(_two_cat_results())
    # 假裝上一幀鎖定的是「緊貼 index 1」的框
    det._prev_bbox = np.array([100.0, 100.0, 110.0, 110.0], dtype=np.float32)
    _kpts, _kconf, bbox, bbox_conf = det.detect(
        np.zeros((16, 16, 3), dtype=np.uint8), single_cat=True
    )
    # single_cat 時無視追蹤，回傳信心最高的 index 0
    assert bbox_conf == pytest.approx(0.95)
    assert bbox[0] == pytest.approx(0.0)


def test_multi_cat_follows_iou_tracking(make_detector):
    det, _ = make_detector(_two_cat_results())
    det._prev_bbox = np.array([100.0, 100.0, 110.0, 110.0], dtype=np.float32)
    _kpts, _kconf, bbox, bbox_conf = det.detect(
        np.zeros((16, 16, 3), dtype=np.uint8), single_cat=False
    )
    # 多貓模式：延續追蹤到緊貼 _prev_bbox 的 index 1（雖然信心較低）
    assert bbox_conf == pytest.approx(0.60)
    assert bbox[0] == pytest.approx(100.0)
