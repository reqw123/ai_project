"""
Unit Test：tools/_playback_bar.py（影片視窗底部的時間拉桿、HH:MM:SS、倍速節流時鐘、快轉讀幀器）

1_run_video_inference.py 的視窗測試模式靠這個模組：時間格式、終端輸入的時間解析、倍速階梯、
依影片時間節流（推論跟不上時不能爆衝追進度；快轉時要追上倍速）、滑鼠拖曳拉桿「放開才跳轉」、
快轉時用 PyAV 跳幀（不能衝過頭）。
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import _playback_bar as pb  # noqa: E402


@pytest.mark.parametrize("sec, text", [
    (0, "00:00:00"),
    (59.99, "00:00:59"),
    (61, "00:01:01"),
    (3600, "01:00:00"),
    (3725.5, "01:02:05"),
    (100 * 3600 + 1, "100:00:01"),
    (-3, "00:00:00"),
])
def test_fmt_hms(sec, text):
    assert pb.fmt_hms(sec) == text


@pytest.mark.parametrize("raw, sec", [
    ("95.5", 95.5),
    ("1:35", 95.0),
    ("1:02:03", 3723.0),
    ("0:00:10.5", 10.5),
    (" 12 ", 12.0),
])
def test_parse_time_input_valid(raw, sec):
    assert pb.parse_time_input(raw) == pytest.approx(sec)


@pytest.mark.parametrize("raw", ["", "abc", "1:2:3:4", "-5", "1:-2"])
def test_parse_time_input_invalid(raw):
    assert pb.parse_time_input(raw) is None


def test_step_speed_walks_ladder_and_clamps():
    assert pb.step_speed(1.0, +1) == 1.5
    assert pb.step_speed(1.0, -1) == 0.5
    assert pb.step_speed(pb.SPEED_STEPS[0], -1) == pb.SPEED_STEPS[0]
    assert pb.step_speed(8.0, +1) == 16.0
    assert pb.SPEED_STEPS[-1] >= 256          # 6 小時影片要能在幾分鐘內快轉完
    assert pb.step_speed(pb.SPEED_STEPS[-1], +1) == pb.SPEED_STEPS[-1]
    assert pb.step_speed(3.0, +1) in pb.SPEED_STEPS  # 不在階梯上的值先對齊再走
    assert pb.speed_label(2.0) == "2x" and pb.speed_label(0.25) == "0.25x" and pb.speed_label(128.0) == "128x"


def test_clock_paces_by_video_time():
    clk = pb.PlaybackClock()
    assert clk.delay_ms(0.0, 1.0, now=100.0) == 1            # 第一幀：設錨點
    # 30fps 1x：下一幀該在 33ms 後出現，已經過 10ms → 再等 ~23ms
    assert clk.delay_ms(1 / 30, 1.0, now=100.010) == pytest.approx(23, abs=1)
    # 2x：同一段影片時間只要一半牆鐘時間
    clk2 = pb.PlaybackClock()
    clk2.delay_ms(0.0, 2.0, now=0.0)
    assert clk2.delay_ms(1.0, 2.0, now=0.1) == pytest.approx(400, abs=1)


def test_clock_does_not_burst_when_behind_and_reanchors_on_seek():
    clk = pb.PlaybackClock()
    clk.delay_ms(0.0, 1.0, now=0.0)
    # 推論太慢：0.1 秒影片花了 1 秒 → 落後很多，不等、重設錨點
    assert clk.delay_ms(0.1, 1.0, now=1.0) == 1
    # 重設後依新錨點節流（不會為了追回落後的 0.9 秒連續狂播）
    assert clk.delay_ms(0.1 + 1 / 30, 1.0, now=1.0) == pytest.approx(33, abs=1)
    # 往前跳 10 分鐘（seek）：不會等 10 分鐘
    assert clk.delay_ms(600.0, 1.0, now=1.05) == 1
    # 往回跳（循環播放、往回 seek）
    assert clk.delay_ms(5.0, 1.0, now=1.10) == 1


def test_due_frame_lets_fast_forward_catch_up():
    clk = pb.PlaybackClock()
    clk.delay_ms(0.0, 128.0, now=0.0, catch_up=True)
    # 每顯示一幀花 50ms：128x 下「現在該播到」的位置是 0.05*128 = 6.4 秒 → 第 192 幀（30fps）
    assert clk.due_frame(30.0, now=0.05) == pytest.approx(192, abs=0.5)
    # 跳過去之後落後很多也不重設錨點（catch_up），下一次照樣從錨點算，seek 花的時間也追得回來
    clk.delay_ms(6.4, 128.0, now=0.3, catch_up=True)
    assert clk.due_frame(30.0, now=0.35) == pytest.approx(0.35 * 128 * 30, abs=0.5)


def test_due_frame_none_without_anchor_or_when_stale():
    clk = pb.PlaybackClock()
    assert clk.due_frame(30.0, now=0.0) is None
    clk.delay_ms(10.0, 4.0, now=0.0)
    assert clk.due_frame(30.0, now=0.1) is not None
    assert clk.due_frame(30.0, now=clk.STALE_SEC + 1) is None   # 暫停很久沒 reset 也不會一口氣快轉過去
    clk.reset()
    assert clk.due_frame(30.0, now=0.1) is None


def test_catch_up_overshoot_is_paid_back():
    # 快轉位置比預期多跳 2 秒：不能當成使用者跳轉而重設錨點，要等時間追上來
    clk = pb.PlaybackClock()
    clk.delay_ms(0.0, 8.0, now=0.0, catch_up=True)
    assert clk.delay_ms(0.8 + 2.0, 8.0, now=0.1, catch_up=True) == pytest.approx(250, abs=2)
    assert clk.due_frame(30.0, now=0.1) == pytest.approx(24, abs=0.5)


def test_clock_actual_speed():
    clk = pb.PlaybackClock()
    t = 0.0
    for i in range(60):          # 每幀 1/30 秒影片、花 1/15 秒牆鐘 → 實際 0.5x
        clk.delay_ms(i / 30, 4.0, now=t)
        t += 1 / 15
    assert clk.actual_speed == pytest.approx(0.5, rel=0.02)
    clk.delay_ms(500.0, 4.0, now=t)   # seek 之後先不顯示舊的實際倍速
    assert clk.actual_speed is None


def test_clock_actual_speed_high_ratio():
    clk = pb.PlaybackClock()
    t = 0.0
    for i in range(60):          # 每顯示一幀跳 6.4 秒影片、花 50ms → 實際 128x，不能被當成 seek 而清掉
        clk.delay_ms(i * 6.4, 128.0, now=t, catch_up=True)
        t += 0.05
    assert clk.actual_speed == pytest.approx(128, rel=0.02)


def _rendered_bar(total=3000, cur=0, width=1280, y_offset=600, fps=30.0):
    bar = pb.SeekBar()
    img = bar.render(width, y_offset=y_offset, frame_idx=cur, total_frames=total, fps=fps,
                     paused=False, speed=1.0)
    return bar, img


def test_render_band_size_and_track_geometry():
    bar, img = _rendered_bar()
    assert img.shape == (pb.band_height(1280), 1280, 3)
    x0, w = bar._track
    assert x0 > 0 and x0 + w < 1280


def test_render_fast_forward_state():
    bar = pb.SeekBar()
    img = bar.render(1280, y_offset=600, frame_idx=100, total_frames=3000, fps=30.0, paused=False,
                     speed=32.0, actual_speed=31.0, fast_forward=True)
    assert img.shape == (pb.band_height(1280), 1280, 3)


def test_drag_previews_and_commits_on_release_only():
    bar, _ = _rendered_bar(total=3001)
    x0, w = bar._track
    y = 600 + 20
    bar.on_mouse(cv2.EVENT_LBUTTONDOWN, x0 + w // 4, y, 0)
    assert bar.dragging and bar.take_pending() is None       # 按下：只預覽
    bar.on_mouse(cv2.EVENT_MOUSEMOVE, x0 + w // 2, y + 200, 0)  # 拖曳中滑鼠跑出拉桿也照樣跟
    assert bar.drag_frame == pytest.approx(1500, abs=2) and bar.take_pending() is None
    bar.on_mouse(cv2.EVENT_LBUTTONUP, x0 + w // 2, y + 200, 0)
    assert not bar.dragging
    assert bar.take_pending() == pytest.approx(1500, abs=2)
    assert bar.take_pending() is None                          # 取走就清空


def test_click_outside_bar_and_clamping():
    bar, _ = _rendered_bar(total=101)
    x0, w = bar._track
    bar.on_mouse(cv2.EVENT_LBUTTONDOWN, x0 + 10, 100, 0)       # 點在影片畫面上（y 不在拉桿）
    assert not bar.dragging
    bar.on_mouse(cv2.EVENT_LBUTTONDOWN, 1, 610, 0)             # 軌道左邊的空白也算點到，夾到第 0 幀
    bar.on_mouse(cv2.EVENT_LBUTTONUP, 1, 610, 0)
    assert bar.take_pending() == 0
    bar.on_mouse(cv2.EVENT_LBUTTONDOWN, 1279, 610, 0)
    bar.on_mouse(cv2.EVENT_LBUTTONUP, 1279, 610, 0)
    assert bar.take_pending() == 100


def test_wheel_steps_seconds_and_accumulates():
    bar, _ = _rendered_bar(total=3000, cur=300, fps=30.0)
    up = 120 << 16
    bar.on_mouse(cv2.EVENT_MOUSEWHEEL, 640, 615, up)
    bar.on_mouse(cv2.EVENT_MOUSEWHEEL, 640, 615, up)
    assert bar.take_pending() == 300 + 2 * int(pb.WHEEL_STEP_SEC * 30)
    down = (-120 & 0xFFFF) << 16
    bar.on_mouse(cv2.EVENT_MOUSEWHEEL, 640, 615, down)
    assert bar.take_pending() == 300 - int(pb.WHEEL_STEP_SEC * 30)


def test_stream_without_length_is_not_seekable():
    bar = pb.SeekBar()
    img = bar.render(1280, y_offset=600, frame_idx=42, total_frames=0, fps=30.0, paused=True, speed=1.0)
    assert img.shape[1] == 1280
    bar.on_mouse(cv2.EVENT_LBUTTONDOWN, 640, 610, 0)
    bar.on_mouse(cv2.EVENT_LBUTTONUP, 640, 610, 0)
    assert bar.take_pending() is None


def test_cancel_clears_pending_drag():
    bar, _ = _rendered_bar()
    bar.on_mouse(cv2.EVENT_LBUTTONDOWN, 640, 610, 0)
    bar.cancel()
    bar.on_mouse(cv2.EVENT_LBUTTONUP, 640, 610, 0)
    assert not bar.dragging and bar.take_pending() is None


# ── 快轉讀幀器（PyAV）：用合成影片確認跳轉落點與順序讀幀 ──
av = pytest.importorskip("av")


def _code_img(i):
    """幀號編碼成左右兩塊亮度（間距 ≥12，轉 YUV 再轉回來也分得出來）。"""
    img = np.empty((96, 160, 3), np.uint8)
    img[:, :80] = (i // 16) * 12
    img[:, 80:] = (i % 16) * 15
    return img


def _patch_means(img):
    return float(img[:, :80].mean()), float(img[:, 80:].mean())


@pytest.fixture(scope="module")
def synth_video(tmp_path_factory):
    """300 幀、30fps、每 30 幀一個關鍵幀；回傳 (路徑, 照順序完整解碼得到的每幀左右亮度＝標準答案)。"""
    path = str(tmp_path_factory.mktemp("ff") / "synth.mp4")
    with av.open(path, "w") as out:
        st = out.add_stream("libx264", rate=30)
        st.width, st.height, st.pix_fmt = 160, 96, "yuv420p"
        st.options = {"g": "30", "keyint_min": "30", "sc_threshold": "0", "crf": "0"}
        for i in range(300):
            for pkt in st.encode(av.VideoFrame.from_ndarray(_code_img(i), format="bgr24")):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    with av.open(path) as c:
        truth = [_patch_means(f.to_ndarray(format="bgr24")) for f in c.decode(c.streams.video[0])]
    assert len(truth) == 300
    return path, truth


def _is_frame(img, idx, truth):
    return np.allclose(_patch_means(img), truth[idx], atol=2.0)


def test_ff_reader_far_jump_lands_on_keyframe_ahead(synth_video):
    path, truth = synth_video
    r = pb.FastForwardReader(path, 30.0, expected_hw=(96, 160))
    idx, img = r.jump(200, current=10)
    assert idx == 180 and r.active                       # 目標前最近的關鍵幀（每 30 幀一個），不必精確
    assert _is_frame(img, 180, truth)
    nxt_idx, nxt_img = r.next()                          # 接著照順序讀
    assert nxt_idx == 181 and _is_frame(nxt_img, 181, truth)
    r.close()


def test_ff_reader_short_and_long_jumps(synth_video):
    path, truth = synth_video
    r = pb.FastForwardReader(path, 30.0, expected_hw=(96, 160))
    r.jump(60, current=0)                                # 落在關鍵幀 60
    idx, img = r.jump(66, current=61)                    # 近的：照順序往下解到目標
    assert idx == 66 and _is_frame(img, 66, truth)
    idx, img = r.jump(86, current=67)                    # 20 幀：照順序解到目標
    assert idx == 86 and _is_frame(img, 86, truth)
    # 遠的跳轉，但目標前的關鍵幀（90）在目前位置之前 → 從 90 解到目標，不能衝到後面的關鍵幀 120
    r.KEYFRAME_MIN_JUMP = 8                               # 合成影片關鍵幀間隔短，調小門檻才測得到這個分支
    r.gop = None
    idx, img = r.jump(118, current=95)
    assert idx == 118 and _is_frame(img, 118, truth)
    nxt_idx, nxt_img = r.next()
    assert nxt_idx == 119 and _is_frame(nxt_img, 119, truth)
    r.close()


def test_ff_reader_never_overshoots_on_first_jump(synth_video):
    # 使用者回報：關鍵幀每 10 秒一個的影片，剛進快轉就從 ~1 秒衝到 10 秒。第一次跳轉（還沒在快轉狀態）
    # 目標之前的關鍵幀在目前位置之前時，要從關鍵幀解到目標，不能停在目標之後的關鍵幀
    path, truth = synth_video
    r = pb.FastForwardReader(path, 30.0, expected_hw=(96, 160))
    idx, img = r.jump(20, current=10)                    # 關鍵幀在 0、30：答案是 20，不是 30
    assert idx == 20 and _is_frame(img, 20, truth)
    r.close()


def test_ff_reader_learns_gop_and_decodes_forward_instead_of_seeking(synth_video):
    path, truth = synth_video
    r = pb.FastForwardReader(path, 30.0, expected_hw=(96, 160))
    r.KEYFRAME_MIN_JUMP = 8
    r.jump(5, current=0)
    for _ in range(70):                                  # 照順序讀過關鍵幀 30、60 → 量到間隔 30
        r.next()
    assert r.gop == 30
    seeks = []
    orig = r._seek_keyframe
    r._seek_keyframe = lambda t: (seeks.append(t), orig(t))[1]
    cur = r._last_idx + 1
    idx, img = r.jump(cur + 25, current=cur)             # 比關鍵幀間隔短：照順序解，不 seek
    assert idx == cur + 25 and _is_frame(img, idx, truth) and seeks == []
    idx, _ = r.jump(idx + 100, current=idx + 1)          # 比間隔長：跳目標前的關鍵幀
    assert seeks and idx <= seeks[-1] and idx % 30 == 0
    r.close()


def test_ff_reader_end_and_size_mismatch(synth_video):
    path, truth = synth_video
    r = pb.FastForwardReader(path, 30.0, expected_hw=(96, 160))
    # 片尾：目標前的關鍵幀（270）在目前位置之前 → 從關鍵幀照順序解到目標
    idx, img = r.jump(299, current=290)
    assert idx == 299 and _is_frame(img, 299, truth) and not r.disabled
    while r.next() is not None:
        pass
    assert not r.active and r.next() is None            # 讀到結尾
    r.close()
    bad = pb.FastForwardReader(path, 30.0, expected_hw=(1080, 1920))
    assert bad.jump(100, current=0) is None and bad.disabled   # 尺寸跟 OpenCV 不一致（旋轉影片）→ 停用退回 OpenCV
