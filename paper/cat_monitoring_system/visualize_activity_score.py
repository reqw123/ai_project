"""
Activity Score 機制視覺化

複製 anomaly_detector + behavior_tracker 的完整計算流程，
用合成位移資料展示 activity_value 與 activity_score 對不同運動模式的反應。

執行：python visualize_activity_score.py
輸出：同目錄下 activity_score_visualization.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── 完全對應 config.py 的參數 ────────────────────────────────────────
EMA_ALPHA             = 1.0    # AnomalyDetectionConfig.EMA_ALPHA（無平滑 = 直接用 motion_score）
ABNORMAL_THRESHOLD_PX = 0.2    # AnomalyDetectionConfig.ABNORMAL_THRESHOLD（單位：像素）
MAX_MOTION            = 20.0   # AnomalyDetectionConfig.MAX_MOTION（像素，正規化上限）
ACTIVITY_WINDOW_SEC   = 3.0    # BehaviorTrackingConfig.ACTIVITY_SCORE_WINDOW_SECONDS
LOW_CONF_WEIGHT       = 0.5    # BehaviorTrackingConfig.LOW_CONFIDENCE_ACTIVITY_WEIGHT
FPS                   = 15.0   # 模擬幀率
DURATION              = 15.0   # 每個場景持續秒數

# abnormal 閾值換算成 activity_value 的等效值（供圖示用）
ABNORMAL_AV = int(min(ABNORMAL_THRESHOLD_PX / MAX_MOTION, 1.0) * 100)  # = 1


# ── 演算法複製 ───────────────────────────────────────────────────────
def disp_to_av(disp_px: float) -> int:
    """
    anomaly_detector.detect() 的核心：
    activity_value = int( clamp(mean_disp / MAX_MOTION, 0, 1) * 100 )
    """
    return int(min(disp_px / max(MAX_MOTION, 1e-6), 1.0) * 100)


def simulate(disp_fn, fps: float = FPS, duration: float = DURATION):
    """
    用合成位移函數跑完整 activity_value → activity_score 管線。

    Returns
    -------
    ts       : (N,) 時間軸（秒）
    disps    : (N,) 每幀位移（像素）
    avs      : (N,) activity_value（0-100）
    scores   : (N,) activity_score（0-100）
    """
    n  = int(duration * fps)
    dt = 1.0 / fps
    ts    = np.arange(n) * dt
    disps = np.array([max(0.0, float(disp_fn(t))) for t in ts])
    avs   = np.array([disp_to_av(d) for d in disps])

    # behavior_tracker.get_activity_score() 複製
    window: list = []
    scores: list = []
    for t, av in zip(ts, avs):
        window.append({"t": t, "av": int(av), "w": LOW_CONF_WEIGHT})
        # 只保留 ACTIVITY_WINDOW_SEC 內的項目
        window = [e for e in window if (t - e["t"]) < ACTIVITY_WINDOW_SEC]
        tw = sum(e["w"] for e in window)
        sc = round(sum(e["av"] * e["w"] for e in window) / tw) if tw > 0 else 50
        scores.append(max(0, min(100, sc)))

    return ts, disps, avs, np.array(scores)


# ── 六種運動場景 ─────────────────────────────────────────────────────
scenarios = [
    {
        "label": "① 漸進加速  (Ramp Up)",
        "desc":  f"位移 0 → {MAX_MOTION:.0f}px 線性增加",
        "fn":    lambda t: t / DURATION * MAX_MOTION,
    },
    {
        "label": "② 漸進減速  (Ramp Down)",
        "desc":  f"位移 {MAX_MOTION:.0f} → 0px 線性下降",
        "fn":    lambda t: MAX_MOTION * (1.0 - t / DURATION),
    },
    {
        "label": "③ 突發動作  (Spike)",
        "desc":  "靜止(1.5px) → 爆發 3 秒(16px) → 靜止",
        "fn":    lambda t: 16.0 if 5.0 <= t <= 8.0 else 1.5,
    },
    {
        "label": "④ 週期性動作  (Periodic)",
        "desc":  "sin 波：模擬有節奏的舔舐 / 搔抓",
        "fn":    lambda t: 10.0 + 9.0 * np.sin(2 * np.pi * t / 2.5),
    },
    {
        "label": "⑤ 階躍切換  (Step)",
        "desc":  "靜止突然切換到高活動量（觀察 score 滯後）",
        "fn":    lambda t: 2.0 if t < DURATION / 2 else 16.0,
    },
    {
        "label": "⑥ 間歇性動作  (Intermittent)",
        "desc":  "短暫爆發 1 秒 → 靜止 3 秒，週期重複",
        "fn":    lambda t: 15.0 if (t % 4.0) < 1.0 else 1.5,
    },
]

# ── 顏色 ─────────────────────────────────────────────────────────────
C = {
    "bg":      "#0f172a",
    "surface": "#1e293b",
    "border":  "#334155",
    "muted":   "#94a3b8",
    "disp":    "#64748b",
    "av":      "#38bdf8",   # sky blue
    "score":   "#f97316",   # orange
    "pos":     "#f97316",
    "neg":     "#818cf8",   # indigo
    "abnorm":  "#ef4444",
    "max_m":   "#fbbf24",
}

# ── 圖形佈局 ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 17))
fig.patch.set_facecolor(C["bg"])

fig.text(0.5, 0.985, "Activity Score 機制視覺化",
         ha="center", va="top", fontsize=15, fontweight="bold", color="white")
fig.text(
    0.5, 0.966,
    f"activity_value = clamp( mean_keypoint_displacement / {MAX_MOTION}px, 0, 1 ) × 100"
    f"     activity_score = {ACTIVITY_WINDOW_SEC}s 加權滾動平均  (weight={LOW_CONF_WEIGHT})\n"
    f"abnormal_threshold = {ABNORMAL_THRESHOLD_PX}px ≈ activity_value ≥ {ABNORMAL_AV}"
    f"     EMA_ALPHA = {EMA_ALPHA}  →  ema_motion = motion_score（無平滑）",
    ha="center", va="top", fontsize=8.5, color=C["muted"],
)

outer = gridspec.GridSpec(3, 2, figure=fig, hspace=0.60, wspace=0.30,
                          top=0.945, bottom=0.095, left=0.06, right=0.97)

for idx, sc in enumerate(scenarios):
    row, col = divmod(idx, 2)
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[row, col], hspace=0.07, height_ratios=[1.35, 1]
    )
    ax_top = fig.add_subplot(inner[0])
    ax_bot = fig.add_subplot(inner[1], sharex=ax_top)

    for ax in (ax_top, ax_bot):
        ax.set_facecolor(C["surface"])
        for sp in ax.spines.values():
            sp.set_color(C["border"])
            sp.set_linewidth(0.7)
        ax.tick_params(colors=C["muted"], labelsize=7)

    ts, disps, avs, scores = simulate(sc["fn"])

    # ── 上子圖：位移（右軸） + activity_value（左軸）──────────────
    ax_d = ax_top.twinx()
    ax_d.fill_between(ts, disps, alpha=0.22, color=C["disp"], step="mid", label="位移 (px)")
    ax_d.set_ylim(0, MAX_MOTION * 1.35)
    ax_d.set_ylabel("位移 (px)", fontsize=7, color=C["disp"])
    ax_d.tick_params(axis="y", labelsize=6, colors=C["disp"])
    for sp in ax_d.spines.values():
        sp.set_color(C["border"])
    # MAX_MOTION 上限線
    ax_d.axhline(MAX_MOTION, color=C["max_m"], ls="--", lw=0.8, alpha=0.6)
    ax_d.text(0.2, MAX_MOTION + 0.3, f"MAX_MOTION={MAX_MOTION}px",
              fontsize=6, color=C["max_m"], va="bottom")

    ax_top.plot(ts, avs, color=C["av"], lw=1.9, label="activity_value", zorder=3)
    ax_top.axhline(100, color=C["max_m"], ls=":", lw=0.7, alpha=0.5)
    ax_top.set_ylim(-4, 112)
    ax_top.set_ylabel("activity_value\n(0–100)", fontsize=7, color=C["av"])
    ax_top.tick_params(axis="y", labelsize=6, colors=C["av"])
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_title(f"{sc['label']}\n{sc['desc']}",
                     fontsize=8.5, color="white", pad=5, loc="left")

    # 合併兩軸圖例
    h1, l1 = ax_top.get_legend_handles_labels()
    h2, l2 = ax_d.get_legend_handles_labels()
    ax_top.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right",
                  facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["muted"], framealpha=0.8)

    # ── 下子圖：activity_score ────────────────────────────────────
    ax_bot.plot(ts, scores, color=C["score"], lw=2.2, label="activity_score", zorder=3)
    ax_bot.axhline(50, color="#475569", ls=":", lw=0.8, label="baseline 50")
    ax_bot.fill_between(ts, scores, 50, where=(scores >= 50),
                        alpha=0.20, color=C["pos"], interpolate=True)
    ax_bot.fill_between(ts, scores, 50, where=(scores < 50),
                        alpha=0.20, color=C["neg"], interpolate=True)
    ax_bot.set_ylim(-4, 112)
    ax_bot.set_ylabel(f"activity_score\n({ACTIVITY_WINDOW_SEC}s win)", fontsize=7, color=C["score"])
    ax_bot.tick_params(axis="y", labelsize=6, colors=C["score"])
    ax_bot.set_xlabel("時間 (秒)", fontsize=7, color=C["muted"])
    ax_bot.legend(fontsize=7, loc="upper right",
                  facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["muted"], framealpha=0.8)

    # ── 場景⑤（Step）標注 3 秒滯後 ───────────────────────────────
    if idx == 4:
        step_t = DURATION / 2
        catch_t = step_t + ACTIVITY_WINDOW_SEC
        catch_i = min(int(catch_t * FPS), len(scores) - 1)
        ax_bot.annotate(
            f"← {ACTIVITY_WINDOW_SEC}s 窗口\n   score 才追上",
            xy=(catch_t, scores[catch_i]),
            xytext=(catch_t + 1.2, 45),
            fontsize=7, color="#fbbf24",
            arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", fc=C["bg"], ec="#fbbf24", lw=0.8),
        )
        # 標示 3s 窗口區間
        ax_bot.axvspan(step_t, catch_t, alpha=0.08, color="#fbbf24", label=f"{ACTIVITY_WINDOW_SEC}s 追趕區間")
        ax_bot.legend(fontsize=7, loc="upper right",
                      facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["muted"], framealpha=0.8)

    # ── 場景⑥（Intermittent）標注每次爆發衰退 ───────────────────
    if idx == 5:
        for burst_start in [0.0, 4.0, 8.0, 12.0]:
            ax_top.axvspan(burst_start, burst_start + 1.0, alpha=0.12,
                           color=C["abnorm"], zorder=1)
            ax_bot.axvspan(burst_start, burst_start + 1.0, alpha=0.12,
                           color=C["abnorm"], zorder=1)


# ── 底部：Transfer Function 曲線（px → activity_value 映射） ─────────
ax_tf = fig.add_axes([0.08, 0.010, 0.26, 0.075])
ax_tf.set_facecolor(C["surface"])
for sp in ax_tf.spines.values():
    sp.set_color(C["border"])
px_x = np.linspace(0, MAX_MOTION * 1.3, 300)
av_y = [disp_to_av(p) for p in px_x]
ax_tf.plot(px_x, av_y, color=C["av"], lw=2.0)
ax_tf.axvline(MAX_MOTION, color=C["max_m"], ls="--", lw=1.2, label=f"MAX_MOTION ({MAX_MOTION}px) = activity_value 100")
ax_tf.axvline(ABNORMAL_THRESHOLD_PX, color=C["abnorm"], ls=":", lw=1.0,
              label=f"abnormal_threshold ({ABNORMAL_THRESHOLD_PX}px) ≈ av {ABNORMAL_AV}")
ax_tf.set_xlabel("mean 位移 (px)", fontsize=7.5, color=C["muted"])
ax_tf.set_ylabel("activity_value", fontsize=7.5, color=C["av"])
ax_tf.set_title("Transfer Function: px → activity_value", fontsize=8, color="white")
ax_tf.tick_params(labelsize=6.5, colors=C["muted"])
ax_tf.legend(fontsize=6.5, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["muted"], framealpha=0.85)
ax_tf.set_ylim(-3, 108)

# ── 底部：3 秒窗口的滯後特性說明 ─────────────────────────────────────
ax_lag = fig.add_axes([0.40, 0.010, 0.26, 0.075])
ax_lag.set_facecolor(C["surface"])
for sp in ax_lag.spines.values():
    sp.set_color(C["border"])
# 模擬：activity_value 在 t=0 從 0 瞬間跳到 100，score 需要多久追上？
n_lag = int(10 * FPS)
av_step = np.where(np.arange(n_lag) >= 0, 100, 0)
ts_lag  = np.arange(n_lag) / FPS
win_lag: list = []
sc_lag: list  = []
for t, av in zip(ts_lag, av_step):
    win_lag.append({"t": t, "av": int(av), "w": LOW_CONF_WEIGHT})
    win_lag = [e for e in win_lag if (t - e["t"]) < ACTIVITY_WINDOW_SEC]
    tw = sum(e["w"] for e in win_lag)
    sc_lag.append(max(0, min(100, round(sum(e["av"] * e["w"] for e in win_lag) / tw) if tw > 0 else 50)))
ax_lag.step(ts_lag, av_step, color=C["av"], lw=1.5, where="post", label="activity_value (突然=100)")
ax_lag.plot(ts_lag, sc_lag, color=C["score"], lw=2.2, label="activity_score")
ax_lag.axhline(100, color=C["max_m"], ls=":", lw=0.7)
ax_lag.axvline(ACTIVITY_WINDOW_SEC, color="#fbbf24", ls="--", lw=0.9, alpha=0.7,
               label=f"t={ACTIVITY_WINDOW_SEC}s (窗口填滿，score=100)")
ax_lag.set_xlabel("時間 (秒)", fontsize=7.5, color=C["muted"])
ax_lag.set_ylabel("值 (0–100)", fontsize=7.5, color=C["muted"])
ax_lag.set_title(f"activity_score 滯後分析：activity_value 瞬間跳至 100", fontsize=8, color="white")
ax_lag.tick_params(labelsize=6.5, colors=C["muted"])
ax_lag.legend(fontsize=6.5, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["muted"], framealpha=0.85)
ax_lag.set_ylim(-3, 108)
ax_lag.set_xlim(0, 8)

# ── 底部：分佈密度 ────────────────────────────────────────────────────
ax_dist = fig.add_axes([0.72, 0.010, 0.24, 0.075])
ax_dist.set_facecolor(C["surface"])
for sp in ax_dist.spines.values():
    sp.set_color(C["border"])
all_avs    = []
all_scores = []
for sc in scenarios:
    _, _, av_s, sc_s = simulate(sc["fn"])
    all_avs.extend(av_s.tolist())
    all_scores.extend(sc_s.tolist())
bins = np.linspace(0, 100, 26)
ax_dist.hist(all_avs, bins=bins, alpha=0.55, color=C["av"], label="activity_value", density=True)
ax_dist.hist(all_scores, bins=bins, alpha=0.55, color=C["score"], label="activity_score", density=True)
ax_dist.set_xlabel("值 (0–100)", fontsize=7.5, color=C["muted"])
ax_dist.set_ylabel("密度", fontsize=7.5, color=C["muted"])
ax_dist.set_title("全場景 activity_value vs score 分佈", fontsize=8, color="white")
ax_dist.tick_params(labelsize=6.5, colors=C["muted"])
ax_dist.legend(fontsize=6.5, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["muted"], framealpha=0.85)

out_path = Path(__file__).parent / "activity_score_visualization.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"已儲存: {out_path}")
plt.show()
