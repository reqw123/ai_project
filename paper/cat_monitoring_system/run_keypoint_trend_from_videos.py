"""
直接對「同資料夾內、全部同一行為類別」的原始影片跑姿態估計 + 正規化，看單一
行為類別各關鍵點的動作幅度，以及跨影片的趨勢是否一致——不靠已存在的訓練資料
集（paper/skeletons/*.json），完全重新推論。

跟 run_keypoint_verification.py 的差別：
- run_keypoint_verification.py 讀已存在的骨架 JSON，做「跟 stop 比較」的兩類別分析。
- 這支直接吃資料夾內的原始 .mp4，重新跑 YOLO 姿態估計，只看「單一類別」自己的
  逐關節動作幅度，並列出每支影片各自的數字，檢查同一類別內部是否一致（趨勢）。

類別名稱預設直接取「資料夾名稱」（例如 .../scratch/ 底下的影片就視為 scratch 類別），
不用手動指定；資料夾名稱不影響判斷時，可用 --class_name 覆蓋。

用法：
    python run_keypoint_trend_from_videos.py
    python run_keypoint_trend_from_videos.py --video_folder "C:\\path\\to\\scratch_videos"
    python run_keypoint_trend_from_videos.py --video_folder "C:\\path\\to\\某資料夾" --class_name scratch
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import importlib.util

_TRAIN_MODULE_PATH = Path(__file__).parent / "0_train_gcn.py"
_COLLECT_MODULE_PATH = Path(__file__).parent / "train_data" / "0_dataset_collect.py"

# 直接執行（不帶任何參數）時用的預設設定：改 CLASS_TO_ANALYZE 這一行就能切換要分析的
# 類別，DATASET_ROOT 底下要有 scratch/lick/shake/walk/stop 這幾個兄弟資料夾（stop 會
# 被自動抓去當基準，不用改）。想指定其他路徑/類別就用 --video_folder / --class_name。
DATASET_ROOT = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\1_貓咪姿勢影片分類\模型專用"
CLASS_TO_ANALYZE = "scratch"  # scratch / lick / shake / walk
VIDEO_FOLDER = str(Path(DATASET_ROOT) / CLASS_TO_ANALYZE)
YOLO_MODEL_PATH = r"C:\ai_project\cat_pose\v11s_121.pt"
TARGET_FPS = 30
IMGSZ = 640
CONF_THRESHOLD = 0.5

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv"}
_JSON_NUM_JOINTS = 17

# 對應 tg.KEYPOINT_NAMES 前 14 個關節（不含尾巴三點）的中文名稱，供 HTML 圖表標籤使用
ZH_NAMES = [
    "鼻子", "左耳", "右耳", "前胸", "中背", "髖部",
    "左前肘", "左前爪", "右前肘", "右前爪",
    "左後膝", "左後爪", "右後膝", "右後爪",
]


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _skeleton_data_to_arrays(skeleton_data):
    """把 extract_skeleton_from_video() 回傳的逐幀 dict list 轉成
    (T, 17, 2) 座標陣列與 (T, 17) 信心陣列，跟 CatSkeletonDataset._load_sequences()
    對骨架 JSON 的解析邏輯一致（同一份 keypoints schema）。"""
    coords, confs = [], []
    for frame in skeleton_data:
        kpts_list = frame.get('keypoints', [])
        if len(kpts_list) == _JSON_NUM_JOINTS:
            coords.append(np.array([[kpt['x'], kpt['y']] for kpt in kpts_list]))
            confs.append(np.array([kpt.get('conf', 1.0) for kpt in kpts_list]))
        else:
            coords.append(np.zeros((_JSON_NUM_JOINTS, 2), dtype=np.float32))
            confs.append(np.zeros((_JSON_NUM_JOINTS,), dtype=np.float32))
    return np.array(coords), np.array(confs)


def _consistency_label(mean_v, std_v):
    if mean_v is None or np.isnan(mean_v) or mean_v <= 1e-9:
        return "n/a"
    cv = std_v / mean_v
    if np.isnan(cv):
        return "n/a"
    if cv < 0.4:
        return "高"
    if cv < 0.8:
        return "中"
    return "低"


def _build_insight(zh_sorted, values_sorted, metric_label, allow_negative=False):
    """自動產生洞察文字：前幾名關節彼此差距是否平緩（分散）或有明顯斷層（集中）。
    metric_label 用來區分是描述「絕對動作幅度」還是「扣除 stop 基準後的差異」。

    allow_negative=True（只用在 diff 視圖，因為 diff 可能是負值）時，會先檢查
    「排序後由高到低，數值明顯高於基準（> 峰值 5%）的關節」有幾個——如果只有一小
    撮、後面就轉負或趨近於 0，代表訊號其實集中在那幾個關節，不該用「差距幾 %」
    這種只適用於「全部都正值、只是差距平緩」情境的講法，否則像「差距117%」這種
    跨過 0 的百分比會誤導成「分散」。"""
    valid = [v for v in values_sorted if v is not None]
    if len(valid) < 2 or valid[0] <= 1e-9:
        return f"樣本不足或無明顯{metric_label}，無法判斷分散或集中。"

    ratio = valid[0] / valid[1] if valid[1] > 1e-9 else float('inf')
    if ratio >= 2.0:
        return (f"第 1 名（{zh_sorted[0]}）{metric_label}是第 2 名（{zh_sorted[1]}）的 {ratio:.1f} 倍，"
                f"訊號高度集中在單一關節。")

    if allow_negative:
        threshold = max(1e-9, valid[0] * 0.05)  # 低於峰值 5% 視為跟基準沒有明顯差異
        positive_count = 0
        for v in valid:
            if v > threshold:
                positive_count += 1
            else:
                break
        if 0 < positive_count < len(valid) and positive_count <= 6:
            names = "、".join(zh_sorted[:positive_count])
            lo, hi = valid[positive_count - 1], valid[0]
            return (f"只有 {positive_count} 個關節{metric_label}明顯高於 stop 基準"
                    f"（{names}，落在 {lo:.4f}–{hi:.4f} 之間），其餘關節都在基準值附近或更低，"
                    f"訊號集中在這幾個關節，不是全身性分散。")

    top_n = min(7, len(valid))
    top_span_pct = (valid[0] - valid[top_n - 1]) / valid[0] * 100.0
    return (f"前 {top_n} 名關節{metric_label}落在 {valid[top_n-1]:.4f}–{valid[0]:.4f} 之間"
            f"（差距 {top_span_pct:.0f}%），沒有單一關節壓倒性領先，訊號分散在多個關節。")


def build_html(class_name, joint_names, agg_mean, agg_std, per_video_means,
                diff_mean=None, diff_se=None, stop_n=None):
    """依實際算出的數字動態產生自包含的 HTML 長條圖＋逐影片表格，
    取代先前手動把數字貼進 HTML 的做法——資料全部從這次執行的結果算出來，
    改資料夾/類別重跑就會自動換一份新圖表，不用手動改 HTML。

    有提供 diff_mean（該類別逐關節動作幅度 − stop 基準）時，預設以 diff 為主要視圖
    （已扣掉「關節離 Mid_Back 距離造成的幾何放大」與姿態估計雜訊等共通基準值，
    是比較能代表「這個行為真正用到哪個關節」的版本），並提供切換回絕對數值的分頁。"""
    has_diff = diff_mean is not None
    order = (np.argsort(-np.nan_to_num(diff_mean, nan=-np.inf)) if has_diff
             else np.argsort(-np.nan_to_num(agg_mean, nan=-np.inf)))

    joints_sorted = [joint_names[j] for j in order]
    zh_sorted = [ZH_NAMES[j] if j < len(ZH_NAMES) else joint_names[j] for j in order]

    abs_means = [None if np.isnan(agg_mean[j]) else round(float(agg_mean[j]), 5) for j in order]
    abs_stds = [None if np.isnan(agg_std[j]) else round(float(agg_std[j]), 5) for j in order]
    abs_consistency = [_consistency_label(agg_mean[j], agg_std[j]) for j in order]
    abs_insight = _build_insight(zh_sorted, abs_means, "動作幅度")

    videos_payload = [
        {
            "name": name,
            "values": [None if np.isnan(m[j]) else round(float(m[j]), 5) for j in order],
        }
        for name, m in per_video_means
    ]

    diff_payload = None
    if has_diff:
        diff_means = [None if np.isnan(diff_mean[j]) else round(float(diff_mean[j]), 5) for j in order]
        diff_ses = [None if np.isnan(diff_se[j]) else round(float(diff_se[j]), 5) for j in order]
        diff_insight = _build_insight(zh_sorted, diff_means, "差異值", allow_negative=True)
        diff_payload = {
            "means": diff_means,
            "ses": diff_ses,
            "insight": diff_insight,
            "stopN": stop_n,
        }

    data = {
        "className": class_name,
        "joints": joints_sorted,
        "zh": zh_sorted,
        "videos": videos_payload,
        "nVideos": len(per_video_means),
        "absolute": {
            "means": abs_means,
            "stds": abs_stds,
            "consistency": abs_consistency,
            "insight": abs_insight,
        },
        "diff": diff_payload,
        "defaultView": "diff" if has_diff else "absolute",
    }
    data_json = json.dumps(data, ensure_ascii=False)

    html = _HTML_TEMPLATE.replace("__DATA_JSON__", data_json).replace("__CLASS_NAME__", class_name)
    return html


_HTML_TEMPLATE = r"""<title>__CLASS_NAME__ 逐關節動作幅度</title>
<style>
  .jroot {
    --surface-1:      #fcfcfb;
    --surface-2:      #f4f3f0;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #898781;
    --gridline:       #e1e0d9;
    --baseline:       #c3c2b7;
    --border:         rgba(11,11,11,0.10);
    --series-accent:  #2a78d6;
    --errorbar:       rgba(11,11,11,0.45);
    --status-good:    #0ca30c;
    --status-warn:    #fab219;
    --tooltip-bg:     #0b0b0b;
    --tooltip-text:   #fcfcfb;
    color-scheme: light;
  }
  @media (prefers-color-scheme: dark) {
    .jroot {
      --surface-1:      #1a1a19;
      --surface-2:      #232322;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #898781;
      --gridline:       #2c2c2a;
      --baseline:       #383835;
      --border:         rgba(255,255,255,0.10);
      --series-accent:  #3987e5;
      --errorbar:       rgba(255,255,255,0.45);
      --status-good:    #0ca30c;
      --status-warn:    #fab219;
      --tooltip-bg:     #fcfcfb;
      --tooltip-text:   #0b0b0b;
      color-scheme: dark;
    }
  }
  :root[data-theme="dark"] .jroot {
    --surface-1: #1a1a19; --surface-2: #232322; --text-primary: #ffffff;
    --text-secondary: #c3c2b7; --text-muted: #898781; --gridline: #2c2c2a;
    --baseline: #383835; --border: rgba(255,255,255,0.10); --series-accent: #3987e5;
    --errorbar: rgba(255,255,255,0.45); --tooltip-bg: #fcfcfb; --tooltip-text: #0b0b0b;
    color-scheme: dark;
  }
  :root[data-theme="light"] .jroot {
    --surface-1: #fcfcfb; --surface-2: #f4f3f0; --text-primary: #0b0b0b;
    --text-secondary: #52514e; --text-muted: #898781; --gridline: #e1e0d9;
    --baseline: #c3c2b7; --border: rgba(11,11,11,0.10); --series-accent: #2a78d6;
    --errorbar: rgba(11,11,11,0.45); --tooltip-bg: #0b0b0b; --tooltip-text: #fcfcfb;
    color-scheme: light;
  }

  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body { background: var(--surface-2, #f4f3f0); }
  .jroot {
    font-family: "Segoe UI", "Microsoft JhengHei", "PingFang TC", system-ui, -apple-system, sans-serif;
    background: var(--surface-2);
    color: var(--text-primary);
    min-height: 100vh;
    padding: 40px 20px 56px;
    display: flex;
    justify-content: center;
  }
  .card {
    width: 100%;
    max-width: 980px;
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 36px 40px 30px;
  }
  .eyebrow {
    font-size: 12px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--series-accent);
  }
  h1 { font-size: 24px; line-height: 1.35; font-weight: 700; margin: 6px 0 8px; text-wrap: balance; letter-spacing: -0.01em; }
  .subtitle { font-size: 14px; line-height: 1.6; color: var(--text-secondary); max-width: 66ch; margin: 0 0 24px; }
  .insight {
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px;
    padding: 12px 14px; margin-bottom: 24px; font-size: 12.5px; line-height: 1.6; color: var(--text-secondary);
  }
  .insight b { color: var(--text-primary); }
  .chart-wrap { overflow-x: auto; margin-top: 6px; }
  svg { display: block; }
  .axis-label { fill: var(--text-muted); font-size: 11px; }
  .gridline { stroke: var(--gridline); stroke-width: 1; }
  .baseline { stroke: var(--baseline); stroke-width: 1.5; }
  .joint-label { fill: var(--text-secondary); font-size: 11.5px; }
  .joint-label.hot { fill: var(--text-primary); font-weight: 700; }
  .bar { fill: var(--series-accent); rx: 3; }
  .bar:hover { filter: brightness(1.08); cursor: pointer; }
  .errbar { stroke: var(--errorbar); stroke-width: 1.4; }
  .value-label { font-size: 10px; font-variant-numeric: tabular-nums; fill: var(--text-secondary); }
  .value-label.peak { font-weight: 700; fill: var(--text-primary); }
  .chip { font-size: 9px; }
  .chip.high { fill: var(--status-good); }
  .chip.mid { fill: var(--status-warn); }
  .chip.low, .chip.na { fill: var(--text-muted); }

  .table-title { font-size: 13px; font-weight: 700; margin: 28px 0 10px; color: var(--text-primary); }
  .table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; }
  table { border-collapse: collapse; width: 100%; font-size: 11.5px; }
  th, td { padding: 6px 10px; text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
  th:first-child, td:first-child { text-align: left; font-variant-numeric: normal; position: sticky; left: 0; background: var(--surface-1); }
  thead th { color: var(--text-muted); font-weight: 600; border-bottom: 1px solid var(--border); background: var(--surface-1); }
  tbody tr:nth-child(odd) td { background: var(--surface-2); }
  tbody tr:nth-child(odd) td:first-child { background: var(--surface-2); }
  td { color: var(--text-secondary); }

  .legend-note { font-size: 12px; color: var(--text-muted); margin: 10px 0 0; }

  .tabs { display: inline-flex; background: var(--surface-2); border: 1px solid var(--border); border-radius: 9px; padding: 3px; gap: 2px; margin-bottom: 4px; }
  .tab { appearance: none; border: none; background: transparent; color: var(--text-secondary); font: inherit; font-size: 12.5px; font-weight: 600; padding: 7px 14px; border-radius: 6px; cursor: pointer; }
  .tab[aria-pressed="true"] { background: var(--surface-1); color: var(--text-primary); box-shadow: 0 1px 2px var(--border); }
  .tab:focus-visible { outline: 2px solid var(--series-accent); outline-offset: 2px; }

  .tooltip {
    position: absolute; pointer-events: none; background: var(--tooltip-bg); color: var(--tooltip-text);
    font-size: 12px; line-height: 1.5; padding: 8px 10px; border-radius: 6px; opacity: 0;
    transform: translate(-50%, -100%); transition: opacity 0.08s ease; white-space: nowrap; z-index: 10;
    font-variant-numeric: tabular-nums;
  }
  .tooltip.show { opacity: 1; }
  .tooltip b { font-weight: 700; }

  .foot { margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--border); font-size: 11.5px; line-height: 1.7; color: var(--text-muted); }

  @media (max-width: 640px) {
    .card { padding: 26px 18px 22px; border-radius: 0; }
  }
</style>

<div class="jroot">
  <div class="card">
    <div class="eyebrow">ST-GCN 關節判別力分析 · 原始影片重新推論</div>
    <h1 id="title">__CLASS_NAME__ 逐關節動作幅度</h1>
    <p class="subtitle" id="subtitle"></p>
    <div class="insight" id="insight"></div>

    <div class="tabs" id="tabs" role="tablist" aria-label="選擇檢視方式" style="display:none">
      <button class="tab" id="tab-diff" role="tab" onclick="setView('diff')">扣除 stop 基準後的差異</button>
      <button class="tab" id="tab-absolute" role="tab" onclick="setView('absolute')">絕對動作幅度（未扣基準）</button>
    </div>

    <div class="chart-wrap">
      <svg id="chart" width="900" height="420" viewBox="0 0 900 420" role="img" id="chart-svg"></svg>
    </div>
    <p class="legend-note" id="legend-note"></p>

    <div class="table-title" id="table-title"></div>
    <div class="table-wrap">
      <table id="video-table"></table>
    </div>

    <div class="foot" id="foot"></div>
  </div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
(function () {
  const DATA = __DATA_JSON__;
  const svg = document.getElementById("chart");
  const tooltip = document.getElementById("tooltip");
  const NS = "http://www.w3.org/2000/svg";

  document.getElementById("subtitle").textContent =
    `對 ${DATA.nVideos} 部「${DATA.className}」原始影片重新估計骨架，看逐關節動作幅度的大小與跨影片一致性。`;
  document.getElementById("table-title").textContent = `逐影片明細（每列一部影片，看趨勢是否一致；絕對數值，未扣 stop 基準）`;

  const hasDiff = DATA.diff !== null;
  if (hasDiff) {
    document.getElementById("tabs").style.display = "inline-flex";
  }

  const W = 900, H = 420;
  const marginTop = 14, marginBottom = 56, marginLeft = 40, marginRight = 14;
  const plotW = W - marginLeft - marginRight;
  const plotH = H - marginTop - marginBottom;
  const n = DATA.joints.length;
  const groupW = plotW / n;
  const barW = Math.min(28, groupW * 0.55);

  function el(tag, attrs) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  function render(view) {
    const isDiff = view === "diff" && hasDiff;
    const series = isDiff ? DATA.diff : DATA.absolute;
    const means = series.means;
    const spreads = isDiff ? series.ses : series.stds;
    const spreadLabel = isDiff ? "標準誤" : "跨影片標準差";

    document.getElementById("insight").innerHTML = `<b>觀察：</b>${series.insight}`;
    document.getElementById("legend-note").innerHTML =
      `單位：動作幅度（正規化座標下相鄰幀歐氏距離平均值）；誤差線＝${spreadLabel}` +
      (isDiff ? "" : `；關節下方色點：<span style="color:var(--status-good)">●</span>高一致性 ` +
        `<span style="color:var(--status-warn)">●</span>中 <span style="color:var(--text-muted)">●</span>低/n·a`);
    document.getElementById("foot").innerHTML = isDiff
      ? `資料來源：<code>run_keypoint_trend_from_videos.py</code> —— 「${DATA.className}」逐關節動作幅度減去 stop
         （靜止基準，n=${DATA.diff.stopN}）逐關節動作幅度後的差異，扣掉「關節離 Mid_Back 距離造成的幾何放大」與
         姿態估計雜訊等兩者共通的基準值，剩下的才是這個行為真正多出來的訊號。誤差線＝兩組平均值之差的標準誤
         （sqrt(SE² + SE_stop²)）。`
      : `資料來源：<code>run_keypoint_trend_from_videos.py</code> —— 直接對資料夾內原始影片重新跑 YOLO 姿態
         估計，套用 flip→orientation→coord 標準化後，計算逐幀關節位移平均值，不經任何 ST-GCN 模型推論，
         也不依賴 paper/skeletons/ 既有訓練資料。這是未扣除 stop 基準的絕對數值，鼻子/耳朵等離 Mid_Back
         較遠的關節會因幾何放大而系統性偏高，跨類別比較請改看「扣除 stop 基準後的差異」分頁。`;

    if (hasDiff) {
      document.getElementById("tab-diff").setAttribute("aria-pressed", isDiff ? "true" : "false");
      document.getElementById("tab-absolute").setAttribute("aria-pressed", isDiff ? "false" : "true");
    }

    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const validMeans = means.filter(v => v !== null);
    const validSpreads = spreads.filter(v => v !== null);
    const maxAbs = Math.max(...validMeans.map((v, idx) => Math.abs(v) + (validSpreads[idx] || 0)));
    const yMax = maxAbs * 1.15 || 1;
    const yMin = isDiff ? -yMax * 0.35 : 0;
    const yScale = (v) => marginTop + plotH * (1 - (v - yMin) / (yMax - yMin));
    const zeroY = yScale(0);

    const ticks = 6;
    for (let i = 0; i <= ticks; i++) {
      const v = yMin + ((yMax - yMin) / ticks) * i;
      const y = yScale(v);
      svg.appendChild(el("line", { x1: marginLeft, x2: W - marginRight, y1: y, y2: y, class: Math.abs(v) < 1e-9 ? "baseline" : "gridline" }));
      const t = el("text", { x: marginLeft - 8, y: y + 3, class: "axis-label", "text-anchor": "end" });
      t.textContent = v.toFixed(3);
      svg.appendChild(t);
    }

    const peak = Math.max(...validMeans);

    DATA.joints.forEach((joint, i) => {
      const cx = marginLeft + i * groupW + groupW / 2;
      const mean = means[i];
      const spread = spreads[i] || 0;
      if (mean === null) return;

      const barY = yScale(Math.max(mean, 0));
      const barH = Math.max(1, Math.abs(yScale(mean) - zeroY));
      const bar = el("rect", { class: "bar", x: cx - barW/2, y: mean >= 0 ? barY : zeroY, width: barW, height: barH });
      bar.addEventListener("mousemove", (e) => showTip(e, DATA.zh[i] + ` (${joint})`, mean, spread, spreadLabel));
      bar.addEventListener("mouseleave", hideTip);
      svg.appendChild(bar);

      if (spread > 0) {
        const yTop = yScale(mean + spread);
        const yBot = yScale(mean - spread);
        svg.appendChild(el("line", { x1: cx, x2: cx, y1: yTop, y2: yBot, class: "errbar" }));
        svg.appendChild(el("line", { x1: cx - 5, x2: cx + 5, y1: yTop, y2: yTop, class: "errbar" }));
        svg.appendChild(el("line", { x1: cx - 5, x2: cx + 5, y1: yBot, y2: yBot, class: "errbar" }));
      }

      const isPeak = mean >= peak - 1e-9;
      const labelY = mean >= 0 ? yScale(mean + spread) - 6 : yScale(mean - spread) + 12;
      const vLabel = el("text", { x: cx, y: labelY, class: "value-label" + (isPeak ? " peak" : ""), "text-anchor": "middle" });
      vLabel.textContent = isDiff ? ((mean >= 0 ? "+" : "") + mean.toFixed(4)) : mean.toFixed(4);
      svg.appendChild(vLabel);

      const lbl = el("text", { x: cx, y: H - marginBottom + 18, class: "joint-label" + (isPeak ? " hot" : ""), "text-anchor": "middle" });
      lbl.textContent = DATA.zh[i];
      svg.appendChild(lbl);
      const lbl2 = el("text", { x: cx, y: H - marginBottom + 31, class: "joint-label", "text-anchor": "middle", style: "font-size:9px" });
      lbl2.textContent = joint;
      svg.appendChild(lbl2);

      if (!isDiff) {
        const cons = DATA.absolute.consistency[i];
        const chipCls = cons === "高" ? "high" : cons === "中" ? "mid" : "na";
        const chip = el("circle", { cx: cx, cy: H - marginBottom + 40, r: 3, class: "chip " + chipCls });
        svg.appendChild(chip);
      }
    });

    function showTip(evt, label, mean, spread, spreadLabel) {
      const sign = mean >= 0 ? "+" : "";
      tooltip.innerHTML = `${label}<br><b>${sign}${mean.toFixed(4)}</b> ± ${spread.toFixed(4)} (${spreadLabel})`;
      tooltip.style.left = (evt.clientX + window.scrollX) + "px";
      tooltip.style.top = (evt.clientY + window.scrollY - 10) + "px";
      tooltip.classList.add("show");
    }
    function hideTip() { tooltip.classList.remove("show"); }
  }

  window.setView = render;
  render(DATA.defaultView);

  // ── 逐影片表格（一律顯示絕對數值，跟類別/stop 差異無關）──
  const table = document.getElementById("video-table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.appendChild(document.createElement("th")).textContent = "影片";
  DATA.joints.forEach((j, i) => {
    const th = document.createElement("th");
    th.textContent = DATA.zh[i];
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  DATA.videos.forEach(v => {
    const tr = document.createElement("tr");
    const nameTd = document.createElement("td");
    nameTd.textContent = v.name;
    tr.appendChild(nameTd);
    v.values.forEach(val => {
      const td = document.createElement("td");
      td.textContent = val === null ? "n/a" : val.toFixed(4);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
})();
</script>
"""


def process_video_folder(folder, label, dc, tg, pose_extractor, n_joints):
    """對資料夾內全部影片重新跑姿態估計＋正規化，回傳
    [(video_stem, per_joint_motion_array), ...]。抽成函式是為了讓 main() 能
    對「目標類別資料夾」跟「stop 基準資料夾」各呼叫一次，共用同一套流程。"""
    videos = sorted(
        (f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTS),
        key=lambda p: p.name.lower(),
    )
    results = []
    for i, video_path in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video_path.name}")
        result = dc.extract_skeleton_from_video(
            video_path, pose_extractor, target_fps=TARGET_FPS, label=label,
        )
        if result is None:
            print("  ✗ 無法處理，略過")
            continue
        skeleton_data, _ = result
        if len(skeleton_data) < 2:
            print("  ⚠ 有效幀數不足，略過")
            continue

        coords, confs = _skeleton_data_to_arrays(skeleton_data)
        coords = tg.interpolate_missing(coords, confs)

        seq = coords[:, :n_joints, :]
        conf_seq = confs[:, :n_joints]
        seq = tg.flip_normalize(seq)
        seq = tg.orientation_normalize(seq)
        seq = tg.normalize_skeleton_coords(seq)

        joint_motion = tg._compute_per_joint_motion(seq, conf_seq)
        results.append((video_path.stem, joint_motion))
    return results


def _find_stop_folder(video_folder):
    """預設拿 video_folder 同層、名叫 stop 的資料夾當基準（跟專案既有的
    .../<某資料集>/{walk,lick,scratch,shake,stop}/ 慣例一致）。"""
    candidate = video_folder.parent / "stop"
    if candidate.exists() and candidate.is_dir() and candidate.resolve() != video_folder.resolve():
        return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="對資料夾內同一類別的原始影片跑姿態估計，看逐關節動作幅度與跨影片趨勢。")
    parser.add_argument('--video_folder', default=VIDEO_FOLDER)
    parser.add_argument('--class_name', default=None,
                         help='行為類別名稱。不指定則直接取資料夾名稱（例如 .../scratch/ → scratch）。')
    parser.add_argument('--yolo_model_path', default=YOLO_MODEL_PATH)
    parser.add_argument('--stop_video_folder', default=None,
                         help='stop（靜止）基準資料夾，用來扣掉「關節離 Mid_Back 距離造成的幾何放大」與姿態'
                              '估計雜訊等共通基準值，算出這個類別真正比 stop 多出來的訊號。不指定則自動找'
                              'video_folder 同層的 stop 資料夾；找不到就只顯示絕對數值。傳空字串 "" 強制跳過。')
    parser.add_argument('--html_out', default=None,
                         help='HTML 輸出路徑。不指定則預設存到腳本同資料夾下的 '
                              'keypoint_trend_<class_name>.html；傳空字串 "" 可跳過產生 HTML。')
    args = parser.parse_args()

    video_folder = Path(args.video_folder)
    if not video_folder.exists():
        print(f"✗ 資料夾不存在: {video_folder}")
        return
    videos = sorted(
        (f for f in video_folder.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTS),
        key=lambda p: p.name.lower(),
    )
    if not videos:
        print(f"✗ 資料夾內找不到影片: {video_folder}")
        return

    class_name = args.class_name or video_folder.name.lower()

    tg = _load_module(_TRAIN_MODULE_PATH, "_train_gcn")
    dc = _load_module(_COLLECT_MODULE_PATH, "_dataset_collect")

    if class_name not in tg.BEHAVIOR_PREFIXES:
        print(f"  ⚠ 資料夾名稱「{video_folder.name}」不是已知行為類別"
              f"（{', '.join(tg.BEHAVIOR_PREFIXES)}），仍會以「{class_name}」繼續，"
              f"如果判斷錯了可用 --class_name 指定正確類別。")

    # 解析 stop 基準資料夾路徑（先不處理，晚點跟目標類別共用同一個 pose_extractor 再跑）
    stop_folder = None
    if class_name == "stop":
        pass  # 本身就是基準，不需要再跟自己比較
    elif args.stop_video_folder == "":
        pass  # 使用者明確要求跳過
    elif args.stop_video_folder:
        stop_folder = Path(args.stop_video_folder)
        if not stop_folder.exists():
            print(f"  ⚠ 指定的 stop 基準資料夾不存在: {stop_folder}，將只顯示絕對數值")
            stop_folder = None
    else:
        stop_folder = _find_stop_folder(video_folder)
        if stop_folder is None:
            print(f"  ⚠ 找不到 stop 基準資料夾（預期在 {video_folder.parent / 'stop'}），"
                  f"將只顯示絕對數值。可用 --stop_video_folder 指定正確路徑。")

    if args.html_out is None:
        html_out_path = Path(__file__).parent / f"keypoint_trend_{class_name}.html"
    elif args.html_out == "":
        html_out_path = None
    else:
        html_out_path = Path(args.html_out)

    print(f"[類別] {class_name}（{'手動指定' if args.class_name else '取自資料夾名稱'}）"
          f"　[資料夾] {video_folder}　共 {len(videos)} 部影片")
    print(f"[YOLO 模型] {args.yolo_model_path}")

    pose_extractor = dc.PoseExtractor(
        model_path=args.yolo_model_path, imgsz=IMGSZ, conf_threshold=CONF_THRESHOLD,
    )

    n_joints = tg.NUM_JOINTS
    joint_names = tg.KEYPOINT_NAMES[:n_joints]

    per_video_means = process_video_folder(video_folder, class_name, dc, tg, pose_extractor, n_joints)

    if not per_video_means:
        print("✗ 沒有任何影片產出有效結果")
        return

    stacked = np.stack([m for _, m in per_video_means])  # (n_videos, n_joints)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        agg_mean = np.nanmean(stacked, axis=0)
        agg_std = np.nanstd(stacked, axis=0)

    SEP = '─' * 78
    print(f"\n{'='*78}")
    print(f"  {class_name} 逐關節動作幅度（{len(per_video_means)} 部影片彙總，不經任何模型推論）")
    print(f"{'='*78}")
    print(f"  {'關節':<12} {'平均值':>10} {'標準差':>10} {'影片間一致性':>14}")
    print(f"  {SEP}")
    order = np.argsort(-np.nan_to_num(agg_mean, nan=-np.inf))
    for j in order:
        m, s = agg_mean[j], agg_std[j]
        cv = (s / m) if (m and not np.isnan(m) and m > 1e-9) else np.nan
        if np.isnan(cv):
            consistency = "n/a"
        elif cv < 0.4:
            consistency = "高"
        elif cv < 0.8:
            consistency = "中"
        else:
            consistency = "低"
        print(f"  {joint_names[j]:<12} {m:>10.4f} {s:>10.4f} {consistency:>14}")

    top_j = order[:8]
    print(f"\n【逐影片明細】（每列一部影片，前 8 個關節，看趨勢是否一致）")
    header = "  " + f"{'影片':<16}" + "".join(f"{joint_names[j][:6]:>9}" for j in top_j)
    print(header)
    print("  " + SEP)
    for name, m in per_video_means:
        row = f"  {name:<16}" + "".join(f"{m[j]:>9.4f}" for j in top_j)
        print(row)

    print(f"\n{'='*78}\n")

    diff_mean = diff_se = None
    stop_n = None
    if stop_folder is not None:
        stop_videos = [f for f in stop_folder.iterdir()
                       if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTS]
        if not stop_videos:
            print(f"  ⚠ stop 基準資料夾內找不到影片: {stop_folder}，只顯示絕對數值")
        else:
            print(f"[stop 基準] {stop_folder}　共 {len(stop_videos)} 部影片，重新估計中...")
            stop_per_video = process_video_folder(stop_folder, 'stop', dc, tg, pose_extractor, n_joints)
            if not stop_per_video:
                print("  ✗ stop 基準資料夾沒有任何影片產出有效結果，只顯示絕對數值")
            else:
                stop_stacked = np.stack([m for _, m in stop_per_video])
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Mean of empty slice')
                    stop_mean = np.nanmean(stop_stacked, axis=0)
                    stop_std = np.nanstd(stop_stacked, axis=0)
                n_class = len(per_video_means)
                stop_n = len(stop_per_video)
                diff_mean = agg_mean - stop_mean
                se_class = agg_std / np.sqrt(max(n_class, 1))
                se_stop = stop_std / np.sqrt(max(stop_n, 1))
                diff_se = np.sqrt(se_class ** 2 + se_stop ** 2)

                print(f"\n{'='*78}")
                print(f"  {class_name} − stop 基準：扣除幾何放大／雜訊共通值後的差異（依差異排序）")
                print(f"{'='*78}")
                print(f"  {'關節':<12} {'差異':>10} {'標準誤':>10}")
                print(f"  {SEP}")
                diff_order = np.argsort(-np.nan_to_num(diff_mean, nan=-np.inf))
                for j in diff_order:
                    d, se = diff_mean[j], diff_se[j]
                    print(f"  {joint_names[j]:<12} {d:>+10.4f} {se:>10.4f}")
                print(f"\n{'='*78}\n")

    if html_out_path is not None:
        html = build_html(class_name, joint_names, agg_mean, agg_std, per_video_means,
                           diff_mean=diff_mean, diff_se=diff_se, stop_n=stop_n)
        html_out_path.parent.mkdir(parents=True, exist_ok=True)
        html_out_path.write_text(html, encoding='utf-8')
        print(f"✓ HTML 圖表已存檔: {html_out_path}")


if __name__ == "__main__":
    main()
