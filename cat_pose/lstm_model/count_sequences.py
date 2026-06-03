"""
Count ST-GCN skeleton JSON files and labeled frames by behavior class.
"""
import json
from pathlib import Path
from collections import defaultdict

# ==================== 設定 ====================
SKELETON_DIR = Path(r"C:\AI_Project\paper\skeletons")
BEHAVIOR_NAMES = ['walk', 'lick', 'scratch', 'shake', 'stop']

SEQUENCE_LENGTH = 16   # 與訓練腳本一致
STRIDE         = SEQUENCE_LENGTH // 2   # = 8
TARGET_SEQS    = 200   # 每類別目標序列數


def _detect_behavior(json_path: Path, data: dict) -> str | None:
    """從 JSON 內容或檔名推斷行為類別。"""
    # 1. 從 video_metadata.video_path 的父資料夾名稱判斷（最可靠）
    video_path = data.get('video_metadata', {}).get('video_path', '')
    if video_path:
        folder = Path(video_path).parent.name.lower()
        if folder in BEHAVIOR_NAMES:
            return folder

    # 2. 從第一幀有效 label 判斷
    for frame in data.get('frames', []):
        lbl = frame.get('label', 'unannotated')
        if lbl != 'unannotated' and lbl in BEHAVIOR_NAMES:
            return lbl

    # 3. 從檔名前綴判斷
    stem = json_path.stem.lower()
    for name in BEHAVIOR_NAMES:
        if stem.startswith(name):
            return name

    return None


def _estimate_sequences(total_frames: int, labeled_frames: int) -> int:
    """估算滑動窗口可切出的有效序列數（樂觀上界）。"""
    if total_frames < SEQUENCE_LENGTH:
        return 0
    windows = (total_frames - SEQUENCE_LENGTH) // STRIDE + 1
    ratio   = labeled_frames / max(total_frames, 1)
    return max(0, int(windows * ratio))


def count_sequences():
    if not SKELETON_DIR.exists():
        print(f"⚠ 找不到資料夾，跳過: {SKELETON_DIR}")
        return

    json_files = sorted(SKELETON_DIR.glob("*.json"))
    if not json_files:
        print(f"📂 {SKELETON_DIR} 內無 JSON 檔案")
        return

    video_counts   = defaultdict(int)
    frame_counts   = defaultdict(int)
    labeled_counts = defaultdict(int)
    seq_estimates  = defaultdict(int)
    unknown_files  = []

    print("=" * 72)
    print(f"  ST-GCN Skeleton Data — {SKELETON_DIR}")
    print("=" * 72)

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ⚠ 讀取失敗，跳過: {json_file.name}  ({e})")
            continue

        behavior = _detect_behavior(json_file, data)
        if behavior is None:
            unknown_files.append(json_file.name)
            continue

        frames         = data.get('frames', [])
        total_f        = len(frames)
        labeled_f      = sum(1 for fr in frames
                             if fr.get('label', 'unannotated') != 'unannotated')
        est_seqs       = _estimate_sequences(total_f, labeled_f)

        video_counts[behavior]   += 1
        frame_counts[behavior]   += total_f
        labeled_counts[behavior] += labeled_f
        seq_estimates[behavior]  += est_seqs

        status = "✓" if labeled_f > 0 else "✗"
        print(f"  {status} [{behavior:<7}] {json_file.name:<40} "
              f"{labeled_f:>4}/{total_f:<5} labeled  ~{est_seqs:>3} seqs")

    # ── 摘要 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Summary by Behavior")
    print("=" * 72)
    print(f"  {'Class':<10} {'Videos':>6}  {'Frames':>7}  {'Labeled':>7}  "
          f"{'~Seqs':>6}  Status")
    print("  " + "─" * 68)

    total_seqs = 0
    for beh in BEHAVIOR_NAMES:
        vids   = video_counts[beh]
        frames = frame_counts[beh]
        lbld   = labeled_counts[beh]
        seqs   = seq_estimates[beh]
        total_seqs += seqs
        ok = "✅" if seqs >= TARGET_SEQS else ("⚠️ " if seqs >= TARGET_SEQS // 2 else "❌")
        print(f"  {ok} {beh:<8}  {vids:>5}  {frames:>8}  {lbld:>8}  {seqs:>7}")

    print("  " + "─" * 68)
    print(f"  {'TOTAL':<10} {sum(video_counts.values()):>6}  "
          f"{sum(frame_counts.values()):>7}  "
          f"{sum(labeled_counts.values()):>7}  "
          f"{total_seqs:>6}")

    if unknown_files:
        print(f"\n  ⚠ 無法判斷類別的檔案（{len(unknown_files)} 個）：")
        for fn in unknown_files:
            print(f"    - {fn}")

    # ── 進度條 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  Progress  (target: {TARGET_SEQS} seqs / class)")
    print("=" * 72)
    bar_len = 40
    total_target = TARGET_SEQS * len(BEHAVIOR_NAMES)
    for beh in BEHAVIOR_NAMES:
        seqs     = seq_estimates[beh]
        progress = min(seqs / TARGET_SEQS, 1.0)
        filled   = int(bar_len * progress)
        bar      = "█" * filled + "░" * (bar_len - filled)
        print(f"  {beh:<8} [{bar}] {seqs}/{TARGET_SEQS} ({progress*100:.0f}%)")

    overall = total_seqs / max(total_target, 1)
    print(f"\n  Overall: {total_seqs}/{total_target} ({overall*100:.0f}%)")
    if total_seqs < total_target:
        print(f"  💡 還需約 {total_target - total_seqs} 個序列達到目標")
    else:
        print("  🎉 已達目標，可開始訓練")
    print("=" * 72)


if __name__ == "__main__":
    count_sequences()
