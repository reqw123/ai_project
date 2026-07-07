"""
檢查每個行為類別的訓練樣本是否集中自少數幾支影片/場景。

背景：懷疑某些類別（例如 scratch）驗證集表現很好、獨立測試集卻掉很多，
可能是「過擬合假象」——模型學到的是少數影片/場景的表面線索，而非真正
可泛化的動作特徵。這裡直接掃描逐幀標註的骨架 JSON，統計每個類別的標註
幀數在各支影片間的分布，用「前 1 支/前 3 支影片佔比」當集中度指標。

用法：
    python eval_class_source_distribution.py
    python eval_class_source_distribution.py --skeleton_dir <path> --top_n 5
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')  # Windows 主控台預設編碼常是 cp950，會把中文印成亂碼

DEFAULT_CONFIG = Path(__file__).parent / "stgcn_config.yaml"


def _load_skeleton_dir_from_config(config_path: Path) -> str:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['SKELETON_DATA_FOLDER']


def scan_label_frame_counts(skeleton_dir: str) -> dict:
    """回傳 {label: {video_id: frame_count}}（排除 'unannotated'）。"""
    per_label = defaultdict(Counter)
    json_files = sorted(Path(skeleton_dir).glob("*.json"))
    for jf in json_files:
        video_id = jf.stem
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)
        frames = data.get('frames', [])
        labels = Counter(fr.get('label', 'unannotated') for fr in frames)
        for label, cnt in labels.items():
            if label == 'unannotated':
                continue
            per_label[label][video_id] += cnt
    return per_label


def print_concentration_report(per_label: dict, top_n: int = 5,
                               top1_flag: float = 0.40, top3_flag: float = 0.70):
    """印出每個類別的來源影片集中度報告，並標記疑似過度集中的類別。"""
    SEP = '─' * 70
    print(f"\n{'═' * 70}")
    print("  類別樣本來源集中度報告（依標註幀數，非切窗後的 sequence 數）")
    print(f"{'═' * 70}")
    flagged = []

    for label in sorted(per_label.keys()):
        counts = per_label[label]
        total = sum(counts.values())
        n_videos = len(counts)
        top_videos = counts.most_common(top_n)
        top1_share = top_videos[0][1] / total if top_videos else 0.0
        top3_share = sum(c for _, c in top_videos[:3]) / total if total else 0.0

        print(f"\n● [{label}]  總標註幀數={total}  來源影片數={n_videos}  "
              f"平均每支影片={total / n_videos:.0f} 幀")
        print(f"  {SEP}")
        for vid, cnt in top_videos:
            print(f"    {vid:<30}  {cnt:>6} 幀  ({cnt / total:.1%})")
        print(f"  Top-1 佔比: {top1_share:.1%}   Top-3 佔比: {top3_share:.1%}")

        if top1_share >= top1_flag or top3_share >= top3_flag:
            flagged.append((label, top1_share, top3_share, n_videos))
            print(f"  ⚠ 集中度偏高（Top-1≥{top1_flag:.0%} 或 Top-3≥{top3_flag:.0%}），"
                  f"驗證集表現可能因場景/個體重疊而虛高，需留意獨立測試集的表現落差。")

    print(f"\n{'═' * 70}")
    if flagged:
        print("  ⚠ 疑似過度集中的類別：")
        for label, t1, t3, nv in flagged:
            print(f"    [{label}]  來源影片數={nv}  Top-1={t1:.1%}  Top-3={t3:.1%}")
    else:
        print("  沒有類別的來源集中度超過門檻。")
    print(f"{'═' * 70}\n")

    return flagged


def main():
    parser = argparse.ArgumentParser(description="檢查各行為類別的訓練樣本來源影片集中度。")
    parser.add_argument('--skeleton_dir', default=None,
                        help="骨架 JSON 資料夾路徑；不指定則從 stgcn_config.yaml 讀 SKELETON_DATA_FOLDER")
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--top_n', type=int, default=5, help="每個類別列出前 N 支貢獻最多的影片")
    parser.add_argument('--top1_flag', type=float, default=0.40)
    parser.add_argument('--top3_flag', type=float, default=0.70)
    args = parser.parse_args()

    skeleton_dir = args.skeleton_dir or _load_skeleton_dir_from_config(Path(args.config))
    print(f"[骨架資料夾] {skeleton_dir}")

    per_label = scan_label_frame_counts(skeleton_dir)
    print_concentration_report(per_label, top_n=args.top_n,
                              top1_flag=args.top1_flag, top3_flag=args.top3_flag)


if __name__ == '__main__':
    main()
