"""
在固定的 test 集（skeletons/test/ 子資料夾）上比較 1～N 個 ST-GCN 模型。

跟 eval_gcn_compare.py 的差別：那支是拿「主要測試/」資料夾的整支影片重跑 YOLO，
並把整支影片視為同一類；這支直接讀骨架 JSON 的逐幀標籤、用跟訓練完全相同的
切窗／正規化流程（重用 0_train_gcn.py 的 CatSkeletonDataset），只評有標註的視窗，
所以數字可以直接跟訓練結束時印出的 test 結果對照。

每個模型的前處理參數（關節數、序列長度、步長、平滑方式、模型結構）都從該 run
資料夾的 params_snapshot.json 讀，不用手動指定。

⚠ 只有「訓練時就使用 train/val/test 資料夾切分」的模型，test 才是真的沒看過的資料。
  在資料夾切分建立之前訓練的舊模型（run_144 以前），test 影片很可能在它們的
  訓練集裡，分數會偏高——腳本會在報表中標示。

用法：
    python eval_gcn_test_set.py ../../stgcn_models/run_145_xy_conf_v_bone_att_on   # 或 run 資料夾完整路徑
    python eval_gcn_test_set.py run_145 run_146          # 可只寫 run 編號或資料夾名
    python eval_gcn_test_set.py                          # 不給參數：列出最近的模型讓你選（Enter＝最新一個）
"""
import argparse
import contextlib
import csv
import importlib.util
import io
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parent
_TRAIN_SCRIPT = _TOOLS / "0_train_gcn.py"
MODELS_ROOT = Path(__file__).resolve().parents[2] / "stgcn_models"
OUT_ROOT = _TOOLS.parent / "cat_monitoring_system" / "eval_results" / "gcn_test_set"


def _load_train_module():
    spec = importlib.util.spec_from_file_location("_gcn_train_ref", str(_TRAIN_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_run(arg):
    """接受 run 資料夾路徑、.pth 路徑、'run_145' 或 '145'，回傳 (run_dir, pth_path)。"""
    p = Path(arg)
    if p.suffix == ".pth" and p.exists():
        return p.parent, p
    if not p.exists():
        num = arg.replace("run_", "")
        hits = sorted(MODELS_ROOT.glob(f"run_{num}_*")) or sorted(MODELS_ROOT.glob(f"{arg}*"))
        if not hits:
            raise SystemExit(f"找不到模型：{arg}")
        p = hits[0]
    pths = sorted(p.glob("*_best_model.pth")) or sorted(p.glob("*.pth"))
    if not pths:
        raise SystemExit(f"{p} 裡沒有 .pth")
    return p, pths[0]


def _mcnemar(correct_a, correct_b):
    from scipy.stats import binomtest
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(b, b + c, 0.5).pvalue)


def _run_dirs_recent(limit=15):
    """最近的 run 資料夾（有 .pth 的），依 run 編號由新到舊。"""
    runs = []
    for d in MODELS_ROOT.glob("run_*"):
        m = __import__("re").match(r"run_(\d+)", d.name)
        if m and d.is_dir() and any(d.glob("*.pth")):
            runs.append((int(m.group(1)), d))
    return [d for _, d in sorted(runs, reverse=True)[:limit]]


def _pick_models_interactively(split_tag):
    """沒給模型參數時：列出最近的 run，標出哪些是用 train/val/test 資料夾切分訓練的（test 才可信），讓使用者選。"""
    runs = _run_dirs_recent()
    if not runs:
        raise SystemExit(f"❌ {MODELS_ROOT} 裡找不到任何訓練好的模型")
    rows = []
    for d in runs:
        ok, test_f1 = False, None
        rl = d / "run_log.json"
        if rl.exists():
            meta = json.loads(rl.read_text(encoding="utf-8")).get("meta", {})
            ok = meta.get("split_source") == split_tag
            test_f1 = meta.get("final_result", {}).get("test_macro_f1")
        rows.append((d, ok, test_f1))

    print("\n最近訓練的模型（✓＝用 train/val/test 資料夾切分訓練，test 分數才可信；⚠＝舊切分，test 可能被訓練過）：")
    for d, ok, f1 in rows:
        num = d.name.split("_")[1]
        extra = f"  訓練時 test macro-F1={f1:.4f}" if f1 is not None else ""
        print(f"  {'✓' if ok else '⚠'} {num:>4}  {d.name}{extra}")
    default = next((d for d, ok, _ in rows if ok), rows[0][0])
    default_num = default.name.split("_")[1]
    ans = input(f"\n要評估哪些模型？輸入 run 編號，多個用空白隔開（例如 147 150；Enter＝{default_num}）：").strip()
    return ans.split() if ans else [default_num]


def main():
    ap = argparse.ArgumentParser(description="在固定 test 集（skeletons/test/）上比較 ST-GCN 行為分類模型")
    ap.add_argument("models", nargs="*", help="run 資料夾／.pth／run 編號；不給則在終端機列出最近的模型讓你選")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print("載入 0_train_gcn.py（重用同一套資料載入與模型定義）...")
    with contextlib.redirect_stdout(io.StringIO()):
        g = _load_train_module()
    import torch
    from sklearn.metrics import confusion_matrix, f1_score, recall_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    skel_root = Path(g.SKELETON_DATA_FOLDER)
    if not (skel_root / "test").is_dir():
        raise SystemExit(f"找不到 {skel_root / 'test'}（先執行 tools/gcn_dataset_manager.py（模式 2））")
    split_tag = g.split_source_tag()
    if not args.models:
        args.models = _pick_models_interactively(split_tag)
    class_names = [n for n, _ in sorted(g.BEHAVIOR_PREFIXES.items(), key=lambda kv: kv[1])]
    labels_range = list(range(len(class_names)))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset_cache = {}
    results = []
    for arg in args.models:
        run_dir, pth = _resolve_run(arg)
        snap = json.loads((run_dir / "params_snapshot.json").read_text(encoding="utf-8"))
        ep, fc = snap["effective_params"], snap["full_config_snapshot"]
        name = run_dir.name.split("_")[1] if run_dir.name.startswith("run_") else run_dir.name
        name = f"Run{name}"

        # 這個模型訓練時是否就用這份切分——不是的話，test 影片可能被它訓練過
        leak = True
        rl = run_dir / "run_log.json"
        if rl.exists():
            src = json.loads(rl.read_text(encoding="utf-8")).get("meta", {}).get("split_source")
            leak = src != split_tag

        key = (ep["num_joints"], ep["sequence_length"], ep["window_stride"], ep["feature_mode"],
               ep.get("smoothing_kind"), ep.get("kp_ema_alpha"),
               ep.get("kalman_process_noise"), ep.get("kalman_measurement_noise"))
        if key not in dataset_cache:
            print(f"\n[載入骨架] joints={key[0]} T={key[1]} stride={key[2]} smoothing={key[4]}")
            with contextlib.redirect_stdout(io.StringIO()):
                ds = g.CatSkeletonDataset(
                    fc["SKELETON_DATA_FOLDER"], sequence_length=key[1], num_joints=key[0],
                    augment=False, feature_mode=key[3], window_stride=key[2],
                    kp_ema_alpha=key[5], smoothing_kind=key[4],
                    kalman_process_noise=key[6], kalman_measurement_noise=key[7])
            idx = [i for i, s in enumerate(ds.sequences) if s["split"] == "test"]
            dataset_cache[key] = (ds, idx)
        ds, idx = dataset_cache[key]

        model = g.STGCN(
            num_classes=ep["num_classes"], in_channels=ep["in_channels"], num_joints=ep["num_joints"],
            spatial_kernel_size=fc["SPATIAL_KERNEL_SIZE"], temporal_kernel_size=fc["TEMPORAL_KERNEL_SIZE"],
            num_layers=fc["NUM_STGCN_LAYERS"], use_attention=ep["use_attention"],
        ).to(device)
        model.load_state_dict(torch.load(pth, map_location=device), strict=True)
        model.eval()

        preds, trues, keys = [], [], []
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=64, shuffle=False)
        with torch.no_grad():
            for x, y in loader:
                preds += model(x.to(device)).argmax(1).cpu().tolist()
                trues += y.tolist()
        keys = [(ds.sequences[i]["video_id"], ds.sequences[i]["start_idx"]) for i in idx]
        preds, trues = np.array(preds), np.array(trues)
        rec = recall_score(trues, preds, labels=labels_range, average=None, zero_division=0)
        f1c = f1_score(trues, preds, labels=labels_range, average=None, zero_division=0)
        results.append(dict(
            name=name + (" ⚠" if leak else ""), run_dir=run_dir, leak=leak, keys=keys,
            preds=preds, trues=trues, acc=float((preds == trues).mean()),
            macro_f1=float(f1_score(trues, preds, labels=labels_range, average="macro", zero_division=0)),
            bal_acc=float(rec.mean()), recall=rec, f1=f1c,
            cm=confusion_matrix(trues, preds, labels=labels_range)))

    # ── 報表 ──
    n_vid = len({k[0] for k in results[0]["keys"]})
    print(f"\ntest 資料夾：{skel_root / 'test'}")
    print(f"test：{n_vid} 支影片 / {len(results[0]['keys'])} 個視窗\n")
    w = max(12, max(len(r["name"]) for r in results) + 2)
    print(f"  {'指標':<16}" + "".join(f"{r['name']:>{w}}" for r in results))
    for lab, k in (("macro-F1", "macro_f1"), ("balanced acc", "bal_acc"), ("accuracy", "acc")):
        print(f"  {lab:<16}" + "".join(f"{r[k]:>{w}.4f}" for r in results))
    for c, cn in enumerate(class_names):
        print(f"  {cn + ' recall':<16}" + "".join(f"{r['recall'][c]:>{w}.3f}" for r in results))
    if any(r["leak"] for r in results):
        print("\n  ⚠ 標記的模型訓練時沒有使用 train/val/test 資料夾切分，test 影片可能在它的訓練集裡，分數會偏高，"
              "不能跟使用資料夾切分訓練的模型直接比較。")

    same_windows = all(r["keys"] == results[0]["keys"] for r in results)
    mcn = []
    if len(results) > 1 and same_windows:
        print("\n  McNemar（兩兩比較，同一批視窗）：")
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                a, b = results[i], results[j]
                bb, cc, p = _mcnemar(a["preds"] == a["trues"], b["preds"] == b["trues"])
                mcn.append((a["name"], b["name"], bb, cc, p))
                print(f"    {a['name']} vs {b['name']}：只有前者對 {bb}、只有後者對 {cc}，p={p:.4f}"
                      + ("  （顯著）" if p < 0.05 else ""))
    elif len(results) > 1:
        print("\n  （各模型的切窗設定不同，視窗無法一一對應，略過 McNemar）")

    # ── 存檔 ──
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    num = len([p for p in OUT_ROOT.iterdir() if p.is_dir()]) + 1
    out = OUT_ROOT / f"{num:03d}_{'_vs_'.join(r['name'].replace(' ⚠', '') for r in results)}"
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["metric"] + [r["name"] for r in results])
        for lab, k in (("macro_f1", "macro_f1"), ("balanced_acc", "bal_acc"), ("accuracy", "acc")):
            wr.writerow([lab] + [f"{r[k]:.4f}" for r in results])
        for c, cn in enumerate(class_names):
            wr.writerow([f"{cn}_recall"] + [f"{r['recall'][c]:.4f}" for r in results])
            wr.writerow([f"{cn}_f1"] + [f"{r['f1'][c]:.4f}" for r in results])
        wr.writerow(["trained_with_this_split"] + [str(not r["leak"]) for r in results])
        for a, b, bb, cc, p in mcn:
            wr.writerow([f"mcnemar_{a}_vs_{b}", f"b={bb}", f"c={cc}", f"p={p:.4f}"])

    with open(out / "per_video.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["video_id", "true", "n_windows"] + [f"{r['name']}_acc" for r in results])
        for r0 in results[:1]:
            vids = sorted({k[0] for k in r0["keys"]})
        for v in vids:
            row, tl, nw = [], None, None
            for r in results:
                m = np.array([k[0] == v for k in r["keys"]])
                nw = int(m.sum())
                tl = class_names[int(r["trues"][m][0])]
                row.append(f"{(r['preds'][m] == r['trues'][m]).mean():.3f}")
            wr.writerow([v, tl, nw] + row)

    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 5))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, results):
        cm = r["cm"]
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(labels_range); ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(labels_range); ax.set_yticklabels(class_names)
        ax.set_title(f"{r['name'].replace(' ⚠', '（訓練未用此切分）')}\nmacro-F1={r['macro_f1']:.3f}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in labels_range:
            for j in labels_range:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(out / "test_confusion_matrices.png", dpi=200)
    plt.close()
    print(f"\n✓ 結果已存到：{out}")
    print(f"  （{datetime.now():%Y-%m-%d %H:%M}，summary.csv / per_video.csv / test_confusion_matrices.png）")


if __name__ == "__main__":
    main()
