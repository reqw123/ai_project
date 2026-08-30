"""
2_train.py — 兩隻固定貓咪的身分辨識 CNN（獨立腳本）
================================================================
完全獨立，不 import 本專案任何自訂模組，也不碰 YOLO / Tracking / Pose /
ST-GCN / FrameProcessor。只用 PyTorch + torchvision + scikit-learn +
matplotlib。

做的事：載入 torchvision 的 MobileNetV3-Small（ImageNet 預訓練），把分類頭
換成 2 類（Cat_A / Cat_B），先凍結 backbone 只訓練新頭，可選擇解凍最後
幾個 feature block 做 fine-tuning，用 CUDA 訓練，最後在 test set 上完整評估。
另外提供單張圖片推論模式（低信心 → 輸出 Unknown，Unknown 不是訓練類別）。

不做：第三類、Re-ID、Metric Learning、ArcFace、Triplet、蒸餾、Web/API/GUI/DB。

─────────────────────────────────────────────────────────────────
執行環境：需要 torch / torchvision（CUDA 版）。本機請用專案的 conda 環境：
    C:\\Users\\lynnc\\anaconda3\\envs\\yolo\\python.exe 2_train.py
（該環境已有 torch 2.5.1+cu121 / torchvision 0.20.1 / scikit-learn / matplotlib）

用法：改下面「設定區」的參數，然後
    python 2_train.py           # MODE="train"：訓練 + 評估
    python 2_train.py           # MODE="predict"：單張圖片推論
─────────────────────────────────────────────────────────────────

預期資料夾（程式自行讀取兩個 class 子資料夾，名稱任意，依字母排序對應 index 0/1）：
    <DATASET_PATH>/
    ├─ cat_A/  *.jpg ...
    └─ cat_B/  *.jpg ...

輸出（每次訓練一個資料夾，放在 MODEL_DIR 底下）：
    <MODEL_DIR>/
    ├─ run_<YYYYMMDD-HHMM>/               資料夾＝訓練開始時間（到分鐘）
    │    ├─ <流水號>.pt                    best 權重（如 001.pt；流水號全域遞增，只留 best）
    │    ├─ class_names.json
    │    ├─ training_history.csv
    │    ├─ loss_curve.png / accuracy_curve.png / confusion_matrix.png
    │    ├─ test_metrics.json
    │    └─ run_meta.json                  含總訓練時長、參數、best epoch、test 準確率
    ├─ latest.pt                           固定路徑，永遠指向最近一次訓練的 best
    └─ latest.txt                          latest.pt 對應哪個 run
"""

# ═══════════════════════════════════════════════════════════════
#  設定區（所有主要參數集中在這裡）
# ═══════════════════════════════════════════════════════════════

# "train"   = 訓練 + 在 test set 上評估
# "predict" = 載入 best 權重，對單張 IMAGE_PATH 做推論
MODE = "train"

# ── 資料 ──
# 指到含兩個 class 子資料夾的目錄。可直接指向 1_build_dataset.py 的
# crops/ 輸出（子資料夾會是「目標貓 / 他貓」）。
DATASET_PATH = r"C:\ai_project\paper\cat_monitoring_system\tools\train_data\cat_identity\dataset\crops"

IMAGE_SIZE = 128            # 訓練 / 推論的正方形輸入邊長
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0            # Windows 上若 DataLoader 卡住，改成 0（小資料集用 0 最穩）
DEVICE = "cuda"           # "cuda" 優先，抓不到自動 fallback 到 "cpu"
RANDOM_SEED = 42

VALIDATION_RATIO = 0.15    # 佔整個資料集的比例
TEST_RATIO = 0.15         # 佔整個資料集的比例；train = 1 - VALIDATION_RATIO - TEST_RATIO

# ── 資料切分方式 ──
# "image" = 每張圖獨立切分（第一版預設）
# "group" = 依「來源影片」切分，同一支影片的幀不會同時出現在 train 和 test
#           （避免高相關的相鄰幀灌水準確率）。group key 由 group_key_from_path()
#           解析檔名 "{video_stem}__f000123.jpg" 得到（見 1_build_dataset.py）。
SPLIT_MODE = "group"

# ── 凍結 / 微調 ──
FREEZE_BACKBONE = True      # True：先凍結 backbone，只訓練新的分類頭
UNFREEZE_LAST_BLOCKS = 1    # 額外解凍 model.features 最後幾個 block 一起訓練（0 = 完全凍結 backbone）

# ── 訓練排程 ──
EARLY_STOPPING_PATIENCE = 12    # 連續幾個 epoch 的 val accuracy 沒進步就停
LR_SCHEDULER_PATIENCE = 4       # val accuracy 幾個 epoch 沒進步就降 LR
LR_SCHEDULER_FACTOR = 0.3
MIN_LR = 1e-6

# ── 輸出 ──
# 每次訓練在 MODEL_DIR 底下開一個資料夾（run_<YYYYMMDD-HHMM>），只留 best 權重
# <流水號>.pt（流水號全域遞增，掃 run_*/*.pt 取最大值 +1）+
# 所有訓練結果數據（曲線 / 混淆矩陣 / history csv / metrics / run_meta）都放在裡面。
# MODEL_DIR 與 yolo_models / stgcn_models 同層。
# 另外會在 MODEL_DIR 根目錄維護一份 latest.pt（指向最近一次訓練的 best），
# 給 3_infer_video.py 之類的腳本用固定路徑引用。
MODEL_DIR = r"C:\ai_project\identity_models"
LATEST_MODEL_NAME = "latest.pt"

# ── predict 模式 ──
IMAGE_PATH = r"test.jpg"
CONFIDENCE_THRESHOLD = 0.80    # 最高 softmax 低於此值 → 輸出 "Unknown"（非訓練類別）
# 留空 = 用 MODEL_DIR/latest.pt；要指定某次訓練就填該 run 資料夾裡的 <流水號>.pt 路徑
PREDICT_MODEL_PATH = r""

# ── 環境變數覆蓋（給 identity_trainer_window.py 這類 GUI 用；命令列直接跑就不會設）──
#   CAT_IDENTITY_DATASET_PATH        → 覆蓋 DATASET_PATH（GUI 每隻貓有各自的資料集資料夾）
#   CAT_IDENTITY_EPOCHS              → 覆蓋 EPOCHS 上限（進階手動調；GUI 不再設，交給 early stop）
#   CAT_IDENTITY_TARGET_DISPLAY_NAME → 目標貓的英文顯示別名，寫進權重檔的 display_names，
#                                      供推論疊框 / CSV 顯示（不影響內部類別名「目標貓/他貓」）
import os as _os_env
_env_dataset = _os_env.getenv("CAT_IDENTITY_DATASET_PATH", "").strip()
if _env_dataset:
    DATASET_PATH = _env_dataset
_env_epochs = _os_env.getenv("CAT_IDENTITY_EPOCHS", "").strip()
if _env_epochs.isdigit() and int(_env_epochs) > 0:
    EPOCHS = int(_env_epochs)
TARGET_DISPLAY_NAME = _os_env.getenv("CAT_IDENTITY_TARGET_DISPLAY_NAME", "").strip() or "target"

# ═══════════════════════════════════════════════════════════════

import sys
import os
import re
import json
import time
import random
import csv as _csv
from datetime import datetime
from pathlib import Path

# ── 套件檢查（給清楚的安裝提示，而不是一句 ModuleNotFoundError）──
_MISSING = []
try:
    import numpy as np
except ImportError:
    _MISSING.append("numpy")
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    _MISSING.append("torch")
try:
    import torchvision
    from torchvision import transforms
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
except ImportError:
    _MISSING.append("torchvision")
try:
    from PIL import Image
except ImportError:
    _MISSING.append("pillow")
try:
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, confusion_matrix,
    )
except ImportError:
    _MISSING.append("scikit-learn")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as _fm
except ImportError:
    _MISSING.append("matplotlib")

if _MISSING:
    print("❌ 缺少套件：", ", ".join(_MISSING))
    print("   本機請改用專案 conda 環境執行，例如：")
    print(r'   & "C:\Users\lynnc\anaconda3\envs\yolo\python.exe" 2_train.py')
    print("   或安裝（CUDA 12.1）：")
    print("   pip install --extra-index-url https://download.pytorch.org/whl/cu121 \\")
    print("       torch==2.5.1+cu121 torchvision==0.20.1+cu121")
    print("   pip install scikit-learn matplotlib pillow numpy")
    sys.exit(1)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ═══════════════════════════════════════════════════════════════
#  共用工具
# ═══════════════════════════════════════════════════════════════
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 兼顧可重現與速度：固定 seed，但仍允許 cudnn 選最快演算法
    torch.backends.cudnn.benchmark = True


def setup_cjk_font():
    """讓 matplotlib 圖上的中文類別名（例如「目標貓 / 他貓」）能正常顯示；
    找不到中文字型就維持預設（標籤會變成方框，但不影響數值與流程）。"""
    for name in ("Microsoft JhengHei", "Microsoft YaHei", "SimHei",
                 "PingFang TC", "Noto Sans CJK TC", "Noto Sans CJK SC"):
        if any(f.name == name for f in _fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def get_device():
    if DEVICE.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if DEVICE.lower() == "cuda":
        print("⚠ 找不到可用的 CUDA，改用 CPU")
    return torch.device("cpu")


def format_hms(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def make_run(model_dir: Path):
    """回傳 (run_id, serial)：
      - run_id = `<YYYYMMDD-HHMM>`（資料夾用；同一分鐘重跑加 `-2`/`-3` 後綴）
      - serial = 全域流水號（模型檔名用，掃現有 run_*/NNN.pt 取最大值 +1，從 1 起）"""
    serial = 0
    for f in model_dir.glob("run_*/*.pt"):
        m = re.match(r"(\d+)(?:_last)?\.pt$", f.name)
        if m:
            serial = max(serial, int(m.group(1)))
    serial += 1

    base = f"{datetime.now():%Y%m%d-%H%M}"
    run_id, n = base, 1
    while (model_dir / f"run_{run_id}").exists():
        n += 1
        run_id = f"{base}-{n}"
    return run_id, serial


def group_key_from_path(path):
    """把檔名 "{video_stem}__f000123.jpg" 還原成來源影片 key；
    沒有 "__f" 樣式的檔名就以整個檔名當 key（等同 image-level）。"""
    stem = Path(path).stem
    return stem.split("__f")[0] if "__f" in stem else stem


def scan_dataset(root):
    """回傳 (samples, class_names)。
    samples = [(image_path:str, label:int), ...]；class_names 依字母排序。"""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"DATASET_PATH 不存在或不是資料夾: {root}")
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if len(class_dirs) != 2:
        raise ValueError(
            f"預期剛好 2 個 class 子資料夾，實際找到 {len(class_dirs)} 個："
            f"{[d.name for d in class_dirs]}"
        )
    class_names = [d.name for d in class_dirs]
    samples = []
    for label, d in enumerate(class_dirs):
        files = [p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        for p in sorted(files):
            samples.append((str(p), label))
    if not samples:
        raise ValueError(f"兩個 class 資料夾裡都沒有圖片檔: {root}")
    return samples, class_names


def split_dataset(samples):
    """依 SPLIT_MODE 切成 train / val / test，三者比例由 VALIDATION_RATIO /
    TEST_RATIO 決定，固定 RANDOM_SEED 可重現。回傳三個 samples 子清單。"""
    paths = np.array([s[0] for s in samples])
    labels = np.array([s[1] for s in samples])
    idx = np.arange(len(samples))
    val_plus_test = VALIDATION_RATIO + TEST_RATIO
    if not (0 < val_plus_test < 1):
        raise ValueError("VALIDATION_RATIO + TEST_RATIO 必須介於 0 和 1 之間")

    mode = SPLIT_MODE
    if mode == "group":
        groups = np.array([group_key_from_path(p) for p in paths])
        per_class_groups = {
            int(l): len(set(groups[labels == l].tolist())) for l in np.unique(labels)
        }
        # 影片層級切分要「同一支影片不同時進 train 和 test」，所以每一類至少要有 2 支
        # 來源影片才切得動；只有 1 支時 GroupShuffleSplit 會直接報錯。
        if min(per_class_groups.values(), default=0) < 2:
            print(
                f"⚠ 影片層級切分(group)需要每類至少 2 支來源影片，目前各類影片數 = {per_class_groups}。\n"
                f"   → 自動改用影格層級切分(image)。注意：同一支影片的相鄰影格會同時落在\n"
                f"     train / val / test，測試準確率會偏高、不代表真實泛化能力。\n"
                f"     多拍幾支「不同時段 / 不同場景」的影片再重新訓練即可獲得可信的評估。"
            )
            mode = "image"

    if mode == "group":
        groups = np.array([group_key_from_path(p) for p in paths])
        gss1 = GroupShuffleSplit(n_splits=1, test_size=val_plus_test, random_state=RANDOM_SEED)
        train_idx, hold_idx = next(gss1.split(idx, labels, groups))
        rel_test = TEST_RATIO / val_plus_test
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_test, random_state=RANDOM_SEED)
        val_rel, test_rel = next(gss2.split(hold_idx, labels[hold_idx], groups[hold_idx]))
        val_idx, test_idx = hold_idx[val_rel], hold_idx[test_rel]
    elif mode == "image":
        train_idx, hold_idx = train_test_split(
            idx, test_size=val_plus_test, random_state=RANDOM_SEED, stratify=labels,
        )
        rel_test = TEST_RATIO / val_plus_test
        val_idx, test_idx = train_test_split(
            hold_idx, test_size=rel_test, random_state=RANDOM_SEED, stratify=labels[hold_idx],
        )
    else:
        raise ValueError(f"未知的 SPLIT_MODE: {SPLIT_MODE!r}（應為 'image' 或 'group'）")

    take = lambda ii: [samples[i] for i in ii]
    return take(train_idx), take(val_idx), take(test_idx)


def build_transforms(train):
    """train=True 給訓練用 augmentation；train=False 只做必要的 resize/normalize。
    刻意不動 hue，saturation 也只給極小幅度——毛色是分辨兩隻貓的關鍵特徵。"""
    resize_to = int(round(IMAGE_SIZE * 1.25))
    if train:
        return transforms.Compose([
            transforms.Resize((resize_to, resize_to)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05, hue=0.0),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((resize_to, resize_to)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CatImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as im:
            im = im.convert("RGB")
        return self.transform(im), label


def build_model(num_classes=2):
    """ImageNet 預訓練 MobileNetV3-Small，分類頭換成 num_classes。"""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    # classifier = Sequential(Linear(576,1024), Hardswish, Dropout, Linear(1024, 1000))
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    if FREEZE_BACKBONE:
        for p in model.features.parameters():
            p.requires_grad = False
        if UNFREEZE_LAST_BLOCKS > 0:
            for block in model.features[-UNFREEZE_LAST_BLOCKS:]:
                for p in block.parameters():
                    p.requires_grad = True
    # 分類頭一律可訓練
    for p in model.classifier.parameters():
        p.requires_grad = True
    return model


def count_labels(samples):
    c = [0, 0]
    for _, y in samples:
        c[y] += 1
    return c


# ═══════════════════════════════════════════════════════════════
#  訓練
# ═══════════════════════════════════════════════════════════════
def run_one_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += x.size(0)
    torch.set_grad_enabled(True)
    return total_loss / total_n, total_correct / total_n


def train_main():
    t_start = time.perf_counter()
    set_seed(RANDOM_SEED)
    device = get_device()

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_id, serial = make_run(model_dir)     # run_id=<YYYYMMDD-HHMM> 資料夾用；serial=模型檔名流水號
    run_dir = model_dir / f"run_{run_id}"    # 這次訓練的專屬資料夾（權重 + 所有結果數據都放這）
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / f"{serial:03d}.pt"       # 只留 best（如 003.pt）；infer 用這個

    samples, class_names = scan_dataset(DATASET_PATH)
    train_s, val_s, test_s = split_dataset(samples)

    ca = count_labels(samples)
    print("=" * 60)
    print("貓咪身分辨識 CNN 訓練")
    print("=" * 60)
    print(f"輸出資料夾       : {run_dir}   （模型 {serial:03d}.pt）")
    print(f"裝置             : {device}")
    if device.type == "cuda":
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print(f"資料夾           : {DATASET_PATH}")
    print(f"類別             : [0] {class_names[0]}   [1] {class_names[1]}")
    print(f"資料集總數       : {len(samples)}")
    print(f"  {class_names[0]:<12}: {ca[0]}")
    print(f"  {class_names[1]:<12}: {ca[1]}")
    print(f"切分方式         : {SPLIT_MODE}  (seed={RANDOM_SEED})")
    print(f"  Train          : {len(train_s)}  {count_labels(train_s)}")
    print(f"  Validation     : {len(val_s)}  {count_labels(val_s)}")
    print(f"  Test           : {len(test_s)}  {count_labels(test_s)}")
    print(f"影像尺寸 / batch : {IMAGE_SIZE} / {BATCH_SIZE}")
    print(f"凍結 backbone    : {FREEZE_BACKBONE}  (解凍最後 {UNFREEZE_LAST_BLOCKS} 個 block)")
    print("=" * 60)

    if min(ca) < 15:
        print("⚠ 有一類樣本數 < 15，訓練可能不穩定，建議補資料")

    (run_dir / "class_names.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        CatImageDataset(train_s, build_transforms(True)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        CatImageDataset(val_s, build_transforms(False)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin,
    )
    test_loader = DataLoader(
        CatImageDataset(test_s, build_transforms(False)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin,
    )

    model = build_model(num_classes=2).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"可訓練參數       : {n_trainable:,} / {n_total:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE, min_lr=MIN_LR,
    )

    history = []
    best_val_acc = -1.0
    best_epoch = 0
    stopped_epoch = EPOCHS
    epochs_no_improve = 0
    t_train_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_one_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_one_epoch(model, val_loader, criterion, device, None)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(va_acc)

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | lr {lr_now:.2e}"
        )
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc, "lr": lr_now,
        })

        ckpt = {
            "model_state": model.state_dict(),
            "arch": "mobilenet_v3_small",
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
            "norm_mean": IMAGENET_MEAN,
            "norm_std": IMAGENET_STD,
            "epoch": epoch,
            "val_acc": va_acc,
            # 顯示別名（僅供疊框 / CSV 顯示；內部類別名仍是「目標貓 / 他貓」）
            "display_names": {"目標貓": TARGET_DISPLAY_NAME, "他貓": "other"},
        }

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(ckpt, best_path)
            print(f"        ↑ 新的最佳 val accuracy: {best_val_acc:.4f}（已存 {best_path.name}）")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping：val accuracy 連續 {EARLY_STOPPING_PATIENCE} 個 epoch 沒進步")
                stopped_epoch = epoch
                break
    else:
        stopped_epoch = EPOCHS

    train_seconds = time.perf_counter() - t_train_start

    # ── 存訓練歷程 + 曲線（全部進 run_dir）──
    with (run_dir / "training_history.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
        w.writeheader()
        w.writerows(history)
    _plot_curves(history, run_dir)

    # ── 用 best 權重在 test set 上評估 ──
    print(f"\n以 best 權重（epoch {best_epoch}, val acc {best_val_acc:.4f}）在 test set 上評估...")
    test_metrics = evaluate(best_path, test_loader, device, run_dir)

    # ── latest.pt：MODEL_DIR 根目錄的固定路徑，永遠指向最近一次訓練的 best ──
    if best_path.exists():
        import shutil as _shutil
        _shutil.copy2(best_path, model_dir / LATEST_MODEL_NAME)
        (model_dir / "latest.txt").write_text(
            str(Path(f"run_{run_id}") / best_path.name) + "\n", encoding="utf-8"
        )

    total_seconds = time.perf_counter() - t_start
    (run_dir / "run_meta.json").write_text(json.dumps({
        "run_id": run_id, "serial": serial,
        "target_display_name": TARGET_DISPLAY_NAME,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "total_seconds": round(total_seconds, 1),
        "total_time": format_hms(total_seconds),
        "train_seconds": round(train_seconds, 1),
        "train_time": format_hms(train_seconds),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "dataset_path": DATASET_PATH,
        "class_names": class_names,
        "class_counts": {class_names[i]: ca[i] for i in range(2)},
        "split_mode": SPLIT_MODE, "seed": RANDOM_SEED,
        "split_sizes": {"train": len(train_s), "val": len(val_s), "test": len(test_s)},
        "best_epoch": best_epoch, "best_val_acc": round(best_val_acc, 4),
        "stopped_epoch": stopped_epoch, "epochs_cap": EPOCHS,
        "params": {
            "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "freeze_backbone": FREEZE_BACKBONE, "unfreeze_last_blocks": UNFREEZE_LAST_BLOCKS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },
        "test_metrics": test_metrics,
        "files": {"best": best_path.name},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 最終總結（放在所有輸出的最後面）──
    test_acc = test_metrics.get("overall_accuracy") if test_metrics else None
    print("\n" + "═" * 60)
    print(f"  訓練完成  ({run_id})")
    print("═" * 60)
    print(f"  總訓練時長   : {format_hms(total_seconds)}  ({total_seconds:.0f} 秒)"
          f"   〔訓練迴圈 {format_hms(train_seconds)}〕")
    print(f"  最佳 epoch   : {best_epoch} / 上限 {EPOCHS}   val acc {best_val_acc:.4f}")
    print(f"  Test 準確率  : {test_acc:.4f}" if test_acc is not None else "  Test 準確率  : （無 test set）")
    print(f"  輸出資料夾   : {run_dir}")
    print(f"  best 權重    : {best_path}")
    print(f"  latest.pt    : {(model_dir / LATEST_MODEL_NAME).resolve()}   ← infer 腳本用這個（best 的複本）")
    print("═" * 60)


def _plot_curves(history, out_dir):
    ep = [h["epoch"] for h in history]
    plt.figure(figsize=(7, 4.5))
    plt.plot(ep, [h["train_loss"] for h in history], label="train")
    plt.plot(ep, [h["val_loss"] for h in history], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "loss_curve.png", dpi=120); plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(ep, [h["train_acc"] for h in history], label="train")
    plt.plot(ep, [h["val_acc"] for h in history], label="val")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "accuracy_curve.png", dpi=120); plt.close()


def _plot_confusion(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(class_names)
    ax.set_yticks([0, 1]); ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion Matrix (test)")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  評估
# ═══════════════════════════════════════════════════════════════
def _load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)  # 自己存的檔，含 class_names 等 metadata
    model = build_model_for_inference(len(ckpt["class_names"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def build_model_for_inference(num_classes):
    """推論用：結構跟 build_model 一致，但不套用凍結設定（載入權重即可）。"""
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(ckpt_path, test_loader, device, out_dir):
    """回傳 test 指標 dict（供 run_meta.json 收錄）；無法評估時回傳 None。"""
    if not Path(ckpt_path).exists():
        print(f"⚠ 找不到 {ckpt_path}，略過評估")
        return None
    model, ckpt = _load_checkpoint(ckpt_path, device)
    class_names = ckpt["class_names"]

    y_true, y_pred = [], []
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        pred = model(x).argmax(1).cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(y.numpy().tolist())
    y_true = np.array(y_true); y_pred = np.array(y_pred)

    if len(y_true) == 0:
        print("⚠ test set 是空的，略過評估")
        return None

    overall = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
    )
    p_c, r_c, f_c, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    per_class_acc = [
        (cm[i, i] / cm[i].sum()) if cm[i].sum() > 0 else float("nan") for i in range(2)
    ]

    print("\n" + "=" * 60)
    print(f"Test 評估（{len(y_true)} 張，權重 epoch={ckpt.get('epoch')}）")
    print("=" * 60)
    print(f"Overall accuracy : {overall:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro)   : {rec:.4f}")
    print(f"F1-score (macro) : {f1:.4f}")
    print("-" * 60)
    for i, name in enumerate(class_names):
        print(f"{name:<14} | acc {per_class_acc[i]:.4f} | "
              f"precision {p_c[i]:.4f} | recall {r_c[i]:.4f} | f1 {f_c[i]:.4f}")
    print("-" * 60)
    print("Confusion matrix (row = true, col = pred):")
    print(f"{'':<14}" + "".join(f"{n:>14}" for n in class_names))
    for i, name in enumerate(class_names):
        print(f"{name:<14}" + "".join(f"{cm[i, j]:>14}" for j in range(2)))
    print("-" * 60)
    print(f"{class_names[0]} accuracy : {per_class_acc[0]:.4f}")
    print(f"{class_names[1]} accuracy : {per_class_acc[1]:.4f}")
    print(f"Overall accuracy : {overall:.4f}")
    print("=" * 60)

    _plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")
    metrics = {
        "overall_accuracy": round(float(overall), 4),
        "precision_macro": round(float(prec), 4),
        "recall_macro": round(float(rec), 4),
        "f1_macro": round(float(f1), 4),
        "per_class": {
            class_names[i]: {
                "accuracy": None if np.isnan(per_class_acc[i]) else round(float(per_class_acc[i]), 4),
                "precision": round(float(p_c[i]), 4),
                "recall": round(float(r_c[i]), 4),
                "f1": round(float(f_c[i]), 4),
            } for i in range(2)
        },
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(y_true)),
    }
    (out_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


# ═══════════════════════════════════════════════════════════════
#  單張圖片推論
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def predict_main():
    device = get_device()
    ckpt_path = Path(PREDICT_MODEL_PATH) if PREDICT_MODEL_PATH else Path(MODEL_DIR) / LATEST_MODEL_NAME
    if not ckpt_path.exists():
        print(f"❌ 找不到權重: {ckpt_path}（請先用 MODE='train' 訓練，或設定 PREDICT_MODEL_PATH）")
        sys.exit(1)
    print(f"使用權重: {ckpt_path}")
    if not Path(IMAGE_PATH).exists():
        print(f"❌ 找不到圖片: {IMAGE_PATH}")
        sys.exit(1)

    model, ckpt = _load_checkpoint(ckpt_path, device)
    class_names = ckpt["class_names"]
    size = ckpt.get("image_size", IMAGE_SIZE)
    tfm = transforms.Compose([
        transforms.Resize((int(round(size * 1.25)), int(round(size * 1.25)))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(ckpt.get("norm_mean", IMAGENET_MEAN), ckpt.get("norm_std", IMAGENET_STD)),
    ])

    with Image.open(IMAGE_PATH) as im:
        x = tfm(im.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    top_i = int(probs.argmax())
    top_p = float(probs[top_i])

    print(f"\n圖片: {IMAGE_PATH}")
    if top_p < CONFIDENCE_THRESHOLD:
        print(f"Prediction: Unknown  (最高信心 {top_p:.4f} < 門檻 {CONFIDENCE_THRESHOLD})")
    else:
        print(f"Prediction: {class_names[top_i]}")
    print(f"Confidence: {top_p:.4f}")
    for i, name in enumerate(class_names):
        print(f"{name}: {probs[i]:.4f}")


# ═══════════════════════════════════════════════════════════════
def main():
    setup_cjk_font()
    if MODE == "train":
        train_main()
    elif MODE == "predict":
        predict_main()
    else:
        print(f"❌ 未知的 MODE: {MODE!r}（應為 'train' 或 'predict'）")
        sys.exit(1)


if __name__ == "__main__":
    main()
