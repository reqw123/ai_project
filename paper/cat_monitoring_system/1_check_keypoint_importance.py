"""
用「已經訓練好的 checkpoint」直接跑 diagnose_keypoint_motion()，不用重新訓練。

用途：驗證「某個行為的判別力集中在哪個關節」這類假設（例如 scratch 是否真的
依賴後腳尖 LH_Paw/RH_Paw，而不是原本猜測的前腳 LF_Paw/RF_Paw），只需要一個
現成的 .pth 檔案，幾秒鐘內就能拿到跟訓練時同一份驗證集切分（同一組
RANDOM_SEED/TRAIN_TEST_SPLIT，見 0_train_gcn.py 的 split_train_val_indices()）
上的逐關節動作幅度比較，不用等一次完整訓練跑完。

用法：
    python 1_check_keypoint_importance.py --model_path <checkpoint.pth> --class_a scratch --class_b stop
    python 1_check_keypoint_importance.py --model_path <checkpoint.pth> --class_a lick --class_b stop --output_dir eval_results

    # 純資料版（不載入模型、不做推論）：驗證「模型判對樣本」的逐關節排名
    # 是否為 5 類別聯合訓練造成的選樣偏差，而不是資料本身的特性
    python 1_check_keypoint_importance.py --mode groundtruth --class_a scratch --class_b stop
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from models.stgcn_model import (
    CatBehaviorSTGCN,
    flip_normalize,
    orientation_normalize,
    normalize_skeleton_coords,
    build_feature_tensor,
)

import importlib.util

_TRAIN_MODULE_PATH = Path(__file__).parent / "0_train_gcn.py"


def _load_train_module():
    """把 0_train_gcn.py 當一般模組載入（跟訓練腳本共用 CatSkeletonDataset／
    split_train_val_indices／diagnose_keypoint_motion／CONFIG 常數），避免另外
    複製一份容易失去同步的程式碼。"""
    spec = importlib.util.spec_from_file_location("_train_gcn", _TRAIN_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_train_gcn"] = module
    spec.loader.exec_module(module)
    return module


CH_TO_FEATURE = {
    2: 'xy',
    3: 'xy_conf',
    5: 'xy_conf_v',
    7: 'xy_conf_v_bone',
    9: 'xy_conf_v_bone_bmotion',
}


def infer_bn_input_channels(model_path: str):
    ck = torch.load(model_path, map_location='cpu')
    sd = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
    if isinstance(sd, dict):
        for k, v in sd.items():
            if k.endswith('bn_input.weight'):
                return int(v.shape[0])
    return None


def main():
    parser = argparse.ArgumentParser(description="用現成 checkpoint 跑逐關節動作幅度診斷，不用重新訓練。")
    parser.add_argument('--mode', choices=['model', 'groundtruth'], default='model',
                         help="model: 用 checkpoint 推論，只統計「判對」的樣本（預設）。"
                              " groundtruth: 不載入模型，用全部資料的 ground truth 標籤統計，"
                              "用來檢查 model 模式的排名是否為訓練選樣偏差。")
    parser.add_argument('--model_path', required=False, help='.pth checkpoint 路徑（--mode model 時必填）')
    parser.add_argument('--class_a', default='scratch')
    parser.add_argument('--class_b', default='stop')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default=None, help='若提供，額外存成 {class_a}_{class_b}_diagnosis.txt/.csv')
    parser.add_argument('--max_examples', type=int, default=8)
    args = parser.parse_args()

    if args.mode == 'model' and not args.model_path:
        raise SystemExit("--mode model 需要 --model_path")

    tg = _load_train_module()

    if args.mode == 'groundtruth':
        print("\n[純資料版模式] 不載入模型、不做推論，只依 ground truth 標籤統計")
        feature_mode = 'xy_conf_v_bone'
        print("\n[載入資料集]")
        full_dataset = tg.CatSkeletonDataset(
            tg.SKELETON_DATA_FOLDER,
            sequence_length=tg.SEQUENCE_LENGTH,
            num_joints=tg.NUM_JOINTS,
            augment=False,
            feature_mode=feature_mode,
            window_stride=tg.WINDOW_STRIDE,
        )
        if len(full_dataset) == 0:
            print("✗ 資料集是空的，請確認 SKELETON_DATA_FOLDER。")
            return
        all_indices = range(len(full_dataset.sequences))
        tg.diagnose_keypoint_motion_groundtruth(
            full_dataset, all_indices,
            args.class_a, args.class_b,
            output_dir=args.output_dir,
        )
        return

    bn_ch = infer_bn_input_channels(args.model_path)
    feature_mode = CH_TO_FEATURE.get(bn_ch, 'xy_conf_v_bone')
    print(f"[偵測] checkpoint bn_input channels={bn_ch} → feature_mode={feature_mode}")

    print("\n[載入資料集]")
    full_dataset = tg.CatSkeletonDataset(
        tg.SKELETON_DATA_FOLDER,
        sequence_length=tg.SEQUENCE_LENGTH,
        num_joints=tg.NUM_JOINTS,
        augment=False,
        feature_mode=feature_mode,
        window_stride=tg.WINDOW_STRIDE,
    )
    if len(full_dataset) == 0:
        print("✗ 資料集是空的，請確認 SKELETON_DATA_FOLDER。")
        return

    print("\n[切分驗證集]（跟 train_model() 同一套邏輯與亂數種子）")
    train_indices, val_indices = tg.split_train_val_indices(full_dataset)

    print(f"\n[載入模型] {args.model_path}")
    classifier = CatBehaviorSTGCN(
        args.model_path,
        device=args.device,
        sequence_length=tg.SEQUENCE_LENGTH,
        normalize=True,
        feature_mode=feature_mode,
        in_channels=bn_ch,
    )

    print(f"\n[推論驗證集] 共 {len(val_indices)} 個 window ...")
    val_labels = []
    val_preds = []
    for idx in val_indices:
        item = full_dataset.sequences[idx]
        seq = item['sequence']
        conf_seq = item['conf_sequence']
        if full_dataset.num_joints < seq.shape[1]:
            seq = seq[:, :full_dataset.num_joints, :]
            conf_seq = conf_seq[:, :full_dataset.num_joints]
        seq = flip_normalize(seq)
        seq = orientation_normalize(seq)
        seq = normalize_skeleton_coords(seq)
        feats = build_feature_tensor(seq, conf_seq, feature_mode)
        pred_id, _, _ = classifier.predict(feats, precomputed=True)
        val_labels.append(item['label'])
        val_preds.append(int(pred_id) if pred_id is not None else -1)

    acc = float(np.mean([p == l for p, l in zip(val_preds, val_labels)]))
    print(f"  驗證集整體準確率: {acc:.1%}（僅供檢查推論是否正常，非正式評估指標）")

    tg.diagnose_keypoint_motion(
        full_dataset, val_indices, val_labels, val_preds,
        args.class_a, args.class_b,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
