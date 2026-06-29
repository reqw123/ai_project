"""
ST-GCN 行為分類封裝（含前處理）

classify() 統一接收 (T,V,2) raw 座標 + (T,V) 信心值，
內部根據模型 in_channels 決定前處理路徑，
呼叫方無需感知 multichannel / legacy 差異。
"""
from models.stgcn_model import (
    CatBehaviorSTGCN,
    flip_normalize,
    orientation_normalize,
    normalize_skeleton_coords,
    build_feature_tensor,
)


class BehaviorClassifier:
    def __init__(self, model_path, device='cuda', sequence_length=16, normalize=True,
                 feature_mode=None, in_channels=None, use_attention=None):
        # feature_mode=None 時從 config 讀取，確保 config 設定始終生效
        if feature_mode is None:
            try:
                from config import STGCNConfig
                feature_mode = STGCNConfig.FEATURE_MODE
            except Exception:
                feature_mode = 'xy'
        self.model = CatBehaviorSTGCN(
            model_path,
            device=device,
            sequence_length=sequence_length,
            normalize=normalize,
            feature_mode=feature_mode,
            in_channels=in_channels,
            use_attention=use_attention,
        )

    @property
    def _is_legacy_4ch(self) -> bool:
        """True = 舊版 4ch (xy_v) 路徑，由 CatBehaviorSTGCN.predict(precomputed=False) 處理。"""
        return self.model.in_channels == 4

    def classify(self, seq_xy, conf_seq=None):
        """
        Args:
            seq_xy:   (T, V, 2)  插值補點後（可含 EMA 平滑）的關鍵點座標
            conf_seq: (T, V)     原始信心值序列（legacy 4ch 模式可傳 None）

        Returns:
            (behavior_id, confidence, class_probs)
        """
        if seq_xy is None:
            return None, 0.0, None

        if self._is_legacy_4ch:
            # 舊路徑：CatBehaviorSTGCN.predict() 內部自行做 flip+orient+normalize+velocity
            return self.model.predict(seq_xy, precomputed=False)

        # 多通道路徑：前處理在此完成，model.predict() 接收已建構好的特徵張量
        m = self.model
        seq = seq_xy.copy()
        if m.normalize:
            seq = flip_normalize(seq)
            seq = orientation_normalize(seq)
            seq = normalize_skeleton_coords(seq)
        seq_features = build_feature_tensor(seq, conf_seq, m.feature_mode)
        return self.model.predict(seq_features, precomputed=True)
