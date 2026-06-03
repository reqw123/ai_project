"""
ST-GCN 行為分類封裝
"""
from models.stgcn_model import CatBehaviorSTGCN

class BehaviorClassifier:
    def __init__(self, model_path, device='cuda', sequence_length=16, normalize=True,
                 feature_mode=None, in_channels=None, use_attention=None):
        # feature_mode=None 時從 config 讀取，確保 config 設定始終生效
        if feature_mode is None:
            try:
                from config import STGCNConfig
                feature_mode = STGCNConfig.FEATURE_MODE
            except Exception:
                feature_mode = 'xy_v'
        self.model = CatBehaviorSTGCN(
            model_path,
            device=device,
            sequence_length=sequence_length,
            normalize=normalize,
            feature_mode=feature_mode,
            in_channels=in_channels,
            use_attention=use_attention,
        )

    def classify(self, keypoints_sequence, precomputed=False):
        if keypoints_sequence is None:
            return None, 0.0, None
        return self.model.predict(keypoints_sequence, precomputed=precomputed)
