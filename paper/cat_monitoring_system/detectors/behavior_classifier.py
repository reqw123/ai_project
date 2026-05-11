"""
ST-GCN 行為分類封裝
"""
from models.stgcn_model import CatBehaviorSTGCN

class BehaviorClassifier:
    def __init__(self, model_path, device='cuda', sequence_length=32, normalize=True, feature_mode='xyv', in_channels=None):
        self.model = CatBehaviorSTGCN(
            model_path,
            device=device,
            sequence_length=sequence_length,
            normalize=normalize,
            feature_mode=feature_mode,
            in_channels=in_channels,
        )

    def classify(self, keypoints_sequence, precomputed=False):
        return self.model.predict(keypoints_sequence, precomputed=precomputed)
