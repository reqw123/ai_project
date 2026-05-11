# ==================== 顏色 ====================
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED   = (0, 0, 255)
BLUE  = (255, 0, 0)

COLOR_HEAD = (255, 255, 0)
COLOR_BODY = (0, 255, 0)
COLOR_LIMB = (255, 0, 0)
COLOR_TAIL = (255, 0, 255)
COLOR_KPT  = (0, 0, 255)

# 四隻腳的顏色（左前、右前、左後、右後）
COLOR_LEFT_FRONT  = (255, 0, 255)   # 洋紅色 (magenta)
COLOR_RIGHT_FRONT = (0, 255, 255)   # 青色 (cyan)
COLOR_LEFT_HIND   = (255, 165, 0)   # 橙色 (orange)
COLOR_RIGHT_HIND  = (0, 255, 0)     # 綠色 (green)

# ==================== 骨架連結 ====================
HEAD_LINKS  = [(0,1),(0,2),(1,2)]
BODY_LINKS  = [(0,3),(3,4),(4,5)]
FRONT_LIMBS = [(3,6),(6,7),(3,8),(8,9)]
HIND_LIMBS  = [(5,10),(10,11),(5,12),(12,13)]
TAIL_LINKS  = [(5,14),(14,15),(15,16)]


import os
import cv2
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Import model architecture from training script
# (In practice, you would import from a separate module)
import sys
sys.path.append(os.path.dirname(__file__))

# ==================== Configuration ====================
TEST_VIDEOS_FOLDER = r"C:\cat_pose\模型測試影片\cat5.mp4"
RESULTS_FOLDER = r"C:\cat_pose\gcn_pose\results"
YOLO_MODEL_PATH = r"C:\cat_pose\no_aug.pt"
MODEL_PATH = r"C:\cat_pose\gcn_pose\models\stgcn_best.pth"
# Inference parameters
SEQUENCE_LENGTH = 32  # Must match training
WINDOW_STRIDE = 16  # Frames to slide for next window
IN_CHANNELS = 2  # x, y coordinates
NUM_JOINTS = 17  # COCO-17 keypoints
IMGSZ = 640
CONF_THRESHOLD = 0.5
KP_CONF_THRESHOLD = 0.3

# Visualization parameters
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
BG_COLOR = (0, 0, 0)
PREDICTION_DISPLAY_FRAMES = 16  # How long to display each prediction

# Class labels
CLASS_NAMES = ["walk", "lying", "lick", "shake"]
CLASS_COLORS = [
    (0, 255, 0),     # walk - Green
    (255, 0, 0),     # lying - Blue (OpenCV BGR: (255,0,0))
    (0, 255, 255),   # lick - Yellow (BGR)
    (0, 0, 255)      # shake - Red
]

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Setup ====================
def setup_directories():
    """Create necessary directories"""
    Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)
    print(f"✓ Results directory created: {RESULTS_FOLDER}")


# ==================== Model Loading ====================
def normalize_adjacency_matrix(adj_matrix):
    """Normalize adjacency matrix (same as in training)"""
    degree = np.sum(adj_matrix, axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    normalized = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    return normalized.astype(np.float32)


class SpatialGraphConv(torch.nn.Module):
    """Spatial Graph Convolution Layer"""
    
    def __init__(self, in_channels, out_channels, kernel_size, adjacency_matrix):
        super(SpatialGraphConv, self).__init__()
        self.kernel_size = kernel_size
        self.register_buffer('A', torch.FloatTensor(adjacency_matrix))
        self.conv = torch.nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=1)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, x):
        N, C, T, V = x.size()
        x = self.conv(x)
        x = x.view(N, self.kernel_size, -1, T, V)
        x = torch.einsum('nkctv,vw->nkctw', x, self.A)
        x = x.sum(dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TemporalConv(torch.nn.Module):
    """Temporal Convolution Layer"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(TemporalConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0)
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class STGCNBlock(torch.nn.Module):
    """ST-GCN Block"""
    
    def __init__(self, in_channels, out_channels, adjacency_matrix,
                 spatial_kernel_size=3, temporal_kernel_size=9, stride=1, residual=True):
        super(STGCNBlock, self).__init__()
        self.residual = residual
        self.sgc = SpatialGraphConv(in_channels, out_channels, spatial_kernel_size, adjacency_matrix)
        self.tcn = TemporalConv(out_channels, out_channels, temporal_kernel_size, stride)
        
        if not residual:
            self.residual_conv = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual_conv = lambda x: x
        else:
            self.residual_conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                torch.nn.BatchNorm2d(out_channels)
            )
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, x):
        res = self.residual_conv(x)
        x = self.sgc(x)
        x = self.tcn(x)
        x = x + res
        x = self.relu(x)
        return x


class STGCN(torch.nn.Module):
    """ST-GCN Model"""
    
    def __init__(self, num_classes, in_channels, num_joints, adjacency_matrix,
                 spatial_kernel_size=3, temporal_kernel_size=9, num_layers=3):
        super(STGCN, self).__init__()
        adj_matrix = normalize_adjacency_matrix(adjacency_matrix)
        
        self.bn_input = torch.nn.BatchNorm2d(in_channels)
        self.stgcn_layers = torch.nn.ModuleList()
        
        self.stgcn_layers.append(
            STGCNBlock(in_channels, 64, adj_matrix, spatial_kernel_size, temporal_kernel_size, stride=1)
        )
        self.stgcn_layers.append(
            STGCNBlock(64, 128, adj_matrix, spatial_kernel_size, temporal_kernel_size, stride=2)
        )
        
        for _ in range(num_layers - 2):
            self.stgcn_layers.append(
                STGCNBlock(128, 128, adj_matrix, spatial_kernel_size, temporal_kernel_size, stride=1)
            )
        
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(128, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.bn_input(x)
        for layer in self.stgcn_layers:
            x = layer(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def load_model(model_path):
    """
    Load trained ST-GCN model
    
    Returns:
        tuple: (model, model_config)
    """
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Try to get model config and adj_matrix from checkpoint, else use defaults
    default_model_config = {
        'num_classes': 4,
        'in_channels': 2,
        'num_joints': 17,
        'spatial_kernel_size': 3,
        'temporal_kernel_size': 9,
        'num_layers': 3
    }
    def_default_adj = np.eye(default_model_config['num_joints'], dtype=np.float32)

    model_config = checkpoint.get('model_config', default_model_config)
    adj_matrix = checkpoint.get('adjacency_matrix', def_default_adj)

    # If only state_dict is present (weights-only), use defaults
    if isinstance(checkpoint, dict) and ('state_dict' in checkpoint or 'model_state_dict' in checkpoint or 'ema_state_dict' in checkpoint):
        pass  # model_config and adj_matrix set above
    elif isinstance(checkpoint, dict):
        # Possibly a pure state_dict
        checkpoint = {'state_dict': checkpoint}

    # Create model
    model = STGCN(
        num_classes=model_config['num_classes'],
        in_channels=model_config['in_channels'],
        num_joints=model_config['num_joints'],
        adjacency_matrix=adj_matrix,
        spatial_kernel_size=model_config.get('spatial_kernel_size', 3),
        temporal_kernel_size=model_config.get('temporal_kernel_size', 9),
        num_layers=model_config.get('num_layers', 3)
    ).to(DEVICE)

    # Load weights
    loaded = False
    if 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
        print("✓ EMA weights loaded for inference")
        loaded = True
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Standard weights loaded for inference")
        loaded = True
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("✓ Weights loaded from 'state_dict' (no config in checkpoint)")
        loaded = True
    elif isinstance(checkpoint, dict):
        try:
            model.load_state_dict(checkpoint)
            print("✓ Weights loaded from checkpoint dict (raw state_dict)")
            loaded = True
        except Exception as e:
            print(f"✗ Failed to load weights: {e}")
    if not loaded:
        raise RuntimeError("No valid state_dict found in checkpoint.")

    model.eval()
    val_acc = checkpoint.get('val_acc', None)
    if val_acc is not None:
        print(f"✓ Model loaded (Val Acc: {val_acc:.4f})")
    else:
        print("✓ Model loaded (Val Acc: N/A)")
    return model, model_config


# ==================== Skeleton Extraction ====================
class VideoSkeletonExtractor:
    """Extract skeletons from video frames using YOLO-Pose"""
    
    def __init__(self, model_path, imgsz=640, conf_threshold=0.5):
        print(f"Loading YOLO-Pose model...")
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        
        try:
            self.model.to("cuda")
            print("✓ YOLO-Pose loaded on GPU")
        except:
            print("✓ YOLO-Pose loaded on CPU")
    
    def extract_from_frame(self, frame):
        """
        Extract keypoints from a single frame
        
        Returns:
            numpy array: Keypoints of shape (num_joints, 2) or None if not detected
        """
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            verbose=False
        )[0]
        
        if results.keypoints is None or len(results.keypoints.xy) == 0:
            return None
        
        keypoints_xy = results.keypoints.xy[0].cpu().numpy()
        keypoints_conf = results.keypoints.conf[0].cpu().numpy()
        
        # Filter low confidence keypoints
        valid_mask = keypoints_conf > KP_CONF_THRESHOLD
        
        # Return keypoints with confidence filtering
        return keypoints_xy if np.sum(valid_mask) > NUM_JOINTS // 2 else None
    
    def extract_from_video(self, video_path):
        """
        Extract skeleton sequence from entire video
        
        Returns:
            list: List of keypoint arrays (or None for frames without detection)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"  Video: {total_frames} frames @ {fps:.2f} fps")
        
        skeleton_sequence = []
        
        pbar = tqdm(total=total_frames, desc="  Extracting skeletons")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            keypoints = self.extract_from_frame(frame)
            skeleton_sequence.append(keypoints)
            
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        detected_frames = sum(1 for s in skeleton_sequence if s is not None)
        print(f"  ✓ Detected skeletons in {detected_frames}/{total_frames} frames")
        
        return skeleton_sequence, fps


# ==================== Preprocessing ====================
def normalize_skeleton(sequence):
    """
    Normalize skeleton coordinates
    Same as in training
    """
    center_joint = 4
    neck_joint = 3
    lower_body_joint = 5
    
    sequence_normalized = []
    
    for keypoints in sequence:
        if keypoints is None:
            sequence_normalized.append(None)
            continue
        
        # Center the skeleton
        center = keypoints[center_joint, :]
        keypoints_centered = keypoints - center
        
        # Scale by body size
        body_size = np.linalg.norm(keypoints[neck_joint, :] - keypoints[lower_body_joint, :])
        if body_size > 1e-6:
            keypoints_centered /= body_size
        
        sequence_normalized.append(keypoints_centered)
    
    return sequence_normalized


def create_sliding_windows(skeleton_sequence, window_length=32, stride=16):
    """
    Create sliding windows from skeleton sequence
    
    Args:
        skeleton_sequence: List of keypoint arrays
        window_length: Number of frames per window
        stride: Number of frames to slide
    
    Returns:
        list: List of dictionaries with 'start_frame', 'end_frame', 'sequence'
    """
    windows = []
    
    # Find valid segments (consecutive frames with detection)
    valid_segments = []
    current_segment = []
    
    for i, keypoints in enumerate(skeleton_sequence):
        if keypoints is not None:
            current_segment.append(i)
        else:
            if len(current_segment) >= window_length:
                valid_segments.append(current_segment)
            current_segment = []
    
    if len(current_segment) >= window_length:
        valid_segments.append(current_segment)
    
    # Create windows from valid segments
    for segment in valid_segments:
        for start_idx in range(0, len(segment) - window_length + 1, stride):
            frame_indices = segment[start_idx:start_idx + window_length]
            
            # Extract keypoints for this window
            window_keypoints = [skeleton_sequence[i] for i in frame_indices]
            window_array = np.array(window_keypoints)  # Shape: (T, V, C)
            
            windows.append({
                'start_frame': frame_indices[0],
                'end_frame': frame_indices[-1],
                'sequence': window_array
            })
    
    return windows


# ==================== Inference ====================
def predict_window(model, window_sequence, device):
    """
    Perform inference on a single window
    
    Args:
        model: Trained ST-GCN model
        window_sequence: Numpy array of shape (T, V, C)
        device: torch device
    
    Returns:
        tuple: (predicted_class, class_probabilities)
    """
    # Convert to tensor and add batch dimension: (1, C, T, V)
    sequence_tensor = torch.FloatTensor(window_sequence).permute(2, 0, 1).unsqueeze(0)
    sequence_tensor = sequence_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(sequence_tensor)
        probabilities = F.softmax(outputs, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        class_probs = probabilities[0].cpu().numpy()
    
    return pred_class, class_probs


# ==================== Visualization ====================
def draw_prediction_on_frame(frame, prediction_text, confidence, color, frame_number=None):
    """
    Draw prediction text on video frame
    
    Args:
        frame: Video frame
        prediction_text: Text to display
        confidence: Confidence score
        color: Text color (BGR)
        frame_number: Optional frame number to display
    """
    h, w = frame.shape[:2]
    # Top-left: only show class prediction
    text = f"{prediction_text}: {confidence:.2%}"
    x, y = 10, 22
    cv2.putText(frame, text, (x, y), FONT, FONT_SCALE, (0,0,0), FONT_THICKNESS + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)

    # Bottom-right: frame, fps, YOLO model, GCN model (smaller font,緊貼下邊界)
    info_lines = []
    if frame_number is not None:
        info_lines.append(f"frame: {frame_number}")
    if 'GLOBAL_FPS' in globals() and GLOBAL_FPS is not None:
        info_lines.append(f"fps: {GLOBAL_FPS:.2f}")
    if 'GLOBAL_YOLO_MODEL_NAME' in globals() and GLOBAL_YOLO_MODEL_NAME:
        info_lines.append(f"yolo model: {GLOBAL_YOLO_MODEL_NAME}")
    if 'GLOBAL_GCN_MODEL_NAME' in globals() and GLOBAL_GCN_MODEL_NAME:
        info_lines.append(f"gcn model: {GLOBAL_GCN_MODEL_NAME}")
    info_font_scale = 0.55
    info_thickness = 1
    outline_thickness = 3
    # 計算最下方一行的 y 座標，讓所有行緊貼下邊界，靠左側
    line_height = 18
    x_info = 10
    y_start = h - 10 - (len(info_lines) - 1) * line_height
    for i, info in enumerate(info_lines):
        yy = y_start + i * line_height
        cv2.putText(frame, info, (x_info, yy), FONT, info_font_scale, (0,0,0), outline_thickness, cv2.LINE_AA)
        cv2.putText(frame, info, (x_info, yy), FONT, info_font_scale, (255,255,255), info_thickness, cv2.LINE_AA)
    return frame


def draw_probability_bars(frame, class_probs):
    """
    Draw probability bars for all classes
    
    Args:
        frame: Video frame
        class_probs: Array of class probabilities
    """
    h, w = frame.shape[:2]
    # 右上角，條狀不重疊，顏色明顯
    bar_width = 180
    bar_height = 22
    start_x = w - bar_width - 30
    start_y = 18
    spacing = 16
    bar_colors = [
        (0,255,0),      # walk - Green
        (255,0,0),      # lying - Blue
        (0,255,255),    # lip - Yellow
        (0,0,255)       # jump - Red
    ]
    for i, (prob, class_name) in enumerate(zip(class_probs, CLASS_NAMES)):
        y = start_y + i * (bar_height + spacing)
        # 背景條
        cv2.rectangle(frame, (start_x, y), (start_x + bar_width, y + bar_height), (40, 40, 40), -1)
        # 機率條
        color = bar_colors[i % len(bar_colors)]
        filled_width = int(bar_width * prob)
        cv2.rectangle(frame, (start_x, y), (start_x + filled_width, y + bar_height), color, -1)
        # 外框
        cv2.rectangle(frame, (start_x, y), (start_x + bar_width, y + bar_height), (200, 200, 200), 1)
        # 類別名稱與機率
        label = f"{class_name}: {prob*100:.1f}%"
        cv2.putText(frame, label, (start_x + 8, y + bar_height - 6), FONT, 0.65, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame, label, (start_x + 8, y + bar_height - 6), FONT, 0.65, (255,255,255), 1, cv2.LINE_AA)
    return frame


def process_video_with_visualization(video_path, skeleton_sequence, predictions, output_path, fps):
    """
    Create annotated video with predictions
    
    Args:
        video_path: Path to input video
        skeleton_sequence: List of skeleton keypoints
        predictions: List of prediction dictionaries
        output_path: Path to save output video
        fps: Video FPS
    """
    while True:
        cap = cv2.VideoCapture(str(video_path))
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Create video writer (only on first loop)
        if not os.path.exists(str(output_path)):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        else:
            out = None
        total_frames = len(skeleton_sequence)
        # Create frame-to-prediction mapping
        frame_predictions = {}
        for pred in predictions:
            for frame_idx_ in range(pred['start_frame'], pred['end_frame'] + 1):
                if frame_idx_ not in frame_predictions:
                    frame_predictions[frame_idx_] = []
                frame_predictions[frame_idx_].append(pred)
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="  Creating annotated video (Press 'q' to stop)")
        interrupted = False
        delay = int(1000 / fps) if fps > 1e-3 else 33
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Get predictions for current frame
            if frame_idx in frame_predictions:
                # Use the most recent prediction
                pred = frame_predictions[frame_idx][-1]
                class_idx = pred['predicted_class']
                class_name = CLASS_NAMES[class_idx]
                confidence = pred['confidence']
                class_probs = pred['class_probs']
                color = CLASS_COLORS[class_idx]
                # Draw prediction
                frame = draw_prediction_on_frame(frame, class_name, confidence, color, frame_idx)
                # Draw probability bars
                frame = draw_probability_bars(frame, class_probs)
            else:
                # No prediction for this frame
                frame = draw_prediction_on_frame(frame, "No Detection", 0.0, (128, 128, 128), frame_idx)
            if out:
                out.write(frame)
            # ====== 即時顯示視窗 ======
            cv2.imshow("ST-GCN Inference", frame)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                print("\nUser interrupted. Closing window...")
                interrupted = True
                break
            frame_idx += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        if out:
            out.release()
            out = None
        cv2.destroyAllWindows()
        if not interrupted:
            print(f"  ✓ Annotated video saved: {output_path}")
        # Ask user if they want to replay
        print("\nPlayback finished.")
        user_input = input("Press [r] to replay, any other key to exit: ")
        if user_input.lower() != 'r':
            break
        print("\nReplaying video...\n")


# ==================== Main Inference Pipeline ====================
def process_test_video(video_path, model, skeleton_extractor):
    """
    Process a single test video
    
    Args:
        video_path: Path to input video
        model: Trained ST-GCN model
        skeleton_extractor: VideoSkeletonExtractor instance
    
    Returns:
        tuple: (predictions_list, skeleton_sequence, fps)
    """
    video_id = video_path.stem
    print(f"\nProcessing video: {video_id}")
    
    # Step 1: Extract skeletons
    skeleton_sequence, fps = skeleton_extractor.extract_from_video(video_path)
    
    # Step 2: Normalize skeletons
    print("  Normalizing skeletons...")
    normalized_sequence = normalize_skeleton(skeleton_sequence)
    
    # Step 3: Create sliding windows
    print(f"  Creating sliding windows (length={SEQUENCE_LENGTH}, stride={WINDOW_STRIDE})...")
    windows = create_sliding_windows(normalized_sequence, SEQUENCE_LENGTH, WINDOW_STRIDE)
    print(f"  ✓ Created {len(windows)} windows")
    
    if len(windows) == 0:
        print("  ✗ No valid windows created (not enough consecutive frames with detection)")
        return [], skeleton_sequence, fps
    
    # Step 4: Run inference on each window
    print("  Running ST-GCN inference...")
    predictions = []
    
    for window in tqdm(windows, desc="  Predicting"):
        pred_class, class_probs = predict_window(model, window['sequence'], DEVICE)
        
        predictions.append({
            'start_frame': window['start_frame'],
            'end_frame': window['end_frame'],
            'predicted_class': pred_class,
            'predicted_label': CLASS_NAMES[pred_class],
            'confidence': class_probs[pred_class],
            'class_probs': class_probs
        })
    
    return predictions, skeleton_sequence, fps


def save_predictions_csv(predictions, output_path):
    """
    Save predictions to CSV file
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save CSV
    """
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'start_frame': p['start_frame'],
            'end_frame': p['end_frame'],
            'predicted_label': p['predicted_label'],
            'confidence': p['confidence'],
            **{f'prob_{CLASS_NAMES[i]}': p['class_probs'][i] for i in range(len(CLASS_NAMES))}
        }
        for p in predictions
    ])
    
    df.to_csv(output_path, index=False)
    print(f"  ✓ Predictions saved: {output_path}")


def main():
    """Main inference pipeline"""
    
    print("="*70)
    print("ST-GCN Real-Time Inference (即時推論)")
    print("="*70)

    setup_directories()
    print(f"\nDevice: {DEVICE}")
    print(f"Sequence length: {SEQUENCE_LENGTH} frames")

    # Load model
    model, model_config = load_model(MODEL_PATH)
    # Load YOLO model
    yolo_model = YOLO(YOLO_MODEL_PATH)
    try:
        yolo_model.to("cuda")
    except:
        pass

    video_path = Path(TEST_VIDEOS_FOLDER)
    if not video_path.exists():
        print(f"\n✗ Test video not found: {TEST_VIDEOS_FOLDER}")
        return

    while True:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(500 / fps) if fps > 1e-3 else 33
        # Set global variables for overlay info
        global GLOBAL_FPS, GLOBAL_GCN_MODEL_NAME, GLOBAL_YOLO_MODEL_NAME
        GLOBAL_FPS = fps
        GLOBAL_GCN_MODEL_NAME = os.path.basename(MODEL_PATH)
        GLOBAL_YOLO_MODEL_NAME = os.path.basename(YOLO_MODEL_PATH)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\n[Real-Time] Processing video: {video_path.name} ({width}x{height} @ {fps:.2f} fps)")

        skeleton_buffer = []  # 存放最近 SEQUENCE_LENGTH 幀的骨架
        frame_idx = 0
        pred_class = None
        class_probs = None
        confidence = 0.0
        color = (128, 128, 128)
        class_name = "No Detection"

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- YOLO骨架偵測 ---
            results = yolo_model.predict(
                frame,
                imgsz=IMGSZ,
                conf=CONF_THRESHOLD,
                verbose=False
            )[0]
            # 畫出骨架與YOLO外框
            if results.keypoints is not None and len(results.keypoints.xy) > 0:
                for kpt, box in zip(results.keypoints.xy, results.boxes):
                    kpt = kpt.cpu().numpy()
                    # 畫YOLO外框（只畫第一隻貓）
                    if box is not None:
                        xyxy = box.xyxy[0].cpu().numpy() if hasattr(box, 'xyxy') else box.cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    # 畫骨架連線（分部位上色）
                    for i, j in HEAD_LINKS:
                        pt1 = tuple(map(int, kpt[i]))
                        pt2 = tuple(map(int, kpt[j]))
                        cv2.line(frame, pt1, pt2, COLOR_HEAD, 2)
                    for i, j in BODY_LINKS:
                        pt1 = tuple(map(int, kpt[i]))
                        pt2 = tuple(map(int, kpt[j]))
                        cv2.line(frame, pt1, pt2, COLOR_BODY, 2)
                    for idx, (i, j) in enumerate(FRONT_LIMBS):
                        pt1 = tuple(map(int, kpt[i]))
                        pt2 = tuple(map(int, kpt[j]))
                        color = COLOR_LEFT_FRONT if i in [6,7] or j in [6,7] else COLOR_RIGHT_FRONT
                        cv2.line(frame, pt1, pt2, color, 2)
                    for idx, (i, j) in enumerate(HIND_LIMBS):
                        pt1 = tuple(map(int, kpt[i]))
                        pt2 = tuple(map(int, kpt[j]))
                        color = COLOR_LEFT_HIND if i in [10,11] or j in [10,11] else COLOR_RIGHT_HIND
                        cv2.line(frame, pt1, pt2, color, 2)
                    for i, j in TAIL_LINKS:
                        pt1 = tuple(map(int, kpt[i]))
                        pt2 = tuple(map(int, kpt[j]))
                        cv2.line(frame, pt1, pt2, COLOR_TAIL, 2)
                    # 畫骨架點
                    for idx, (x, y) in enumerate(kpt):
                        if idx in [6,7]:
                            cv2.circle(frame, (int(x), int(y)), 4, COLOR_LEFT_FRONT, -1)
                        elif idx in [8,9]:
                            cv2.circle(frame, (int(x), int(y)), 4, COLOR_RIGHT_FRONT, -1)
                        elif idx in [10,11]:
                            cv2.circle(frame, (int(x), int(y)), 4, COLOR_LEFT_HIND, -1)
                        elif idx in [12,13]:
                            cv2.circle(frame, (int(x), int(y)), 4, COLOR_RIGHT_HIND, -1)
                        else:
                            cv2.circle(frame, (int(x), int(y)), 3, COLOR_KPT, -1)
                # 只取第一隻貓
                keypoints_xy = results.keypoints.xy[0].cpu().numpy()
                skeleton_buffer.append(keypoints_xy)
                if len(skeleton_buffer) > SEQUENCE_LENGTH:
                    skeleton_buffer.pop(0)
            else:
                skeleton_buffer.append(None)
                if len(skeleton_buffer) > SEQUENCE_LENGTH:
                    skeleton_buffer.pop(0)

            # --- GCN推論 ---
            # 只要最近 SEQUENCE_LENGTH 幀都有骨架才推論
            if len(skeleton_buffer) == SEQUENCE_LENGTH and all(s is not None for s in skeleton_buffer):
                # Normalization (同train)
                seq = np.array(skeleton_buffer)
                center_joint = 4
                neck_joint = 3
                lower_body_joint = 5
                # Center
                seq_centered = seq - seq[:,center_joint: center_joint+1,:]
                # Scale
                body_sizes = np.linalg.norm(seq[:,neck_joint,:2] - seq[:,lower_body_joint,:2], axis=1)
                avg_body_size = np.mean(body_sizes)
                if avg_body_size > 1e-6:
                    seq_centered[:,:,:2] /= avg_body_size
                # (T,V,C) -> (1,C,T,V)
                seq_tensor = torch.FloatTensor(seq_centered).permute(2,0,1).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    outputs = model(seq_tensor)
                    probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
                    pred_class = int(np.argmax(probs))
                    class_probs = probs
                    confidence = probs[pred_class]
                    class_name = CLASS_NAMES[pred_class]
                    color = CLASS_COLORS[pred_class]
            # --- 畫推論結果 ---
            if pred_class is not None:
                frame = draw_prediction_on_frame(frame, class_name, confidence, color, frame_idx)
                frame = draw_probability_bars(frame, class_probs)
            else:
                frame = draw_prediction_on_frame(frame, "No Detection", 0.0, (128,128,128), frame_idx)

            cv2.imshow("ST-GCN Real-Time Inference", frame)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                print("\nUser interrupted. Closing window...")
                cap.release()
                cv2.destroyAllWindows()
                return
            frame_idx += 1
        cap.release()
        cv2.destroyAllWindows()
        # 自動重播（無需按鍵）
        print("\nReplaying video...\n")


# ==================== Main Entry Point ====================
if __name__ == "__main__":
    # Check dependencies
    try:
        import torch
        import cv2
        import pandas
        from ultralytics import YOLO
        print("✓ All dependencies available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install required packages:")
        print("  pip install torch torchvision")
        print("  pip install opencv-python")
        print("  pip install pandas")
        print("  pip install ultralytics")
        exit(1)
    
    main()