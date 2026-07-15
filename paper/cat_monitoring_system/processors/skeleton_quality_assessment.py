"""
Skeleton Quality Assessment（骨架品質評估）── 獨立、可插拔模組

移植自 test_bone_length_stability.py（模式2，GUI 視覺偵測）驗證過的計算
邏輯，把「GCN 分類為主、幾何判斷為輔」雙重判定機制帶進正式推論管線
（processors/frame_processor.py）。三項指標各自看的角度：

  midback_offset_ratio ─ MidBack 偏離 Chest-Hip 虛擬中點的距離 ÷ Chest-Hip
                          距離，偏移量超過解剖合理性上限視為可疑。
  midback_angle        ─ Chest-MidBack-Hip 夾角（取窗口最後一幀），太接近
                          180 度（幾乎共線）或太小（夾角過尖）都視為可疑。
  body_axis_score_jitter ─ Body Axis Proportion Analysis 單幀分數在窗口內
                          的振幅，振幅越大代表骨架偵測越不穩定/反覆跳動。

低耦合設計（刻意，不是疏漏）：
  - 這個檔案只依賴 numpy 跟 models.stgcn_model.interpolate_missing，不
    import FrameProcessor/config.py 任何東西，也不知道呼叫端是誰——單純
    「餵窗口進來、吐判定結果出去」的純函式模組，可以整個被刪除或搬到
    別的專案，不會牽動其他程式碼。
  - 是否啟用整套機制的總開關在 config.py 的 SQAConfig.ENABLE_SQA_DUAL_
    JUDGMENT（呼叫端負責檢查，這裡不重複判斷）；3 項指標各自要不要參與
    判定，則由下面 ENABLE_MIDBACK_OFFSET_CHECK/ENABLE_MIDBACK_ANGLE_CHECK/
    ENABLE_SCORE_JITTER_CHECK 這三個模組內變數個別控制——兩層開關刻意分開
    管理，一個決定「這整套機制要不要開」，另一個決定「開了之後細節要
    看哪幾項」。

Fail-safe 承諾（呼叫端可以完全信任）：
  - 唯一對外的進入點 evaluate_window() 保證不拋出例外——任何內部計算
    錯誤都會被攔截，回傳 (True, {})，也就是「判定為可信、不覆蓋」。
  - 這代表這個模組萬一有 bug，最壞結果只是雙重判定不生效（退回只信任
    ST-GCN 原本的分類結果，等同這個功能不存在時的行為），絕不會讓主系統
    的推論管線中斷或跑出更糟的結果。
"""
import numpy as np

from models.stgcn_model import interpolate_missing

# ============================================================================
# ===== 三項指標各自的啟用開關（哪些指標要參與「是否可信」的判定）=====
# ============================================================================
# 把某一項改成 False，該指標就只是純計算/不參與 evaluate_window() 的判定，
# 不影響其餘兩項運行，也不會出錯（_enabled_thresholds() 只收集開著的項目）。
ENABLE_MIDBACK_OFFSET_CHECK = True
ENABLE_MIDBACK_ANGLE_CHECK = True
ENABLE_SCORE_JITTER_CHECK = True

# ============================================================================
# ===== 門檻/參考值常數（跟 test_bone_length_stability.py 目前使用的數值同步）=====
# ============================================================================
BONE_CONF_THRESHOLD = 0.3     # 骨段兩端關鍵點信心低於此值，該幀不納入該項計算
MIN_VALID_FRAMES_MIDBACK_OFFSET = 1  # midback_offset_ratio 至少要有幾幀有效才採信

CANDIDATE_MIDBACK_OFFSET_THRESHOLD = 1.0  # 方向：越大越可疑

CANDIDATE_MIDBACK_ANGLE_LOW_THRESHOLD = 20.0    # 度，低於此值視為夾角過尖，可疑
CANDIDATE_MIDBACK_ANGLE_HIGH_THRESHOLD = 160.0  # 度，高於此值視為過直/接近共線，可疑

BODY_AXIS_REFERENCE_RATIOS = {
    "chest_midback": 0.75,   # Chest-MidBack / Chest-Hip
    "midback_hip": 0.70,     # MidBack-Hip / Chest-Hip
}
BODY_AXIS_ERROR_SIGMA = 0.70  # geometry_error → body_axis_score 指數衰減的尺度常數
BODY_AXIS_MIN_VALID_SAMPLES = 3          # 窗口內至少要有幾幀有效才採信振幅值
BODY_AXIS_SCORE_JITTER_THRESHOLD = 20.0  # 方向：越大越可疑

# 校準說明：以上門檻值目前是依 test_bone_length_stability.py 少量影片
# （個位數支）的校準結果設定，不是從大量真實資料統計出來的基準值，正式
# 套用前建議先在 GUI/背景模式肉眼比對過覆蓋規則是否合理（見
# SQAConfig.ENABLE_SQA_DUAL_JUDGMENT 的說明），之後應該用更多影片持續
# 重新校準這幾個常數。


def compute_midback_angle(kpts, kpt_conf, conf_thresh=BONE_CONF_THRESHOLD):
    """算 Chest(3)-MidBack(4)-Hip(5) 這個夾角（以 MidBack 為頂點），單位度。
    標註慣例是把 MidBack 點在貓背部拱起的最高點，所以這三點正常情況下是
    一個三角形、不是一直線——夾角本來就會小於 180 度，實際大小跟貓拱背
    弧度/姿勢有關，並非越接近 180 度就代表偵測越正確；太接近 180 度視為
    三點幾乎共線（MidBack 關鍵點可能消失/飄移到 Chest-Hip 連線上），太小
    視為夾角過尖，兩端都可能代表關鍵點錯位或偵測失效。
    任一點信心不足或兩個向量長度太短（幾乎重疊）時回傳 None。"""
    if kpt_conf[3] < conf_thresh or kpt_conf[4] < conf_thresh or kpt_conf[5] < conf_thresh:
        return None
    chest = kpts[3, :2]
    midback = kpts[4, :2]
    hip = kpts[5, :2]
    v1 = chest - midback
    v2 = hip - midback
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_angle = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def compute_bone_stability_overlay(seq_window, conf_window):
    """seq_window: (T, 17, 2) 原始（未插值）座標；conf_window: (T, 17)。
    回傳這個窗口的骨架穩定度資訊：midback_offset_ratio、midback_angle
    （Chest-MidBack-Hip 目前這一刻的夾角，取窗口最後一幀）。
    """
    seq = interpolate_missing(seq_window, conf_window, threshold=0.1)

    chest_hip_valid = (conf_window[:, 3] >= BONE_CONF_THRESHOLD) & (conf_window[:, 5] >= BONE_CONF_THRESHOLD)

    midback_valid = chest_hip_valid & (conf_window[:, 4] >= BONE_CONF_THRESHOLD)
    midback_offset_ratio = float("nan")
    if np.any(midback_valid):
        virtual_pt = (seq[:, 3, :2] + seq[:, 5, :2]) / 2.0
        raw_offset = np.linalg.norm(seq[:, 4, :2] - virtual_pt, axis=1)
        body_size_per_frame = np.linalg.norm(seq[:, 3, :2] - seq[:, 5, :2], axis=1)
        frame_ok = midback_valid & (body_size_per_frame > 1e-6)
        if int(np.sum(frame_ok)) >= MIN_VALID_FRAMES_MIDBACK_OFFSET:
            ratio_vals = raw_offset[frame_ok] / body_size_per_frame[frame_ok]
            midback_offset_ratio = float(np.mean(ratio_vals))

    midback_angle = compute_midback_angle(seq[-1], conf_window[-1], conf_thresh=BONE_CONF_THRESHOLD)
    if midback_angle is None:
        midback_angle = float("nan")

    return {
        "midback_offset_ratio": midback_offset_ratio,
        "midback_angle": float(midback_angle),
    }


def compute_body_axis_geometry(kpts, chest_joint=3, midback_joint=4, hip_joint=5):
    """單幀「身體主軸幾何比例分析」。只吃單一幀的 17 個關鍵點座標
    （kpts: (17, 2)），不使用信心值/時間序列/ST-GCN 輸出。

    計算流程：
      1. 算 Chest-MidBack、MidBack-Hip 兩段的歐氏距離。
      2. 用 Chest-Hip 距離正規化：R2 = Chest-MidBack / Chest-Hip，
         R3 = MidBack-Hip / Chest-Hip。
      3. Geometry Vector = [R2, R3]，跟 BODY_AXIS_REFERENCE_RATIOS 算歐氏
         距離得到 geometry_error。
      4. geometry_error 用指數衰減轉成 body_axis_score（0~100）。

    body_axis_score 本身純背景計算，不直接參與門檻判斷——異常判斷改用
    compute_body_axis_score_jitter() 算出的抖動幅度。
    """
    kpts = np.asarray(kpts, dtype=np.float64)
    chest_hip_dist = float(np.linalg.norm(kpts[chest_joint] - kpts[hip_joint]))

    if chest_hip_dist < 1e-6:
        return {
            "chest_midback_ratio": float("nan"),
            "midback_hip_ratio": float("nan"),
            "geometry_vector": [float("nan"), float("nan")],
            "geometry_error": float("nan"),
            "body_axis_score": float("nan"),
        }

    chest_midback = float(np.linalg.norm(kpts[chest_joint] - kpts[midback_joint]))
    midback_hip = float(np.linalg.norm(kpts[midback_joint] - kpts[hip_joint]))

    r2 = chest_midback / chest_hip_dist
    r3 = midback_hip / chest_hip_dist
    geometry_vector = [r2, r3]

    diffs = np.array([
        r2 - BODY_AXIS_REFERENCE_RATIOS["chest_midback"],
        r3 - BODY_AXIS_REFERENCE_RATIOS["midback_hip"],
    ])
    geometry_error = float(np.linalg.norm(diffs))
    body_axis_score = float(100.0 * np.exp(-geometry_error / BODY_AXIS_ERROR_SIGMA))

    return {
        "chest_midback_ratio": r2,
        "midback_hip_ratio": r3,
        "geometry_vector": geometry_vector,
        "geometry_error": geometry_error,
        "body_axis_score": body_axis_score,
    }


def compute_body_axis_score_jitter(kpts_window):
    """量測 body_axis_score 在一個時間窗口內的變化振幅。

    kpts_window: (T, 17, 2)，每一幀的關鍵點座標，跟 compute_body_axis_
    geometry() 要求一致：不使用信心值、不使用 ST-GCN 輸出。

    Returns: (amplitude, valid_sample_count)。amplitude = 窗口內所有有效
    body_axis_score 的 max - min（振幅，不是逐幀差值的平均）；有效幀數
    低於 BODY_AXIS_MIN_VALID_SAMPLES 時，amplitude 回傳 NaN。
    """
    kpts_window = np.asarray(kpts_window, dtype=np.float64)
    t_len = kpts_window.shape[0]
    scores = np.array([
        compute_body_axis_geometry(kpts_window[t])["body_axis_score"]
        for t in range(t_len)
    ])
    valid_scores = scores[np.isfinite(scores)]
    valid_sample_count = int(valid_scores.size)
    if valid_sample_count < BODY_AXIS_MIN_VALID_SAMPLES:
        return float("nan"), valid_sample_count
    amplitude = float(np.max(valid_scores) - np.min(valid_scores))
    return amplitude, valid_sample_count


def _is_bad(value, threshold, direction):
    """通用門檻判斷。
    direction="above"：value > threshold 才算異常。
    direction="below"：value < threshold 才算異常。
    direction="outside_range"：threshold 是 (low, high) 兩元組，value 在
    範圍外（< low 或 > high）才算異常——midback_angle 這種「兩端都可疑」
    的指標用這個方向。"""
    if direction == "outside_range":
        low, high = threshold
        return value < low or value > high
    return (value > threshold) if direction == "above" else (value < threshold)


def _enabled_thresholds() -> dict:
    """依三個 ENABLE_*_CHECK 變數目前的值，組出這次要參與判定的指標登記表
    ——每次呼叫都重新檢查（不是模組載入時算一次快取），所以執行期改變
    這三個變數的值會立刻生效，不需要重新載入模組。"""
    thresholds = {}
    if ENABLE_MIDBACK_OFFSET_CHECK:
        thresholds["midback_offset_ratio"] = (CANDIDATE_MIDBACK_OFFSET_THRESHOLD, "above")
    if ENABLE_MIDBACK_ANGLE_CHECK:
        thresholds["midback_angle"] = (
            (CANDIDATE_MIDBACK_ANGLE_LOW_THRESHOLD, CANDIDATE_MIDBACK_ANGLE_HIGH_THRESHOLD),
            "outside_range",
        )
    if ENABLE_SCORE_JITTER_CHECK:
        thresholds["body_axis_score_jitter"] = (BODY_AXIS_SCORE_JITTER_THRESHOLD, "above")
    return thresholds


def evaluate_window(seq_window, conf_window):
    """唯一對外進入點。seq_window/conf_window：(T, 17, 2)/(T, 17)，跟餵給
    ST-GCN 分類器的是同一個滑動窗口、同一份未正規化的原始像素座標（呼叫端
    不需要另外做插值/正規化，這裡跟 compute_bone_stability_overlay/
    compute_body_axis_score_jitter 內部會自己處理）。

    回傳 (reliable, details)：
      reliable ─ True 代表這個窗口沒有任何啟用中的指標超標（或全部指標都
                 被關閉/資料不足無法判斷，保守視為可信，不否決 ST-GCN
                 結果）；False 代表至少一項啟用中的指標超標。
      details  ─ dict，包含三項指標的原始數值跟 "failed_checks"（超標的
                 指標名稱清單），供呼叫端需要時記錄/除錯用；不需要細節時
                 可以忽略這個回傳值。

    Fail-safe：任何內部例外都會被攔截，回傳 (True, {})——絕不拋出例外，
    絕不讓呼叫端因為這個函式而中斷。
    """
    try:
        ovl = compute_bone_stability_overlay(seq_window, conf_window)
        jitter, _valid_sample_count = compute_body_axis_score_jitter(seq_window)
        ovl["body_axis_score_jitter"] = jitter

        failed = []
        for key, (threshold, direction) in _enabled_thresholds().items():
            value = ovl.get(key, float("nan"))
            if not np.isfinite(value):
                continue
            if _is_bad(value, threshold, direction):
                failed.append(key)

        ovl["failed_checks"] = failed
        return len(failed) == 0, ovl
    except Exception:
        return True, {}
