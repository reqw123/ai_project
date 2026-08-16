"""
共用視覺化常數（17-kpt 貓骨架連線與配色）。

直接複製自 paper/cat_monitoring_system/utils/constants.py 的骨架/配色定義
（該檔案是 paper 端目前實際使用中的版本，其餘跟 ST-GCN 行為分類相關的常數
不適用於本專案，未一併複製）。cat_pose/ 底下多支腳本原本各自複製貼上一份
幾乎相同的骨架連線與配色，這支模組讓它們改成統一從這裡 import，之後要調整
配色只需要改一個地方。

如果 paper 那邊的 EAR_DISTANCE_* 常數之後又更新，記得手動同步過來這份副本
（兩邊是獨立檔案，沒有共用套件路徑，無法直接 import 過去）。
"""

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

COLOR_HEAD = (255, 255, 0)
COLOR_KPT = (0, 0, 255)

# ===== 17-kpt 骨架連線（索引對應 nose/ear_tip*2/chest/mid_back/hip/
# 前肢*2/後肢*2/tail_base/mid/tip） =====
# fmt: off
EAR_DISTANCE_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 2),
    (0, 3), (3, 4), (4, 5),
    (3, 6), (6, 7), (3, 8), (8, 9),
    (5, 10), (10, 11), (5, 12), (12, 13),
    (5, 14), (14, 15), (15, 16),
]
# fmt: on

# 索引 3/4/5（Chest/Mid_Back/Hip）原本都落在黃綠色系，彼此只差 R/G 通道
# 幾十，畫面上小圓點很難分辨，改用黃／洋紅／綠三個色相分離的顏色，跟相鄰的
# 頭部（紅橙）、前肢（青藍）、後肢（紫）都不衝突。
# fmt: off
EAR_DISTANCE_KP_COLORS = [
    (255, 80, 80), (255, 160, 40), (255, 160, 40),
    (255, 230, 0), (255, 0, 200), (0, 220, 0),
    (60, 200, 255), (60, 120, 255), (60, 200, 255), (60, 120, 255),
    (180, 80, 255), (120, 40, 255), (180, 80, 255), (120, 40, 255),
    (80, 220, 180), (60, 180, 140), (40, 140, 100),
]

EAR_DISTANCE_EDGE_COLORS = [
    (255, 120, 60), (255, 120, 60), (255, 120, 60),
    (220, 220, 60), (200, 220, 60), (160, 220, 60),
    (102, 85, 255), (102, 85, 255), (255, 68, 204), (255, 68, 204),
    (255, 170, 34), (255, 170, 34), (0, 153, 255), (0, 153, 255),
    (80, 200, 160), (60, 170, 130), (40, 140, 100),
]
# fmt: on

# ===== 依身體部位分組的骨架連結／配色 =====
# 跟上面「單一扁平列表＋逐邊配色」（EAR_DISTANCE_*）是同一套骨架拓撲的另一種
# 表示法：每個身體部位分組各配一個固定顏色（HEAD_LINKS 用 COLOR_HEAD 畫、
# BODY_LINKS 用 COLOR_BODY 畫...），給 tello_drone_archive/tello_cat_pose.py（已停用的
# 空拍機方案，移入 tello_drone_archive/ 保留）／0_video_pose_viewer.py／
# pose_single_image_test.py／eda_realtime_anomaly_viewer.py／
# 自動標註工具/auto_labeling_capture.py 這種「呼叫 draw_links(frame, kpts, conf,
# LINKS, COLOR)」風格的繪製函式共用。
COLOR_BODY = (0, 255, 0)
COLOR_TAIL = (255, 0, 255)
COLOR_LEFT_FRONT = (255, 0, 255)
COLOR_RIGHT_FRONT = (0, 255, 255)
COLOR_LEFT_HIND = (255, 165, 0)
COLOR_RIGHT_HIND = (0, 255, 0)

HEAD_LINKS = [(0, 1), (0, 2), (1, 2)]
BODY_LINKS = [(0, 3), (3, 4), (4, 5)]
TAIL_LINKS = [(5, 14), (14, 15), (15, 16)]
LEFT_FRONT_LINKS = [(3, 6), (6, 7)]
RIGHT_FRONT_LINKS = [(3, 8), (8, 9)]
LEFT_HIND_LINKS = [(5, 10), (10, 11)]
RIGHT_HIND_LINKS = [(5, 12), (12, 13)]
