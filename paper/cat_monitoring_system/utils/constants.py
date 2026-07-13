
"""
常量定義
"""

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)

COLOR_HEAD = (255, 255, 0)
COLOR_BODY = (0, 255, 0)
COLOR_LIMB = (255, 0, 0)
COLOR_TAIL = (255, 0, 255)
COLOR_HIP_TAIL = (0, 165, 255)  # 橘黃色，用於 Hip→Tail_Root，與尾巴段區隔
COLOR_KPT = (0, 0, 255)

# BBox 視覺樣式（主系統與 EMA 推論共用）
BBOX_OUTER_COLOR = BLACK
BBOX_INNER_COLOR = COLOR_HEAD
BBOX_OUTER_THICKNESS = 4
BBOX_INNER_THICKNESS = 2

# 四隻腳的顏色（左前、右前、左後、右後）
COLOR_LEFT_FRONT = (255, 0, 255) # 洋紅色 (magenta)
COLOR_RIGHT_FRONT = (0, 255, 255) # 青色 (cyan)
COLOR_LEFT_HIND = (255, 165, 0) # 橙色 (orange)
COLOR_RIGHT_HIND = (0, 255, 0) # 綠色 (green)

# ==================== 骨架連結 ====================
HEAD_LINKS = [(0,1),(0,2),(1,2)]
BODY_LINKS = [(0,3),(3,4),(4,5)]
FRONT_LIMBS = [(3,6),(6,7),(3,8),(8,9)]
HIND_LIMBS = [(5,10),(10,11),(5,12),(12,13)]
HIP_TAIL_LINK = [(5, 14)]           # Hip → Tail_Root（接身體用，獨立顏色）
TAIL_LINKS = [(14,15),(15,16)]      # Tail_Root → Tail_Mid → Tail_Tip
ALL_SKELETON = HEAD_LINKS + BODY_LINKS + FRONT_LIMBS + HIND_LIMBS + HIP_TAIL_LINK + TAIL_LINKS

# ===== 耳距監測腳本的骨架視覺樣式 =====
EAR_DISTANCE_SKELETON_EDGES = [
	(0, 1), (0, 2), (1, 2),
	(0, 3), (3, 4), (4, 5),
	(3, 6), (6, 7), (3, 8), (8, 9),
	(5, 10), (10, 11), (5, 12), (12, 13),
	(5, 14), (14, 15), (15, 16),
]

# 索引 3/4/5（Chest/Mid_Back/Hip）原本都落在黃綠色系（255,255,60 /
# 200,255,60 / 100,255,100），彼此只差 R/G 通道幾十，畫面上小圓點很難分辨，
# 改用黃／洋紅／綠三個色相分離的顏色，跟相鄰的頭部（紅橙）、前肢（青藍）、
# 後肢（紫）都不衝突。此常數同時被主專案 Visualizer（processors/visualizer.py）
# 與 1_measure_ear_distance_single_video.py 共用，改這裡兩邊同步生效。
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

BEHAVIOR_CLASSES = ['walk', 'lick', 'scratch', 'shake', 'stop']
BEHAVIOR_COLORS = {0: (0, 255, 0), 1: (0, 255, 255), 2: (255, 165, 0), 3: (0, 0, 255), 4: (0, 165, 255)}
BEHAVIOR_TEXT_MAP = {0: "走動", 1: "舔舐", 2: "搔抓", 3: "甩頭", 4: "靜止"}
BEHAVIOR_EMOJI_MAP = {
    0: "🐾",   # walk    — 腳印，貓咪在移動
    1: "👅",   # lick    — 舌頭，正在舔舐理毛
    2: "🦶",   # scratch — 肉掌，正在搔抓
    3: "🌀",   # shake   — 旋轉，甩頭動作
    4: "💤",   # stop    — 睡著符號，靜止休息
}

# ── 三層狀態 sentinel ────────────────────────────────────────
# Layer 1  YOLO 偵測層
NOT_VISIBLE_ID = -2       # YOLO 未偵測到貓 → ST-GCN 不執行
NOT_VISIBLE_TEXT = "NO_CAT"
NOT_VISIBLE_DISPLAY_TEXT = "不在畫面"
NOT_VISIBLE_EMOJI = "🔍"  # 放大鏡：找不到貓

# Layer 3  推論失效層（有貓、有骨架，但模型信心不足）
LOW_CONF_ID = -1
LOW_CONF_TEXT = "LOW_CONF"
LOW_CONF_EMOJI = "❓"     # 問號：模型不確定

# Layer 2  ST-GCN 行為輸出層（id 0–4，見 BEHAVIOR_* 上方）
