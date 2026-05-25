
"""
常量定義（與 cat_monitoring_stgcn_integrated.py 完全一致）
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

# 其餘常量保留
BEHAVIOR_CLASSES = ['walk', 'lick', 'scratch', 'shake']
BEHAVIOR_COLORS = {0: (0, 255, 0), 1: (0, 255, 255), 2: (255, 165, 0), 3: (0, 0, 255)}
BEHAVIOR_TEXT_MAP = {0: "走動", 1: "舔舐", 2: "搔抓", 3: "甩頭"}
BEHAVIOR_EMOJI_MAP = {0: "🐾", 1: "🧼", 2: "🐈", 3: "🐈↺"}
KP_CONF_THRES = 0.5

# 信心门檣：所有類別機率低於此即輸出"目前正常"
CONFIDENCE_THRESHOLD = 0.27
LOW_CONF_ID = -1          # sentinel：信心度不足
LOW_CONF_TEXT = "Normal"
LOW_CONF_EMOJI = "😴"
# 當行為信心小於此值時，前端/顯示層級會把標籤視為 "目前正常"
BEHAVIOR_MIN_CONFIDENCE = 0.60
