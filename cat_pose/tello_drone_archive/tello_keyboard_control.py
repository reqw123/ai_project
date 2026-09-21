import pygame
import cv2
import numpy as np
import time
from djitellopy import Tello

# =========================
# 基本設定
# =========================

SPEED = 60          # 飛行速度 (-100~100)
INTERVAL = 0.05     # RC 發送頻率 (20Hz)

# 控制通道
lr = 0      # 左右
fb = 0      # 前後
ud = 0      # 上下
yaw = 0     # 旋轉

running = True
in_air = False


# =========================
# 初始化 Tello
# =========================

tello = Tello()
tello.connect()

print("Battery:", tello.get_battery())

tello.streamon()
frame_read = tello.get_frame_read()


# =========================
# pygame 初始化
# =========================

pygame.init()
screen = pygame.display.set_mode((640, 480))  # 與相機解析度一致
pygame.display.set_caption("Tello Keyboard Control")
clock = pygame.time.Clock()

print("""
=========== 控制鍵 ===========
T = 起飛
L = 降落

W = 前進
S = 後退
A = 左平移
D = 右平移

R = 上升
F = 下降

Q = 左旋轉
E = 右旋轉

ESC = 離開程式
=============================
""")

# =========================
# HUD（按鍵提示 + 電量）
# =========================

BATTERY_POLL_SEC = 3.0   # 電量更新間隔

# 找得到中文字型就用中文，否則退回英文，避免顯示成方框
_cjk_font = pygame.font.match_font("microsoftjhenghei,msjh,simhei,simsun,notosanscjktc")
if _cjk_font:
    hud_font = pygame.font.Font(_cjk_font, 14)
    HELP_TEXT = "T起飛 L降落 | WASD移動 | QE旋轉 | RF升降 | ESC離開"
    STATE_TEXT = {True: "飛行中", False: "地面"}
    STALE_TEXT = "串流中斷"
else:
    hud_font = pygame.font.Font(None, 18)
    HELP_TEXT = "T takeoff  L land | WASD move | QE yaw | RF up/down | ESC quit"
    STATE_TEXT = {True: "AIR", False: "GROUND"}
    STALE_TEXT = "NO VIDEO"

# 底部提示條只需渲染一次
_help_text_surf = hud_font.render(HELP_TEXT, True, (235, 235, 235))
help_bar = pygame.Surface((640, _help_text_surf.get_height() + 6), pygame.SRCALPHA)
help_bar.fill((0, 0, 0, 140))
help_bar.blit(_help_text_surf, (6, 3))

battery = None
_last_battery_poll = 0.0

STREAM_STALE_SEC = 2.0   # 超過這麼久沒有新影像幀就視為串流中斷
last_frame = None
last_frame_time = time.time()
stream_stale = False


def update_battery():
    """定時更新電量，失敗時保留舊值（避免 UDP 偶發逾時讓 HUD 消失）。"""
    global battery, _last_battery_poll
    now = time.time()
    if now - _last_battery_poll < BATTERY_POLL_SEC:
        return
    _last_battery_poll = now
    try:
        battery = tello.get_battery()
    except Exception:
        pass


def make_badge(text, color):
    label = hud_font.render(text, True, color)
    badge = pygame.Surface((label.get_width() + 12, label.get_height() + 6), pygame.SRCALPHA)
    badge.fill((0, 0, 0, 140))
    badge.blit(label, (6, 3))
    return badge


def draw_hud(surface):
    # 右上角：狀態 + 電量
    if battery is None:
        color, bat_str = (180, 180, 180), "--%"
    else:
        bat_str = f"{battery}%"
        color = (90, 220, 90) if battery > 50 else (240, 200, 60) if battery > 20 else (240, 80, 80)
    badge = make_badge(f"{STATE_TEXT[in_air]}  {bat_str}", color)
    x = surface.get_width() - badge.get_width() - 6
    surface.blit(badge, (x, 6))

    # 串流中斷警示（貼在電量角標左邊）
    if stream_stale:
        warn = make_badge(STALE_TEXT, (240, 80, 80))
        surface.blit(warn, (x - warn.get_width() - 6, 6))

    # 底部：按鍵提示
    surface.blit(help_bar, (0, surface.get_height() - help_bar.get_height()))

# =========================
# 主控制迴圈
# =========================

while running:

    # ===== 顯示攝影機（pygame 視窗，避免焦點問題）=====
    frame = frame_read.frame
    if frame is not last_frame:   # 每張新解碼的幀都是新物件
        last_frame = frame
        last_frame_time = time.time()
    stream_stale = time.time() - last_frame_time > STREAM_STALE_SEC
    frame = cv2.resize(frame, (640, 480))
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))  # djitellopy 已是 RGB，只需軸對調 HWC→WHC
    screen.blit(surface, (0, 0))
    update_battery()
    draw_hud(screen)
    pygame.display.update()

    # ===== 處理 pygame 事件 =====
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:

            # T = 起飛
            if event.key == pygame.K_t:
                if not in_air:
                    tello.takeoff()
                    in_air = True

            # L = 降落
            if event.key == pygame.K_l:
                if in_air:
                    tello.land()
                    in_air = False

            # ESC = 結束
            if event.key == pygame.K_ESCAPE:
                running = False

    # ===== 持續偵測按鍵 =====
    keys = pygame.key.get_pressed()

    lr = fb = ud = yaw = 0

    # W = 前進
    if keys[pygame.K_w]:
        fb = SPEED

    # S = 後退
    if keys[pygame.K_s]:
        fb = -SPEED

    # A = 左移
    if keys[pygame.K_a]:
        lr = -SPEED

    # D = 右移
    if keys[pygame.K_d]:
        lr = SPEED

    # R = 上升
    if keys[pygame.K_r]:
        ud = SPEED

    # F = 下降
    if keys[pygame.K_f]:
        ud = -SPEED

    # Q = 左旋轉
    if keys[pygame.K_q]:
        yaw = -SPEED

    # E = 右旋轉
    if keys[pygame.K_e]:
        yaw = SPEED

    # ===== 發送控制指令 =====
    if in_air:
        tello.send_rc_control(lr, fb, ud, yaw)

    clock.tick(int(1 / INTERVAL))  # 控制迴圈頻率，取代 time.sleep


# =========================
# 安全關閉
# =========================

# 先關視窗，再做降落/關串流（這些指令最久可能等數秒，不能讓視窗卡著）
pygame.quit()

try:
    if in_air:
        print("降落中...")
        tello.land()
    tello.streamoff()
    tello.end()
except Exception as e:
    print("關閉時發生錯誤:", e)