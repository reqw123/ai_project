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

t = 上升
l = 下降

Q = 左旋轉
E = 右旋轉

X = 離開程式
=============================
""")

# =========================
# 主控制迴圈
# =========================

while running:

    # ===== 顯示攝影機（pygame 視窗，避免焦點問題）=====
    frame = frame_read.frame
    frame = cv2.resize(frame, (640, 480))
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))  # djitellopy 已是 RGB，只需軸對調 HWC→WHC
    screen.blit(surface, (0, 0))
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

            # X = 結束
            if event.key == pygame.K_x:
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

if in_air:
    tello.land()

tello.streamoff()
tello.end()

pygame.quit()