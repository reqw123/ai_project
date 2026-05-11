import cv2

video_path = r"C:\cat_pose\back1.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ 無法開啟影片")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()

print(f"解析度: {width} x {height}")
print(f"FPS: {fps}")
print(f"總幀數: {frames}")
