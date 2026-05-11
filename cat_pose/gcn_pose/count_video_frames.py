"""
Count total frames for each category folder
==========================================
This script counts the total number of frames (all videos combined) in each of the four category folders.

Usage: python count_video_frames.py
"""

import cv2
from pathlib import Path

# Define your category folders
VIDEO_FOLDERS = [
    ("walk", r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\walk"),
    ("lying", r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\lying"),
    ("lick", r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\lick"),
    ("shake", r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\shake"),
]

VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv']

def count_frames_in_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Warning] Failed to open video: {video_path}")
        return 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def main():
    print("="*60)
    print("Total Frame Count Per Category")
    print("="*60)
    grand_total = 0
    for category, folder in VIDEO_FOLDERS:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"[Warning] Folder not found: {folder}")
            continue
        video_files = [f for f in folder_path.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]
        category_total = 0
        for video_file in video_files:
            frames = count_frames_in_video(video_file)
            print(f"  {category:6} | {video_file.name:30} : {frames:6} frames")
            category_total += frames
        print(f"[Category: {category}] Total frames: {category_total}")
        print("-"*60)
        grand_total += category_total
    print(f"[All Categories] Grand total frames: {grand_total}")
    print("="*60)

if __name__ == "__main__":
    main()
