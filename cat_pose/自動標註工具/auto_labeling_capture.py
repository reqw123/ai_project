#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cat Pose Video Annotation Tool - Interactive Editing Version
Press Space to freeze frame and edit, Press S to save and continue playing
"""

import os
import cv2
import numpy as np
import shutil
from ultralytics import YOLO
from pathlib import Path


# ==================== Colors and Skeleton Links ====================
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

COLOR_HEAD = (255, 255, 0)
COLOR_BODY = (0, 255, 0)
COLOR_LIMB = (255, 0, 0)
COLOR_TAIL = (255, 0, 255)

COLOR_LEFT_FRONT  = (255, 0, 255)
COLOR_RIGHT_FRONT = (0, 255, 255)
COLOR_LEFT_HIND   = (255, 165, 0)
COLOR_RIGHT_HIND  = (0, 255, 0)

HEAD_LINKS = [(0,1), (0,2), (1,2)]
BODY_LINKS = [(0,3), (3,4), (4,5)]
TAIL_LINKS = [(5,14), (14,15), (15,16)]


# ==================== Skeleton Drawing Functions ====================
def draw_links_fast(frame, kpts, visibility, links, color, thickness=2):
    """Draw skeleton connections"""
    for a, b in links:
        if a < len(visibility) and b < len(visibility):
            if visibility[a] > 0 and visibility[b] > 0:
                pt1 = (int(kpts[a][0]), int(kpts[a][1]))
                pt2 = (int(kpts[b][0]), int(kpts[b][1]))
                cv2.line(frame, pt1, pt2, color, thickness)


def draw_skeleton_fast(frame, kpts, visibility, thickness=2):
    """Draw complete skeleton"""
    if len(kpts) < 17:
        return
    
    draw_links_fast(frame, kpts, visibility, HEAD_LINKS, COLOR_HEAD, thickness)
    draw_links_fast(frame, kpts, visibility, BODY_LINKS, COLOR_BODY, thickness)
    draw_links_fast(frame, kpts, visibility, [(3,6), (6,7)], COLOR_LEFT_FRONT, thickness)
    draw_links_fast(frame, kpts, visibility, [(3,8), (8,9)], COLOR_RIGHT_FRONT, thickness)
    draw_links_fast(frame, kpts, visibility, [(5,10), (10,11)], COLOR_LEFT_HIND, thickness)
    draw_links_fast(frame, kpts, visibility, [(5,12), (12,13)], COLOR_RIGHT_HIND, thickness)
    draw_links_fast(frame, kpts, visibility, TAIL_LINKS, COLOR_TAIL, thickness)
    
    # Draw keypoints
    for i, (x, y) in enumerate(kpts):
        if visibility[i] > 0:
            if i in [6, 7]:
                color = COLOR_LEFT_FRONT
            elif i in [8, 9]:
                color = COLOR_RIGHT_FRONT
            elif i in [10, 11]:
                color = COLOR_LEFT_HIND
            elif i in [12, 13]:
                color = COLOR_RIGHT_HIND
            else:
                color = RED
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)


# ==================== Interactive Editor ====================
class PoseEditor:
    """Pose editor for interactive keypoint adjustment"""
    
    kpt_names = [
        "Nose", "Left Ear", "Right Ear", "Neck", "Body Center", "Hip",
        "Left Front Paw 1", "Left Front Paw 2", "Right Front Paw 1", "Right Front Paw 2",
        "Left Hind Leg 1", "Left Hind Leg 2", "Right Hind Leg 1", "Right Hind Leg 2",
        "Tail 1", "Tail 2", "Tail 3"
    ]
    
    def __init__(self, frame, kpts, visibility, frame_idx, total_kpts=17):
        self.frame = frame.copy()
        self.kpts = kpts.copy().astype(np.float32)
        self.visibility = visibility.copy().astype(np.int32)
        self.frame_idx = frame_idx
        self.total_kpts = total_kpts
        
        self.selected = 0
        self.dragging = False
        self.last_mouse_pos = None
        self.saved = False
        self.zoom_scale = 1.0  # Add zoom capability
        
        h, w = self.frame.shape[:2]
        self.h, self.w = h, w
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for dragging keypoints"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find nearest keypoint (adjust for zoom)
            kpts_zoom = self.kpts * self.zoom_scale
            dists = np.linalg.norm(kpts_zoom - np.array([x, y]), axis=1)
            idx = np.argmin(dists)
            # Activate dragging if within 15 pixels
            if dists[idx] < 15:
                self.selected = idx
                self.dragging = True
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                # Update keypoint position with boundary check (adjust for zoom)
                x_orig = x / self.zoom_scale
                y_orig = y / self.zoom_scale
                self.kpts[self.selected][0] = np.clip(x_orig, 0, self.w - 1)
                self.kpts[self.selected][1] = np.clip(y_orig, 0, self.h - 1)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
    
    def draw_editor(self):
        """Draw editing view"""
        # Apply zoom
        if self.zoom_scale != 1.0:
            h, w = self.frame.shape[:2]
            new_w = int(w * self.zoom_scale)
            new_h = int(h * self.zoom_scale)
            disp = cv2.resize(self.frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            kpts_zoom = self.kpts * self.zoom_scale
        else:
            disp = self.frame.copy()
            kpts_zoom = self.kpts.copy()
        
        h, w = disp.shape[:2]
        
        # Draw skeleton with zoomed coordinates
        draw_skeleton_fast(disp, kpts_zoom, self.visibility, thickness=2)
        
        # Draw keypoints
        for i, (x, y) in enumerate(kpts_zoom):
            # Set color based on visibility
            if self.visibility[i] == 2:
                color = GREEN  # Visible
            elif self.visibility[i] == 1:
                color = YELLOW  # Occluded
            else:
                color = (128, 128, 128)  # Invisible
            
            # Draw point - smaller size (radius 3-4)
            radius = 4 if i == self.selected else 3
            cv2.circle(disp, (int(x), int(y)), radius, color, -1)
            
            # Add white border to selected point only
            if i == self.selected:
                cv2.circle(disp, (int(x), int(y)), radius + 1, WHITE, 1)
            
            # Draw point number
            cv2.putText(disp, str(i), (int(x) + 6, int(y) - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
        
        # Information panel at top
        info_text = [
            f"Frame: {self.frame_idx} | Zoom: {self.zoom_scale:.1f}x",
            f"Point: {self.selected} - {self.kpt_names[self.selected]} | Pos: ({self.kpts[self.selected][0]:.0f}, {self.kpts[self.selected][1]:.0f}) | Vis: {['Inv', 'Occ', 'Vis'][self.visibility[self.selected]]}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(disp, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
            y_offset += 28
        
        # Controls hint at bottom (2 lines)
        h_disp = disp.shape[0]
        control_text = [
            "LMB=Move  A/D=Switch  W/S=Visibility  0/1/2=Set  +/-=Zoom  S=Save  Q=Cancel",
            "Green=Visible  Yellow=Occluded  Gray=Invisible"
        ]
        
        y_bottom = h_disp - 35
        for i, text in enumerate(control_text):
            cv2.putText(disp, text, (10, y_bottom + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return disp
    
    def run(self):
        """Run the editor"""
        win = f"Edit Pose - Frame {self.frame_idx} (Press S to save and continue)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 960)
        cv2.setMouseCallback(win, self.mouse_callback)
        
        print(f"\n[Edit] Frame {self.frame_idx}...")
        print("Controls: LMB drag=move | A/D=switch | W=toggle visibility | S=save")
        
        while True:
            disp = self.draw_editor()
            cv2.imshow(win, disp)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('s') or key == ord('S'):
                # Save and exit
                self.saved = True
                cv2.destroyWindow(win)
                print(f"[Saved] Frame {self.frame_idx}")
                return True
            
            elif key == ord('a') or key == ord('A'):
                # Previous keypoint
                self.selected = (self.selected - 1) % self.total_kpts
            
            elif key == ord('d') or key == ord('D'):
                # Next keypoint
                self.selected = (self.selected + 1) % self.total_kpts
            
            elif key == ord('w') or key == ord('W'):
                # Cycle visibility forward (0->1->2->0)
                self.visibility[self.selected] = (self.visibility[self.selected] + 1) % 3
            
            elif key == ord('0'):
                self.visibility[self.selected] = 0
            elif key == ord('1'):
                self.visibility[self.selected] = 1
            elif key == ord('2'):
                self.visibility[self.selected] = 2
            
            elif key == ord('+') or key == ord('='):
                # Zoom in
                self.zoom_scale = min(self.zoom_scale * 1.2, 4.0)
            
            elif key == ord('-') or key == ord('_'):
                # Zoom out
                self.zoom_scale = max(self.zoom_scale / 1.2, 0.5)
            
            elif key == 27 or key == ord('q') or key == ord('Q'):  # ESC
                cv2.destroyWindow(win)
                print(f"[Cancel] Frame {self.frame_idx}")
                return False
        
        cv2.destroyWindow(win)
        return False


# ==================== Helper Functions ====================
def get_bbox_from_kpts(kpts, vis, w, h, margin=0.05):
    """Calculate bounding box from keypoints"""
    valid_kpts = kpts[vis > 0]
    if len(valid_kpts) == 0:
        return 0.5, 0.5, 0.5, 0.5
    
    x_min = valid_kpts[:, 0].min()
    y_min = valid_kpts[:, 1].min()
    x_max = valid_kpts[:, 0].max()
    y_max = valid_kpts[:, 1].max()
    
    w_box = x_max - x_min
    h_box = y_max - y_min
    
    x_min = max(0, x_min - w_box * margin)
    y_min = max(0, y_min - h_box * margin)
    x_max = min(w, x_max + w_box * margin)
    y_max = min(h, y_max + h_box * margin)
    
    bbox_center_x = np.clip(((x_min + x_max) / 2) / w, 0, 1)
    bbox_center_y = np.clip(((y_min + y_max) / 2) / h, 0, 1)
    bbox_width = np.clip((x_max - x_min) / w, 0, 1)
    bbox_height = np.clip((y_max - y_min) / h, 0, 1)
    
    return bbox_center_x, bbox_center_y, bbox_width, bbox_height


def create_label_string(class_id, bbox, kpts, vis, w, h, total_kpts=17):
    """Generate YOLO format label string"""
    label_parts = [str(class_id)]
    label_parts.extend([f"{v:.16f}" for v in bbox])
    
    for i in range(total_kpts):
        kpt_x = np.clip(kpts[i, 0] / w, 0, 1)
        kpt_y = np.clip(kpts[i, 1] / h, 0, 1)
        label_parts.extend([f"{kpt_x:.16f}", f"{kpt_y:.16f}", str(vis[i])])
    
    return " ".join(label_parts)


# ==================== Main Program ====================
def main():
    """Main program entry"""
    
    # ==================== Configuration ====================
    MODEL_PATH = r"C:/cat_pose/2222.pt"
    VIDEO_PATH = r"C:/cat_pose/test1.mp4"
    OUTPUT_DIR = r"C:\cat_pose\自動標註工具\auto_capture"
    CLASS_ID = 0
    TOTAL_KPTS = 17
    
    # Create folders
    images_dir = Path(OUTPUT_DIR) / "images"
    labels_dir = Path(OUTPUT_DIR) / "labels"
    yolo_infer_dir = Path(OUTPUT_DIR) / "yolo_infer_images"
    
    # Buffer
    temp_images = []
    temp_labels = []
    temp_infer_images = []
    
    # Load model
    print("\n" + "="*60)
    print("Cat Pose Video Annotation Tool - Interactive Edition")
    print("="*60)
    print("[Loading] Model...")
    
    try:
        model = YOLO(MODEL_PATH)
        model.to("cuda")
        print("[OK] Model loaded (GPU)!")
    except Exception as e:
        print(f"[Failed] GPU loading failed: {e}")
        try:
            model = YOLO(MODEL_PATH)
            print("[OK] Model loaded (CPU)!")
        except Exception as e2:
            print(f"[Failed] Model loading failed: {e2}")
            return
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {VIDEO_PATH}")
        return
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[Info] Video information:")
    print(f"       Resolution: {width}x{height}")
    print(f"       Total frames: {total_frames}")
    print(f"       FPS: {fps:.2f}")
    
    frame_idx = 0
    save_idx = 0
    user_scale = 1.0
    paused = False
    
    print("\n" + "="*60)
    print("Playback Controls:")
    print("  Space  - Freeze frame and enter edit mode")
    print("  P      - Pause/Resume playback")
    print("  Left   - Previous frame")
    print("  Right  - Next frame")
    print("  +/-    - Zoom in/out preview")
    print("  Q      - Quit")
    print("="*60 + "\n")
    
    frame = None
    ret = True
    
    # Main loop
    while cap.isOpened():
        if not paused and ret:
            ret, frame = cap.read()
            if not ret:
                # Loop video when finished
                print("\n[Loop] Video finished, restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                ret, frame = cap.read()
                if not ret:
                    break
                continue
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        
        if frame is None:
            continue
        
        # Auto inference
        result = model.predict(frame, imgsz=640, conf=0.5, half=False, verbose=False)[0]
        
        # Display frame
        disp_frame = frame.copy()
        h0, w0 = disp_frame.shape[:2]
        scale_disp = min(1920 / w0, 1080 / h0, 1.0) * user_scale
        
        if scale_disp != 1.0:
            disp_frame = cv2.resize(disp_frame, (int(w0 * scale_disp), int(h0 * scale_disp)), 
                                   interpolation=cv2.INTER_AREA)
        
        # Draw inference results
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            kpts = result.keypoints.xy[0].cpu().numpy()
            kpt_conf = result.keypoints.conf[0].cpu().numpy()
            kpts_disp = kpts * scale_disp
            draw_skeleton_fast(disp_frame, kpts_disp, kpt_conf)
            
            if result.boxes is not None and len(result.boxes.xyxy) > 0:
                box = result.boxes.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = (box * scale_disp).astype(int)
                cv2.rectangle(disp_frame, (x1, y1), (x2, y2), YELLOW, 2)
                conf = result.boxes.conf[0].item()
                cv2.putText(disp_frame, f"Conf: {conf:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
        
        # Display info
        cv2.putText(disp_frame, f"Frame: {frame_idx}/{total_frames}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)
        cv2.putText(disp_frame, f"Annotated: {save_idx}", (20, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)
        status = "PAUSED" if paused else "PLAYING"
        status_color = RED if paused else GREEN
        cv2.putText(disp_frame, f"Status: {status}", (20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(disp_frame, "Space=Edit  P=Pause  Arrow=Frame  +/-=Zoom  Q=Quit", (20, 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Cat Pose Annotation", disp_frame)
        
        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\n[Exit] Quitting...")
            break
        elif key == ord('p') or key == ord('P'):
            paused = not paused
            print(f"[Status] {'Paused' if paused else 'Resumed'}")
        elif key == 81 or key == 65361:  # Left arrow
            if frame_idx > 0:
                frame_idx -= 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
        elif key == 83 or key == 65363:  # Right arrow
            if frame_idx < total_frames - 1:
                frame_idx += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
        elif key == ord('+') or key == ord('='):
            user_scale = min(user_scale + 0.1, 2.5)
            print(f"[Zoom] {user_scale:.2f}x")
        elif key == ord('-') or key == ord('_'):
            user_scale = max(user_scale - 0.1, 0.2)
            print(f"[Zoom] {user_scale:.2f}x")
        elif key == 32:  # Space - Enter edit mode
            # Check if keypoints detected
            if not (hasattr(result, 'keypoints') and result.keypoints is not None and 
                   hasattr(result.keypoints, 'xy') and len(result.keypoints.xy) > 0):
                print(f"[Warning] Frame {frame_idx}: No keypoints detected")
                continue
            
            # Extract keypoints and confidence
            kpts = result.keypoints.xy[0].cpu().numpy()
            kpt_conf = result.keypoints.conf[0].cpu().numpy()
            
            # Generate visibility labels (based on confidence)
            vis = np.array([2 if c > 0.6 else (1 if c > 0.3 else 0) for c in kpt_conf], dtype=np.int32)
            
            # Enter editor
            editor = PoseEditor(frame, kpts, vis, frame_idx, TOTAL_KPTS)
            if editor.run():
                # Save editing result
                save_idx += 1
                img_name = f"cat{save_idx}.jpg"
                label_name = f"cat{save_idx}.txt"
                h, w = frame.shape[:2]
                
                kpts_edited = editor.kpts
                vis_edited = editor.visibility
                
                # Calculate bounding box
                bbox = get_bbox_from_kpts(kpts_edited, vis_edited, w, h)
                
                # Generate label string
                label_str = create_label_string(CLASS_ID, bbox, kpts_edited, vis_edited, w, h, TOTAL_KPTS)
                
                # Save to buffer
                temp_images.append((frame.copy(), img_name))
                temp_labels.append((label_str, label_name))
                
                # Generate visualization
                visual_frame = frame.copy()
                draw_skeleton_fast(visual_frame, kpts_edited, vis_edited, thickness=2)
                for i, (x, y) in enumerate(kpts_edited):
                    if vis_edited[i] == 2:
                        color = GREEN
                    elif vis_edited[i] == 1:
                        color = YELLOW
                    else:
                        color = (128, 128, 128)
                    cv2.circle(visual_frame, (int(x), int(y)), 5, color, -1)
                    cv2.putText(visual_frame, str(i), (int(x)+8, int(y)-8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
                
                temp_infer_images.append((visual_frame, img_name))
        
        if not paused:
            frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Batch save
    if len(temp_images) == 0:
        print("\n[Done] No annotations")
        return
    
    print("\n" + "="*60)
    print("[Saving] Files...")
    print("="*60)
    
    # Clear and recreate folders
    for d in [images_dir, labels_dir, yolo_infer_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    
    # Write images
    for img, img_name in temp_images:
        cv2.imwrite(str(images_dir / img_name), img)
    print(f"[OK] Images saved to: {images_dir}")
    
    # Write labels
    for label_str, label_name in temp_labels:
        with open(labels_dir / label_name, 'w', encoding='utf-8') as f:
            f.write(label_str)
    print(f"[OK] Labels saved to: {labels_dir}")
    
    # Write visualizations
    for infer_img, infer_name in temp_infer_images:
        cv2.imwrite(str(yolo_infer_dir / infer_name), infer_img)
    print(f"[OK] Visualizations saved to: {yolo_infer_dir}")
    
    print("\n" + "="*60)
    print(f"[Complete] {len(temp_images)} images annotated")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Interrupted] User interrupted")
    except Exception as e:
        print(f"\n[Error] Execution failed: {e}")
        import traceback
        traceback.print_exc()