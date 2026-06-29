import os
import cv2
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]


class VideoRenameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("影片觀看改名工具")
        self.root.geometry("1100x700")

        self.folder = ""
        self.files = []
        self.index = 0

        self.cap = None
        self.playing = False
        self.current_photo = None

        self.counter = {}

        self.build_ui()
        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("1", lambda e: self.prev_video())
        self.root.bind("2", lambda e: self.next_video())
        self.root.bind("<Delete>", lambda e: self.delete_current())

    def build_ui(self):
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=8)

        tk.Button(top, text="選擇影片資料夾", command=self.select_folder).pack(side="left")
        self.folder_label = tk.Label(top, text="尚未選擇資料夾", anchor="w")
        self.folder_label.pack(side="left", padx=10)

        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main, width=280)
        left.pack(side="left", fill="y", padx=10)

        tk.Label(left, text="影片清單").pack(anchor="w")
        self.listbox = tk.Listbox(left, width=40)
        self.listbox.pack(fill="y", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)

        center = tk.Frame(main)
        center.pack(side="left", fill="both", expand=True)

        self.video_label = tk.Label(center, bg="black")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        self.info_label = tk.Label(center, text="目前影片：無", font=("Arial", 12))
        self.info_label.pack(pady=5)

        controls = tk.Frame(center)
        controls.pack(pady=5)

        tk.Button(controls, text="上一部 1", width=12, command=self.prev_video).pack(side="left", padx=5)
        tk.Button(controls, text="播放 / 暫停 Space", width=18, command=self.toggle_play).pack(side="left", padx=5)
        tk.Button(controls, text="下一部 2", width=12, command=self.next_video).pack(side="left", padx=5)

        rename = tk.LabelFrame(center, text="改名")
        rename.pack(fill="x", padx=10, pady=10)

        tk.Label(rename, text="前綴：").grid(row=0, column=0, padx=5, pady=8)
        self.prefix_entry = tk.Entry(rename, width=20)
        self.prefix_entry.grid(row=0, column=1, padx=5)
        self.prefix_entry.insert(0, "walk")

        tk.Label(rename, text="編號位數：").grid(row=0, column=2, padx=5)
        self.digits_entry = tk.Entry(rename, width=8)
        self.digits_entry.grid(row=0, column=3, padx=5)
        self.digits_entry.insert(0, "3")

        tk.Button(rename, text="改名並跳下一部", command=self.rename_current).grid(
            row=0, column=4, padx=10
        )

        # 刪除按鈕（紅色，Del 快捷鍵提示）
        tk.Button(
            rename,
            text="🗑 刪除此影片  Del",
            fg="white",
            bg="#c0392b",
            activebackground="#922b21",
            activeforeground="white",
            width=16,
            command=self.delete_current,
        ).grid(row=0, column=5, padx=10)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return

        self.folder = folder
        self.folder_label.config(text=folder)
        self.load_files()

    def load_files(self):
        self.files = [
            f for f in os.listdir(self.folder)
            if Path(f).suffix.lower() in VIDEO_EXTS
        ]
        self.files.sort()

        self.listbox.delete(0, tk.END)
        for f in self.files:
            self.listbox.insert(tk.END, f)

        self.index = 0
        if self.files:
            self.listbox.selection_set(0)
            self.open_video(0)

    def open_video(self, index):
        if not self.files:
            return

        self.index = max(0, min(index, len(self.files) - 1))
        self.playing = False

        if self.cap:
            self.cap.release()

        path = os.path.join(self.folder, self.files[self.index])
        self.cap = cv2.VideoCapture(path)

        self.info_label.config(
            text=f"目前影片：[{self.index + 1}/{len(self.files)}] {self.files[self.index]}"
        )

        self.show_frame()

        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.index)
        self.listbox.see(self.index)

    def show_frame(self):
        if not self.cap:
            return

        ret, frame = self.cap.read()

        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w = frame.shape[:2]
        max_w = 760
        max_h = 480
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        frame = cv2.resize(frame, (new_w, new_h))
        img = Image.fromarray(frame)
        self.current_photo = ImageTk.PhotoImage(img)

        self.video_label.config(image=self.current_photo)

        if self.playing:
            self.root.after(30, self.show_frame)

    def toggle_play(self):
        if not self.cap:
            return

        self.playing = not self.playing

        if self.playing:
            self.show_frame()

    def prev_video(self):
        self.open_video(self.index - 1)

    def next_video(self):
        self.open_video(self.index + 1)

    def on_list_select(self, event):
        sel = self.listbox.curselection()
        if sel:
            self.open_video(sel[0])

    def get_next_number(self, prefix):
        nums = []

        for f in os.listdir(self.folder):
            name = Path(f).stem
            ext = Path(f).suffix.lower()

            if ext not in VIDEO_EXTS:
                continue

            if name.startswith(prefix):
                tail = name.replace(prefix, "", 1)
                if tail.isdigit():
                    nums.append(int(tail))

        if nums:
            return max(nums) + 1

        return 1

    def rename_current(self):
        if not self.files:
            return

        prefix = self.prefix_entry.get().strip()
        if not prefix:
            messagebox.showwarning("錯誤", "請輸入前綴，例如 walk")
            return

        try:
            digits = int(self.digits_entry.get())
        except ValueError:
            digits = 3

        old_name = self.files[self.index]
        old_path = os.path.join(self.folder, old_name)
        ext = Path(old_name).suffix

        number = self.get_next_number(prefix)
        new_name = f"{prefix}{number:0{digits}d}{ext}"
        new_path = os.path.join(self.folder, new_name)

        if self.cap:
            self.cap.release()
            self.cap = None

        try:
            os.rename(old_path, new_path)
        except PermissionError:
            messagebox.showerror("錯誤", "影片可能正在被其他程式使用，請關閉後再試")
            return
        except FileExistsError:
            messagebox.showerror("錯誤", f"{new_name} 已存在")
            return

        self.load_files()

        if self.index < len(self.files):
            self.open_video(self.index)
        elif self.files:
            self.open_video(len(self.files) - 1)

    def delete_current(self):
        if not self.files:
            return

        target = self.files[self.index]

        # 第一次確認
        confirmed = messagebox.askyesno(
            "確認刪除",
            f"確定要永久刪除以下影片嗎？\n\n{target}",
            icon="warning",
        )
        if not confirmed:
            return

        # 第二次確認（防誤觸）
        confirmed2 = messagebox.askyesno(
            "再次確認",
            f"此操作無法復原，真的要刪除嗎？\n\n{target}",
            icon="warning",
        )
        if not confirmed2:
            return

        target_path = os.path.join(self.folder, target)

        # 先釋放影片資源，否則 Windows 會鎖住檔案
        self.playing = False
        if self.cap:
            self.cap.release()
            self.cap = None

        try:
            os.remove(target_path)
        except PermissionError:
            messagebox.showerror("錯誤", "影片正在被其他程式使用，無法刪除")
            return
        except FileNotFoundError:
            messagebox.showerror("錯誤", "找不到檔案，可能已被移動或刪除")
            return

        # 刪除後重新載入，並停在同一位置（或最後一部）
        del_index = self.index
        self.load_files()

        if self.files:
            next_index = min(del_index, len(self.files) - 1)
            self.open_video(next_index)
        else:
            self.info_label.config(text="目前影片：無")
            self.video_label.config(image="")


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoRenameGUI(root)
    root.mainloop()