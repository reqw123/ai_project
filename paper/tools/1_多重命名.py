import os
import cv2
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
        self.total_frames = 0
        self.fps = 0.0
        self.updating_timeline = False

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

        # 影片清單 + 捲軸，方便快速往下拉
        listbox_frame = tk.Frame(left)
        listbox_frame.pack(fill="both", expand=True)

        list_scrollbar = tk.Scrollbar(listbox_frame, orient="vertical")
        self.listbox = tk.Listbox(
            listbox_frame,
            width=40,
            yscrollcommand=list_scrollbar.set,
        )
        list_scrollbar.config(command=self.listbox.yview)

        self.listbox.pack(side="left", fill="both", expand=True)
        list_scrollbar.pack(side="right", fill="y")

        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)

        center = tk.Frame(main)
        center.pack(side="left", fill="both", expand=True)

        self.video_label = tk.Label(center, bg="black")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        # 可點擊／拖曳的影片時間軸（數值以影格位置表示）
        timeline = tk.Frame(center)
        timeline.pack(fill="x", padx=16, pady=(0, 2))

        self.time_label = tk.Label(timeline, text="00:00 / 00:00", width=15, anchor="w")
        self.time_label.pack(side="left", padx=(0, 8))

        self.timeline_scale = tk.Scale(
            timeline,
            from_=0,
            to=1,
            orient="horizontal",
            showvalue=False,
            resolution=1,
            command=self.seek_from_timeline,
        )
        self.timeline_scale.pack(side="left", fill="x", expand=True)

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

        # 流水號從這個數字開始往上排（例如填 3 → 3, 4, 5…），不補零
        tk.Label(rename, text="起始序號：").grid(row=0, column=2, padx=5)
        self.start_entry = tk.Entry(rename, width=8)
        self.start_entry.grid(row=0, column=3, padx=5)
        self.start_entry.insert(0, "1")

        tk.Button(rename, text="改名並跳下一部", command=self.rename_current).grid(
            row=0, column=4, padx=10
        )

        # 一鍵整批：清單內所有影片原地改名（留在同一個資料夾），用同一個前綴＋
        # 流水號一次改完（從「起始序號」開始往上排）
        tk.Button(
            rename,
            text="⚡ 全部影片一鍵改名（前綴＋流水號）",
            command=self.rename_all,
        ).grid(row=1, column=0, columnspan=5, sticky="w", padx=5, pady=(0, 8))

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
            and os.path.isfile(os.path.join(self.folder, f))
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
        self.total_frames = max(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        if self.fps <= 0:
            self.fps = 30.0

        self.updating_timeline = True
        self.timeline_scale.config(to=max(1, self.total_frames - 1))
        self.timeline_scale.set(0)
        self.updating_timeline = False
        self.update_time_label(0)

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
        self.update_timeline_position()

        if self.playing:
            self.root.after(30, self.show_frame)

    def seek_from_timeline(self, value):
        """點擊或拖曳時間軸後，跳至對應影格。"""
        if self.updating_timeline or not self.cap:
            return

        target_frame = int(float(value))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        # 播放中的 after 迴圈會繼續運作；此處只更新預覽，避免拖曳時建立多個播放迴圈。
        was_playing = self.playing
        self.playing = False
        self.show_frame()
        self.playing = was_playing

    def update_timeline_position(self):
        if not self.cap:
            return

        # read() 後的位置是「下一張影格」，因此減 1 顯示目前畫面的位置。
        current_frame = max(0, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        if self.total_frames:
            current_frame = min(current_frame, self.total_frames - 1)

        self.updating_timeline = True
        self.timeline_scale.set(current_frame)
        self.updating_timeline = False
        self.update_time_label(current_frame)

    def update_time_label(self, current_frame):
        current_seconds = current_frame / self.fps if self.fps else 0
        total_seconds = self.total_frames / self.fps if self.fps else 0
        self.time_label.config(
            text=f"{self.format_time(current_seconds)} / {self.format_time(total_seconds)}"
        )

    @staticmethod
    def format_time(seconds):
        seconds = int(seconds)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

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

    def read_start(self):
        """讀「起始序號」欄位；不是正整數就回傳 None（呼叫端跳錯誤訊息）。"""
        try:
            n = int(self.start_entry.get().strip())
        except ValueError:
            return None
        return n if n >= 0 else None

    def set_start(self, n):
        self.start_entry.delete(0, tk.END)
        self.start_entry.insert(0, str(n))

    def rename_all(self):
        """把清單內所有影片依目前排序，以「前綴＋流水號」原地改寫檔名（留在同一個
        資料夾，不搬到別的地方）。編號從「起始序號」開始往上排（不補零），
        例如起始序號 3 → prefix3, prefix4, prefix5…"""
        if not self.files:
            messagebox.showinfo("提示", "目前資料夾沒有影片")
            return

        prefix = self.prefix_entry.get().strip()
        if not prefix:
            messagebox.showwarning("錯誤", "請輸入前綴，例如 walk")
            return

        start = self.read_start()
        if start is None:
            messagebox.showwarning("錯誤", "起始序號請輸入 0 以上的整數，例如 3")
            return

        plan = []  # (舊檔名, 新檔名)
        for i, old_name in enumerate(self.files):
            new_name = f"{prefix}{start + i}{Path(old_name).suffix}"
            plan.append((old_name, new_name))

        # 只跟「這批之後仍留在資料夾裡、不屬於這次改名清單」的檔案比對衝突——
        # self.files 裡的舊檔名這批結束後全部會變成新檔名，不算衝突對象。
        others = set(os.listdir(self.folder)) - set(self.files)
        conflicts = [n for _, n in plan if n in others]
        if conflicts:
            messagebox.showerror(
                "錯誤", f"資料夾內已有同名檔案（例如 {conflicts[0]}），已取消，沒有改動任何檔案"
            )
            return

        first_line = f"{plan[0][0]}  →  {plan[0][1]}"
        last_line = f"{plan[-1][0]}  →  {plan[-1][1]}"
        preview = first_line if len(plan) == 1 else "\n".join([first_line, "…", last_line])
        confirmed = messagebox.askyesno(
            "確認整批改名",
            f"將 {len(plan)} 部影片原地改名（留在同一個資料夾）：\n\n"
            f"{preview}\n\n確定嗎？",
        )
        if not confirmed:
            return

        # 改名前先釋放影片資源，否則 Windows 會鎖住檔案
        self.playing = False
        if self.cap:
            self.cap.release()
            self.cap = None

        done = 0
        error = None
        for old_name, new_name in plan:
            try:
                os.rename(os.path.join(self.folder, old_name), os.path.join(self.folder, new_name))
                done += 1
            except OSError as e:
                error = f"{old_name}：{e}"
                break

        # 起始序號接到已改名的下一個，之後再改名不會撞號
        self.set_start(start + done)

        self.load_files()
        if not self.files:
            self.info_label.config(text="目前影片：無")
            self.video_label.config(image="")

        if error:
            messagebox.showerror("部分完成", f"已改名 {done}/{len(plan)} 部，中途失敗：\n{error}")
        else:
            messagebox.showinfo("完成", f"已改名 {done} 部：{plan[0][1]} ~ {plan[-1][1]}")

    def rename_current(self):
        if not self.files:
            return

        prefix = self.prefix_entry.get().strip()
        if not prefix:
            messagebox.showwarning("錯誤", "請輸入前綴，例如 walk")
            return

        number = self.read_start()
        if number is None:
            messagebox.showwarning("錯誤", "起始序號請輸入 0 以上的整數，例如 3")
            return

        old_name = self.files[self.index]
        old_path = os.path.join(self.folder, old_name)
        ext = Path(old_name).suffix

        new_name = f"{prefix}{number}{ext}"
        new_path = os.path.join(self.folder, new_name)

        # 改名前先釋放影片資源，否則 Windows 會鎖住檔案
        if self.cap:
            self.cap.release()
            self.cap = None

        if os.path.exists(new_path):
            messagebox.showerror("錯誤", f"{new_name} 已存在")
            return

        try:
            os.rename(old_path, new_path)
        except PermissionError:
            messagebox.showerror("錯誤", "影片可能正在被其他程式使用，請關閉後再試")
            return
        except FileExistsError:
            messagebox.showerror("錯誤", f"{new_name} 已存在")
            return

        # 下一部接著用下一個序號
        self.set_start(number + 1)

        # 原地改名後，這支影片還是留在同一個資料夾（只是換了名字），不會像舊版
        # 「移到已改名資料夾」那樣自動從清單消失，重新排序後的位置也不一定緊接在
        # 原本位置——這裡改記「原本清單裡的下一部影片檔名」，reload 後找它現在的
        # 新位置接著看，才是真正的「自動接續到下一部還沒改名的影片」，而不是
        # 用清單索引位置去猜（改名後排序一變就會猜錯，跳到不相干的影片）。
        next_name = (
            self.files[self.index + 1] if self.index + 1 < len(self.files) else None
        )
        self.load_files()

        if self.files:
            if next_name is not None and next_name in self.files:
                self.open_video(self.files.index(next_name))
            else:
                self.open_video(min(self.index, len(self.files) - 1))
        else:
            self.info_label.config(text="目前影片：無")
            self.video_label.config(image="")

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
