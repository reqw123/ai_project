#!/usr/bin/env python3
"""以圖形介面統計多個資料夾內 Python 檔案的程式碼行數。"""

from __future__ import annotations

import io
import os
import queue
import threading
import tokenize
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk


IGNORED_TOKEN_TYPES = {
    tokenize.ENCODING,
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
}


def count_file_lines(file_path: Path) -> tuple[int, int]:
    """回傳（含註解與空白的全部行數, 排除註解與空白的有效行數）。"""
    source = file_path.read_bytes()
    all_lines = len(source.splitlines())
    code_lines: set[int] = set()

    for token in tokenize.tokenize(io.BytesIO(source).readline):
        if token.type in IGNORED_TOKEN_TYPES:
            continue

        start_line, end_line = token.start[0], token.end[0]
        code_lines.update(range(start_line, end_line + 1))

    return all_lines, len(code_lines)


class MultiFolderDialog:
    """以樹狀清單提供 Ctrl／Shift 多選資料夾功能。"""

    DUMMY_VALUE = "__load_children__"

    def __init__(self, parent: tk.Tk) -> None:
        self.parent = parent
        self.result: list[Path] = []

        self.window = tk.Toplevel(parent)
        self.window.title("一次選擇多個資料夾")
        self.window.geometry("900x610")
        self.window.minsize(680, 440)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._cancel)

        container = ttk.Frame(self.window, padding=16)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="一次選擇多個資料夾",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            container,
            text="展開資料夾後，按住 Ctrl 可逐項選取；按住 Shift 可選取一段範圍。",
        ).pack(anchor="w", pady=(4, 10))

        tree_frame = ttk.Frame(container)
        tree_frame.pack(fill="both", expand=True)
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(
            tree_frame,
            columns=("path",),
            show="tree headings",
            selectmode="extended",
        )
        self.tree.heading("#0", text="資料夾")
        self.tree.heading("path", text="完整路徑")
        self.tree.column("#0", width=300, minwidth=180, stretch=True)
        self.tree.column("path", width=560, minwidth=300, stretch=True)

        vertical_scrollbar = ttk.Scrollbar(
            tree_frame,
            orient="vertical",
            command=self.tree.yview,
        )
        horizontal_scrollbar = ttk.Scrollbar(
            tree_frame,
            orient="horizontal",
            command=self.tree.xview,
        )
        self.tree.configure(
            yscrollcommand=vertical_scrollbar.set,
            xscrollcommand=horizontal_scrollbar.set,
        )
        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal_scrollbar.grid(row=1, column=0, sticky="ew")

        self.selection_text = tk.StringVar(value="已選取 0 個資料夾")
        footer = ttk.Frame(container)
        footer.pack(fill="x", pady=(10, 0))
        ttk.Label(footer, textvariable=self.selection_text).pack(side="left")
        ttk.Button(footer, text="取消", command=self._cancel).pack(side="right")
        ttk.Button(
            footer,
            text="加入選取",
            command=self._confirm,
            style="Accent.TButton",
        ).pack(side="right", padx=(0, 8))

        self.tree.bind("<<TreeviewOpen>>", self._load_opened_item)
        self.tree.bind("<<TreeviewSelect>>", self._update_selection_text)
        self.tree.bind("<Return>", lambda _event: self._confirm())

        self._insert_roots()
        self._center_over_parent()

    @staticmethod
    def _system_roots() -> list[tuple[str, Path]]:
        home = Path.home().resolve()
        roots: list[tuple[str, Path]] = [("使用者資料夾", home)]

        if os.name == "nt":
            import ctypes

            drive_mask = ctypes.windll.kernel32.GetLogicalDrives()
            for index in range(26):
                if drive_mask & (1 << index):
                    drive = Path(f"{chr(65 + index)}:\\")
                    if os.path.normcase(str(drive)) != os.path.normcase(str(home)):
                        roots.append((f"磁碟機 {drive}", drive))
        else:
            root_path = Path("/")
            if root_path != home:
                roots.append(("檔案系統 /", root_path))

        return roots

    def _insert_roots(self) -> None:
        for label, path in self._system_roots():
            item_id = self.tree.insert(
                "",
                "end",
                text=label,
                values=(str(path),),
            )
            self._insert_dummy(item_id)

    def _insert_dummy(self, parent_id: str) -> None:
        self.tree.insert(
            parent_id,
            "end",
            text="讀取中……",
            values=(self.DUMMY_VALUE,),
        )

    def _load_opened_item(self, _event: tk.Event) -> None:
        item_id = self.tree.focus()
        if not item_id:
            return

        children = self.tree.get_children(item_id)
        if len(children) != 1:
            return

        child_values = self.tree.item(children[0], "values")
        if not child_values or child_values[0] != self.DUMMY_VALUE:
            return

        self.tree.delete(children[0])
        item_values = self.tree.item(item_id, "values")
        if not item_values:
            return

        folder = Path(item_values[0])
        try:
            subfolders = sorted(
                (path for path in folder.iterdir() if path.is_dir()),
                key=lambda path: path.name.lower(),
            )
        except (OSError, PermissionError):
            return

        for subfolder in subfolders:
            child_id = self.tree.insert(
                item_id,
                "end",
                text=subfolder.name or str(subfolder),
                values=(str(subfolder),),
            )
            self._insert_dummy(child_id)

    def _update_selection_text(self, _event: tk.Event | None = None) -> None:
        valid_count = 0
        for item_id in self.tree.selection():
            values = self.tree.item(item_id, "values")
            if values and values[0] != self.DUMMY_VALUE:
                valid_count += 1
        self.selection_text.set(f"已選取 {valid_count} 個資料夾")

    def _confirm(self) -> None:
        selected_paths: list[Path] = []
        seen: set[str] = set()

        for item_id in self.tree.selection():
            values = self.tree.item(item_id, "values")
            if not values or values[0] == self.DUMMY_VALUE:
                continue

            path = Path(values[0])
            key = os.path.normcase(str(path))
            if path.is_dir() and key not in seen:
                seen.add(key)
                selected_paths.append(path)

        if not selected_paths:
            messagebox.showwarning(
                "尚未選取",
                "請先選取至少一個資料夾。",
                parent=self.window,
            )
            return

        self.result = selected_paths
        self.window.destroy()

    def _cancel(self) -> None:
        self.result = []
        self.window.destroy()

    def _center_over_parent(self) -> None:
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = self.parent.winfo_rootx() + max((self.parent.winfo_width() - width) // 2, 0)
        y = self.parent.winfo_rooty() + max((self.parent.winfo_height() - height) // 2, 0)
        self.window.geometry(f"{width}x{height}+{x}+{y}")

    def show(self) -> list[Path]:
        self.window.grab_set()
        self.window.focus_set()
        self.parent.wait_window(self.window)
        return self.result


class PythonLineCounterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Python 程式碼行數統計工具")
        self.root.geometry("1220x760")
        self.root.minsize(850, 580)

        self.recursive_var = tk.BooleanVar(value=True)
        self.folder_count_var = tk.StringVar(value="資料夾數：0")
        self.file_count_var = tk.StringVar(value="Python 檔案數：0")
        self.all_lines_var = tk.StringVar(value="全部行數：0")
        self.code_lines_var = tk.StringVar(value="有效程式碼行數：0")
        self.progress_text_var = tk.StringVar(value="尚未開始")
        self.status_var = tk.StringVar(value="請加入一個或多個資料夾")

        self.worker_events: queue.Queue[tuple] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.is_running = False

        self._configure_style()
        self._build_ui()

    def _configure_style(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("Title.TLabel", font=("Microsoft JhengHei UI", 18, "bold"))
        style.configure("Section.TLabel", font=("Microsoft JhengHei UI", 10, "bold"))
        style.configure("Summary.TLabel", font=("Microsoft JhengHei UI", 10, "bold"))
        style.configure("Treeview", rowheight=28, font=("Microsoft JhengHei UI", 10))
        style.configure("Treeview.Heading", font=("Microsoft JhengHei UI", 10, "bold"))
        style.configure("Accent.TButton", font=("Microsoft JhengHei UI", 10, "bold"))
        style.map(
            "Accent.TButton",
            background=[("active", "#1D4ED8"), ("!disabled", "#2563EB")],
            foreground=[("!disabled", "white")],
        )

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=18)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="Python 程式碼行數統計工具",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            container,
            text="可加入多個資料夾，同時統計全部實體行數與排除註解、空白後的有效行數。",
        ).pack(anchor="w", pady=(4, 12))

        folder_section = ttk.LabelFrame(container, text="統計資料夾", padding=10)
        folder_section.pack(fill="x")

        folder_list_frame = ttk.Frame(folder_section)
        folder_list_frame.pack(side="left", fill="both", expand=True)
        folder_list_frame.rowconfigure(0, weight=1)
        folder_list_frame.columnconfigure(0, weight=1)

        self.folder_listbox = tk.Listbox(
            folder_list_frame,
            height=5,
            selectmode=tk.EXTENDED,
            activestyle="none",
            font=("Microsoft JhengHei UI", 10),
        )
        folder_vertical_scrollbar = ttk.Scrollbar(
            folder_list_frame,
            orient="vertical",
            command=self.folder_listbox.yview,
        )
        folder_horizontal_scrollbar = ttk.Scrollbar(
            folder_list_frame,
            orient="horizontal",
            command=self.folder_listbox.xview,
        )
        self.folder_listbox.configure(
            yscrollcommand=folder_vertical_scrollbar.set,
            xscrollcommand=folder_horizontal_scrollbar.set,
        )
        self.folder_listbox.grid(row=0, column=0, sticky="nsew")
        folder_vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        folder_horizontal_scrollbar.grid(row=1, column=0, sticky="ew")

        folder_buttons = ttk.Frame(folder_section)
        folder_buttons.pack(side="left", fill="y", padx=(10, 0))

        self.add_folder_button = ttk.Button(
            folder_buttons,
            text="選擇多個資料夾",
            command=self.add_folder,
        )
        self.add_folder_button.pack(fill="x")

        self.remove_folder_button = ttk.Button(
            folder_buttons,
            text="移除選取",
            command=self.remove_selected_folders,
        )
        self.remove_folder_button.pack(fill="x", pady=(7, 0))

        self.clear_folders_button = ttk.Button(
            folder_buttons,
            text="清空全部",
            command=self.clear_folders,
        )
        self.clear_folders_button.pack(fill="x", pady=(7, 0))

        action_frame = ttk.Frame(container)
        action_frame.pack(fill="x", pady=(10, 12))

        self.recursive_checkbutton = ttk.Checkbutton(
            action_frame,
            text="包含所有子資料夾",
            variable=self.recursive_var,
        )
        self.recursive_checkbutton.pack(side="left")

        ttk.Label(
            action_frame,
            text="在選擇視窗中按住 Ctrl／Shift 即可一次多選",
        ).pack(side="left", padx=(18, 0))

        self.analyze_button = ttk.Button(
            action_frame,
            text="開始統計",
            command=self.start_analysis,
            style="Accent.TButton",
        )
        self.analyze_button.pack(side="right")

        progress_frame = ttk.Frame(container)
        progress_frame.pack(fill="x", pady=(0, 12))
        progress_frame.columnconfigure(0, weight=1)

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode="determinate",
            maximum=1,
            value=0,
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            progress_frame,
            textvariable=self.progress_text_var,
            width=25,
            anchor="e",
        ).grid(row=0, column=1, padx=(10, 0))

        summary_frame = ttk.Frame(container)
        summary_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(summary_frame, textvariable=self.folder_count_var, style="Summary.TLabel").pack(side="left")
        ttk.Label(summary_frame, textvariable=self.file_count_var, style="Summary.TLabel").pack(side="left", padx=(20, 0))
        ttk.Label(summary_frame, textvariable=self.all_lines_var, style="Summary.TLabel").pack(side="left", padx=(20, 0))
        ttk.Label(summary_frame, textvariable=self.code_lines_var, style="Summary.TLabel").pack(side="left", padx=(20, 0))

        table_frame = ttk.Frame(container)
        table_frame.pack(fill="both", expand=True)
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(
            table_frame,
            columns=("index", "all_lines", "code_lines", "path"),
            show="headings",
        )
        self.tree.heading("index", text="編號")
        self.tree.heading("all_lines", text="全部行數（含註解／空白）")
        self.tree.heading("code_lines", text="有效行數（排除註解／空白）")
        self.tree.heading("path", text="完整檔案路徑")
        self.tree.column("index", width=65, minwidth=60, anchor="center", stretch=False)
        self.tree.column("all_lines", width=185, minwidth=170, anchor="center", stretch=False)
        self.tree.column("code_lines", width=195, minwidth=180, anchor="center", stretch=False)
        self.tree.column("path", width=760, minwidth=350, anchor="w", stretch=True)
        self.tree.tag_configure("even", background="#F4F7FB")

        result_vertical_scrollbar = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.tree.yview,
        )
        result_horizontal_scrollbar = ttk.Scrollbar(
            table_frame,
            orient="horizontal",
            command=self.tree.xview,
        )
        self.tree.configure(
            yscrollcommand=result_vertical_scrollbar.set,
            xscrollcommand=result_horizontal_scrollbar.set,
        )
        self.tree.grid(row=0, column=0, sticky="nsew")
        result_vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        result_horizontal_scrollbar.grid(row=1, column=0, sticky="ew")

        ttk.Label(
            container,
            textvariable=self.status_var,
            anchor="w",
        ).pack(fill="x", pady=(9, 0))

    def add_folder(self) -> None:
        selected_folders = MultiFolderDialog(self.root).show()
        if not selected_folders:
            return

        current_folders = self._get_folders()
        existing_keys = {os.path.normcase(str(path)) for path in current_folders}
        added_count = 0

        for selected_folder in selected_folders:
            resolved_folder = str(selected_folder.resolve())
            key = os.path.normcase(resolved_folder)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            self.folder_listbox.insert(tk.END, resolved_folder)
            added_count += 1

        self._update_folder_count()
        if added_count:
            self.status_var.set(f"已加入 {added_count} 個資料夾，可直接開始統計")
        else:
            self.status_var.set("選取的資料夾都已存在於清單中")

    def remove_selected_folders(self) -> None:
        selected_indices = self.folder_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("尚未選取", "請先選取要移除的資料夾。")
            return

        for index in reversed(selected_indices):
            self.folder_listbox.delete(index)
        self._update_folder_count()
        self.status_var.set("已移除選取的資料夾")

    def clear_folders(self) -> None:
        if self.folder_listbox.size() == 0:
            return
        self.folder_listbox.delete(0, tk.END)
        self._update_folder_count()
        self.status_var.set("資料夾清單已清空")

    def _get_folders(self) -> list[Path]:
        return [Path(value) for value in self.folder_listbox.get(0, tk.END)]

    def _update_folder_count(self) -> None:
        self.folder_count_var.set(f"資料夾數：{self.folder_listbox.size()}")

    def _clear_results(self) -> None:
        for item_id in self.tree.get_children():
            self.tree.delete(item_id)
        self.file_count_var.set("Python 檔案數：0")
        self.all_lines_var.set("全部行數：0")
        self.code_lines_var.set("有效程式碼行數：0")

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.add_folder_button.configure(state=state)
        self.remove_folder_button.configure(state=state)
        self.clear_folders_button.configure(state=state)
        self.analyze_button.configure(state=state)
        self.recursive_checkbutton.configure(state=state)

    def start_analysis(self) -> None:
        if self.is_running:
            return

        folders = self._get_folders()
        if not folders:
            messagebox.showwarning("尚未加入資料夾", "請先加入至少一個資料夾。")
            return

        self._clear_results()
        self._set_controls_enabled(False)
        self.is_running = True
        self.progress_bar.configure(mode="indeterminate", maximum=1, value=0)
        self.progress_bar.start(12)
        self.progress_text_var.set("正在搜尋 .py 檔案……")
        self.status_var.set("正在整理檔案清單並排除重複項目")

        self.worker_thread = threading.Thread(
            target=self._analysis_worker,
            args=(folders, self.recursive_var.get()),
            daemon=True,
        )
        self.worker_thread.start()
        self.root.after(50, self._poll_worker_events)

    def _analysis_worker(self, folders: list[Path], recursive: bool) -> None:
        files_by_key: dict[str, Path] = {}
        errors: list[str] = []

        for folder in folders:
            if not folder.is_dir():
                errors.append(f"資料夾不存在或無法存取：{folder}")
                continue

            try:
                iterator = folder.rglob("*.py") if recursive else folder.glob("*.py")
                for file_path in iterator:
                    try:
                        if not file_path.is_file():
                            continue
                        resolved_path = file_path.resolve()
                        key = os.path.normcase(str(resolved_path))
                        files_by_key[key] = resolved_path
                    except OSError as exc:
                        errors.append(f"無法檢查：{file_path}\n  {exc}")
            except OSError as exc:
                errors.append(f"無法搜尋：{folder}\n  {exc}")

        files = sorted(files_by_key.values(), key=lambda path: os.path.normcase(str(path)))
        self.worker_events.put(("discovered", len(files)))

        successful_files = 0
        total_all_lines = 0
        total_code_lines = 0

        for file_index, file_path in enumerate(files, start=1):
            try:
                all_lines, code_lines = count_file_lines(file_path)
            except (OSError, SyntaxError, UnicodeError, tokenize.TokenError) as exc:
                errors.append(f"無法統計：{file_path}\n  {exc}")
            else:
                successful_files += 1
                total_all_lines += all_lines
                total_code_lines += code_lines
                self.worker_events.put(
                    (
                        "result",
                        successful_files,
                        all_lines,
                        code_lines,
                        str(file_path),
                    )
                )

            self.worker_events.put(("progress", file_index, len(files)))

        self.worker_events.put(
            (
                "done",
                successful_files,
                total_all_lines,
                total_code_lines,
                errors,
                len(files),
            )
        )

    def _poll_worker_events(self) -> None:
        handled_events = 0
        while handled_events < 200:
            try:
                event = self.worker_events.get_nowait()
            except queue.Empty:
                break

            handled_events += 1
            event_type = event[0]

            if event_type == "discovered":
                total_files = event[1]
                self.progress_bar.stop()
                self.progress_bar.configure(
                    mode="determinate",
                    maximum=max(total_files, 1),
                    value=0,
                )
                self.progress_text_var.set(f"0 / {total_files}（0%）")
                self.status_var.set(f"找到 {total_files} 個不重複的 Python 檔案")

            elif event_type == "result":
                _, index, all_lines, code_lines, file_path = event
                tag = "even" if index % 2 == 0 else ""
                self.tree.insert(
                    "",
                    "end",
                    values=(index, all_lines, code_lines, file_path),
                    tags=(tag,),
                )

            elif event_type == "progress":
                _, current, total = event
                percentage = round(current / total * 100) if total else 100
                self.progress_bar.configure(value=current)
                self.progress_text_var.set(f"{current} / {total}（{percentage}%）")

            elif event_type == "done":
                self._finish_analysis(*event[1:])

        if self.is_running or not self.worker_events.empty():
            self.root.after(50, self._poll_worker_events)

    def _finish_analysis(
        self,
        successful_files: int,
        total_all_lines: int,
        total_code_lines: int,
        errors: list[str],
        discovered_files: int,
    ) -> None:
        self.is_running = False
        self._set_controls_enabled(True)
        self.file_count_var.set(f"Python 檔案數：{successful_files}")
        self.all_lines_var.set(f"全部行數：{total_all_lines:,}")
        self.code_lines_var.set(f"有效程式碼行數：{total_code_lines:,}")

        if discovered_files == 0:
            self.progress_bar.configure(value=0)
            self.progress_text_var.set("沒有可統計的檔案")
            self.status_var.set("選取的資料夾中找不到 .py 檔案")
            if errors:
                self._show_errors(errors)
            else:
                messagebox.showinfo("沒有結果", "選取的資料夾中找不到 .py 檔案。")
            return

        if errors:
            self.status_var.set(f"統計完成，但有 {len(errors)} 個項目發生錯誤")
            self._show_errors(errors)
        else:
            self.status_var.set("統計完成")

    @staticmethod
    def _show_errors(errors: list[str]) -> None:
        error_preview = "\n\n".join(errors[:10])
        if len(errors) > 10:
            error_preview += f"\n\n……另有 {len(errors) - 10} 個錯誤未顯示。"
        messagebox.showwarning("部分項目無法統計", error_preview)


def main() -> None:
    root = tk.Tk()
    app = PythonLineCounterApp(root)
    root.after(150, app.add_folder)
    root.mainloop()


if __name__ == "__main__":
    main()
