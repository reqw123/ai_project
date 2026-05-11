import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

def rename_files():
    try:
        # 獲取使用者輸入
        folder_path = path_entry.get()
        start_num = int(start_entry.get())
        prefix = prefix_entry.get()
        source_extension = source_combobox.get()
        target_extension = target_combobox.get()
        
        # 驗證輸入
        if not folder_path.strip():
            messagebox.showerror("錯誤", "請選擇資料夾路徑")
            return
            
        if not os.path.exists(folder_path):
            messagebox.showerror("錯誤", "資料夾路徑不存在")
            return
            
        if not source_extension:
            messagebox.showerror("錯誤", "請選擇原始檔案類型")
            return
            
        if not target_extension:
            messagebox.showerror("錯誤", "請選擇目標檔案類型")
            return
        
        # 獲取指定副檔名的檔案，支援中文路徑
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(f'.{source_extension.lower()}')]
        
        if not files:
            messagebox.showwarning("警告", f"在指定資料夾中找不到 .{source_extension} 檔案")
            return
        
        # 按順序重命名檔案
        success_count = 0
        for i, filename in enumerate(files, start=start_num):
            # 建立新的檔案名稱
            if prefix.strip():
                new_name = f"{prefix}_{i}.{target_extension}"
            else:
                new_name = f"{i}.{target_extension}"

            # 獲取完整的檔案路徑（支援中文）
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            try:
                # 重命名檔案（支援中文路徑）
                os.rename(old_path, new_path)
                success_count += 1
                result_text.insert(tk.END, f"重新命名: '{filename}' → '{new_name}'\n")
                result_text.see(tk.END)
                root.update()
            except Exception as e:
                result_text.insert(tk.END, f"失敗: '{filename}' - {str(e)}\n")
                result_text.see(tk.END)
                root.update()
        
        # 顯示完成訊息
        result_text.insert(tk.END, f"\n完成！成功重新命名 {success_count}/{len(files)} 個檔案\n")
        result_text.insert(tk.END, "=" * 60 + "\n")
        result_text.see(tk.END)
        messagebox.showinfo("完成", f"檔案重新命名完成！\n成功: {success_count} 個\n總共: {len(files)} 個")
        
    except ValueError:
        messagebox.showerror("錯誤", "請輸入有效的起始編號")
    except Exception as e:
        messagebox.showerror("錯誤", f"發生錯誤: {str(e)}")

def browse_folder():
    """選擇資料夾"""
    folder_path = filedialog.askdirectory()
    if folder_path:
        path_entry.delete(0, tk.END)
        path_entry.insert(0, folder_path)

def clear_results():
    """清空結果顯示區域"""
    result_text.delete(1.0, tk.END)

def update_status(event=None):
    """更新狀態欄"""
    status_label.config(text="就緒")

# 創建主窗口
root = tk.Tk()
root.title("檔案重新命名工具 - 大字體版")
root.geometry("900x800")

# 設定大字體
LARGE_FONT = ("Microsoft JhengHei", 14)
ENTRY_FONT = ("Microsoft JhengHei", 13)
BUTTON_FONT = ("Microsoft JhengHei", 12)
RESULT_FONT = ("Microsoft JhengHei", 11)
COMBO_FONT = ("Microsoft JhengHei", 12)

# 創建主框架
main_frame = ttk.Frame(root, padding="25")
main_frame.pack(fill=tk.BOTH, expand=True)

# 輸入區域 - 使用網格對齊
# 第0行：資料夾路徑
path_label = ttk.Label(main_frame, text="資料夾路徑:", font=LARGE_FONT)
path_label.grid(row=0, column=0, sticky=tk.W, pady=15, padx=(0, 15))

path_frame = ttk.Frame(main_frame)
path_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=15)

path_entry = tk.Entry(path_frame, font=ENTRY_FONT, width=35, bd=2, relief=tk.SOLID)
path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

browse_btn = tk.Button(path_frame, text="瀏覽", command=browse_folder, 
                      font=BUTTON_FONT, bg="#2196F3", fg="white",
                      padx=15, pady=5, bd=2, relief=tk.RAISED)
browse_btn.pack(side=tk.RIGHT, padx=(10, 0))

# 第1行：起始編號
start_label = ttk.Label(main_frame, text="起始編號:", font=LARGE_FONT)
start_label.grid(row=1, column=0, sticky=tk.W, pady=15, padx=(0, 15))

start_entry = tk.Entry(main_frame, font=ENTRY_FONT, width=20, bd=2, relief=tk.SOLID)
start_entry.insert(0, "10001")
start_entry.grid(row=1, column=1, sticky=tk.W, pady=15) 

# 第2行：檔案前綴
prefix_label = ttk.Label(main_frame, text="檔案前綴:", font=LARGE_FONT)
prefix_label.grid(row=2, column=0, sticky=tk.W, pady=15, padx=(0, 15))

prefix_entry = tk.Entry(main_frame, font=ENTRY_FONT, width=25, bd=2, relief=tk.SOLID)
prefix_entry.insert(0, "image")
prefix_entry.grid(row=2, column=1, sticky=tk.W, pady=15)

# 第3行：原始檔案類型
source_label = ttk.Label(main_frame, text="原始檔案類型:", font=LARGE_FONT)
source_label.grid(row=3, column=0, sticky=tk.W, pady=15, padx=(0, 15))

source_combobox = ttk.Combobox(main_frame, font=COMBO_FONT, width=12, state="readonly")
source_combobox['values'] = ('txt', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'zip', 'rar', 'mp4', 'mov')
source_combobox.set('png')
source_combobox.grid(row=3, column=1, sticky=tk.W, pady=15)

# 第4行：目標檔案類型
target_label = ttk.Label(main_frame, text="目標檔案類型:", font=LARGE_FONT)
target_label.grid(row=4, column=0, sticky=tk.W, pady=15, padx=(0, 15))

target_combobox = ttk.Combobox(main_frame, font=COMBO_FONT, width=12, state="readonly")
target_combobox['values'] = ('txt', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'zip', 'rar', 'mp4', 'mov')
target_combobox.set('jpg')
target_combobox.grid(row=4, column=1, sticky=tk.W, pady=15)

# 按鈕框架
button_frame = ttk.Frame(main_frame)
button_frame.grid(row=5, column=0, columnspan=2, pady=25)

rename_btn = tk.Button(button_frame, text="開始重新命名", command=rename_files,
                      font=BUTTON_FONT, bg="#4CAF50", fg="white",
                      padx=20, pady=8, bd=3, relief=tk.RAISED)
rename_btn.pack(side=tk.LEFT, padx=(0, 20))

clear_btn = tk.Button(button_frame, text="清空結果", command=clear_results,
                     font=BUTTON_FONT, bg="#FF9800", fg="white",
                     padx=20, pady=8, bd=3, relief=tk.RAISED)
clear_btn.pack(side=tk.LEFT)

# 分隔線
separator = ttk.Separator(main_frame, orient='horizontal')
separator.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=20)

# 結果顯示區域
result_label = ttk.Label(main_frame, text="執行結果:", font=LARGE_FONT)
result_label.grid(row=7, column=0, sticky=tk.W, pady=(10, 5))

result_frame = ttk.Frame(main_frame)
result_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))

result_text = tk.Text(result_frame, height=18, width=85, font=RESULT_FONT, wrap=tk.WORD,
                     bd=3, relief=tk.SOLID, padx=10, pady=10)
scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=result_text.yview)
result_text.configure(yscrollcommand=scrollbar.set)

result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# 狀態欄
status_frame = ttk.Frame(main_frame)
status_frame.grid(row=9, column=0, columnspan=2, sticky=(tk.W, tk.E))

status_label = ttk.Label(status_frame, text="就緒", font=LARGE_FONT)
status_label.pack(side=tk.LEFT)

# 配置網格權重
main_frame.columnconfigure(1, weight=1)
main_frame.rowconfigure(8, weight=1)

# 綁定事件
for entry in [path_entry, start_entry, prefix_entry]:
    entry.bind('<FocusOut>', update_status)

source_combobox.bind('<<ComboboxSelected>>', update_status)
target_combobox.bind('<<ComboboxSelected>>', update_status)

# 啟動程式
root.mainloop()




















"""
import os
def rename_jpg_files(folder_path, prefix="image"):
    # 獲取資料夾中的所有檔案
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    # 按順序重命名檔案
    for i, filename in enumerate(files, start=10001):
        # 建立新的檔案名稱
        new_name = f"{i}.png"

        # 獲取完整的檔案路徑
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)

        # 重命名檔案
        os.rename(old_path, new_path)
        print(f"Renamed '{filename}' to '{new_name}'")


# 設定資料夾路徑
folder_path = r"C:\arduino\50"
rename_jpg_files(folder_path)

"""