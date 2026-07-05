
import os
import tkinter as tk
from tkinter import ttk, messagebox

# 設定超大字體
XXL_FONT = ("Microsoft JhengHei", 22)
XXL_BUTTON_FONT = ("Microsoft JhengHei", 20)
XXL_RESULT_FONT = ("Microsoft JhengHei", 18)

def create_files():
    try:
        # 獲取使用者輸入
        start_num = int(start_entry.get())
        file_count = int(count_entry.get())
        content = content_entry.get()
        directory = directory_entry.get()
        
        # 驗證輸入
        if file_count <= 0:
            messagebox.showerror("錯誤", "文件數量必須大於0")
            return
            
        if not directory.strip():
            messagebox.showerror("錯誤", "請輸入目錄名稱")
            return
        
        # 創建目錄
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # 創建進度條
        progress_bar['maximum'] = file_count
        progress_bar['value'] = 0
        root.update()
        
        # 產生文件
        success_count = 0
        for i in range(start_num, start_num + file_count):
            filename = f"{directory}/{i}.txt"
            try:
                with open(filename, "w", encoding='utf-8') as file:
                    file.write(content)
                success_count += 1
                result_text.insert(tk.END, f"已創建: {filename}\n")
            except Exception as e:
                result_text.insert(tk.END, f"創建失敗 {filename}: {str(e)}\n")
            
            # 更新進度條
            progress_bar['value'] = i - start_num + 1
            root.update()
        
        # 顯示完成訊息
        result_text.insert(tk.END, f"\n成功創建 {success_count}/{file_count} 個文件！\n")
        result_text.insert(tk.END, "-" * 50 + "\n")
        result_text.see(tk.END)
        messagebox.showinfo("完成", f"文件創建完成！\n成功: {success_count} 個\n失敗: {file_count - success_count} 個")
        
    except ValueError:
        messagebox.showerror("錯誤", "請輸入有效的數字")
    except Exception as e:
        messagebox.showerror("錯誤", f"發生錯誤: {str(e)}")

def clear_results():
    result_text.delete(1.0, tk.END)

# 創建主窗口
root = tk.Tk()
root.title("批量文件生成器 - 超大字體版")
root.geometry("1200x1100")

# 創建主框架
main_frame = ttk.Frame(root, padding="40")
main_frame.pack(fill=tk.BOTH, expand=True)

# 輸入欄位 - 使用 tk.Entry 確保字體大小正確
ttk.Label(main_frame, text="起始編號:", font=XXL_FONT).grid(row=0, column=0, sticky=tk.W, pady=25)
start_entry = tk.Entry(main_frame, font=XXL_FONT, width=15, bd=3, relief=tk.SOLID)
start_entry.insert(0, "10001")
start_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=25, padx=(20,0))

ttk.Label(main_frame, text="文件數量:", font=XXL_FONT).grid(row=1, column=0, sticky=tk.W, pady=25)
count_entry = tk.Entry(main_frame, font=XXL_FONT, width=15, bd=3, relief=tk.SOLID)
count_entry.insert(0, "20")
count_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=25, padx=(20,0))

ttk.Label(main_frame, text="文件內容:", font=XXL_FONT).grid(row=2, column=0, sticky=tk.W, pady=25)
content_entry = tk.Entry(main_frame, font=XXL_FONT, width=25, bd=3, relief=tk.SOLID)
content_entry.insert(0, "10")
content_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=25, padx=(20,0))

ttk.Label(main_frame, text="目錄名稱:", font=XXL_FONT).grid(row=3, column=0, sticky=tk.W, pady=25)
directory_entry = tk.Entry(main_frame, font=XXL_FONT, width=20, bd=3, relief=tk.SOLID)
directory_entry.insert(0, "txt_files")
directory_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=25, padx=(20,0))

# 按鈕框架 - 使用 tk.Button 確保字體大小正確
button_frame = ttk.Frame(main_frame)
button_frame.grid(row=4, column=0, columnspan=2, pady=40)

create_btn = tk.Button(button_frame, text="創建文件", command=create_files, 
                      font=XXL_BUTTON_FONT, bg="#4CAF50", fg="white", 
                      padx=20, pady=10, bd=3, relief=tk.RAISED)
create_btn.pack(side=tk.LEFT, padx=(0, 30))

clear_btn = tk.Button(button_frame, text="清空結果", command=clear_results,
                     font=XXL_BUTTON_FONT, bg="#f44336", fg="white",
                     padx=20, pady=10, bd=3, relief=tk.RAISED)
clear_btn.pack(side=tk.LEFT)

# 進度條
ttk.Label(main_frame, text="進度:", font=XXL_FONT).grid(row=5, column=0, sticky=tk.W, pady=(30,15))
progress_bar = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, mode='determinate', length=700)
progress_bar.grid(row=5, column=1, sticky=(tk.W, tk.E), pady=(30,15), padx=(20,0))

# 結果顯示區域
ttk.Label(main_frame, text="創建結果:", font=XXL_FONT).grid(row=6, column=0, sticky=tk.W, pady=(30,15))
result_frame = ttk.Frame(main_frame)
result_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(20,0))

# 文字顯示區域
result_text = tk.Text(result_frame, height=20, width=80, font=XXL_RESULT_FONT, wrap=tk.WORD,
                     bd=3, relief=tk.SOLID, padx=10, pady=10)
scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=result_text.yview)
result_text.configure(yscrollcommand=scrollbar.set)

result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# 狀態欄
status_frame = ttk.Frame(main_frame)
status_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(25,0))

status_label = ttk.Label(status_frame, text="就緒", font=XXL_FONT)
status_label.pack(side=tk.LEFT)

# 配置網格權重
main_frame.columnconfigure(1, weight=1)
main_frame.rowconfigure(7, weight=1)

# 綁定事件更新狀態
def update_status(event=None):
    status_label.config(text="輸入完成")

# 綁定輸入框的焦點離開事件
for entry in [start_entry, count_entry, content_entry, directory_entry]:
    entry.bind('<FocusOut>', update_status)

# 啟動程式
root.mainloop()

























"""
import os

# 設定目錄名稱
directory = "txt_files"
if not os.path.exists(directory):
    os.makedirs(directory)

# 產生100個.txt文件
for i in range(10001,10021):
    filename = f"{directory}/{i}.txt"  # 設定文件名稱
    with open(filename, "w") as file:
        file.write(f"10")  # 文件內容
    print(f"Created: {filename}")

print("100 files have been created successfully.")
"""
