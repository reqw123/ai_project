import os

# === 指定要搜尋的資料夾路徑 ===
folder_path = r"C:\tinycnn\120\labels"  # ← 改成你的資料夾路徑
output_file = os.path.join(folder_path, "all_texts_output.txt")

# === 開啟輸出檔案（覆蓋舊內容） ===
with open(output_file, "w", encoding="utf-8") as out:
    # 遍歷所有子資料夾與檔案
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".txt") and filename != os.path.basename(output_file):
                file_path = os.path.join(root, filename)

                # 分隔線
                header = f"\n===== {file_path} =====\n"
                print(header)
                out.write(header)

                # 嘗試讀取內容
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(file_path, "r", encoding="ansi", errors="ignore") as f:
                        content = f.read()

                # 輸出到終端與檔案
                print(content)
                out.write(content + "\n")

print(f"\n✅ 所有文字已輸出到: {output_file}")
