# -*- coding: utf-8 -*-
"""移除 v2 flow 的 CSV 寫入（行為統計累積器第4輸出固定回傳 null）"""
import json, re

V2_PATH    = r'c:\ai_project\paper\cat_health_v2_flow.json'
ACCUM_ID   = '8feb3527c9bca065'

with open(V2_PATH, 'r', encoding='utf-8') as f:
    nodes = json.load(f)

for n in nodes:
    if n.get('id') != ACCUM_ID:
        continue
    func = n['func']

    # 移除 doCsv / v2_last_csv / csvMsg 整個區塊，第4輸出固定 null
    # 找到 "// 每 5 分鐘寫一次 CSV" 開始到 "return [" 之前，整段替換
    OLD_CSV_BLOCK = re.search(
        r'\n// 每 5 分鐘寫一次 CSV\n.*?(?=\nreturn \[)',
        func, re.DOTALL
    )
    if OLD_CSV_BLOCK:
        func = func[:OLD_CSV_BLOCK.start()] + '\n' + func[OLD_CSV_BLOCK.end():]
        print('Removed: doCsv / csvMsg block')
    else:
        print('WARNING: csv block pattern not matched, trying line-by-line')
        # fallback：逐行移除相關行
        lines = func.split('\n')
        keep = []
        skip_keywords = ['v2_last_csv', 'doCsv', 'csvMsg', '每 5 分鐘']
        for line in lines:
            if any(kw in line for kw in skip_keywords):
                continue
            keep.append(line)
        func = '\n'.join(keep)

    # 確保 return 陣列第4項是 null
    func = re.sub(
        r'return \[msg, msg, msg, csvMsg\];',
        'return [msg, msg, msg, null];',
        func
    )
    func = re.sub(
        r'return \[msg, msg, msg, \w+\s*\?\s*\{[^}]+\}\s*:\s*null\];',
        'return [msg, msg, msg, null];',
        func
    )

    n['func'] = func
    print('Fixed: v2 行為統計累積器 CSV output → null')

with open(V2_PATH, 'w', encoding='utf-8') as f:
    json.dump(nodes, f, ensure_ascii=False, indent=4)

print('Done.')
