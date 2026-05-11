#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接修改 flows.json，將行為映射從舊的 6 種改為新的 4 種
舊：搔抓動作/理毛清潔/站立走動/坐下觀察/躺下休息/背對鏡頭
新：scratch/lick/walk/shake
"""

import json
import re

# 讀取 flows.json
with open('flows.json', 'r', encoding='utf-8') as f:
    flows = json.load(f)

print("開始修改 flows.json...")

for i, node in enumerate(flows):
    node_id = node.get('id', '')
    
    # 修改 1: 行為時間軸 (24a9bba3ff43269b)
    if node_id == '24a9bba3ff43269b':
        print(f"\n✓ 找到節點 1: 行為時間軸 (index {i})")
        fmt = node.get('format', '')
        
        # 替換 ng-class 的行為映射
        old_pattern = r"'scratch':\s*item\.behavior\s*===\s*'搔抓動作',\s*'groom':\s*item\.behavior\s*===\s*'理毛清潔',\s*'stand':\s*item\.behavior\s*===\s*'站立走動',\s*'sit':\s*item\.behavior\s*===\s*'坐下觀察',\s*'lay':\s*item\.behavior\s*===\s*'躺下休息',\s*'back':\s*item\.behavior\s*===\s*'背對鏡頭'"
        
        new_pattern = "'scratch': item.behavior === 'scratch', 'lick': item.behavior === 'lick', 'walk': item.behavior === 'walk', 'shake': item.behavior === 'shake'"
        
        fmt_modified = re.sub(old_pattern, new_pattern, fmt)
        
        if fmt_modified != fmt:
            print("  已更新 ng-class 行為映射")
            node['format'] = fmt_modified
        else:
            print("  警告：未找到舊的 ng-class 模式")
    
    # 修改 2: 詳細統計 (35fe802639dcdd84)
    if node_id == '35fe802639dcdd84':
        print(f"\n✓ 找到節點 2: 詳細統計 (index {i})")
        fmt = node.get('format', '')
        
        # 替換 groom 為 lick
        fmt_modified = fmt.replace("'groom':", "'lick':")
        fmt_modified = fmt_modified.replace('理毛清潔', '舔舐')
        fmt_modified = fmt_modified.replace('💇 理毛', '👅 舔舐')
        fmt_modified = fmt_modified.replace('groom_time', 'lick_time')
        
        # 替換 getTotalBehaviors 函式
        old_func = r"scope\.getTotalBehaviors\s*=\s*function\(\)\s*\{\s*const\s+stats\s*=\s*scope\.msg\.payload\.today_stats;\s*return\s+\(stats\.walk\s*\|\|\s*0\)\s*\+\s*\(stats\.scratch\s*\|\|\s*0\)\s*\+\s*\(stats\.groom\s*\|\|\s*0\);\s*\};"
        
        new_func = "scope.getTotalBehaviors = function() { const stats = scope.msg.payload.today_stats; return (stats.walk || 0) + (stats.scratch || 0) + (stats.lick || 0) + (stats.shake || 0); };"
        
        fmt_modified = re.sub(old_func, new_func, fmt_modified, flags=re.DOTALL)
        
        # 檢查是否有 walk 和 shake 的統計卡片
        if 'walk' not in fmt_modified:
            print("  警告：未找到 walk 統計卡片，需要手動檢查")
        if 'shake' not in fmt_modified:
            print("  警告：未找到 shake 統計卡片，需要手動檢查")
        
        if fmt_modified != fmt:
            print("  已更新統計卡片和 getTotalBehaviors 函式")
            node['format'] = fmt_modified
    
    # 修改 3: 健康警示 (b1c28e8141987478)
    if node_id == 'b1c28e8141987478':
        print(f"\n✓ 找到節點 3: 健康警示 (index {i})")
        fmt = node.get('format', '')
        
        # 更新統計顯示，將 groom 改為 lick
        fmt_modified = fmt.replace("stats.groom", "stats.lick")
        fmt_modified = fmt_modified.replace("stats.groom_time", "stats.lick_time")
        fmt_modified = fmt_modified.replace('理毛清潔', '舔舐')
        fmt_modified = fmt_modified.replace('groom', 'lick')
        
        if fmt_modified != fmt:
            print("  已更新統計顯示")
            node['format'] = fmt_modified

# 寫回 flows.json
with open('flows.json', 'w', encoding='utf-8') as f:
    json.dump(flows, f, ensure_ascii=False, indent=4)

print("\n✅ flows.json 已成功修改！")
print("修改摘要：")
print("  1. 行為時間軸: scratch/lick/walk/shake 映射")
print("  2. 詳細統計: groom → lick, 更新 getTotalBehaviors()")
print("  3. 健康警示: 更新統計顯示")
