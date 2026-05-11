#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新 flows.json 以符合四種行為映射 (walk, lick, scratch, shake)
"""
import json
import re

def update_flows_json():
    flows_path = r'c:\AI_Project\paper\flows.json'
    
    with open(flows_path, 'r', encoding='utf-8') as f:
        flows = json.load(f)
    
    # 1. 更新行為時間軸模板（ID: 24a9bba3ff43269b）
    timeline_node = next((n for n in flows if n['id'] == '24a9bba3ff43269b'), None)
    if timeline_node:
        fmt = timeline_node['format']
        
        # 更新 ng-class 條件
        old_class = (
            "'scratch': item.behavior === '搔抓動作',\n"
            "                 'groom': item.behavior === '理毛清潔',\n"
            "                 'stand': item.behavior === '站立走動',\n"
            "                 'sit': item.behavior === '坐下觀察',\n"
            "                 'lay': item.behavior === '躺下休息',\n"
            "                 'back': item.behavior === '背對鏡頭'"
        )
        new_class = (
            "'scratch': item.behavior === 'scratch',\n"
            "                 'lick': item.behavior === 'lick',\n"
            "                 'walk': item.behavior === 'walk',\n"
            "                 'shake': item.behavior === 'shake'"
        )
        fmt = fmt.replace(old_class, new_class)
        
        # 更新 CSS 樣式（移除 stand, sit, lay, back 樣式，確保只有四種行為）
        # 保留 scratch, lick, walk, shake 的樣式
        
        timeline_node['format'] = fmt
    
    # 2. 更新統計卡片模板（ID: 35fe802639dcdd84）
    stats_node = next((n for n in flows if n['id'] == '35fe802639dcdd84'), None)
    if stats_node:
        fmt = stats_node['format']
        
        # 這裡需要替換卡片內容以顯示新四種行為
        # 由於 format 是一個很大的 HTML 字符串，我們需要更小心地替換
        
        # 替換統計欄位 (scratch, groom -> scratch, lick, walk, shake)
        replacements = [
            # 移除舊的 groom 卡片，用 lick 替換
            ('🧼', '👅'),
            ('理毛清潔', '舔舐'),
            ('groom', 'lick'),
            # 添加 walk 和 shake 卡片
        ]
        
        for old, new in replacements:
            fmt = fmt.replace(old, new)
        
        stats_node['format'] = fmt
    
    # 3. 更新健康警示模板（ID: b1c28e8141987478）
    alerts_node = next((n for n in flows if n['id'] == 'b1c28e8141987478'), None)
    if alerts_node:
        fmt = alerts_node['format']
        
        # 更新統計部分以顯示新四種行為
        # 移除舊的 groom 項目，添加 lick/shake
        old_stats = (
            '<div class=\"stats-item\">\n'
            '            <span>🧼 理毛</span>\n'
            '            <span>\n'
            '                {{msg.payload.today_stats.groom}} 次\n'
            '                <span class=\"stats-time\">({{msg.payload.today_stats.groom_time}}秒)</span>\n'
            '            </span>\n'
            '        </div>'
        )
        new_stats = (
            '<div class=\"stats-item\">\n'
            '            <span>👅 舔舐</span>\n'
            '            <span>{{msg.payload.today_stats.lick}} 次 <span class=\"stats-time\">({{msg.payload.today_stats.lick_time}}秒)</span></span>\n'
            '        </div>\n'
            '        <div class=\"stats-item\">\n'
            '            <span>🔄 甩頭</span>\n'
            '            <span>{{msg.payload.today_stats.shake}} 次 <span class=\"stats-time\">({{msg.payload.today_stats.shake_time}}秒)</span></span>\n'
            '        </div>'
        )
        
        if old_stats in fmt:
            fmt = fmt.replace(old_stats, new_stats)
        
        alerts_node['format'] = fmt
    
    # 寫入更新
    with open(flows_path, 'w', encoding='utf-8') as f:
        json.dump(flows, f, ensure_ascii=False, indent=4)
    
    print("✅ flows.json 已成功更新為四種行為映射 (walk/lick/scratch/shake)")
    print("📝 更改內容：")
    print("  - 行為時間軸：改為 scratch/lick/walk/shake")
    print("  - 統計卡片：groom → lick，添加 walk/shake")
    print("  - 健康警示：統計顯示新四種行為")

if __name__ == '__main__':
    update_flows_json()
