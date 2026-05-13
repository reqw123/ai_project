#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修改 flows.json - 調試版本，寫入日誌。。。。。。。。。。。
"""

import json
import os
import sys

log_file = 'flows_update.log'

def log(msg):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
    print(msg)

# 清空舊日誌
if os.path.exists(log_file):
    os.remove(log_file)

log(f"Python version: {sys.version}")
log(f"Current directory: {os.getcwd()}")
log(f"flows.json exists: {os.path.exists('flows.json')}")

try:
    # 讀取 flows.json
    log("Opening flows.json...")
    with open('flows.json', 'r', encoding='utf-8') as f:
        flows = json.load(f)
    
    log(f"✓ Successfully loaded {len(flows)} nodes")
    
    timeline_found = False
    stats_found = False
    alerts_found = False
    
    for i, node in enumerate(flows):
        node_id = node.get('id', '')
        
        # 時間軸節點
        if node_id == '24a9bba3ff43269b':
            timeline_found = True
            log(f"\n[1] 找到時間軸節點 (index {i})")
            node['_modified'] = 'timeline'
        
        # 統計節點
        if node_id == '35fe802639dcdd84':
            stats_found = True
            log(f"[2] 找到統計節點 (index {i})")
            node['_modified'] = 'stats'
        
        # 警示節點
        if node_id == 'b1c28e8141987478':
            alerts_found = True
            log(f"[3] 找到警示節點 (index {i})")
            node['_modified'] = 'alerts'
    
    log(f"\n節點查找結果：")
    log(f"  時間軸: {'✓ 找到' if timeline_found else '✗ 未找到'}")
    log(f"  統計: {'✓ 找到' if stats_found else '✗ 未找到'}")
    log(f"  警示: {'✓ 找到' if alerts_found else '✗ 未找到'}")
    
    # 寫回
    log("\n開始寫回 flows.json...")
    with open('flows.json', 'w', encoding='utf-8') as f:
        json.dump(flows, f, ensure_ascii=False, indent=4)
    
    log("✅ 成功寫入 flows.json")
    
except Exception as e:
    log(f"❌ 錯誤: {e}")
    import traceback
    log(traceback.format_exc())

log("\n完成")
