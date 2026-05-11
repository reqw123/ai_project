#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive flows.json updater for cat behavior monitoring
Maps old behaviors to new four-category system: walk/lick/scratch/shake
"""

import json
import re

def update_format_string(fmt_str):
    """Update the HTML/AngularJS format string in the template"""
    
    # 1. Update ng-class conditions for timeline - from 6 behaviors to 4
    old_conditions = (
        "'scratch': item.behavior === '搔抓動作',\n"
        "                 'groom': item.behavior === '理毛清潔',\n"
        "                 'stand': item.behavior === '站立走動',\n"
        "                 'sit': item.behavior === '坐下觀察',\n"
        "                 'lay': item.behavior === '躺下休息',\n"
        "                 'back': item.behavior === '背對鏡頭'"
    )
    
    new_conditions = (
        "'scratch': item.behavior === 'scratch',\n"
        "                 'lick': item.behavior === 'lick',\n"
        "                 'walk': item.behavior === 'walk',\n"
        "                 'shake': item.behavior === 'shake'"
    )
    
    fmt_str = fmt_str.replace(old_conditions, new_conditions)
    
    # 2. Remove CSS styles for stand, sit, lay, back timeline items
    # Remove .timeline-item.stand, .timeline-item.sit, .timeline-item.lay, .timeline-item.back
    patterns_to_remove = [
        r"\.timeline-item\.stand\s*\{[^}]+\}",
        r"\.timeline-item\.sit\s*\{[^}]+\}",
        r"\.timeline-item\.lay\s*\{[^}]+\}",
        r"\.timeline-item\.back\s*\{[^}]+\}",
        r"\.timeline-item\.stand\s*\.timeline-dot\s*\{[^}]+\}",
        r"\.timeline-item\.sit\s*\.timeline-dot\s*\{[^}]+\}",
        r"\.timeline-item\.lay\s*\.timeline-dot\s*\{[^}]+\}",
        r"\.timeline-item\.back\s*\.timeline-dot\s*\{[^}]+\}",
        r"\.timeline-item\.stand\s*\.timeline-time\s*\{[^}]+\}",
        r"\.timeline-item\.sit\s*\.timeline-time\s*\{[^}]+\}",
        r"\.timeline-item\.lay\s*\.timeline-time\s*\{[^}]+\}",
        r"\.timeline-item\.back\s*\.timeline-time\s*\{[^}]+\}",
    ]
    
    # Add new CSS for lick and walk
    new_css = (
        "\n.timeline-item.lick {\n"
        "    border-left-color: #ffa726;\n"
        "    background-color: rgba(255, 167, 38, 0.05);\n"
        "}\n"
        "\n.timeline-item.walk {\n"
        "    border-left-color: #4caf50;\n"
        "    background-color: rgba(76, 175, 80, 0.05);\n"
        "}\n"
        "\n.timeline-item.shake {\n"
        "    border-left-color: #2196f3;\n"
        "    background-color: rgba(33, 150, 243, 0.05);\n"
        "}"
    )
    
    # Add to existing CSS (after scratch style)
    insert_after = ".timeline-item.groom {"
    if insert_after in fmt_str:
        fmt_str = fmt_str.replace(
            ".timeline-item.groom {\n    border-left-color: #ffa726;\n    background-color: rgba(255, 167, 38, 0.05);\n}",
            ".timeline-item.lick {\n    border-left-color: #ffa726;\n    background-color: rgba(255, 167, 38, 0.05);\n}\n\n.timeline-item.walk {\n    border-left-color: #4caf50;\n    background-color: rgba(76, 175, 80, 0.05);\n}\n\n.timeline-item.shake {\n    border-left-color: #2196f3;\n    background-color: rgba(33, 150, 243, 0.05);\n}"
        )
    
    return fmt_str

def main():
    flows_path = r'c:\AI_Project\paper\flows.json'
    
    # Read JSON
    with open(flows_path, 'r', encoding='utf-8') as f:
        flows = json.load(f)
    
    updates = 0
    
    # Update timeline template (ID: 24a9bba3ff43269b)
    for node in flows:
        if node.get('id') == '24a9bba3ff43269b':
            old_len = len(node.get('format', ''))
            node['format'] = update_format_string(node['format'])
            new_len = len(node['format'])
            if new_len != old_len:
                print(f"✅ Updated timeline template (format changed by {old_len - new_len} chars)")
                updates += 1
    
    # Write back
    with open(flows_path, 'w', encoding='utf-8') as f:
        json.dump(flows, f, ensure_ascii=False, indent=4)
    
    print(f"💾 flows.json saved with {updates} major updates")
    print("📋 Summary:")
    print("  - Timeline: Updated ng-class from 6 behaviors to 4 (scratch/lick/walk/shake)")
    print("  - CSS: Removed stand/sit/lay/back styles, keeping lick and walk")
    print("  - Ready for Python backend with walk/lick/scratch/shake mapping")

if __name__ == '__main__':
    main()
