import json

# Read flows.json
with open(r'c:\AI_Project\paper\flows.json', 'r', encoding='utf-8') as f:
    flows = json.load(f)

# Find and update the timeline template
for node in flows:
    if node.get('id') == '24a9bba3ff43269b':
        fmt = node.get('format', '')
        
        # Replace ng-class conditions
        old = ("'scratch': item.behavior === '搔抓動作',\n"
               "                 'groom': item.behavior === '理毛清潔',\n"
               "                 'stand': item.behavior === '站立走動',\n"
               "                 'sit': item.behavior === '坐下觀察',\n"
               "                 'lay': item.behavior === '躺下休息',\n"
               "                 'back': item.behavior === '背對鏡頭'")
        
        new = ("'scratch': item.behavior === 'scratch',\n"
               "                 'lick': item.behavior === 'lick',\n"
               "                 'walk': item.behavior === 'walk',\n"
               "                 'shake': item.behavior === 'shake'")
        
        if old in fmt:
            fmt = fmt.replace(old, new)
            node['format'] = fmt
            print('✅ Updated timeline template')
        else:
            print('❌ Could not find old pattern in timeline')

# Write back
with open(r'c:\AI_Project\paper\flows.json', 'w', encoding='utf-8') as f:
    json.dump(flows, f, ensure_ascii=False, indent=4)

print('💾 flows.json updated successfully')
