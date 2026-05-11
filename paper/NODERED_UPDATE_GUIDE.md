# Node-RED 配置更新指南 - 四種行為映射 (walk/lick/scratch/shake)

## 概述
Python 端（`behavior_tracker.py`）已定義四種行為：
- **walk** (ID 0): 走動
- **lick** (ID 1): 舔舐/理毛
- **scratch** (ID 2): 搔抓
- **shake** (ID 3): 甩頭

Node-RED flows.json 需要更新以匹配此映射。

## Python 端發送的數據結構
```json
{
  "today_stats": {
    "walk": <次數>,
    "walk_time": <秒數>,
    "scratch": <次數>,
    "scratch_time": <秒數>,
    "lick": <次數>,
    "lick_time": <秒數>,
    "shake": <次數>,
    "shake_time": <秒數>,
    "active_time": <總活動時間>,
    "rest_time": <休息時間>,
    "normal": <walk 的別名>,
    "normal_time": <walk_time 的別名>,
    "groom": <lick 的別名>,
    "groom_time": <lick_time 的別名>
  },
  "alerts": [<警示列表>]
}
```

## 需要更新的 flows.json 節點

### 1. 行為時間軸 (ID: 24a9bba3ff43269b)
**修改項目**：
- 舊 ng-class 條件（6種行為）改為新的（4種行為）
- 舊：`'scratch'/'groom'/'stand'/'sit'/'lay'/'back'`
- 新：`'scratch'/'lick'/'walk'/'shake'`

**替換內容**：
```html
<!-- 舊 -->
ng-class="{
    'scratch': item.behavior === '搔抓動作',
    'groom': item.behavior === '理毛清潔',
    'stand': item.behavior === '站立走動',
    'sit': item.behavior === '坐下觀察',
    'lay': item.behavior === '躺下休息',
    'back': item.behavior === '背對鏡頭'
}"

<!-- 新 -->
ng-class="{
    'scratch': item.behavior === 'scratch',
    'lick': item.behavior === 'lick',
    'walk': item.behavior === 'walk',
    'shake': item.behavior === 'shake'
}"
```

### 2. 詳細統計卡片 (ID: 35fe802639dcdd84)
**修改項目**：
- 保留 scratch 卡片，標籤改為 "搔抓"
- 將 groom 卡片改為 lick：`'👅 舔舐'`，數據源改為 `lick`
- 新增 walk 卡片：`'🚶 走動'`
- 新增 shake 卡片：`'🔄 甩頭'`
- 更新 getTotalBehaviors() 函式

```javascript
scope.getTotalBehaviors = function() {
    const stats = scope.msg.payload.today_stats;
    return (stats.walk || 0) + (stats.scratch || 0) + (stats.lick || 0) + (stats.shake || 0);
};
```

### 3. 健康警示 (ID: b1c28e8141987478)
**修改項目**：
- 更新統計顯示區段，改為顯示四種行為
- 舊：搔抓 + 理毛
- 新：走動 + 搔抓 + 舔舐 + 甩頭

```html
<!-- 新統計行 -->
<div class="stats-item">
    <span>🚶 走動</span>
    <span>{{msg.payload.today_stats.walk}} 次</span>
</div>
<div class="stats-item">
    <span>🐾 搔抓</span>
    <span>{{msg.payload.today_stats.scratch}} 次</span>
</div>
<div class="stats-item">
    <span>👅 舔舐</span>
    <span>{{msg.payload.today_stats.lick}} 次</span>
</div>
<div class="stats-item">
    <span>🔄 甩頭</span>
    <span>{{msg.payload.today_stats.shake}} 次</span>
</div>
```

## CSS 樣式更新

### 移除的樣式
- `.timeline-item.stand`
- `.timeline-item.sit`
- `.timeline-item.lay`
- `.timeline-item.back`
- 以及它們對應的 `.timeline-dot` 和 `.timeline-time` 樣式

### 新增/保留的樣式
- `.timeline-item.scratch` ✓ (保留，紅色#ff6b6b)
- `.timeline-item.lick` ✓ (改名自groom，橙色#ffa726)
- `.timeline-item.walk` ✓ (新增，綠色#4caf50)
- `.timeline-item.shake` ✓ (新增，藍色#2196f3)

## 自動化更新方式

### 使用 Python 腳本
```python
import json

with open('flows.json', 'r', encoding='utf-8') as f:
    flows = json.load(f)

for node in flows:
    if node.get('id') == '24a9bba3ff43269b':  # Timeline
        fmt = node.get('format', '')
        old = "'scratch': item.behavior === '搔抓動作',\n..." 
        new = "'scratch': item.behavior === 'scratch',\n..."
        node['format'] = fmt.replace(old, new)

with open('flows.json', 'w', encoding='utf-8') as f:
    json.dump(flows, f, ensure_ascii=False, indent=4)
```

## 驗證步驟

1. ✅ 在 Node-RED 中導入更新後的 flows.json
2. ✅ 確認行為時間軸顯示 walk/lick/scratch/shake
3. ✅ 確認統計卡片顯示四種行為的計數和時間
4. ✅ 啟動 Python 側的行為監測，確認數據正確流入 Node-RED
5. ✅ 檢查警示系統根據搔抓/舔舐/甩頭頻率提示

## 相容性說明

Python 端為保持相容性，提供了以下別名：
- `stats["normal"]` = `stats["walk"]`
- `stats["groom"]` = `stats["lick"]`

舊的 Node-RED 面板可臨時使用這些別名過渡，但建議更新為新的字段名。
