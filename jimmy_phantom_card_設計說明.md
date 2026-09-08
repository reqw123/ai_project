# 吉米 · WIND MIRAGE — SSR 全息殘影卡　設計說明書

> 對應檔案：`jimmy_phantom_card.html`（單一自足檔，約 960 KB，3 張場景圖 + 1 張去背主體全部以 base64 內嵌）
> 最後更新：2026-09-08

---

## 目錄

1. [設計定位與核心氛圍](#1-設計定位與核心氛圍)
2. [色彩系統](#2-色彩系統)
3. [字體排版](#3-字體排版)
4. [版面結構與尺寸](#4-版面結構與尺寸)
5. [層與層 · 景深座標設計（核心）](#5-層與層--景深座標設計核心)
6. [摳圖分離：主體去背](#6-摳圖分離主體去背)
7. [幻影殘影層（時空殘影）](#7-幻影殘影層時空殘影)
8. [全息膜光學特效](#8-全息膜光學特效)
9. [動態金屬流光外框](#9-動態金屬流光外框)
10. [毛玻璃技能面板](#10-毛玻璃技能面板)
11. [互動邏輯（JavaScript）](#11-互動邏輯javascript)
12. [效能、相容性與降級](#12-效能相容性與降級)
13. [客製化指南](#13-客製化指南)
14. [資產清單與來源](#14-資產清單與來源)
15. [與原始 spec 的取捨對照](#15-與原始-spec-的取捨對照)

---

## 1. 設計定位與核心氛圍

**「Dark Fantasy × 現代 TCG（集換式卡牌）UI」**

| 面向 | 手法 |
|---|---|
| 深淵感 | 卡片底為深邃黑 `#09121a` + 徑向漸層；整個頁面背景也是由中心向外壓暗的徑向漸層，四周飄著模糊光塵（`body::before`，22s 緩慢位移）。 |
| 魔法感 | 主色翠綠 `#10b981` + 青色 `#06b6d4`，貫穿到殘影、發光、裝飾線、面板上緣高光。 |
| 史詩感 | 標題/稱號/技能名用襯線字 **Cinzel**；金屬流光外框模擬高反差黃金／黃銅反光。 |
| 卡牌真實感 | 技能區用毛玻璃（Glassmorphism）模擬實體 UI 面板；全息膜用 `color-dodge` 疊在卡面上模擬閃卡鍍膜。 |
| **景深 / 立體感** | 卡片本體 3D 傾斜，內容分成「窗內較深的環境 + 卡面 UI + 浮在最上層的去背主體」三個景深帶，各帶不同的視差係數。 |

互動一句話：**移動滑鼠 / 傾斜裝置 → 卡片朝指標傾斜（±25°），各層依景深產生不同幅度的視差，貓浮在卡面之上獨立移動並投下落地陰影。**

---

## 2. 色彩系統

以 CSS 變數集中管理（`:root`）：

```css
--emerald: #10b981;   /* 魔法主色（翠綠）：技能名、裝飾線、發光 */
--cyan:    #06b6d4;    /* 魔法副色（青）：稱號、風屬性、殘影 */
--abyss:   #09121a;    /* 深淵黑：卡片底、霧的收尾色 */
```

其他固定色：

| 用途 | 色值 |
|---|---|
| 頁面背景徑向漸層 | `#10222e → #0a1520 → #05090e → #03060a` |
| 卡片內層徑向漸層 | `#142833 → var(--abyss) → #05090d` |
| 金框漸層 | `#241a05 / #8a6a1c / #d8b352 / #fff6d8 / #b98d2e / #5c4610 / #9a7a24 / #f4e2a0 / #7c5f18`（115° 線性，`background-size: 280%` 掃動） |
| 毛玻璃面板底 | `rgba(15,23,42,.85)` |
| 面板上緣高光 | `inset 0 1px 0 rgba(255,255,255,.14)` |
| SSR 徽章 | `#ffe9a8 → #d4af37 → #a97d1e`（金） |
| WIND 徽章 | `rgba(6,182,212,.6) → rgba(16,185,129,.42)`（青綠） |
| 全息彩虹停點 | 洋紅 `#ff008c` → 青 `#00e0ff` → 紫 `#783cff` → 金黃 `#ffd600` → 翠 `#00ffa8`（各 `.42` alpha） |

**文字一律加 `text-shadow`**（多半是 `0 1px 2~3px #000` + 一層對應色的外發光），確保在全息／殘影／照片等複雜背景上不糊。

---

## 3. 字體排版

```html
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600;700;900&family=Noto+Sans+TC:wght@400;500;700;900&display=swap" rel="stylesheet">
```

| 元素 | 字體 | 字重 | 字級 | 字距 |
|---|---|---|---|---|
| 稱號 `WIND MIRAGE` | Cinzel | 600 | 10.5px | `.34em` |
| 角色名 `吉米` | Cinzel → Noto Sans TC | 900 | 30px | `.05em` |
| 技能名 `✦ 三千風影` | Cinzel → Noto Sans TC | 700 | 15px | — |
| 技能英名 `Phantom Strike` | Noto Sans TC | 500 | 10.5px | `.12em` |
| 技能描述 | Noto Sans TC | 400 | 11.5px（行高 1.6） | — |
| 徽章 `WIND 風 / SSR` | Cinzel | 700 | 11px | `.16em` / `.26em` |

- `Cinzel` 沒有中文字形，`'Cinzel','Noto Sans TC'` 的 fallback 讓「吉米」自動落到 Noto Sans TC 900。
- `display=swap`：字型載入前先用系統 serif / 微軟正黑體頂著，不擋渲染。沒有網路時整體仍可讀，只是少了史詩襯線味。

---

## 4. 版面結構與尺寸

```
.stage                     透視容器 perspective:1300px；上方 padding 60px 留白給探出的貓頭
└─ .card                   3D 傾斜主體（本身「不加 filter」，才留得住 preserve-3d）
   ├─ .card__frame         金屬流光外框（padding 6px 當邊框厚度；卡片投影掛在這層）
   │  └─ .card__inner      深淵黑內層，border-radius:14px + overflow:hidden（裁掉溢出）
   │     ├─ .scene         「場景窗」，卡片上緣起、高 60%，overflow:hidden
   │     │  ├─ .scene__bg      最深：原場景照大幅模糊/壓暗（環境霧）
   │     │  ├─ .ghost--warn    中景：青藍「警戒」殘影
   │     │  ├─ .ghost--past    中景後：翠綠「沉睡」殘影
   │     │  └─ .scene__mist    窗內霧化 overlay（底部漸層融進深淵黑）
   │     ├─ .holo          全息膜（color-dodge，卡面光學層）
   │     │  ├─ .holo__spectrum  彩虹漸層 + 隨指標的柔和反光
   │     │  └─ .holo__dust      星芒微塵點陣
   │     └─ .panel         毛玻璃技能面板，卡片下緣起、高 46%
   ├─ .subject             ★ 去背的貓，.card 直屬層（不在 .card__inner 內、不被裁切）
   └─ .topbar              WIND / SSR 徽章，.card 直屬層（永遠可讀）
```

**尺寸**

```css
--card-w: clamp(272px, min(86vw, 58vh), 372px);   /* 寬度同時受視窗寬與高限制，避免直的卡片超出螢幕 */
aspect-ratio: 100 / 140;                           /* 標準 TCG 比例 ≈ 5:7 */
--frame: 6px;                                      /* 金框厚度 */
```

**為什麼把 `.subject` 與 `.topbar` 拉出 `.card__inner`？**
`.card__inner` 有 `overflow:hidden`，會裁掉任何溢出的東西，也會強制 `transform-style: flat`（壓平 3D 子層）。要讓貓能「探出卡面、破框、獨立浮起」，就必須把它放到 `.card` 直屬層。同理，`.topbar` 拉出來才能用 `translateZ` 疊到貓之上。

---

## 5. 層與層 · 景深座標設計（核心）

### 5.1 兩套「深度」機制並用

| 機制 | 用在哪 | 為什麼 |
|---|---|---|
| **真 3D（`translateZ` + `preserve-3d`）** | `.subject`、`.topbar`（都是 `.card` 直屬層） | `.card` 一旋轉，離鏡頭近（`translateZ` 大）的層因為旋轉半徑的槓桿更長，位移天生比卡面大 → **真實的視差**，不用另外算。 |
| **假 2D 視差（JS 每幀寫 `transform: translate()`）** | `.scene__bg`、`.ghost--warn`、`.ghost--past` | 這些層在 `.scene`（`overflow:hidden`）裡，被強制壓平、吃不到 `translateZ`。改用 JS 依指標位移給不同係數的 2D 平移，視覺上一樣有遠近。 |

> **關鍵前提**：`.card` 本身**絕對不能有 `filter`**。`filter` 會讓元素強制 `transform-style: flat`，`.subject` 的 `translateZ` 就會失效。所以卡片的投影（drop-shadow）改掛在 `.card__frame` 上，`.card` 只留 `transform` + `transform-style: preserve-3d`。

### 5.2 完整景深堆疊表（近 → 遠）

| # | 層 | 深度座標 | 位移方式 | 傾斜 | 說明 |
|---|---|---|---|---|---|
| 1 | `.topbar`（WIND / SSR） | `translateZ(66px)`，`z-index:21` | 跟卡片一起（無額外視差） | 隨卡片 | 比主體再高一點，稀有度/屬性永遠讀得到 |
| 2 | **`.subject`（去背貓）** | `translateZ(60px)`，`z-index:20` | JS：`--sx = ox·9px`、`--sy = oy·7px − mag·3px`；`--sscale = 1 + mag·0.025` | **只轉卡片的 50%**（`--srx/--sry = −rx/−ry · 0.5`），刻意「自成一個平面」 | 浮在卡面上、壓過外框與面板；`drop-shadow` 把落地陰影拉開＝「與卡片分離」最關鍵線索 |
| 3 | `.holo`（全息膜） | 卡面 `z-index:3`（Z 0） | 只有內部彩虹/微塵隨指標移動，層本身不位移 | 隨卡片 | 鍍膜貼在卡面上，所以在主體之下 |
| 4 | `.panel`（毛玻璃面板） | 卡面 `z-index:2`（Z 0） | 不位移 | 隨卡片 | 卡片表面 UI |
| 5 | `.ghost--warn`（青「警戒」殘影） | 窗內 `z-index:2` | 預設 `translate(11%, −18%)` + JS `bx·38 / by·33`，`scale(1.16)` | 隨卡片（壓平） | 中景殘影 |
| 6 | `.ghost--past`（綠「沉睡」殘影） | 窗內 `z-index:3`（在 warn 之上，用 screen 疊光） | 預設 `translate(−28%, 16%)` + JS `bx·64 / by·56`，`scale(1.26)` | 隨卡片（壓平） | 位移最大＝飄最遠 |
| 7 | `.scene__bg`（環境霧） | 窗內 `z-index:1`（最底） | JS `bx·8 / by·7`，`scale(1.28)` | 隨卡片（壓平） | `blur(8px) brightness(.42)`，退到最深當背景 |

> `ox / oy` = 指標相對卡片中心的正規化座標（−1 ~ 1）；`bx = −ox、by = −oy`（反向）；`mag = √(ox²+oy²)` 夾在 0~1。
> **位移係數越大 = 離鏡頭越遠**（背景霧 8px 最少 → 沉睡殘影 64px 最多）；反向平移製造「往深處退」的視差。

### 5.3 「主體與卡片分離」是怎麼做到的

四個線索疊加，缺一不可：

1. **落地陰影**（最重要）：`.subject` 的 `filter` 裡三層 `drop-shadow`
   ```css
   drop-shadow(0 32px 22px rgba(0,0,0,.66))   /* 主陰影，往下拉 32px */
   drop-shadow(0 12px 12px rgba(0,0,0,.48))   /* 接觸陰影，貼近本體 */
   drop-shadow(0 0 18px rgba(6,182,212,.30))  /* 青色輪廓光，跟繁忙背景拉開 */
   ```
2. **`translateZ(60px)`**：在 `.card` 的 `preserve-3d` 空間裡真的往鏡頭方向浮出來，透視放大 ≈ `1300 / (1300−60) ≈ 1.05×`。
3. **獨立視差 + 少轉一半**：貓比卡片少轉 50%，滑鼠掃過時貓和卡面「錯開」，眼睛就判定它們不在同一個平面。
4. **破框**：貓的前腳會壓過右側金框、身體壓在面板頂緣 → 明確的前後遮擋關係。

---

## 6. 摳圖分離：主體去背

### 6.1 產製流程

| 步驟 | 工具 / 參數 |
|---|---|
| 1. 語意分割 | `ultralytics` `yolo11m-seg.pt`，`imgsz=1024`、`conf=0.25`、`retina_masks=True`，取面積最大的 `cat` 遮罩 |
| 2. 遮罩整形 | `MaxFilter(5)` → `MinFilter(5)` 補小洞；再 `MinFilter(3)` 內縮 1px（去掉舊背景的亮邊），`GaussianBlur(1.6)` 羽化 |
| 3. 上 alpha | 原圖轉 RGBA、`putalpha(遮罩)` |
| 4. 裁切 | 依 alpha 的 `getbbox()` + 8px padding |
| 5. 縮圖 | 最大寬 760px（卡片實際只顯示約 260px，2× 供 retina） |
| 6. 輸出 | PNG（`optimize=True`）→ base64 → 內嵌成 `.subject > img` 的 `src` |

成品：`760 × 543` 透明 PNG，約 472 KB（base64 約 629 KB）。

### 6.2 為什麼要去背

原本三張照片直接疊，只看得到「一隻貓」——因為不透明的頂層本體把下面兩層完全蓋住，而且三張構圖太像、糊成一團。去背之後：

- **本體** = 銳利、真實色彩、浮在最上層的視覺錨點。
- **背景窗** = 模糊環境霧 + 兩層彩色殘影，退到窗內較深的景深，當「時空殘影」的舞台。

### 6.3 在卡片裡的擺放

```css
.subject{
  position:absolute; left:50%; bottom:50%; width:75%;
  z-index:20; pointer-events:none;
  transform:
    translate(-50%, 0)
    translate3d(var(--sx,0px), var(--sy,0px), 60px)   /* 浮起 */
    rotateX(var(--srx,0deg)) rotateY(var(--sry,0deg)) /* 少轉一半 */
    scale(var(--sscale,1));
  transition: transform .1s ease-out;
  filter: /* 三層落地陰影 + */ contrast(1.08) saturate(1.09) brightness(1.02);
}
.subject img{ display:block; width:100%; height:auto; }
```

- `width:75%` + 貓是橫躺（比例 ≈ 1.4）→ 高度約卡片的 43%，頭在上緣附近、腳掌壓在面板頂緣。
- `bottom:50%`：貓的下緣落在場景窗與面板的交界，看起來是「從霧裡走出來、趴到面板前方」。
- 想換照片時，換 `.subject > img` 的 `src`，再視新圖比例微調 `width` / `bottom` 即可。

---

## 7. 幻影殘影層（時空殘影）

對應技能「✦ 三千風影 — 在戰場留下沉睡與警戒的時空殘影」。

| 層 | 來源照片 | `opacity` | `mix-blend-mode` | `filter` | 預設錯位 + `scale` |
|---|---|---|---|---|---|
| `.ghost--warn`（過渡·警戒） | 抬頭警戒 | `.42` | `screen` | `sepia(1) hue-rotate(188deg) saturate(3.2) brightness(.9) contrast(1.12) drop-shadow(0 0 10px rgba(0,220,255,.7))` | `translate(11%, −18%) scale(1.16)` |
| `.ghost--past`（過去·沉睡） | 蜷曲沉睡 | `.4` | `screen` | `sepia(1) hue-rotate(96deg) saturate(4) brightness(.9) contrast(1.1) blur(.6px) drop-shadow(0 0 11px rgba(0,255,150,.7))` | `translate(−28%, 16%) scale(1.26)` |
| `.scene__bg`（環境霧） | 覺醒本體那張 | 1 | normal | `blur(8px) brightness(.42) saturate(.62) contrast(1.06)` | `scale(1.28)` |

設計重點：

- **`sepia(1) → hue-rotate`** 把照片統一染成單一魔法色（綠 96° / 青 188°），`saturate` 拉濃、`brightness(.9)` 壓一點避免 `screen` 疊出死白。
- **`screen` 疊在環境霧之上**：亮處（貓身）留下濃色剪影，暗處保持暗，另一個姿勢的輪廓才看得出來。
- **預設就大幅錯位**（−28% / +11%）：不動滑鼠時三種姿勢已經疊得出來；動滑鼠時再各自以不同係數拖曳分離。
- **`.scene__mist`**：窗內 overlay，底部 `linear-gradient` + `inset box-shadow` 把場景下緣融進深淵黑，貓像「從霧裡浮現」。

---

## 8. 全息膜光學特效

`.holo`（獨立一層，`mix-blend-mode: color-dodge`，`pointer-events:none`）

```css
.holo{
  position:absolute; inset:0; z-index:3;
  mix-blend-mode: color-dodge;
  opacity:.32;   /* JS 會依傾斜幅度改成 0.14 + mag·0.30 */
  mask-image: linear-gradient(180deg, #000 0 52%, rgba(0,0,0,.22) 64%, rgba(0,0,0,.14) 100%);
}
```

- **遮罩**：全息在場景窗（上 52%）全強度、到面板區大幅衰減 → 技能文字保持可讀。
- **兩部分**：
  1. `.holo__spectrum`：`linear-gradient(var(--holo-ang) …)` 五色彩虹（`background-size: 320%`），角度 `--holo-ang` 由 JS 用 `atan2(oy, ox)·180/π + 110` 算出 → **隨指標方位旋轉**；另疊一塊 `radial-gradient` 白色柔光跟著指標（`--gx/--gy`）。
  2. `.holo__dust`：三組不同大小的 `radial-gradient` 點陣（`background-size: 17 / 26 / 39px`），稀疏亮點模擬星芒微塵；`@keyframes dustDrift` 12s 緩慢飄移，另隨指標小幅位移（`--dx/--dy`）。
- `@property --holo-ang { syntax:'<angle>' }`：讓角度能被平滑補間（不支援的瀏覽器會直接跳，優雅降級）。
- **相位隨視角變，不只隨時間**——這是「閃卡」和「單純漸層動畫」的差別。

---

## 9. 動態金屬流光外框

```css
.card__frame{
  padding: var(--frame);                          /* 6px = 邊框厚度 */
  background: linear-gradient(115deg, /* 9 個金/銅停點 */);
  background-size: 280% 280%;
  animation: sheen 7s linear infinite;            /* background-position 0% → 280% 掃動 */
  filter:
    brightness(var(--sheen,1)) saturate(1.06)     /* --sheen 由 JS = 1 + mag·0.55，傾斜時金框更亮 */
    drop-shadow(0 34px 48px rgba(0,0,0,.62))       /* ← 卡片本體投影掛在這裡，不掛 .card */
    drop-shadow(0 9px 18px rgba(0,0,0,.55));
  box-shadow:
    inset 0 0 0 1px rgba(255,244,210,.42),         /* 內側 1px 亮金線 */
    inset 0 0 16px rgba(0,0,0,.55),
    0 0 22px rgba(212,175,55,.14);                 /* 外側微弱金色光暈 */
}
```

- `115deg` + 多個高低交替的停點 → 掃動時像金屬表面反光。
- 傾斜時 `--sheen` 拉高 `brightness`，模擬「轉動時金框吃到光」。

---

## 10. 毛玻璃技能面板

```css
.panel{
  position:absolute; left:0; right:0; bottom:0; height:46%;
  background: rgba(15, 23, 42, .85);
  backdrop-filter: blur(6px) saturate(1.15);
  border-top: 1px solid rgba(125,232,220,.30);        /* 青綠上緣描邊 */
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,.14),              /* 1px 亮色上緣高光 → 實體 UI 面板感 */
    0 -18px 40px rgba(0,0,0,.42);
}
```

面板內容：

- `.eyebrow` 稱號（青、字距 `.34em`）
- `.name` 角色名（Cinzel 900 30px，底下一條 `::after` 綠→青漸層細線）
- `.skill`：左側 `::before` 一條 3px 翠綠裝飾線（帶 `box-shadow` 發光）
  - `.skill__name` 技能名（翠綠）+ `span` 英文名（灰藍小字）
  - `.skill__desc` 描述，`<b>` 強調（`80%`、`【肉球突擊】`）用白字 + 綠色外發光

---

## 11. 互動邏輯（JavaScript）

一支 IIFE，`requestAnimationFrame` 迴圈。核心是 **目標值 `tgt` → 現值 `cur` 每幀線性插值（`k=0.14`）**，再套 CSS `transition: transform .1s ease-out` 做二段平滑，消弭游標抖動。

### 11.1 狀態

```js
var cur = { rx, ry, ox, oy, gx, gy, mag };   // 現值（畫面實際用的）
var tgt = { … };                              // 目標值（事件寫入的）
```

### 11.2 指標 → 傾斜角

```js
function aim(clientX, clientY){
  var r  = card.getBoundingClientRect();
  var nx = (clientX - r.left) / r.width;      // 0~1
  var ny = (clientY - r.top ) / r.height;
  var ox = clamp(nx*2-1, -1, 1);              // -1~1
  var oy = clamp(ny*2-1, -1, 1);
  tgt.ry = clamp( ox * 25, -25, 25);          // 繞 Y：朝指標傾（±25° 上限）
  tgt.rx = clamp(-oy * 25, -25, 25);          // 繞 X：朝指標傾
  tgt.gx = clamp(nx*100, 0, 100);             // 全息/反光位置（%）
  tgt.gy = clamp(ny*100, 0, 100);
  tgt.mag = Math.min(1, Math.hypot(ox, oy));  // 傾斜強度 0~1
}
```

`MAX_DEG = 25`，`clamp` 硬夾 ±25°。

### 11.3 每幀輸出（`frame()`）

| 目標 | 寫入 |
|---|---|
| 卡片傾斜 | `--rx / --ry`（deg）、`--sheen = 1 + mag·0.55` |
| 主體 | `--sx = ox·9px`、`--sy = oy·7 − mag·3px`、`--srx/--sry = −rx/−ry·0.5`、`--sscale = 1 + mag·0.025` |
| 環境霧 | `transform: translate(bx·8, by·7) scale(1.28)` |
| 青殘影 | `translate(calc(11% + bx·38), calc(−18% + by·33)) scale(1.16)` |
| 綠殘影 | `translate(calc(−28% + bx·64), calc(16% + by·56)) scale(1.26)` |
| 全息 | `--holo-ang = atan2(oy,ox)·180/π + 110`、`--gx/--gy`、`--sweep = 50 + ox·55%`、`holo.opacity = 0.14 + mag·0.30`、微塵 `--dx/--dy = ox/oy·−16px` |

### 11.4 事件

| 事件 | 行為 |
|---|---|
| `pointermove`（window，passive） | 呼叫 `aim()`；`pointermove` 同時涵蓋滑鼠 / 觸控 / 手寫筆 |
| `pointerleave`（card）/ `blur`（window）/ `visibilitychange` 隱藏 | `release()` 回到中立、恢復閒置自轉 |
| `deviceorientation` | 手機傾斜：`gamma/28`、`(beta−42)/28` 映射到 `ox/oy`（iOS 需點一下觸發 `requestPermission`） |

### 11.5 閒置自轉

沒有指標互動時（`!pointerActive`），用 `Math.sin(t·0.7)` / `Math.cos(t·0.52)` 給 `tgt` 一個緩慢的 8° / 5.5° 呼吸擺動，卡片不會死板停著。

### 11.6 減少動態

`@media (prefers-reduced-motion: reduce)`：關閉 `sheen` / `dustDrift` / `floatDust` 動畫、`transition` 設為 `none`、JS 的插值 `k` 設為 `1`（瞬間到位）、閒置自轉停用。

---

## 12. 效能、相容性與降級

| 項目 | 說明 |
|---|---|
| **單一檔案** | 所有 HTML/CSS/JS + 4 張圖（base64）都在一個 `.html`，約 960 KB。搬到任何地方雙擊就開，不依賴外部圖檔。 |
| Google Fonts | 唯一的外部請求。沒網路 → fallback 到系統 serif / 微軟正黑體，功能不受影響。 |
| `mix-blend-mode` / `backdrop-filter` / `color-dodge` | 現代 Chromium / Firefox / Safari 皆支援。老瀏覽器：全息會變成單純半透明疊色、毛玻璃變不透明底 —— 退化但不壞。 |
| `@property` | Chrome/Edge、Safari 16.4+、Firefox 128+。不支援 → 全息角度改成瞬跳（仍會動）。 |
| `transform-style: preserve-3d` | 關鍵：`.card` 不能有 `filter` / `overflow:hidden` / `opacity<1`，否則 `.subject` 的 `translateZ` 立體感消失。已刻意把卡片投影移到 `.card__frame`。 |
| GPU | `will-change: transform`、`backface-visibility:hidden`；每幀只改 CSS 變數與少數 `transform` 字串，rAF 單迴圈。 |
| 觸控 | `pointermove` + `deviceorientation` 皆支援；`.subject` / `.holo` / `.topbar` 都 `pointer-events:none`，不擋任何互動。 |

---

## 13. 客製化指南

### 換一張主體照片
1. 用同一套流程去背成透明 PNG（見 §6.1），或自備去背圖。
2. 換掉 `.subject > img` 的 `src`。
3. 視新圖比例調 `.subject` 的 `width`（橫躺約 75%、站姿可降到 45~55%）與 `bottom`（讓頭接近上緣、腳掌壓面板頂緣）。

### 換配色（例如改成火屬性）
改 `:root` 的 `--emerald` / `--cyan`，再同步：殘影 `filter` 的 `hue-rotate`、全息彩虹停點、徽章漸層、`WIND 風` 文案。

### 調景深強度
| 想要 | 改哪 |
|---|---|
| 貓浮更高 | `.subject` 的 `translateZ(60px)` 調大（同時 `width` 略減，透視會放大它） |
| 貓移動更明顯 | `frame()` 裡 `--sx / --sy` 的係數（`ox·9`、`oy·7`）調大 |
| 貓更「黏」在卡上 | `--srx/--sry` 的 `·0.5` 調大到 `·0.8`（少轉一點 → 比較跟卡片） |
| 傾斜更誇張 | `MAX_DEG`（目前 25）；`transition: transform .1s` 的時間 |
| 殘影飄更遠 | `.ghost--past` / `.ghost--warn` 的預設 `translate()` 與 `frame()` 裡的 `bx·64 / bx·38` 係數 |
| 背景更模糊/更暗 | `.scene__bg` 的 `blur(8px)` / `brightness(.42)` |

### 調全息強度
`.holo` 的 `opacity:.32` 與 `frame()` 裡 `0.14 + mag·0.30`；彩虹濃度改停點的 alpha（目前 `.42`）。

---

## 14. 資產清單與來源

| 資產 | 來源 | 在卡片裡的角色 |
|---|---|---|
| `螢幕擷取畫面 2026-09-08 162900`（覺醒·睜眼直視） | 使用者截圖 | 去背後 → `.subject` 主體；原圖模糊後 → `.scene__bg` 環境霧 |
| `螢幕擷取畫面 2026-09-08 162909`（警戒·抬頭） | 使用者截圖 | `.ghost--warn` 青藍殘影 |
| `螢幕擷取畫面 2026-09-08 162946`（沉睡·蜷曲） | 使用者截圖 | `.ghost--past` 翠綠殘影 |
| Cinzel / Noto Sans TC | Google Fonts（CDN） | 字體 |
| 去背模型 `yolo11m-seg.pt` | Ultralytics（僅產製階段用，不含在成品裡） | 產生主體遮罩 |

所有圖片在成品中皆為 base64 內嵌，不需外部檔案。

---

## 15. 與原始 spec 的取捨對照

| 原始要求 | 實作 | 原因 |
|---|---|---|
| 三張殘影 `opacity: 0.35 / 0.6 / 1`、`本體 opacity:1` 在最上層 | 本體改成**去背 PNG 浮在最上層**；殘影 `opacity` 降到 `.4 / .42`、以 `screen` 疊在**環境霧之上** | 不透明的滿版本體會完全蓋住殘影，看不到「不同姿勢」。改成去背 + 分層才同時滿足「本體最上層」與「看得到殘影」。 |
| 頂層本體 `box-shadow: inset 0 -30px 30px rgba(0,0,0,.9)` | 改用 `.scene__mist` overlay（`inset box-shadow` + 底部漸層） | `inset box-shadow` 畫不到 `<img>` 這種置換元素上；overlay 達到同樣「底部融入深色 UI」效果。 |
| 三張圖放同一個 `.artwork-container`、`object-fit:cover` | 場景窗 `.scene` 內放 `.scene__bg` + 兩層 `.ghost`（都 `object-fit:cover`）；本體另外拉出成 `.subject` | 為了景深分離（見 §4、§5）。 |
| `transform: rotateX() rotateY()` + `transition: transform 0.1s ease-out` | ✅ 完全照做，另加 rAF 插值做二段平滑 | — |
| 全息 `mix-blend-mode: color-dodge` + 彩虹 + 星芒 | ✅ | — |
| 毛玻璃 `rgba(15,23,42,.85)` + `blur(6px)` + 1px 上緣高光 | ✅ | — |
| 金屬流光外框、Cinzel + Noto Sans TC、文案 | ✅ | — |

---

*本文件對應 `jimmy_phantom_card.html` 於 2026-09-08 的版本；改動 CSS/JS 數值時建議一併更新 §5、§7、§11 的對照表。*
