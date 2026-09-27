"""把飼主語音紀錄需要的修改寫進「健康監測 v2」流程（P4 面板）。

    主要用法（09-27 起）：python -m iot.voice.node_red.build_p4_patch --apply-to C:/ai_project/node-red-flows/cat_health_v4_flow.json
    （那份檔案的節點 ID＝1880，改完匯入選「取代」）。下面「從 1880 產生修補檔」是舊用法，保留給流程檔對不上時用。

    cd paper/cat_monitoring_system
    python -m iot.voice.node_red.build_p4_patch                 # 從 http://127.0.0.1:1880/flows 讀（只讀取）
    python -m iot.voice.node_red.build_p4_patch --flows 匯出.json

為什麼從 1880 讀：repo 的 node-red-flows/cat_health_v4_flow.json 跟 1880 上實際在跑的節點 ID 不同，
直接匯入整個檔案會重複／蓋錯；這裡以 1880 上的節點為底，只改需要的地方。

修補內容（v7.2，2026-09-27，使用者要求）：
1. 「使用者設定管理器」的 save_event_tag：同一天改成**合併**（原本整天覆蓋，會把語音記的標籤蓋掉）。
2. 「P4 個體基線面板」近期事件：標籤後面顯示次數（「嘔吐 × 2 · 甩頭 × 1」，語音記的才有次數）。
3. 新增「← 飼主語音紀錄」link in＋「飼主語音紀錄 → 事件標記＋更新 P4」function（放在 P4 旁邊）：
   接橋接分頁送來的紀錄 → 寫進 v2_event_tags（情境事件用 P4 自己的標籤、飼主觀察用類別名稱）→ 立即更新 P4
   （P4 平常要等 Python 行為資料進來才更新；這裡跟「設定變更→立即回填P4面板」組同一份資料）。
P4 的資料格式只有這個檔案知道；橋接分頁只管把紀錄送到「← 飼主語音紀錄」。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import urllib.request
from pathlib import Path

from iot.voice.node_red.build_flow import P4_LINK_IN_ID

OUT = Path(__file__).resolve().parent / "p4_patch.json"
MERGE_ID = hashlib.sha256(b"owner-reports-p4:merge").hexdigest()[:16]   # 第一次加進流程時用；之後沿用流程裡的 ID
LINK_IN_NAME = "← 飼主語音紀錄"          # 橋接分頁（build_flow.py）靠這個名字找它的 ID
MERGE_NAME = "飼主語音紀錄 → 事件標記＋更新 P4"
RETURN_NAME = "↩ 回到飼主語音紀錄"        # link out（回傳）：橋接分頁用動態 link call 呼叫「← 飼主語音紀錄」，這裡回應
RETURN_ID = hashlib.sha256(b"owner-reports-p4:return").hexdigest()[:16]
MARK = "飼主語音紀錄 v7.2"

# P4 面板 evtTagDefs 的標籤文字；key 跟 iot/voice/ontology.py 的 event 類相同
P4_TAG_LABELS = {"vet": "帶去看醫生", "food": "換了貓糧", "stranger": "陌生人來訪",
                 "stress": "情緒緊張", "medicine": "服藥/治療", "other": "其他"}

MGR_FROM = """    let events = global.get("gfile").get('v2_event_tags') || [];
    events = events.filter(function(e){ return e.date !== date; });
    events.push({date: date, tags: tags, note: note});"""
MGR_TO = """    // 飼主語音紀錄 v7.2（2026-09-27）：同一天改成「合併」——原本整天覆蓋，會把語音記的標籤（看醫生、嘔吐…）蓋掉
    let events = global.get("gfile").get('v2_event_tags') || [];
    let entry = events.find(function(e){ return e && e.date === date; });
    if (!entry) { entry = {date: date, tags: [], note: ''}; events.push(entry); }
    if (!Array.isArray(entry.tags)) entry.tags = [];
    tags.forEach(function(t){ if (entry.tags.indexOf(t) < 0) entry.tags.push(t); });
    if (note && String(entry.note || '').indexOf(note) < 0) entry.note = entry.note ? entry.note + '；' + note : note;"""

P4_TAGS_FROM = "{{e.tags.join(' · ')}}"
P4_TAGS_TO = "{{p4TagText(e)}}"
P4_WATCH = "scope.$watch('msg.payload',function(p){"
# 近期事件每一列：語音紀錄清單（可展開）。時間用 AngularJS 的 date 濾鏡（voice[].ts 是秒）
P4_ROW_FROM = "{{p4TagText(e)}}</span><span class=\"p4-evt-note\" ng-if=\"e.note\"> — {{e.note}}</span></div>"
P4_ROW_TO = ("{{p4TagText(e)}}</span><span class=\"p4-evt-note\" ng-if=\"e.note\"> — {{e.note}}</span>"
             "<details class=\"p4-evt-voice\" ng-if=\"e.voice.length\"><summary>🎙 語音紀錄 {{e.voice.length}} 筆</summary>"
             "<div class=\"p4-evt-voice-row\" ng-repeat=\"v in e.voice track by $index\">"
             "<span class=\"p4-evt-voice-time\">{{v.ts*1000 | date:'HH:mm'}}</span> {{v.label}}<span class=\"p4-evt-voice-text\">「{{v.text}}」</span>"
             "</div></details></div>")
P4_CSS_FROM = ".p4-evt-note{color:#8ce6d8;font-style:italic}"
P4_CSS_TO = (P4_CSS_FROM + ".p4-evt-voice{margin-top:4px;color:#b9c7d6}.p4-evt-voice summary{cursor:pointer;color:#93eadc;font-size:11px}"
             ".p4-evt-voice-row{padding:2px 0 2px 14px;font-size:11px;line-height:1.5}.p4-evt-voice-time{color:#8da2b8;margin-right:4px}"
             ".p4-evt-voice-text{color:#c7d6e3}")
P4_FUNC = ("// 飼主語音紀錄 v7.2：標籤後面加次數（語音記的才有 counts）\n  "
           "scope.p4TagText=function(e){var c=(e&&e.counts)||{};return ((e&&e.tags)||[]).map(function(t){return c[t]?t+' × '+c[t]:t;}).join(' · ');};\n  ")

# ── v7.3（09-27）：次數＝重新數「語音紀錄 voice[]＋手動紀錄 manual[]」，不再 +1 累加 ──────────────
# 使用者回報：P4 那一天「🎙 語音紀錄 6 筆」但只顯示「其他 × 2」，手動按的標籤也沒有次數。原因：
#   v7.1 的語音紀錄沒有次數、沒有 label；手動「記錄今日事件」只補標籤不計數。
# 改成每次寫入都依紀錄清單重算，次數一定跟清單對得上；舊資料第一次寫入（或部署時）自動補齊。
# 下面兩段 JS 放進「使用者設定管理器」和「飼主語音紀錄 → 事件標記＋更新 P4」兩個節點（同一份程式）。
RECOUNT_JS = r"""
// 飼主語音紀錄 v7.3：某一天的次數＝語音紀錄（voice）＋手動紀錄（manual，按一次「記錄今日事件」算一次）重新數
const P4_LBL = LABELS;
const p4Recount = function (e) {
    if (!Array.isArray(e.tags)) e.tags = [];
    e.voice = (Array.isArray(e.voice) ? e.voice : []).map(function (v) {   // v7.1 的紀錄沒有 label → 依類別補
        return v && v.label ? v : Object.assign({}, v, { label: P4_LBL[v && v.category] || P4_LBL.other });
    });
    if (!Array.isArray(e.manual)) {
        // 第一次（舊資料）：語音沒記到的標籤＝以前手動按的，各算一次
        const fromVoice = e.voice.map(function (v) { return v.label; });
        const old = e.tags.filter(function (t) { return fromVoice.indexOf(t) < 0; });
        e.manual = old.length ? [{ ts: null, tags: old, legacy: true }] : [];
        // 以前語音原句塞在備註（「🎙 …」），語音清單裡已經有了 → 從備註拿掉，只留飼主自己打的字
        if (e.note) {
            const texts = e.voice.map(function (v) { return String(v.text || '').trim(); });
            e.note = String(e.note).split('；').filter(function (s) {
                return !(/^🎙/.test(s) && texts.indexOf(s.replace(/^🎙\s*/, '').trim()) >= 0);
            }).join('；');
        }
    }
    const c = {};
    e.voice.forEach(function (v) { c[v.label] = (c[v.label] || 0) + 1; });
    e.manual.forEach(function (m) { (m.tags || []).forEach(function (t) { c[t] = (c[t] || 0) + 1; }); });
    Object.keys(c).forEach(function (t) { if (e.tags.indexOf(t) < 0) e.tags.push(t); });
    e.counts = c;
    return e;
};
const p4CountText = function (e) {
    return (e.tags || []).map(function (t) { return t + ' × ' + ((e.counts || {})[t] || 0); }).join('、');
};
// 立即更新 P4：組跟「設定變更→立即回填P4面板」同一份資料（Dashboard 會把最後一則重播給新開的頁面，所以要完整）
const p4Payload = function (g) {
    const bl = g.get('v2_baseline') || null;
    const cfg = g.get('v2_user_settings') || {};
    const out = {};
    if (bl && bl.metrics) { out.baseline = bl; out.v2_insufficient = false; }
    else { out.v2_insufficient = true; out.required_days = (bl && bl.required_days) || cfg.baseline_days || 7; }
    out.deviation = global.get('v2_deviation') || null;
    out.user_settings = cfg;
    out.history_days = (g.get('v2_daily_history') || []).length;
    out.v2_daily_history = (g.get('v2_daily_history') || []).slice(-30);
    out.v2_event_tags = (g.get('v2_event_tags') || []).slice(-14);
    out.v2_excluded_dates = g.get('v2_excluded_dates') || [];
    return out;
};
"""

MERGE_FUNC = r"""
// 飼主語音紀錄 v7.3（語音辨識 v7 修補 26 → 🎙 飼主語音紀錄分頁 → 這裡）：寫進事件標記，並立即更新 P4 面板
//   msg.payload = { text, ts, reports:[{ report_id, kind: "event"|"observation", category, label, day: "YYYY-MM-DD" }] }
//   * 情境事件用 P4 自己的標籤（帶去看醫生…），飼主觀察用類別名稱（嘔吐、甩頭…）；同一天合併
//   * 每一筆的時間、類別、原句放在 voice 陣列，P4「近期事件」那一天下面可以展開「🎙 語音紀錄 N 筆」看（最多留 200 筆）
//     09-27 起不再寫進備註：同一天記很多筆時備註會被截斷、後面的看不到（使用者回報）；備註只留給飼主自己打的字
//   * v7.3：次數由 p4Recount 依 voice＋manual 重新數（手動「記錄今日事件」也算）
const g = global.get("gfile");
// 09-27：橋接分頁改用動態 link call 呼叫（依名稱，不怕匯入時 ID 被換掉）→ 輸出 2 一定要把原本的 msg 送回「↩ 回到飼主語音紀錄」
if (!g || typeof g.get !== "function") { node.status({ fill: "grey", shape: "ring", text: "沒有 gfile，略過" }); return [null, msg]; }
RECOUNT
const list = g.get('v2_event_tags') || [];
const added = [];
for (const r of (msg.payload && msg.payload.reports) || []) {
    const [y, m, d] = String(r.day || "").split("-").map(Number);
    if (!y || !m || !d) continue;
    const date = new Date(y, m - 1, d).toLocaleDateString('zh-TW');
    let e = list.find(x => x && x.date === date);
    if (!e) { e = { date, tags: [], note: '' }; list.push(e); }
    p4Recount(e);   // 舊資料先補齊（第一次），再加這一筆
    const label = r.kind === "event" ? (P4_LBL[r.category] || P4_LBL.other) : String(r.label || P4_LBL[r.category] || r.category);
    const text = String(msg.payload.text || "");
    e.voice = e.voice.concat([{ ts: msg.payload.ts, report_id: r.report_id, category: r.category, label, text }]).slice(-200);
    p4Recount(e);
    added.push(label);
}
if (!added.length) return [null, msg];
list.sort((a, b) => new Date(a.date) - new Date(b.date));
g.set('v2_event_tags', list.slice(-90));
node.status({ fill: "green", shape: "dot", text: "已記錄：" + added.join("、") });
return [{ payload: p4Payload(g) }, msg];
"""

# 部署時把舊資料補齊一次（只動語音記過的日子），並推一次 P4，畫面馬上是正確的次數
MERGE_INIT = r"""
// 飼主語音紀錄 v7.3：部署／啟動時補齊舊資料的次數（只動語音記過的日子：有 voice 或 counts 的那幾天）
const g = global.get("gfile");
if (g && typeof g.get === "function") {
RECOUNT
    const list = g.get('v2_event_tags') || [];
    let n = 0;
    list.forEach(function (e) {
        if (!e || !(Array.isArray(e.voice) || e.counts)) return;
        const before = JSON.stringify(e);
        p4Recount(e);
        if (JSON.stringify(e) !== before) n++;
    });
    if (n) {
        g.set('v2_event_tags', list);
        setTimeout(function () {   // 流程都啟動後推一次 P4（失敗只警告，不能讓 Node-RED 掛掉）
            try { node.send([{ payload: p4Payload(g) }, null]); node.log('事件標記次數補齊：' + n + ' 天，已推送 P4'); }
            catch (err) { node.warn('事件標記次數補齊：' + n + ' 天；推送 P4 失敗：' + err.message); }
        }, 1500);
    }
}
"""

# 「使用者設定管理器」的 save_event_tag 整段換成這一段（v7.3）。輸出 3＝直接更新 P4（原本存完不通知 P4，
# 要等下一次即時資料才看得到）；輸出 1、2 原本的線（基線重算、Discord）不動
MGR_BRANCH_START = "if (action === 'save_event_tag') {"
MGR_BRANCH = r"""if (action === 'save_event_tag') {
    // 飼主語音紀錄 v7.3（2026-09-27）：同一天「合併」（原本整天覆蓋，會把語音記的標籤蓋掉）；
    // 按一次「記錄今日事件」＝一筆手動紀錄（manual），次數跟語音紀錄一起重新數；存完直接更新 P4（輸出 3）
    let tags = msg.payload.tags || [];
    let note = (msg.payload.note || '').trim();
    let date = msg.payload.date || new Date().toLocaleDateString('zh-TW');
    if (tags.length === 0 && !note) return [null, null, null];
    const g = global.get("gfile");
RECOUNT
    let events = g.get('v2_event_tags') || [];
    let entry = events.find(function(e){ return e && e.date === date; });
    if (!entry) { entry = {date: date, tags: [], note: ''}; events.push(entry); }
    p4Recount(entry);   // 舊資料先補齊（第一次），再加這一筆
    entry.manual = entry.manual.concat([{ ts: Date.now() / 1000, tags: tags.slice(), note: note }]).slice(-200);
    if (note && String(entry.note || '').indexOf(note) < 0) entry.note = entry.note ? entry.note + '；' + note : note;
    p4Recount(entry);
    if (events.length > 90) events = events.slice(-90);
    g.set('v2_event_tags', events);
    node.warn('事件標記已儲存：' + date + ' [' + tags.join(',') + ']，今天：' + p4CountText(entry));
    return [null, null, { payload: p4Payload(g) }];
}"""
MARK_V73 = "飼主語音紀錄 v7.3"


def _indent(js: str, n: int) -> str:
    return "\n".join((" " * n + line) if line.strip() else "" for line in js.strip("\n").split("\n"))


def _labels() -> dict[str, str]:
    """類別 key → P4 上顯示的文字：情境事件用 P4 自己的標籤，飼主觀察用類別名稱（嘔吐、甩頭…）。"""
    from iot.voice.ontology import OBSERVATION_CATEGORIES
    return {**P4_TAG_LABELS, **{c.key: c.label for c in OBSERVATION_CATEGORIES}}


def _recount_js(indent: int) -> str:
    return _indent(RECOUNT_JS.replace("LABELS", json.dumps(_labels(), ensure_ascii=False)), indent)


def _patch_mgr(func: str) -> str:
    """save_event_tag 整段換成 v7.3（原版或 v7.2 都可以）；已經是 v7.3 就照舊。"""
    if MARK_V73 in func:
        return func
    start = func.find(MGR_BRANCH_START)
    end = func.find("\n}", start)
    if start < 0 or end < 0 or func.count(MGR_BRANCH_START) != 1:
        raise SystemExit("使用者設定管理器：找不到 save_event_tag 那一段（1880 上的節點跟預期不同，請確認）")
    old = func[start:end + 2]
    if "v2_event_tags" not in old or not (MGR_FROM in old or MARK in old):
        raise SystemExit("使用者設定管理器：save_event_tag 那一段跟預期不同，請確認")
    return func[:start] + MGR_BRANCH.replace("RECOUNT", _recount_js(4)) + func[end + 2:]


def load_flows(src: str) -> list[dict]:
    if src.startswith("http://") or src.startswith("https://"):
        with urllib.request.urlopen(src, timeout=10) as r:   # 只讀取（GET）
            return json.loads(r.read().decode("utf-8"))
    return json.loads(Path(src).read_text(encoding="utf-8"))


def _one(flows: list[dict], name: str) -> dict:
    hit = [n for n in flows if n.get("name") == name]
    if len(hit) != 1:
        raise SystemExit(f"找不到或不只一個「{name}」（{len(hit)} 個）")
    return hit[0]


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{what}：要改的地方預期 1 處，實際 {n} 處（1880 上的節點跟預期不同，請確認）")
    return text.replace(old, new)


def build(flows: list[dict]) -> list[dict]:
    mgr = copy.deepcopy(_one(flows, "使用者設定管理器"))
    mgr["func"] = _patch_mgr(mgr["func"])
    p4 = copy.deepcopy(_one(flows, "P4 個體基線面板"))
    # 輸出 3 → P4（存完事件標記直接更新畫面）；原本的輸出 1、2 不動
    wires = [list(w) for w in mgr.get("wires", [])] + [[] for _ in range(max(0, 3 - len(mgr.get("wires", []))))]
    if p4["id"] not in wires[2]:
        wires[2] = wires[2] + [p4["id"]]
    mgr["wires"], mgr["outputs"] = wires, max(3, int(mgr.get("outputs", 2) or 2))
    if "p4TagText" not in p4["format"]:
        p4["format"] = _replace_once(p4["format"], P4_TAGS_FROM, P4_TAGS_TO, "P4 近期事件標籤")
        p4["format"] = _replace_once(p4["format"], P4_WATCH, P4_FUNC + P4_WATCH, "P4 訊息處理")
    if "p4-evt-voice" not in p4["format"]:   # 09-27：語音紀錄清單（可展開）
        p4["format"] = _replace_once(p4["format"], P4_ROW_FROM, P4_ROW_TO, "P4 近期事件列")
        p4["format"] = _replace_once(p4["format"], P4_CSS_FROM, P4_CSS_TO, "P4 樣式")
    x, y = int(p4.get("x", 600)), int(p4.get("y", 300))
    # 兩個新節點已經在流程裡（例如匯入 1880 時 Node-RED 換了 ID）→ 沿用它的 ID 與位置，不會重複加
    old_link = _existing(flows, p4["z"], "link in", LINK_IN_NAME)
    old_merge = _existing(flows, p4["z"], "function", MERGE_NAME)
    old_ret = _existing(flows, p4["z"], "link out", RETURN_NAME)
    ret_id = old_ret["id"] if old_ret else RETURN_ID
    link_id = old_link["id"] if old_link else P4_LINK_IN_ID
    merge_id = old_merge["id"] if old_merge else MERGE_ID
    link_in = {"id": link_id, "type": "link in", "z": p4["z"], "name": LINK_IN_NAME, "links": [],
               "x": old_link["x"] if old_link else max(120, x - 520), "y": old_link["y"] if old_link else y + 120,
               "wires": [[merge_id]]}
    merge = {"id": merge_id, "type": "function", "z": p4["z"], "name": MERGE_NAME,
             "func": MERGE_FUNC.replace("RECOUNT", _recount_js(0)).strip() + "\n",
             "outputs": 2, "timeout": 0, "noerr": 0, "initialize": MERGE_INIT.replace("RECOUNT", _recount_js(4)).strip() + "\n",
             "finalize": "", "libs": [],
             "x": old_merge["x"] if old_merge else max(300, x - 260), "y": old_merge["y"] if old_merge else y + 120,
             "wires": [[p4["id"]], [ret_id]]}
    ret = {"id": ret_id, "type": "link out", "z": p4["z"], "name": RETURN_NAME, "mode": "return", "links": [],
           "x": old_ret["x"] if old_ret else merge["x"] + 260, "y": old_ret["y"] if old_ret else merge["y"] + 40, "wires": []}
    return [mgr, p4, link_in, merge, ret]


def _existing(flows: list[dict], z: str, type_: str, name: str):
    hit = [n for n in flows if n.get("z") == z and n.get("type") == type_ and n.get("name") == name]
    if len(hit) > 1:
        raise SystemExit(f"「{name}」有 {len(hit)} 個，請先刪掉多的")
    return hit[0] if hit else None


def apply_to_file(path: str) -> list[str]:
    """把修補**直接寫進**一份完整的流程檔（例如 repo 的 node-red-flows/cat_health_v4_flow.json）：
    兩個原有節點原地替換（陣列位置不變）、兩個新節點加在最後（已經有就更新）；其他節點一個字都不動。
    寫回時照原檔的格式（這個檔是 indent 2、不跳脫中文、結尾沒有換行），重跑結果一樣（冪等）。"""
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    flows = json.loads(raw)
    patched = build(flows)
    by_id = {n["id"]: n for n in patched}
    out = [by_id.pop(n["id"], n) for n in flows]   # 原有的兩個節點原地換掉
    out += list(by_id.values())                     # 新節點（第一次）加在最後
    changed = [n.get("name", n["id"]) for n in patched if json.dumps(n, sort_keys=True) != json.dumps(
        next((o for o in flows if o["id"] == n["id"]), None), sort_keys=True)]
    text = json.dumps(out, ensure_ascii=False, indent=2) + ("\n" if raw.endswith("\n") else "")
    if text != raw:
        p.write_text(text, encoding="utf-8")
    return changed


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="產生健康監測 P4 的飼主語音紀錄修補檔")
    ap.add_argument("--flows", default="http://127.0.0.1:1880/flows", help="1880 的 /flows（只讀取）或匯出的 JSON 檔")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--apply-to", help="直接改寫這份完整流程檔（例如 node-red-flows/cat_health_v4_flow.json），不產生修補檔")
    a = ap.parse_args(argv)
    if a.apply_to:
        changed = apply_to_file(a.apply_to)
        print(f"已寫進 {a.apply_to}：" + ("、".join(changed) if changed else "（已經是最新，沒有變動）"))
        return
    nodes = build(load_flows(a.flows))
    Path(a.out).write_text(json.dumps(nodes, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"已產生 {a.out}：" + "、".join(f"{n.get('name')}（{n['type']}）" for n in nodes))


if __name__ == "__main__":
    main()
