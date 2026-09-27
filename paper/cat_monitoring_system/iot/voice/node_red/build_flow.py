"""產生 Node-RED 分頁「🎙 飼主語音紀錄（貓咪系統）」：owner_reports_flow.json。

    cd paper/cat_monitoring_system
    python -m iot.voice.node_red.build_flow

這個分頁是**轉接層（adapter）**，只做兩件事，其他知識都在 Python 服務：
1. **橋接**：語音分頁用動態 link call 呼叫名稱 ``owner-report``（跟主機動作的 ``host-action`` 同一套做法），
   這裡把句子發到 MQTT ``cat/voice/utterance``，等 ``cat/voice/result`` 回來再 return 給語音分頁。
   服務離線時馬上回「沒有記錄」，不讓語音分頁等到逾時。
2. **交給健康監測 P4**：有紀錄就用動態 link call 依名稱呼叫「← 飼主語音紀錄」（健康監測流程裡、P4 面板旁邊），
   由那邊寫進事件標記、立即更新 P4。依名稱呼叫，健康監測流程重新匯入、ID 變了也接得上。v7.2 起不再有自己的 Dashboard 分頁（使用者要求：直接寫進 P4）。

語音分頁（C:\\arduino\\語音辨識-v7）不知道 MQTT／Python 的存在；Python 服務不知道 Node-RED 流程的長相；
這個分頁不知道 P4 的資料格式（那是 P4 修補檔的事）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "owner_reports_flow.json"
PREFIX = "cat/voice"
LINK_NAME = "owner-report"
BROKER = {"host": "192.168.0.171", "port": "1883"}
# 健康監測流程（個體化基線）裡接語音紀錄的 link in 名稱：用「動態 link call」依名稱呼叫，不依賴節點 ID
# （09-27：使用者重新匯入健康監測流程時 Node-RED 換了 ID，原本依 ID 的 link out 就斷了）
P4_LINK_NAME = "← 飼主語音紀錄"
P4_LINK_IN_ID = hashlib.sha256(b"owner-reports-p4:link_in").hexdigest()[:16]   # 第一次加進健康監測流程時用的 ID（build_p4_patch.py）


def nid(key: str) -> str:
    return hashlib.sha256(("owner-reports:" + key).encode()).hexdigest()[:16]


TAB, BROKER_ID = nid("tab"), nid("broker")

SEND_FUNC = r"""
// 語音分頁送來的一句話（msg.ownerReport = { requestId, text, deviceId, source, mode, ts }）→ MQTT 給紀錄服務
//   mode "explicit"＝飼主先說了「我要記錄貓咪」（v7.2 起語音分頁只送這種）
// 回應回來時「結果 → 回到語音分頁」節點用 requestId 找回這個 msg，經 link out（return）交回 link call。
const r = msg.ownerReport || {};
const id = String(r.requestId || msg._msgid);
if (!flow.get("vr_online")) {   // 服務沒在線：馬上回「沒有記錄」，不讓語音分頁等到逾時
    msg.ownerResult = { id, ok: false, matched: false, reports: [], error: "紀錄服務沒有在線（python -m iot.voice）" };
    node.status({ fill: "grey", shape: "ring", text: "服務離線：" + String(r.text || "").slice(0, 12) });
    return [null, msg];
}
const pending = flow.get("vr_pending") || {};
const now = Date.now();
for (const k of Object.keys(pending)) if (now - pending[k].at > 15000) delete pending[k];
pending[id] = { at: now, msg };
flow.set("vr_pending", pending);
node.status({ fill: "blue", shape: "dot", text: "送出：" + String(r.text || "").slice(0, 16) });
return [{ topic: TOPIC, payload: { id, text: String(r.text || ""), ts: r.ts || now, mode: r.mode === "explicit" ? "explicit" : "",
    deviceId: String(r.deviceId || ""), source: String(r.source || "voice") } }, null];
"""

RESULT_FUNC = r"""
// 紀錄服務的結果 → 找回原本的 msg，經 link out（return）交回語音分頁；有紀錄就另外送給健康監測 P4（寫進事件標記）
const res = msg.payload || {};
const pending = flow.get("vr_pending") || {};
const p = pending[res.id];
if (!p) return [null, null];   // 別台 Node-RED 的請求、或已經逾時
delete pending[res.id];
flow.set("vr_pending", pending);
const out = p.msg;
out.ownerResult = res;
node.status({ fill: res.matched ? "green" : "grey", shape: "dot",
    text: res.matched ? (res.reports || []).map(x => x.label).join("、") : res.cancelled ? "已取消" : "沒有紀錄：" + String(res.text || "").slice(0, 12) });
const reports = res.ok && res.matched ? (res.reports || []) : [];
return [out, reports.length ? { target: "← 飼主語音紀錄", payload: { text: res.text, ts: res.ts, reports } } : null];
"""

P4_FAIL_FUNC = r"""
// 動態 link call 找不到「← 飼主語音紀錄」（健康監測流程沒匯入／沒啟用）或 10 秒沒回應：紀錄已經存進 python -m iot.voice，只是 P4 沒更新
const m = String(msg.error && msg.error.message || "");
node.warn("飼主語音紀錄沒寫進健康監測 P4：" + (/timeout/i.test(m) ? "10 秒沒回應" : "找不到「← 飼主語音紀錄」（健康監測流程要匯入並啟用）") + "；紀錄本身已存檔");
node.status({ fill: "red", shape: "ring", text: "P4 沒更新：" + m.slice(0, 20) });
return null;
"""

STATUS_FUNC = r"""
// 紀錄服務上線／離線（retained；離線是服務設的 MQTT 遺囑）→ 「送出」節點看這個決定要不要等
const online = !!(msg.payload && msg.payload.online);
flow.set("vr_online", online);
node.status({ fill: online ? "green" : "red", shape: "dot", text: online ? "紀錄服務在線" : "紀錄服務離線（python -m iot.voice）" });
return null;
"""


def fn(key, name, func, outputs, wires, x, y):
    return {"id": nid(key), "type": "function", "z": TAB, "name": name, "func": func.strip() + "\n", "outputs": outputs,
            "timeout": 0, "noerr": 0, "initialize": "", "finalize": "", "libs": [], "x": x, "y": y, "wires": wires}


def mqtt_in(key, name, topic, wires, x, y):
    return {"id": nid(key), "type": "mqtt in", "z": TAB, "name": name, "topic": topic, "qos": "1", "datatype": "json",
            "broker": BROKER_ID, "nl": False, "rap": True, "rh": 0, "inputs": 0, "x": x, "y": y, "wires": [wires]}


def build() -> list[dict]:
    t = lambda name: f"{PREFIX}/{name}"   # noqa: E731
    send = fn("send", "送出一句話 → 紀錄服務", SEND_FUNC.replace("TOPIC", json.dumps(t("utterance"))), 2,
              [[nid("mqtt_utt")], [nid("link_return")]], 470, 100)
    result = fn("result", "結果 → 回到語音分頁／交給 P4", RESULT_FUNC, 2, [[nid("link_return")], [nid("to_p4")]], 490, 180)
    p4_fail = fn("p4_fail", "寫進 P4 失敗", P4_FAIL_FUNC, 1, [[]], 1080, 260)
    status = fn("status", "服務狀態", STATUS_FUNC, 1, [[]], 450, 280)
    return [
        {"id": TAB, "type": "tab", "label": "🎙 飼主語音紀錄（貓咪系統）", "disabled": False,
         "info": "由 paper/cat_monitoring_system/iot/voice/node_red/build_flow.py 產生，請改那支腳本再重新產生。\n"
                 "需要：python -m iot.voice（飼主語音紀錄服務）、語音辨識 v7.2 流程（修補 26：先說「我要記錄貓咪」）、"
                 "P4 修補檔 p4_patch.json（「← 飼主語音紀錄」寫進事件標記、立即更新 P4）。"},
        {"id": nid("comment"), "type": "comment", "z": TAB,
         "name": "語音分頁（link call「owner-report」）⇄ MQTT cat/voice/* ⇄ python -m iot.voice；有紀錄 → 健康監測 P4（← 飼主語音紀錄）",
         "info": "", "x": 420, "y": 40, "wires": []},
        {"id": BROKER_ID, "type": "mqtt-broker", "name": "飼主語音紀錄 broker", "broker": BROKER["host"], "port": BROKER["port"],
         "clientid": "", "autoConnect": True, "usetls": False, "protocolVersion": "4", "keepalive": "60", "cleansession": True,
         "autoUnsubscribe": True, "birthTopic": "", "birthQos": "0", "birthPayload": "", "birthMsg": {}, "closeTopic": "",
         "closeQos": "0", "closePayload": "", "closeMsg": {}, "willTopic": "", "willQos": "0", "willPayload": "", "willMsg": {},
         "userProps": "", "sessionExpiry": ""},
        {"id": nid("link_in"), "type": "link in", "z": TAB, "name": LINK_NAME, "links": [], "x": 175, "y": 100, "wires": [[send["id"]]]},
        send,
        {"id": nid("mqtt_utt"), "type": "mqtt out", "z": TAB, "name": "cat/voice/utterance", "topic": "", "qos": "1", "retain": "false",
         "respTopic": "", "contentType": "", "userProps": "", "correl": "", "expiry": "", "broker": BROKER_ID, "x": 790, "y": 100, "wires": []},
        mqtt_in("mqtt_result", "cat/voice/result", t("result"), [result["id"]], 190, 180),
        result,
        {"id": nid("link_return"), "type": "link out", "z": TAB, "name": "回到語音分頁", "mode": "return", "links": [], "x": 790, "y": 150, "wires": []},
        {"id": nid("to_p4"), "type": "link call", "z": TAB, "name": "→ 健康監測 P4（依名稱：← 飼主語音紀錄）", "links": [],
         "linkType": "dynamic", "timeout": "10", "x": 830, "y": 220, "wires": [[]]},
        {"id": nid("to_p4_catch"), "type": "catch", "z": TAB, "name": "寫進 P4 逾時／找不到", "scope": [nid("to_p4")],
         "uncaught": False, "x": 820, "y": 260, "wires": [[p4_fail["id"]]]},
        p4_fail,
        mqtt_in("mqtt_status", "cat/voice/status", t("status"), [status["id"]], 190, 280),
        status,
    ]


# ── 換成 1880 上實際的 ID（09-27）───────────────────────────────────────────
# 使用者匯入時 Node-RED 換過 ID，產生的檔案要用 1880 的 ID，「取代」匯入才會換掉同一批節點。
# 對照表 owner_reports_ids_1880.json：{產生的 ID: 1880 的 ID}。ID 又變了 → python -m iot.voice.node_red.build_flow --sync-ids
# （從 1880 GET /flows 讀；Node-RED 關著就讀 ~/.node-red/flows.json；依「類型＋名稱」一對一配對）
PIN_FILE = OUT.with_name("owner_reports_ids_1880.json")
_REF_KEYS = ("id", "z", "g", "broker", "group", "tab")


def remap_ids(nodes: list[dict], pins: dict[str, str]) -> list[dict]:
    out = []
    for n in json.loads(json.dumps(nodes)):
        for k in _REF_KEYS:
            if isinstance(n.get(k), str) and n[k] in pins:
                n[k] = pins[n[k]]
        if isinstance(n.get("wires"), list):
            n["wires"] = [[pins.get(w, w) for w in port] for port in n["wires"]]
        for k in ("links", "scope", "nodes"):
            if isinstance(n.get(k), list):
                n[k] = [pins.get(w, w) for w in n[k]]
        out.append(n)
    return out


def _load_live_flows() -> list[dict]:
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:1880/flows", timeout=3) as r:
            print("讀 1880 GET /flows")
            data = json.loads(r.read().decode("utf-8"))
    except OSError:
        f = Path.home() / ".node-red" / "flows.json"
        print(f"1880 沒開，讀 {f}")
        data = json.loads(f.read_text(encoding="utf-8"))
    return data["flows"] if isinstance(data, dict) else data


def sync_ids(nodes: list[dict], live: list[dict]) -> dict[str, str]:
    tab = next(n for n in nodes if n["type"] == "tab")
    live_tabs = [n for n in live if n["type"] == "tab" and n.get("label") == tab["label"]]
    if len(live_tabs) != 1:
        raise SystemExit(f"1880 上「{tab['label']}」分頁有 {len(live_tabs)} 個")
    pins = {tab["id"]: live_tabs[0]["id"]}
    for n in nodes:
        if n["type"] == "tab":
            continue
        cand = [l for l in live if l["type"] == n["type"] and l.get("name", "") == n.get("name", "")
                and (l.get("z") == live_tabs[0]["id"] if n.get("z") else not l.get("z"))]
        if len(cand) != 1:
            raise SystemExit(f"{n['type']}「{n.get('name', '')}」在 1880 上有 {len(cand)} 個，無法一對一配對")
        pins[n["id"]] = cand[0]["id"]
    return {a: b for a, b in pins.items() if a != b}


def main(argv: list[str] | None = None) -> None:
    import sys
    argv = sys.argv[1:] if argv is None else argv
    nodes = build()
    if "--sync-ids" in argv:
        pins = sync_ids(nodes, _load_live_flows())
        PIN_FILE.write_text(json.dumps(pins, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"已寫入 {PIN_FILE}：{len(pins)} 個 ID 跟產生的不同")
    else:
        pins = json.loads(PIN_FILE.read_text(encoding="utf-8")) if PIN_FILE.exists() else {}
    nodes = remap_ids(nodes, pins)
    ids = [n["id"] for n in nodes]
    assert len(ids) == len(set(ids)), "節點 ID 重複"
    wired = {w for n in nodes for out in n.get("wires", []) for w in out}
    missing = wired - set(ids)
    assert not missing, f"連線指到不存在的節點：{missing}"
    OUT.write_text(json.dumps(nodes, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"已產生 {OUT}（{len(nodes)} 個節點；換成 1880 ID {len(pins)} 個；交給 P4：動態 link call「{P4_LINK_NAME}」）")


if __name__ == "__main__":
    main()
