"""P4 事件標記的次數（v7.3）：用 node 實際執行產生出來的 Node-RED 程式（管理器 save_event_tag、語音合併、部署時補齊）。

起點是 09-27 使用者 P4 上的真實資料：「🎙 語音紀錄 6 筆」但只顯示「其他 × 2」、手動按的標籤沒有次數、
語音原句被塞在備註裡。沒有 node 就跳過。
"""

import json
import shutil
import subprocess

import pytest

from iot.voice.node_red import build_p4_patch as bp
from iot.voice.tests.test_p4_patch import _flows

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="沒有 node")

LEGACY = {
    "date": "2026/9/27",
    "tags": ["帶去看醫生", "其他", "情緒緊張", "陌生人來訪", "服藥/治療"],
    "note": "🎙 今天帶貓咪去看醫生；🎙 幫我記錄今天幫貓咪洗澡；🎙 記錄貓咪；🎙 幫我記錄貓咪甩；4",
    "voice": [
        {"ts": 1790441895.761, "category": "vet", "text": "今天帶貓咪去看醫生"},
        {"ts": 1790442086.929, "category": "other", "text": "幫我記錄今天幫貓咪洗澡"},
        {"ts": 1790442093.282, "category": "other", "text": "記錄貓咪"},
        {"ts": 1790442474.53, "category": "other", "text": "幫我記錄貓咪甩"},
        {"ts": 1790452985.399, "report_id": 9, "category": "other", "label": "其他", "text": "今天肚子痛"},
        {"ts": 1790453006.871, "report_id": 10, "category": "other", "label": "其他", "text": "今天一隻摔頭"},
    ],
    "counts": {"其他": 2},
}
OLD_MANUAL_DAY = {"date": "2026/9/8", "tags": ["測試資料", "情緒緊張"], "note": "只有手動"}

RUNNER = r"""
const fs = require('fs');
const S = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const store = { v2_event_tags: S.events, v2_user_settings: { baseline_days: 7 } };
const gfile = { get: k => store[k], set: (k, v) => { store[k] = JSON.parse(JSON.stringify(v)); } };
const sent = [];
const node = { status() {}, warn() {}, log() {}, send: m => sent.push(m) };
const global = { get: k => (k === 'gfile' ? gfile : null), set() {} };
const run = (code, msg) => new Function('msg', 'node', 'global', 'setTimeout', code)(msg, node, global, f => f());
const mgr = msg => run(S.mgr, { payload: Object.assign({ action: 'save_event_tag', date: '2026/9/27', note: '' }, msg) });
const voice = (category, label, kind) => run(S.merge, { payload: { text: '句子', ts: 1790460000,
    reports: [{ report_id: 99, kind, category, label, day: '2026-09-27' }] } });
const day = d => JSON.parse(JSON.stringify(store.v2_event_tags.find(e => e.date === d)));
const out = {};
run(S.init, {});                                   // 部署：補齊舊資料
out.afterInit = day('2026/9/27'); out.oldManual = day('2026/9/8'); out.initSent = sent.length;
const before = JSON.stringify(store.v2_event_tags);
run(S.init, {}); out.initIdempotent = JSON.stringify(store.v2_event_tags) === before;
out.mgrRet = mgr({ tags: ['情緒緊張'] });            // 手動按一次「情緒緊張」
out.afterManual = day('2026/9/27');
voice('vomit', '嘔吐', 'observation');              // 語音記一次嘔吐
voice('vet', '看醫生', 'event');                     // 語音記一次看醫生（P4 標籤是「帶去看醫生」）
out.afterVoice = day('2026/9/27');
mgr({ tags: ['其他', '換了貓糧'], note: '晚上補記' }); // 手動再按一次（兩個標籤＋備註）
out.final = day('2026/9/27');
mgr({ tags: [], note: '' });                          // 什麼都沒選：不記
out.noop = JSON.stringify(day('2026/9/27')) === JSON.stringify(out.final);
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    mgr, _p4, _li, merge, _ret = bp.build(_flows())
    start = mgr["func"].index(bp.MGR_BRANCH_START)
    branch = mgr["func"][start:mgr["func"].index("\n}", start) + 2]
    d = tmp_path_factory.mktemp("p4")
    (d / "state.json").write_text(json.dumps({
        "events": [OLD_MANUAL_DAY, LEGACY],
        "mgr": "const action = msg.payload.action;\n" + branch + "\nreturn [null, null];",
        "merge": merge["func"], "init": merge["initialize"],
    }, ensure_ascii=False), encoding="utf-8")
    (d / "run.js").write_text(RUNNER, encoding="utf-8")
    p = subprocess.run([NODE, str(d / "run.js"), str(d / "state.json")], capture_output=True, text=True,
                       encoding="utf-8", timeout=30)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def test_deploy_fixes_legacy_counts(result):
    e = result["afterInit"]
    # 6 筆語音：看醫生 1、其他 5；以前手動按的三個各 1
    assert e["counts"] == {"帶去看醫生": 1, "其他": 5, "情緒緊張": 1, "陌生人來訪": 1, "服藥/治療": 1}
    assert all(v["label"] for v in e["voice"]) and len(e["voice"]) == 6
    assert e["note"] == "4"                                      # 備註裡的語音原句拿掉（清單裡有），飼主打的字留著
    assert e["manual"] == [{"ts": None, "tags": ["情緒緊張", "陌生人來訪", "服藥/治療"], "legacy": True}]
    assert result["initSent"] == 1                               # 部署時推一次 P4
    assert result["initIdempotent"]
    assert result["oldManual"] == OLD_MANUAL_DAY                 # 語音沒碰過的日子不動


def test_manual_save_counts_and_refreshes_p4(result):
    e = result["afterManual"]
    assert e["counts"]["情緒緊張"] == 2
    ret = result["mgrRet"]
    assert ret[0] is None and ret[1] is None                    # 基線重算、Discord 的線不動
    tags = next(x for x in ret[2]["payload"]["v2_event_tags"] if x["date"] == "2026/9/27")
    assert tags["counts"]["情緒緊張"] == 2                       # 輸出 3 直接把新資料送給 P4


def test_voice_and_manual_mix(result):
    e = result["afterVoice"]
    assert e["counts"]["嘔吐"] == 1 and e["counts"]["帶去看醫生"] == 2 and "嘔吐" in e["tags"]
    f = result["final"]
    assert f["counts"] == {"帶去看醫生": 2, "其他": 6, "情緒緊張": 2, "陌生人來訪": 1, "服藥/治療": 1,
                           "嘔吐": 1, "換了貓糧": 1}
    assert f["note"] == "4；晚上補記"
    # 次數一定等於紀錄筆數
    n_voice = len(f["voice"])
    n_manual_tags = sum(len(m["tags"]) for m in f["manual"])
    assert sum(f["counts"].values()) == n_voice + n_manual_tags
    assert result["noop"]
