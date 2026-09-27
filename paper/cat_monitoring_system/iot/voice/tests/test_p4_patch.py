"""P4 修補（node_red/build_p4_patch.py）：產生修補節點、直接寫進完整流程檔（只動兩個節點＋加兩個、冪等）。"""

import json

import pytest

from iot.voice.node_red import build_p4_patch as bp
from iot.voice.node_red.build_flow import P4_LINK_IN_ID


def _flows():
    return [
        {"id": "tab2", "type": "tab", "label": "第2層 行為分析引擎"},
        {"id": "other", "type": "function", "z": "tab2", "name": "別的節點", "func": "return msg;", "wires": [[]]},
        {"id": "mgr", "type": "function", "z": "tab2", "name": "使用者設定管理器", "x": 100, "y": 100,
         "func": "if (action === 'save_event_tag') {\n" + bp.MGR_FROM + "\n}\n", "wires": [["x"], ["y"]]},
        {"id": "p4", "type": "ui_template", "z": "tab2", "name": "P4 個體基線面板", "x": 900, "y": 300,
         "format": "<style>" + bp.P4_CSS_FROM + "</style><div class=\"p4-evt-hist-row\"><span class=\"p4-evt-tags-text\">"
                   + bp.P4_TAGS_FROM + "</span><span class=\"p4-evt-note\" ng-if=\"e.note\"> — {{e.note}}</span></div>"
                   + "<script>" + bp.P4_WATCH + "})</script>", "wires": [["mgr"]]},
    ]


def test_build_changes_only_the_two_snippets():
    mgr, p4, link_in, merge, ret = bp.build(_flows())
    assert bp.MARK_V73 in mgr["func"] and "p4Recount(entry)" in mgr["func"] and "events.filter" not in mgr["func"]
    assert "{{p4TagText(e)}}" in p4["format"] and "scope.p4TagText=" in p4["format"]
    # 原本的輸出 1、2 不動；加輸出 3 → P4（存完事件標記直接更新畫面）
    assert mgr["wires"] == [["x"], ["y"], ["p4"]] and mgr["outputs"] == 3 and p4["wires"] == [["mgr"]]
    assert "p4Recount(e)" in merge["initialize"]                                    # 部署時補齊舊資料
    assert link_in["id"] == P4_LINK_IN_ID and link_in["z"] == "tab2" and link_in["wires"] == [[merge["id"]]]
    assert merge["wires"] == [["p4"], [ret["id"]]] and merge["z"] == "tab2" and merge["outputs"] == 2
    assert ret["type"] == "link out" and ret["mode"] == "return" and ret["name"] == bp.RETURN_NAME   # 動態 link call 的回傳
    assert "{{p4TagText(e)}}" in p4["format"] and "🎙 語音紀錄 {{e.voice.length}} 筆" in p4["format"]
    assert "'🎙 ' +" not in merge["func"] and ".slice(-200)" in merge["func"]   # 不再把原句塞進備註（會被截斷）
    # 已經修補過的節點再跑一次不會壞掉
    again = bp.build([_flows()[0], mgr, p4])
    assert again[0]["func"] == mgr["func"] and again[1]["format"] == p4["format"]


def test_upgrades_v72_manager_in_place():
    # 1880／repo 上已經是 v7.2（合併但不計數、不通知 P4）→ 整段換成 v7.3，其他分支一個字都不動
    flows = _flows()
    flows[2]["func"] = ("if (action === 'test_discord') {\n    return [null, msg];\n}\n\n"
                        "if (action === 'save_event_tag') {\n" + bp.MGR_TO + "\n    return [null, null];\n}\n\n"
                        "if (action === 'set_baseline_period') {\n    return [msg, null];\n}\n")
    mgr = bp.build(flows)[0]
    assert bp.MARK_V73 in mgr["func"] and bp.MGR_TO not in mgr["func"]
    assert mgr["func"].startswith("if (action === 'test_discord') {\n    return [null, msg];\n}\n\n")
    assert mgr["func"].endswith("if (action === 'set_baseline_period') {\n    return [msg, null];\n}\n")
    assert bp.build([flows[0], mgr, flows[3]])[0]["func"] == mgr["func"]           # 再跑一次不變


def test_existing_new_nodes_are_reused_by_name(tmp_path):
    # 匯入 1880 時 Node-RED 可能換了兩個新節點的 ID：再跑一次要沿用流程裡的 ID，不能重複加
    path = tmp_path / "flow.json"
    path.write_text(json.dumps(_flows(), ensure_ascii=False, indent=2), encoding="utf-8")
    bp.apply_to_file(str(path))
    flows = json.loads(path.read_text(encoding="utf-8"))
    for n in flows:   # 模擬 1880 換了 ID
        if n.get("name") == bp.LINK_IN_NAME:
            n["id"] = "live_link"; n["wires"] = [["live_merge"]]; n["x"] = 11
        elif n.get("name") == bp.MERGE_NAME:
            n["id"] = "live_merge"
    path.write_text(json.dumps(flows, ensure_ascii=False, indent=2), encoding="utf-8")
    assert bp.apply_to_file(str(path)) == []                                         # 沒有變動
    after = json.loads(path.read_text(encoding="utf-8"))
    assert len(after) == 7 and [n["id"] for n in after if n.get("name") == bp.LINK_IN_NAME] == ["live_link"]


def test_bridge_calls_p4_by_name_not_by_id():
    # 09-27：健康監測流程重新匯入時 Node-RED 換了 ID → 依 ID 的 link out 斷掉。改成動態 link call 依名稱呼叫
    from iot.voice.node_red import build_flow as bf
    nodes = bf.build()
    call = next(n for n in nodes if n["type"] == "link call")
    assert call["linkType"] == "dynamic" and call["links"] == []
    result = next(n for n in nodes if n.get("name") == "結果 → 回到語音分頁／交給 P4")
    assert 'target: "' + bp.LINK_IN_NAME + '"' in result["func"]
    assert not any(n["type"] == "link out" and n.get("mode") == "link" for n in nodes)   # 沒有依 ID 的連線


def test_build_refuses_unexpected_node():
    flows = _flows()
    flows[2]["func"] = "完全不同的程式"
    with pytest.raises(SystemExit):
        bp.build(flows)


def test_apply_to_file_only_touches_target_nodes(tmp_path):
    path = tmp_path / "flow.json"
    raw = json.dumps(_flows(), ensure_ascii=False, indent=2)
    path.write_text(raw, encoding="utf-8")
    changed = bp.apply_to_file(str(path))
    assert set(changed) == {"使用者設定管理器", "P4 個體基線面板", "← 飼主語音紀錄", "飼主語音紀錄 → 事件標記＋更新 P4", "↩ 回到飼主語音紀錄"}
    after = json.loads(path.read_text(encoding="utf-8"))
    assert [n["id"] for n in after[:4]] == ["tab2", "other", "mgr", "p4"]          # 原有順序不變
    assert after[1] == _flows()[1]                                                   # 別的節點一個字都沒動
    assert len(after) == 7
    text = path.read_text(encoding="utf-8")
    assert text == json.dumps(after, ensure_ascii=False, indent=2)                   # 格式照原檔（indent 2、中文不跳脫）
    assert bp.apply_to_file(str(path)) == [] and path.read_text(encoding="utf-8") == text   # 冪等
