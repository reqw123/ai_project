"""橋接分頁產生器：換成 1880 上實際的 ID（09-27：使用者匯入時 Node-RED 換過 ID，「取代」匯入要靠同一批 ID）。"""

import pytest

from iot.voice.node_red import build_flow as bf


def _live_copy(nodes, prefix="live"):
    """模擬 1880：同樣的節點，但每個 ID 都被 Node-RED 換掉。"""
    m = {n["id"]: f"{prefix}{i:02d}" for i, n in enumerate(nodes)}
    return bf.remap_ids(nodes, m), m


def test_sync_ids_maps_every_node_by_type_and_name():
    nodes = bf.build()
    live, expect = _live_copy(nodes)
    pins = bf.sync_ids(nodes, live)
    assert pins == expect
    pinned = bf.remap_ids(nodes, pins)
    assert pinned == live                                   # 換完跟 1880 一模一樣（含 wires、catch scope、broker）


def test_remap_ids_updates_all_references():
    nodes = bf.build()
    pinned = bf.remap_ids(nodes, _live_copy(nodes)[1])
    ids = {n["id"] for n in pinned}
    for n in pinned:
        assert not n.get("z") or n["z"] in ids
        assert all(w in ids for port in n.get("wires", []) for w in port)
        assert all(s in ids for s in n.get("scope") or [])
        if n["type"].startswith("mqtt ") and n.get("broker"):
            assert n["broker"] in ids


def test_sync_ids_refuses_ambiguous_match():
    nodes = bf.build()
    live, _ = _live_copy(nodes)
    dup = dict(next(n for n in live if n["type"] == "function"), id="extra")
    with pytest.raises(SystemExit):
        bf.sync_ids(nodes, live + [dup])                    # 1880 上有兩個同名節點：不猜，直接停


def test_sync_ids_needs_the_tab():
    nodes = bf.build()
    live = [n for n in _live_copy(nodes)[0] if n["type"] != "tab"]
    with pytest.raises(SystemExit):
        bf.sync_ids(nodes, live)
