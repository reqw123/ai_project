"""協調層：一句話 → 分類 → 存檔 → 組回覆。純邏輯，不碰 MQTT（runner 注入 publish）。

``ReportService`` 的三個入口都吃「已經解析好的 dict」、回傳要發出去的 dict，
測試可以直接呼叫，不需要 broker 或 paho。
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from datetime import datetime
from typing import Any, Optional

from iot.voice import ontology, store
from iot.voice.classifier import classify
from iot.voice.config import VoiceReportConfig as _C

_log = logging.getLogger(__name__)

_MAX_TEXT = 300


class ReportService:
    def __init__(self, db_path: Optional[str] = None, log_unmatched: Optional[bool] = None,
                 recent_limit: Optional[int] = None, dedup_size: Optional[int] = None):
        self._db_path = db_path
        self._log_unmatched = _C.LOG_UNMATCHED if log_unmatched is None else log_unmatched
        self._recent_limit = recent_limit or _C.RECENT_LIMIT
        self._dedup: OrderedDict[str, dict] = OrderedDict()
        self._dedup_size = dedup_size or _C.DEDUP_CACHE_SIZE

    # ── 一句話 ───────────────────────────────────────────────────────────
    def handle_utterance(self, payload: dict[str, Any]) -> dict[str, Any]:
        """回傳分類結果（一定有 ``id``、``ok``、``matched``、``reports``、``reply``）。"""
        request_id = str(payload.get("id") or "").strip()[:64]
        text = str(payload.get("text") or "").strip()[:_MAX_TEXT]
        if not request_id:
            return {"id": "", "ok": False, "matched": False, "reports": [], "reply": "", "error": "缺少 id"}
        cached = self._dedup.get(request_id)
        if cached is not None:
            return cached
        ts = _ts_seconds(payload.get("ts"))
        device_id = str(payload.get("deviceId") or "")[:48]
        source = str(payload.get("source") or "")[:24]
        mode = "explicit" if payload.get("mode") == "explicit" else ""   # v7.2：飼主先說了「我要記錄貓咪」

        result = classify(text, now=ts, explicit=bool(mode))
        out: dict[str, Any] = {
            "id": request_id, "ok": True, "matched": result.matched, "reason": result.reason,
            "text": text, "ts": ts, "mode": mode, "cancelled": result.cancelled,
            "reports": [], "reply": "", "confirm_key": "",
        }
        if result.cancelled:
            out.update(reply="好的，已取消記錄。", confirm_key="cancel")
        try:
            ids = store.save(request_id, ts, result, device_id=device_id, source=source,
                             log_unmatched=self._log_unmatched or bool(mode), db_path=self._db_path, mode=mode)
        except Exception as exc:  # noqa: BLE001 — 存檔失敗要讓 Node-RED 知道（韌體唸「沒有完成」）
            _log.warning("存檔失敗（%s）：%s", request_id, exc)
            out.update(ok=False, error="紀錄存檔失敗")
            return out
        for rid, r in zip(ids, result.reports):
            out["reports"].append({
                "report_id": rid, "kind": r.kind, "category": r.category, "label": r.label,
                "day": r.day, "day_offset": r.day_offset, "matched": r.matched,
                "related_behaviors": list(r.related_behaviors),
            })
        if result.reports:
            only = result.reports[0] if len(result.reports) == 1 else None
            out["confirm_key"] = only.category if only else "multi"
            out["reply"] = only.confirm if only else "已記錄：" + "、".join(r.label for r in result.reports) + "。"
            _log.info("紀錄 %s：「%s」→ %s", request_id, text, "、".join(r.label for r in result.reports))
        self._remember(request_id, out)
        return out

    def _remember(self, request_id: str, out: dict) -> None:
        self._dedup[request_id] = out
        while len(self._dedup) > self._dedup_size:
            self._dedup.popitem(last=False)

    # ── 刪除（儀表板）────────────────────────────────────────────────────
    def handle_delete(self, payload: dict[str, Any]) -> bool:
        try:
            report_id = int(payload.get("report_id"))
        except (TypeError, ValueError):
            return False
        restore = payload.get("restore") is True
        ok = store.set_deleted(report_id, deleted=not restore, db_path=self._db_path)
        _log.info("%s紀錄 #%s：%s", "復原" if restore else "刪除", report_id, "成功" if ok else "找不到")
        return ok

    # ── 儀表板用的最近紀錄 ────────────────────────────────────────────────
    def recent_snapshot(self, now: Optional[float] = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        today = datetime.fromtimestamp(now).date().isoformat()
        rows = store.recent_reports(self._recent_limit, db_path=self._db_path)
        return {
            "updated": now,
            "today": today,
            "today_counts": store.day_counts(today, db_path=self._db_path),
            "categories": ontology.describe(),
            "reports": [{
                "report_id": r["id"], "ts": r["ts"], "day": r["day"], "kind": r["kind"],
                "category": r["category"], "label": r["label"], "text": r["text"],
                "related_behaviors": [x for x in r["related"].split(",") if x],
                "device_id": r["device_id"], "source": r["source"],
            } for r in rows],
        }


def _ts_seconds(value: Any) -> float:
    """Node-RED 送毫秒（Date.now()），也接受秒；沒有／壞掉＝現在。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return time.time()
    if v <= 0:
        return time.time()
    return v / 1000.0 if v > 1e11 else v
