"""把一句飼主的話（語音辨識後的文字）分類成 0～多筆紀錄。純函式、無副作用、不碰 I/O。

判斷規則（依序；論文第四章 E1 評估的就是這支）
────────────────────────────────────────────
1. **正規化**：常見簡體字轉繁體（Whisper 偶爾輸出簡體）、去掉空白與標點。
2. **排除**：
   * 問句（「貓咪今天有沒有吐？」「…嗎／呢」）——問句是想得到回答，不是回報；
   * 取消／否定整句（「不要記錄」「取消」「刪掉」）。
3. **主詞規則**：句子要提到貓（``VoiceReportConfig.CAT_ALIASES``：貓／猫／咪…），或有
   「記錄／紀錄／記一下」觸發詞。避免「我今天去看醫生」「我很緊張」被記成貓的事件。
4. **類別比對**：逐類比對 ontology 的樣式；一般樣式前面緊接「沒／沒有／不／未／別」
   視為否定、不算（「貓咪今天沒有吐」）；``negation_ok`` 的樣式本身就是否定說法（「沒吃」）。
   一句可以同時命中多類（「貓咪吐了而且拉肚子」→ 嘔吐＋如廁異常），每類一筆。
5. **其他事件**：有「記錄」觸發詞但對不上任何類別 → 一筆 ``other`` 事件（原句當備註）。
6. **哪一天**：「昨天／昨晚」→ 前一天、「前天」→ 前兩天，其餘（今天、剛剛、沒說）→ 當天。

明確記錄模式（``explicit=True``，v7.2 起語音終端只用這個）
────────────────────────────────────────────────────
飼主先說觸發句「我要記錄貓咪」（語音分頁判斷、ESP32 提示「請說貓咪發生了什麼事」），下一句才送來，
或觸發句後面直接接內容（「我要記錄貓咪，牠剛剛吐了」→ 送「牠剛剛吐了」）。飼主已經表明要記錄，所以：
* 不做問句排除、不做主詞規則（「剛剛吐了」不用再說一次貓咪）；
* 說「取消／算了／不用了」→ ``cancelled``（ESP32 唸「好的，已取消記錄。」）；
* 對不上任何類別 → 一筆 ``other``（原句當備註），不會丟掉飼主想記的事。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterable, Optional

from iot.voice import ontology
from iot.voice.config import VoiceReportConfig as _C

# Whisper 偶爾輸出簡體；只轉這個功能用得到的字（樣式、主詞、時間詞、否定詞）
_S2T = str.maketrans(dict(pair for pair in (
    ("猫", "貓"), ("医", "醫"), ("药", "藥"), ("换", "換"), ("饲", "飼"), ("粮", "糧"), ("来", "來"),
    ("访", "訪"), ("紧", "緊"), ("张", "張"), ("吓", "嚇"), ("惊", "驚"), ("烟", "煙"), ("装", "裝"),
    ("诊", "診"), ("预", "預"), ("针", "針"), ("检", "檢"), ("结", "結"), ("扎", "紮"), ("绝", "絕"),
    ("干", "乾"), ("饭", "飯"), ("没", "沒"), ("进", "進"), ("厕", "廁"), ("泻", "瀉"), ("痒", "癢"),
    ("头", "頭"), ("脸", "臉"), ("发", "發"), ("秃", "禿"), ("摇", "搖"), ("脏", "髒"), ("乱", "亂"),
    ("软", "軟"), ("懒", "懶"), ("虚", "虛"), ("无", "無"), ("气", "氣"), ("动", "動"), ("稳", "穩"),
    ("记", "記"), ("录", "錄"), ("纪", "紀"), ("给", "給"), ("这", "這"), ("里", "裡"), ("还", "還"),
    ("过", "過"), ("后", "後"), ("吗", "嗎"), ("么", "麼"), ("为", "為"), ("时", "時"), ("间", "間"),
    ("晚", "晚"), ("疗", "療"), ("吃", "吃"), ("胃", "胃"), ("觉", "覺"), ("惧", "懼"), ("虑", "慮"),
    ("雷", "雷"), ("鞭", "鞭"), ("搬", "搬"), ("诊", "診"), ("术", "術"), ("疫", "疫"), ("跛", "跛"),
    ("欢", "歡"), ("见", "見"), ("刚", "剛"), ("才", "才"), ("删", "刪"), ("错", "錯"), ("图", "圖"),
    ("儿", "兒"), ("会", "會"), ("说", "說"), ("帮", "幫"), ("东", "東"), ("西", "西"), ("毛", "毛"),
    ("状", "狀"), ("况", "況"), ("喂", "餵"), ("闹", "鬧"), ("吵", "吵"), ("恹", "懨"), ("鸣", "鳴"),
) if pair[0] != pair[1]))
_PUNCT = re.compile(r"[\s，。、！？!?,.：:；;「」『』\"'～~…（）()\-—]+")
_QUESTION = re.compile(r"(嗎|呢|吧)$|有沒有|是不是|會不會|要不要|怎麼辦|為什麼|多少|幾次|幾天")
_QUESTION_MARK = re.compile(r"[?？]")
_CANCEL = re.compile(r"不要記|別記|不用記|取消|刪掉|刪除|記錯")
# 明確模式的取消：開頭就是取消的說法、而且整句很短（「算了，不用記了」「取消」）；長句裡的「不要」不算
_CANCEL_EXPLICIT = re.compile(r"^(那|好|嗯|喔|欸)?(取消|算了|不用了|不用記|不要記|不記了|沒事|先不要|不要了|刪掉|記錯)")
_CANCEL_EXPLICIT_MAX = 10
_RECORD_TRIGGER = re.compile(r"(幫我)?(記錄|紀錄|記一下|記下來|記起來)")
_NEGATION_BEFORE = re.compile(r"(沒有?|不|未|別|並沒有?)(再|在|太|有)?$")
_DAY_OFFSET = ((re.compile(r"前天"), -2), (re.compile(r"昨天|昨晚|昨日|昨夜"), -1))


def normalize(text: str) -> str:
    """簡轉繁（有限字集）＋去空白標點。"""
    return _PUNCT.sub("", str(text or "").translate(_S2T))


@dataclass(frozen=True)
class Report:
    kind: str                   # ontology.EVENT / OBSERVATION
    category: str
    label: str
    day: str                    # 這件事是哪一天（ISO，YYYY-MM-DD）
    day_offset: int             # 0＝當天、-1＝昨天、-2＝前天
    matched: str                # 命中的那段文字（除錯、論文附錄）
    related_behaviors: tuple[str, ...] = ()
    confirm: str = ""


@dataclass(frozen=True)
class Classification:
    text: str                   # 原句
    normalized: str
    reports: tuple[Report, ...] = ()
    reason: str = ""            # 沒有紀錄時的原因：empty / question / cancel / no_subject / no_category
    explicit: bool = False      # 明確記錄模式（飼主先說了「我要記錄貓咪」）
    cancelled: bool = False     # 明確記錄模式裡說「取消／算了」

    @property
    def matched(self) -> bool:
        return bool(self.reports)


def _day_offset(t: str) -> int:
    for pattern, offset in _DAY_OFFSET:
        if pattern.search(t):
            return offset
    return 0


def _first_hit(t: str, cat: ontology.Category) -> Optional[str]:
    """回傳這類第一個「沒被否定」的命中文字；沒有回 None。"""
    for pattern in cat.compiled_neg_ok:
        m = pattern.search(t)
        if m:
            return m.group(0)
    for pattern in cat.compiled:
        for m in pattern.finditer(t):
            if not _NEGATION_BEFORE.search(t[max(0, m.start() - 4):m.start()]):
                return m.group(0)
    return None


def classify(
    text: str,
    now: Optional[float] = None,
    cat_aliases: Optional[Iterable[str]] = None,
    explicit: bool = False,
) -> Classification:
    """``now``＝說話時間（epoch 秒，預設現在；用本機時區決定「哪一天」）。
    ``explicit``＝明確記錄模式（見模組說明）。"""
    t = normalize(text)
    if not t:
        return Classification(text=text, normalized=t, reason="empty", explicit=explicit)
    if explicit:
        if _CANCEL_EXPLICIT.search(t) and len(t) <= _CANCEL_EXPLICIT_MAX:
            return Classification(text=text, normalized=t, reason="cancel", explicit=True, cancelled=True)
        triggered = True   # 對不上類別也記成「其他」
    else:
        if _QUESTION_MARK.search(str(text or "")) or _QUESTION.search(t):
            return Classification(text=text, normalized=t, reason="question")
        if _CANCEL.search(t):
            return Classification(text=text, normalized=t, reason="cancel")
        aliases = tuple(cat_aliases) if cat_aliases is not None else _C.CAT_ALIASES
        triggered = bool(_RECORD_TRIGGER.search(t))
        if not triggered and not any(a and a in t for a in aliases):
            return Classification(text=text, normalized=t, reason="no_subject")

    offset = _day_offset(t)
    when = datetime.fromtimestamp(time.time() if now is None else now)
    day = (when.date() + timedelta(days=offset)).isoformat()

    reports = []
    for cat in ontology.ALL_CATEGORIES:
        hit = _first_hit(t, cat)
        if hit:
            reports.append(Report(
                kind=cat.kind, category=cat.key, label=cat.label, day=day, day_offset=offset,
                matched=hit, related_behaviors=cat.related_behaviors, confirm=cat.confirm,
            ))
    if not reports and triggered:
        other = ontology.get("other")
        reports.append(Report(
            kind=other.kind, category=other.key, label=other.label, day=day, day_offset=offset,
            matched="", confirm=other.confirm,
        ))
    return Classification(
        text=text, normalized=t, reports=tuple(reports),
        reason="" if reports else "no_category", explicit=explicit,
    )


def day_of(ts: float) -> date:
    return datetime.fromtimestamp(ts).date()
