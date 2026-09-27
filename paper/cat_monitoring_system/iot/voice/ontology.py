"""飼主語音紀錄的類別本體（ontology）——所有類別、中文標籤、觸發樣式的唯一定義處。

兩種紀錄
────────
* **event（情境事件，整天層級）**：會讓貓咪行為「非因疾病而改變」的已知生活事件，
  對應論文的混淆因子（Stella et al. 2011；Amat et al. 2016）。類別 key 刻意沿用
  Node-RED 儀表板既有的事件標記（``cat_health_v4_flow.json`` 的 ``v2_event_tags``：
  vet / food / stranger / stress / medicine / other），語音標記跟按鈕標記是同一套語意。
* **observation（飼主觀察，時間點層級）**：飼主親眼看到的異常徵象，帶時間戳，
  當作「弱標註（weak label）」與系統行為偏離預警做時間對照；也涵蓋攝影機拍不到、
  或不在五類行為（walk/lick/scratch/shake/stop）內的事件（嘔吐、食慾、如廁）。
  ``related_behaviors`` 標出它對應系統的哪些行為類別，供分析時對齊。

樣式規則
────────
樣式是正規表示式，比對「正規化後」的句子（繁體、去空白標點，見 classifier.py）。
``negation_ok=True`` 表示樣式本身就是否定說法（例如「沒吃」），不再做否定檢查；
其他樣式前面緊接「沒／沒有／不／未／別」時視為否定，不算（「貓咪今天沒有吐」）。

新增類別只要在這裡加一筆；classifier / store / Node-RED 儀表板都不用改
（韌體若要唸專屬確認句，另在韌體提示音表加一句，沒有就唸通用的「已記錄。」）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

EVENT = "event"
OBSERVATION = "observation"


@dataclass(frozen=True)
class Category:
    key: str
    kind: str                      # EVENT / OBSERVATION
    label: str                     # 中文標籤（儀表板、確認句用）
    patterns: tuple[str, ...]      # 正規表示式（比對正規化後的句子）
    negation_ok: tuple[str, ...] = ()   # 本身就是否定說法的樣式（不做否定檢查）
    related_behaviors: tuple[str, ...] = ()   # 對應系統行為類別（僅 observation）
    confirm: str = ""              # 確認句（Node-RED 顯示、韌體提示音同一句）
    compiled: tuple[re.Pattern, ...] = field(default=(), compare=False, repr=False)
    compiled_neg_ok: tuple[re.Pattern, ...] = field(default=(), compare=False, repr=False)


def _cat(key, kind, label, patterns, negation_ok=(), related=(), confirm=""):
    return Category(
        key=key, kind=kind, label=label, patterns=tuple(patterns),
        negation_ok=tuple(negation_ok), related_behaviors=tuple(related),
        confirm=confirm or f"已記錄：{label}。",
        compiled=tuple(re.compile(p) for p in patterns),
        compiled_neg_ok=tuple(re.compile(p) for p in negation_ok),
    )


# ── A. 情境事件（整天）────────────────────────────────────────────────────
EVENT_CATEGORIES: tuple[Category, ...] = (
    _cat("vet", EVENT, "看醫生", [
        r"看(醫生|獸醫)", r"(動物)?醫院", r"獸醫", r"(回|複)診",
        r"(打|施打)(預防針|疫苗)", r"疫苗", r"健康檢查|檢查身體|體檢",
        r"結紮|絕育|洗牙|抽血|住院|開刀|手術",
    ]),
    _cat("food", EVENT, "換飼料", [
        r"換(了|成)?(新的?|一?(種|款|個|包|袋))?(的)?(飼料|貓糧|糧|罐頭|乾乾|主食|吃的|食物|零食)",
        r"新(的)?(飼料|貓糧|罐頭|乾乾|主食)",
    ], confirm="已記錄：換飼料。"),
    _cat("stranger", EVENT, "有客人來訪", [
        r"(客人|訪客|陌生人|朋友|親戚|外人|別人|家人)(來|到|在)",
        r"有人來", r"(家裡|今天)有(客人|訪客|朋友|親戚|陌生人|人來)",
        r"(來了|有)(客人|訪客|陌生人)",
    ]),
    _cat("stress", EVENT, "情緒緊張", [
        r"緊張|害怕|焦慮|受(到)?(驚|驚嚇)|嚇到|被嚇",
        r"打雷|鞭炮|煙火|搬家(?!具)|施工|裝潢",   # 「搬家具」不是搬家
        r"躲(起來|在|到)",
    ]),
    _cat("medicine", EVENT, "服藥治療", [
        r"(吃|餵|服|給|投)(了)?藥", r"(擦|點|滴|上|換)(眼|耳)?藥",
        r"打針|治療",
    ]),
    # other：句子裡有「記錄」觸發詞，但對不上任何類別時才用（note＝原句），見 classifier.py
    _cat("other", EVENT, "其他事件", [], confirm="好的，已記錄。"),
)

# ── B. 飼主觀察（時間點）──────────────────────────────────────────────────
OBSERVATION_CATEGORIES: tuple[Category, ...] = (
    _cat("vomit", OBSERVATION, "嘔吐", [
        r"嘔吐|乾嘔|反芻", r"(在|有|又|一直|狂|剛剛|剛才)?吐(了|出|毛球|東西|過|很多)",
        r"(又|一直|狂|在|有)吐",
    ]),
    _cat("appetite", OBSERVATION, "食慾或飲水異常", [
        r"吃(得)?(很|比較|好|超)?少", r"食慾(不好|不佳|差|變差|很差)", r"挑食",
        r"喝(很多|超多|好多)水|一直喝水|水喝(很|超|好)?多",
    ], negation_ok=[
        r"(沒|不|不太|不怎麼|都沒|都不)(有)?(吃|進食|喝水)", r"沒(有)?胃口", r"不想吃",
    ], related=()),
    _cat("litter", OBSERVATION, "如廁異常", [
        r"拉肚子|腹瀉|軟便|水便|便秘|大不出來|尿不出來|血尿|尿血|頻尿",
        r"(尿|尿尿|小便)(很多|變多|很頻繁|次數變多)", r"亂(尿|尿尿|大便|上廁所)",
        r"(廁所|貓砂(盆)?)(裡)?(待|蹲)(很久|好久|太久)",
    ], negation_ok=[
        r"(沒|都沒)(有)?(大便|尿尿|上廁所|排便)",
    ]),
    _cat("itch", OBSERVATION, "抓癢或舔毛", [
        r"(一直|狂|很常|拼命|不停|常常)(在)?(抓|舔|咬)",
        r"抓癢|搔癢|抓(耳朵|脖子|臉|身體)", r"舔毛(舔)?(很久|很多|不停)|過度理毛",
        r"掉毛|禿(了|一塊)|皮膚(紅|發炎|破)|咬毛",
    ], related=("lick", "scratch")),
    _cat("head_shake", OBSERVATION, "甩頭", [
        r"甩頭|甩耳朵|搖頭晃腦|歪頭", r"耳朵(很|有點)?(髒|臭|紅)",
        r"(一直|狂|又|在)?摔頭",   # Whisper 常把「甩頭」聽成「摔頭」（09-27 實際紀錄：「今天一隻摔頭」）
    ], related=("shake",)),
    _cat("lethargy", OBSERVATION, "精神不好", [
        r"無精打采|懶洋洋|病懨懨|沒力氣|虛弱",
        r"(一直|整天)(都)?(在)?睡", r"活動(力|量)?(變)?(少|差)",
    ], negation_ok=[
        r"沒(有)?精神", r"(不太|都不|不想)動",
    ], related=("walk", "stop")),
    _cat("mobility", OBSERVATION, "走路異常", [
        r"跛(腳)?|一跛一跛|瘸", r"走路(怪|奇怪|怪怪的|不穩|歪)", r"跳不上去",
    ], related=("walk",)),
)

ALL_CATEGORIES: tuple[Category, ...] = EVENT_CATEGORIES + OBSERVATION_CATEGORIES
_BY_KEY = {c.key: c for c in ALL_CATEGORIES}


def get(key: str) -> Category | None:
    return _BY_KEY.get(key)


def describe() -> list[dict]:
    """給 Node-RED 儀表板、論文附錄用：類別清單（不含正規表示式物件）。"""
    return [
        {"key": c.key, "kind": c.kind, "label": c.label, "confirm": c.confirm,
         "related_behaviors": list(c.related_behaviors)}
        for c in ALL_CATEGORIES
    ]
