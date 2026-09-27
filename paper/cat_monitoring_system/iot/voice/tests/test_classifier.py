"""分類規則（論文第四章 E1 的種子測試集：之後實際錄音的句子也加在這裡）。"""

from datetime import datetime

import pytest

from iot.voice.classifier import classify, normalize

NOW = datetime(2026, 9, 27, 21, 30).timestamp()


def cats(text):
    return sorted(r.category for r in classify(text, now=NOW).reports)


# ── A. 情境事件 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text, expected", [
    ("今天帶貓咪去看醫生", ["vet"]),
    ("貓咪去動物醫院打預防針", ["vet"]),
    ("貓咪今天回診", ["vet"]),
    ("貓咪今天結紮了", ["vet"]),
    ("貓咪換飼料了", ["food"]),
    ("幫貓咪換了新的飼料", ["food"]),
    ("貓咪今天開始吃新罐頭", ["food"]),
    ("今天家裡有客人，貓咪一直躲起來", ["stranger", "stress"]),
    ("朋友來家裡玩貓咪很緊張", ["stranger", "stress"]),
    ("外面打雷貓咪嚇到了", ["stress"]),
    ("貓咪今天在吃藥", ["medicine"]),
    ("剛剛幫貓咪點眼藥", ["medicine"]),
    ("幫我記錄今天幫貓咪洗澡", ["other"]),
    ("記錄一下今天搬家具", ["other"]),
])
def test_events(text, expected):
    assert cats(text) == expected


# ── B. 飼主觀察 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text, expected", [
    ("貓咪剛剛吐了", ["vomit"]),
    ("貓咪又吐毛球", ["vomit"]),
    ("貓咪今天沒吃飯", ["appetite"]),
    ("貓咪最近吃很少", ["appetite"]),
    ("貓咪一直喝水", ["appetite"]),
    ("貓咪拉肚子", ["litter"]),
    ("貓咪在貓砂盆蹲很久", ["litter"]),
    ("貓咪今天都沒尿尿", ["litter"]),
    ("貓咪一直抓耳朵", ["itch"]),
    ("貓咪一直舔肚子", ["itch"]),
    ("貓咪一直甩頭", ["head_shake"]),
    ("貓咪今天沒精神", ["lethargy"]),
    ("貓咪整天都在睡", ["lethargy"]),
    ("貓咪走路一跛一跛的", ["mobility"]),
    ("貓咪吐了而且拉肚子", ["litter", "vomit"]),
])
def test_observations(text, expected):
    assert cats(text) == expected


# ── 不應該記錄 ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text, reason", [
    ("", "empty"),
    ("貓咪今天有沒有吐？", "question"),
    ("貓咪今天還好嗎", "question"),
    ("為什麼貓咪一直抓耳朵", "question"),
    ("不要記錄貓咪吐了", "cancel"),
    ("取消剛剛那筆", "cancel"),
    ("我今天去看醫生", "no_subject"),       # 主詞規則：不是貓的事
    ("我好緊張", "no_subject"),
    ("請幫我開燈", "no_subject"),
    ("貓咪好可愛", "no_category"),
    ("貓咪今天沒有吐", "no_category"),       # 否定
    ("貓咪沒再抓耳朵了", "no_category"),
    ("打開貓咪那套程式", "no_category"),
])
def test_not_recorded(text, reason):
    c = classify(text, now=NOW)
    assert not c.matched
    assert c.reason == reason


# ── 哪一天 ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text, day, offset", [
    ("今天帶貓咪去看醫生", "2026-09-27", 0),
    ("昨天帶貓咪去看醫生", "2026-09-26", -1),
    ("貓咪昨晚吐了", "2026-09-26", -1),
    ("前天貓咪換飼料", "2026-09-25", -2),
])
def test_day(text, day, offset):
    (r,) = classify(text, now=NOW).reports
    assert (r.day, r.day_offset) == (day, offset)


def test_simplified_chinese_from_whisper():
    assert cats("猫咪今天去看医生") == ["vet"]
    assert cats("猫咪吐了") == ["vomit"]
    assert cats("帮我记录猫咪洗澡") == ["other"]


def test_custom_cat_name():
    assert not classify("小花吐了", now=NOW).matched
    assert [r.category for r in classify("小花吐了", now=NOW, cat_aliases=("小花",)).reports] == ["vomit"]


def test_related_behaviors_and_confirm():
    (r,) = classify("貓咪一直抓耳朵", now=NOW).reports
    assert r.related_behaviors == ("lick", "scratch")
    assert r.confirm == "已記錄：抓癢或舔毛。"
    assert r.matched  # 命中的那段文字（附錄、除錯用）


def test_normalize():
    assert normalize("貓咪，今天 吐了！") == "貓咪今天吐了"
