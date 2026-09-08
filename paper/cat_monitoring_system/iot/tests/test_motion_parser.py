import pytest

from iot.sensors.base import ParseError
from iot.sensors.motion import MotionParser


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"active": True}, True),
        ({"active": False}, False),
        ({"motion": 1}, True),
        ({"motion": 0}, False),
        ({"state": "on"}, True),
        ({"state": "off"}, False),
        ({"event": "enter"}, True),
        ({"event": "leave"}, False),
        ({"detected": "yes"}, True),
    ],
)
def test_parse_variants(payload, expected):
    assert MotionParser().parse("hall", payload).active is expected


def test_parse_sets_source_and_kind():
    e = MotionParser().parse("hall", {"active": True})
    assert e.source_id == "hall"
    assert e.kind == "motion"


def test_parse_rejects_unknown():
    with pytest.raises(ParseError):
        MotionParser().parse("hall", {"foo": "bar"})
