"""IoT 子系統（感測器、飼主語音紀錄）的啟動／停止／狀態——設定視窗「📡 IoT 子系統」分頁的邏輯層（不含 Tk）。

跟 ``process_manager.py``（main.py／獨立腳本）刻意分開：
* 那邊同一時間只准一支行程（避免搶 GPU、輸出混在一起）；IoT 服務很輕、要跟 main.py 同時跑，**每個服務各自一個行程**。
* IoT 服務在背景執行：輸出寫到紀錄檔（``paper/logs/iot/<服務>.log``）、不佔終端機面板；從狀態檔＋行程清單認得出來
  （在 cmd 手動執行的也認得）。09-27 起**關掉設定視窗時會強制停止全部 IoT 服務**（``stop_all``，使用者要求）。
* 只用 subprocess 啟動 ``python -m iot``／``python -m iot.voice``，**不 import** ``iot`` 套件（維持 iot 子系統
  「主系統不 import 它」的規則，見 ``cat_monitoring_system/iot/README.md``）。

各感測器分開啟動靠 ``CAT_MONITORING_IOT_KINDS``（``iot/config.py``）：每種感測器一個 ``python -m iot`` 行程，
client id 自動加上種類，互不干擾；手動 ``python -m iot``（不設 KINDS）＝全部感測器一個行程，這裡也認得。

09-27 起：感測器服務啟動時會帶上「⚙ 參數設定」存的 ``iot/config.py`` 覆寫（``iot_config_overrides.py``），
改完用 ``restart_sensors()`` 重新啟動才會生效。
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from settings_gui import iot_config_overrides

try:
    import psutil
except ImportError:  # pragma: no cover — yolo_new 環境有（ultralytics 相依）
    psutil = None

_PAPER_DIR = Path(__file__).resolve().parents[1]
CAT_DIR = _PAPER_DIR / "cat_monitoring_system"          # python -m iot 的工作目錄
LOG_DIR = _PAPER_DIR / "logs" / "iot"
STATE_FILE = Path(__file__).resolve().parent / "iot_services_state.json"
_LOG_ROTATE_BYTES = 5 * 1024 * 1024
_KINDS_ENV = "CAT_MONITORING_IOT_KINDS"


@dataclass(frozen=True)
class Service:
    key: str
    title: str
    desc: str
    module: str                          # python -m <module>
    kind: str = ""                       # 感測器種類（python -m iot 的 KINDS）；語音紀錄＝空
    env: dict = field(default_factory=dict)

    @property
    def log_path(self) -> Path:
        return LOG_DIR / f"{self.key}.log"


def _sensor(kind, title, desc):
    return Service(key=kind, title=title, desc=desc, module="iot", kind=kind, env={_KINDS_ENV: kind})


# 卡片每列兩張、照這個順序排（09-27 使用者要求）：環境＋體表溫度同一列（溫濕度和體表溫度是同一台外出包
# cat-petbox-carrier；環境另有 esp32_room 送空氣品質＋光照，所以環境卡片會列兩個 IP），語音紀錄排最後
SERVICES: tuple[Service, ...] = (
    _sensor("env", "🌡 環境感測",
            "溫濕度（外出包，跟體表溫度同一台）＋空氣品質 MQ-135、光照（esp32_room）→ cat/iot/env/…；"
            "超出舒適範圍發告警"),
    _sensor("bodytemp", "🐾 體表溫度",
            "MLX90614 非接觸測溫（外出包，跟環境感測同一台 → cat/iot/bodytemp/carrier）：初篩用，不是發燒診斷"),
    _sensor("motion", "👣 移動偵測",
            "PIR 移動感測（HC-SR501，esp32_room → cat/iot/motion/…）：貓有沒有經過某個固定位置"),
    _sensor("weight", "⚖ 食盆秤重",
            "HX711 秤重（esp32_weight → cat/iot/weight/…）：進食／加料事件、太久沒進食告警"),
    Service(key="voice", title="🎙 飼主語音紀錄", module="iot.voice",
            desc="對語音終端說「我要記錄貓咪」→ 分類、存檔，寫進健康監測 P4 的事件標記（語音辨識 v7.2）"),
)
_BY_KEY = {s.key: s for s in SERVICES}


def get(key: str) -> Service:
    return _BY_KEY[key]


# ── 狀態 ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Status:
    running: bool
    pid: int = 0
    managed: bool = False            # True＝這個設定視窗啟動的；False＝外面（cmd）啟動的
    shared_kinds: tuple = ()         # 同一個 python -m iot 行程還處理哪些感測器（全部一起跑時）
    started_at: float = 0.0

    def text(self) -> str:
        if not self.running:
            return "已停止"
        s = f"執行中（PID {self.pid}"
        if not self.managed:
            s += "，外部啟動"
        if len(self.shared_kinds) > 1:
            s += "，" + "／".join(get(k).title.split(" ", 1)[-1] for k in self.shared_kinds if k in _BY_KEY) + " 同一個行程"
        return s + "）"


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def _cmd_module(cmdline: list) -> Optional[str]:
    """``python -m iot.voice`` → "iot.voice"；不是 python -m 就回 None。"""
    if not cmdline or "python" not in Path(cmdline[0]).name.lower():
        return None
    for i, a in enumerate(cmdline[:-1]):
        if a == "-m":
            return cmdline[i + 1]
    return None


def _kinds_of(proc, known: tuple) -> tuple:
    """python -m iot 行程處理哪些感測器（讀它的環境變數；讀不到或沒設＝這個模組的全部感測器）。"""
    try:
        raw = proc.environ().get(_KINDS_ENV, "")
    except Exception:  # noqa: BLE001 — AccessDenied 等
        raw = ""
    kinds = tuple(k.strip() for k in raw.split(",") if k.strip() in known)
    return kinds or known


def scan() -> dict[str, Status]:
    """所有服務目前的狀態。先看狀態檔記的（這個視窗啟動的），再掃行程清單補上外面啟動的。"""
    out = {s.key: Status(False) for s in SERVICES}
    if psutil is None:
        return out
    state = _load_state()
    managed = {}
    for key, rec in state.items():
        try:
            p = psutil.Process(int(rec["pid"]))
            if p.is_running() and abs(p.create_time() - float(rec["create_time"])) < 1.0:
                managed[p.pid] = key
        except Exception:  # noqa: BLE001 — 行程已結束、PID 被重用
            continue
    by_module: dict[str, list] = {}
    for s in SERVICES:
        by_module.setdefault(s.module, []).append(s)
    for p in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            svcs = by_module.get(_cmd_module(p.info["cmdline"] or []))
            if not svcs:
                continue
            is_managed = p.pid in managed
            known = tuple(s.kind for s in svcs if s.kind)
            if known:   # 感測器 hub：看它處理哪些種類（全部一起跑的行程，每張卡片都算它）
                kinds = _kinds_of(p, known)
                hit = [s for s in svcs if s.kind in kinds]
            else:
                kinds, hit = (), svcs
            for s in hit:
                if not out[s.key].running or is_managed:   # 同一種有兩個行程時，優先顯示這個視窗啟動的
                    out[s.key] = Status(True, p.pid, is_managed, kinds, started_at=p.info["create_time"])
        except Exception:  # noqa: BLE001
            continue
    return out


# ── 啟動／停止 ────────────────────────────────────────────────────────────


def _ensure_bom(log: Path) -> None:
    """紀錄檔開頭補 UTF-8 BOM：記事本一看就知道是 UTF-8，中文不會變亂碼（只在服務沒在寫的時候呼叫）。"""
    try:
        if not log.exists() or log.stat().st_size == 0:
            log.write_bytes(b"\xef\xbb\xbf")
            return
        with open(log, "rb") as f:
            head = f.read(3)
        if head != b"\xef\xbb\xbf":
            log.write_bytes(b"\xef\xbb\xbf" + log.read_bytes())
    except OSError:
        pass


def _rotate(log: Path) -> None:
    try:
        if log.exists() and log.stat().st_size > _LOG_ROTATE_BYTES:
            old = log.with_suffix(".log.1")
            if old.exists():
                old.unlink()
            log.rename(old)
    except OSError:
        pass


def start(key: str, python: Optional[str] = None) -> tuple[bool, str]:
    """背景啟動一個服務；回傳 (成功與否, 訊息)。已經在跑（包含被別的行程一起處理）就不重複啟動。"""
    svc = get(key)
    st = scan().get(key)
    if st and st.running:
        return False, f"{svc.title} 已經在執行（PID {st.pid}）。"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _rotate(svc.log_path)
    _ensure_bom(svc.log_path)
    env = os.environ.copy()
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
    env.pop(_KINDS_ENV, None)
    # 感測器 hub：設定視窗「⚙ 參數設定」存的 iot/config.py 覆寫（語音紀錄有自己的 config，不套）
    overrides = iot_config_overrides.env_for_start() if svc.kind else {}
    env.update(overrides)
    env.update(svc.env)
    flags = 0
    if os.name == "nt":
        # 背景常駐：自己的行程群組、不開主控台視窗；關掉設定視窗不會連帶結束
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    exe = python or sys.executable
    try:
        with open(svc.log_path, "a", encoding="utf-8") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 由設定視窗啟動：{exe} -m {svc.module}"
                      f"{'（' + _KINDS_ENV + '=' + svc.kind + '）' if svc.kind else ''} =====\n")
            for k, v in sorted(overrides.items()):
                shown = "***" if k.endswith("PASSWORD") else v
                log.write(f"      參數覆寫：{k.removeprefix('CAT_MONITORING_IOT_')} = {shown}\n")
            log.flush()
            proc = subprocess.Popen([exe, "-m", svc.module], cwd=str(CAT_DIR), env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)
    except OSError as e:
        return False, f"啟動失敗：{e}"
    state = _load_state()
    create_time = time.time()
    if psutil is not None:
        try:
            create_time = psutil.Process(proc.pid).create_time()
        except Exception:  # noqa: BLE001
            pass
    state[key] = {"pid": proc.pid, "create_time": create_time, "started_at": time.time(), "module": svc.module}
    _save_state(state)
    return True, f"{svc.title} 已啟動（PID {proc.pid}），紀錄檔：{svc.log_path}"


def stop(key: str, pid: Optional[int] = None, timeout: float = 5.0) -> tuple[bool, str]:
    """停止這個服務所在的行程（全部感測器一起跑的行程會一起停）。
    ``pid``＝畫面上看到的那個行程；給了就只停它（確認期間換成別的行程時不會停錯）。"""
    svc = get(key)
    st = scan().get(key)
    if not st or not st.running:
        return True, f"{svc.title} 沒有在執行。"
    if pid is not None and st.pid != pid:
        return False, f"{svc.title} 現在是 PID {st.pid}，不是要停的 {pid}，請重新整理後再試。"
    if psutil is None:
        return False, "缺少 psutil，無法停止。"
    try:
        p = psutil.Process(st.pid)
        p.terminate()   # Windows：TerminateProcess。MQTT 遺囑會讓 Node-RED 知道離線；SQLite 交易不會寫一半
        try:
            p.wait(timeout)
        except psutil.TimeoutExpired:
            p.kill()
            p.wait(2)
    except psutil.NoSuchProcess:
        pass
    except Exception as e:  # noqa: BLE001
        return False, f"停止失敗：{e}"
    state = _load_state()
    for k in [k for k, rec in state.items() if int(rec.get("pid", 0)) == st.pid]:
        state.pop(k)
    _save_state(state)
    try:
        with open(svc.log_path, "a", encoding="utf-8") as log:
            log.write(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} 由設定視窗停止（PID {st.pid}） =====\n")
    except OSError:
        pass
    return True, f"{svc.title} 已停止（PID {st.pid}）。"


def stop_all(reason: str = "由設定視窗停止", timeout: float = 5.0) -> list[int]:
    """強制停止所有在執行的 IoT 服務行程（這個視窗啟動的、在 cmd 手動啟動的都算），一起送、一起等；回傳停掉的 PID。
    09-27 使用者要求：關掉設定視窗時，執行中的 IoT 服務一律跟著關。"""
    if psutil is None:
        return []
    statuses = scan()
    pids = sorted({st.pid for st in statuses.values() if st.running})
    procs = []
    for pid in pids:
        try:
            p = psutil.Process(pid)
            p.terminate()
            procs.append(p)
        except psutil.NoSuchProcess:
            pass
        except Exception:  # noqa: BLE001
            continue
    _gone, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except Exception:  # noqa: BLE001
            pass
    if alive:
        psutil.wait_procs(alive, timeout=2)
    state = _load_state()
    for k in [k for k, rec in state.items() if int(rec.get("pid", 0)) in pids]:
        state.pop(k)
    _save_state(state)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for s in SERVICES:
        st = statuses.get(s.key)
        if st and st.running:
            try:
                with open(s.log_path, "a", encoding="utf-8") as log:
                    log.write(f"===== {stamp} {reason}（PID {st.pid}） =====\n")
            except OSError:
                pass
    return pids


def restart_sensors() -> list[str]:
    """重新啟動執行中的感測器服務（改了 ⚙ 參數設定後用：iot/config.py 只在啟動時讀一次）。
    同一個行程處理多種感測器的只停一次；重新啟動後每種各自一個行程。回傳訊息清單。"""
    statuses = scan()
    keys = [s.key for s in SERVICES if s.kind and statuses[s.key].running]
    msgs, stopped = [], set()
    for k in keys:
        pid = statuses[k].pid
        if pid in stopped:
            continue
        ok, msg = stop(k, pid=pid)
        stopped.add(pid)
        if not ok:
            msgs.append(msg)
    for k in keys:
        msgs.append(start(k)[1])
    return msgs


def running_titles() -> list[str]:
    """目前在執行的服務名稱（關閉設定視窗時顯示用）。"""
    return [s.title for s in SERVICES if scan()[s.key].running] if psutil is not None else []


def log_size(key: str) -> int:
    """紀錄檔目前大小（bytes）；沒有檔案回 0。紀錄框「清空」記這個位置，之後只顯示它後面的新紀錄。"""
    try:
        return get(key).log_path.stat().st_size
    except OSError:
        return 0


def tail(key: str, lines: int = 200, since: int = 0) -> str:
    """紀錄檔最後幾行（檔案很大也只讀尾端）。since＝只看這個位置之後的內容（紀錄框「清空」用；
    檔案被輪替變小了就當作沒清過）。"""
    path = get(key).log_path
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if since > size:
                since = 0
            f.seek(max(since, size - 64 * 1024))
            data = f.read().decode("utf-8", errors="replace").lstrip("\ufeff")
    except OSError:
        return "（還沒有紀錄檔：這個服務還沒從設定視窗啟動過）"
    if since and not data.strip():
        return "（已清空，等待新的紀錄……完整紀錄仍在檔案裡，可按「用記事本開啟」）"
    return "\n".join(data.splitlines()[-lines:])


_DEFAULT_DB = CAT_DIR / "iot" / "data" / "iot_hub.db"   # iot/config.py 的 DB_PATH 預設（那是運算式，ast 讀不到）


def db_info() -> tuple[Path, int, float]:
    """IoT 資料庫（路徑、大小 bytes（含 -wal／-shm）、原始讀數保留天數）。設定視窗顯示用，不 import iot。"""
    path = Path(iot_config_overrides.effective("CAT_MONITORING_IOT_DB_PATH", str(_DEFAULT_DB)) or _DEFAULT_DB)
    size = 0
    for p in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            size += p.stat().st_size
        except OSError:
            pass
    try:
        days = float(iot_config_overrides.effective("CAT_MONITORING_IOT_DATA_RETENTION_DAYS", "30"))
    except ValueError:
        days = 30.0
    return path, size, days


# 紀錄行開頭的時間（logging 的 "2026-09-27 14:17:09,476"；設定視窗寫的 "===== 2026-09-27 12:02:38 …" 也算）
_LOG_TS = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:,(\d{3}))?")


def merge_logs(texts: dict[str, str]) -> list[tuple[str, str]]:
    """{服務: 紀錄文字} → [(服務, 一行)]，依時間合併（紀錄框「整合檢視」用）。
    沒有時間的行（Traceback、續行）跟著同一個檔案的上一行走；同一時間照原本順序。"""
    rows = []
    for order, (key, text) in enumerate(texts.items()):
        ts = ""
        for i, line in enumerate(text.splitlines()):
            m = _LOG_TS.match(line) or (_LOG_TS.search(line, 0, 30) if line.startswith("=====") else None)
            if m:
                ts = f"{m.group(1)},{m.group(2) or '000'}"
            rows.append((ts, order, i, key, line))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return [(r[3], r[4]) for r in rows]


def merged_tail(lines: int = 400, since: Optional[dict] = None) -> list[tuple[str, str]]:
    """全部服務紀錄檔的最後幾行、依時間合併。還沒有紀錄檔的服務跳過；since＝各服務「清空」的位置。"""
    since = since or {}
    texts = {}
    for s in SERVICES:
        if not s.log_path.exists():
            continue
        text = tail(s.key, lines=lines, since=since.get(s.key, 0))
        if text.startswith("（"):   # 「已清空，等待新的紀錄」之類的提示，不是紀錄
            continue
        texts[s.key] = text
    return merge_logs(texts)[-lines:]


# ── 實際連線：服務連到哪個 broker、哪支 ESP32 是哪個 IP ────────────────────
#
# MQTT 訊息不帶發送者 IP，所以靠「主機名稱」對應：韌體開機時 WiFi.setHostname("cat-<種類>-<節點>")，
# 路由器的 DNS 會記下來（例：192.168.0.148 → cat-petbox-carrier.home），這裡用 DNS 反查 IP 就知道是哪一支。
# 外出包（petbox）同時送 env 和 bodytemp，兩張卡片都會顯示它的 IP。
# 語音終端不走 broker（v4.0 起用 HTTP／WebSocket 直接連 Node-RED 1880），所以另外看連進 1880 的連線，
# 而且只收認得的主機名稱（手機、瀏覽器也會連 1880）。

_HOST_KINDS = {
    "env": ("env",), "motion": ("motion",), "weight": ("weight",),
    "petbox": ("env", "bodytemp"), "voice": ("voice",),
    "room": ("env", "motion"),   # esp32_room：空氣品質＋光照＋移動偵測同一片（09-27 合併）
}
_NODE_RED_PORT = 1880
_DNS_TTL_OK, _DNS_TTL_FAIL = 300.0, 60.0
_dns_cache: dict[str, tuple[float, str]] = {}


@dataclass(frozen=True)
class Device:
    ip: str
    hostname: str = ""               # DNS 反查到的名稱（去掉網域）；""＝查不到
    keys: tuple = ()                 # 對應到哪些服務卡片；()＝認不得（還沒燒新韌體、或不是這個系統的裝置）


@dataclass(frozen=True)
class MqttLinks:
    """實際的連線（讀作業系統的 TCP 連線表，不是看設定值）。"""
    service_targets: tuple = ()      # 執行中的 IoT 服務實際連到的 broker 位址（"ip:port"）
    devices: Optional[tuple] = ()    # 連進本機 broker／Node-RED 的 Device；None＝broker 不在本機，看不到
    connected_pids: tuple = ()       # 已連上 broker 的服務行程 PID（逐一檢查每個服務用）
    node_red: Optional[bool] = None  # Node-RED（node.exe）有沒有連上 broker；None＝沒查（純邏輯測試）

    def ips_for(self, key: str) -> list[str]:
        return [d.ip for d in self.devices or () if key in d.keys]

    def unknown(self) -> list:
        return [d for d in self.devices or () if not d.keys]


def device_keys(hostname: str) -> tuple:
    """``cat-petbox-carrier`` → ("env", "bodytemp")；不是 cat-<種類>-… 的回 ()。"""
    parts = hostname.lower().split("-")
    return _HOST_KINDS.get(parts[1], ()) if len(parts) >= 2 and parts[0] == "cat" else ()


def _reverse_dns(ip: str) -> str:
    """IP → 主機名稱（去掉網域），有快取：反查失敗可能要等好幾秒，不能每 2 秒查一次。"""
    now = time.monotonic()
    hit = _dns_cache.get(ip)
    if hit and hit[0] > now:
        return hit[1]
    try:
        name = socket.gethostbyaddr(ip)[0].split(".", 1)[0]
    except OSError:
        name = ""
    _dns_cache[ip] = (now + (_DNS_TTL_OK if name else _DNS_TTL_FAIL), name)
    return name


def clear_dns_cache() -> None:
    """清掉主機名稱快取（設定視窗「⟳ 重新整理」按鈕：ESP32 燒錄改名後不用等 5 分鐘）。"""
    _dns_cache.clear()


def _ip_key(ip: str) -> tuple:
    return tuple(int(p) if p.isdigit() else 0 for p in ip.split("."))


def _links_from(conns, service_pids, port: int, local_ips: set, resolve=_reverse_dns) -> MqttLinks:
    """純邏輯（方便測試）：conns 是 psutil.net_connections() 的結果，resolve 是 IP → 主機名稱。"""
    targets, pids, broker_ips, node_red_ips, listening = set(), set(), set(), set(), False
    for c in conns:
        if c.status == "LISTEN" and c.laddr and c.laddr.port == port:
            listening = True
        if c.status != "ESTABLISHED" or not c.laddr or not c.raddr:
            continue
        if c.pid in service_pids and c.raddr.port == port:
            targets.add(f"{c.raddr.ip}:{c.raddr.port}")
            pids.add(c.pid)
            continue
        remote = c.raddr.ip
        if remote in local_ips or remote.startswith("127.") or remote == "::1":
            continue
        if c.laddr.port == port:
            broker_ips.add(remote)
        elif c.laddr.port == _NODE_RED_PORT:
            node_red_ips.add(remote)
    devices = {}
    for ip in broker_ips:
        name = resolve(ip)
        devices[ip] = Device(ip, name, device_keys(name))
    for ip in node_red_ips - broker_ips:
        name = resolve(ip)
        if device_keys(name):
            devices[ip] = Device(ip, name, device_keys(name))
    ordered = tuple(devices[ip] for ip in sorted(devices, key=_ip_key))
    return MqttLinks(tuple(sorted(targets)), ordered if listening else None, tuple(sorted(pids)))


def mqtt_links(statuses: dict[str, Status], port: Optional[int] = None) -> Optional[MqttLinks]:
    """執行中服務連到哪個 broker、哪些裝置連進本機 broker（含主機名稱對應）。沒有 psutil 或讀不到連線表回 None。
    可能要反查 DNS（有快取），只在背景執行緒呼叫。"""
    if psutil is None:
        return None
    try:
        port = int(port or iot_config_overrides.effective("CAT_MONITORING_IOT_MQTT_PORT", "1883"))
    except ValueError:
        port = 1883
    try:
        conns = psutil.net_connections("tcp")
        local_ips = {a.address for addrs in psutil.net_if_addrs().values() for a in addrs}
    except Exception:  # noqa: BLE001 — AccessDenied 等
        return None
    pids = {st.pid for st in statuses.values() if st.running}
    links = _links_from(conns, pids, port, local_ips)
    node_red = False
    for c in conns:   # Node-RED 負責把告警轉到 Discord：它沒連上 broker，告警就送不出去
        if c.status == "ESTABLISHED" and c.raddr and c.raddr.port == port and c.pid:
            try:
                if psutil.Process(c.pid).name().lower().startswith("node"):
                    node_red = True
                    break
            except Exception:  # noqa: BLE001
                continue
    return MqttLinks(links.service_targets, links.devices, links.connected_pids, node_red)


def broker_reachable(host: Optional[str] = None, port: Optional[int] = None, timeout: float = 0.8) -> tuple[bool, str]:
    """MQTT broker 連不連得到（IoT 服務都靠它）。位址照服務自己的預設／環境變數。"""
    host = host or iot_config_overrides.effective("CAT_MONITORING_IOT_MQTT_HOST", "192.168.0.171")
    try:
        port = int(port or iot_config_overrides.effective("CAT_MONITORING_IOT_MQTT_PORT", "1883"))
    except ValueError:
        port = 1883
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"{host}:{port}"
    except OSError:
        return False, f"{host}:{port}"
# ── 完整連線檢查（「⟳ 重新整理」按鈕，結果印到設定視窗底部的全域終端機，有問題的用紅字）─────────

_LEVEL_ORDER = {"ok": 0, "warn": 1, "bad": 2}


def _worst(*levels: str) -> str:
    return max(levels, key=lambda lv: _LEVEL_ORDER.get(lv, 0))


def connection_report(statuses: dict[str, Status], broker: tuple[bool, str], links: Optional[MqttLinks],
                      local_ifaces: dict[str, str]) -> list[tuple[str, str]]:
    """純邏輯：回傳 [(等級, 文字)]，等級 ok／warn／bad／info。local_ifaces 是本機 IP → 網卡名稱。"""
    out: list[tuple[str, str]] = [("info", f"── IoT 連線檢查 {time.strftime('%H:%M:%S')} ──")]
    ok, where = broker
    host = where.rsplit(":", 1)[0]
    if ok:
        note = ""
        if host in local_ifaces and not host.startswith("127."):
            note = f"（本機「{local_ifaces[host]}」網卡的 IP：這張網卡斷線時 ESP32 會連不上）"
        out.append(("ok", f"● MQTT broker（設定）{where} 連得到{note}"))
    else:
        out.append(("bad", f"✕ MQTT broker（設定）{where} 連不上：服務、ESP32 的資料都進不來"))
    if links is None:
        out.append(("warn", "△ 讀不到連線表（沒有 psutil？），服務／裝置的連線狀態無法檢查"))
    else:
        if links.service_targets:
            out.append(("info", f"  服務目前實際連上：{'、'.join(links.service_targets)}"))
        if links.node_red is True:
            out.append(("ok", "● Node-RED 已連上 broker（告警可轉到 Discord）"))
        elif links.node_red is False:
            out.append(("bad", "✕ Node-RED 沒連上 broker：告警送不到 Discord"))
    for svc in SERVICES:
        st = statuses.get(svc.key) or Status(False)
        if not st.running:
            s_lv, s_txt = "warn", "服務未啟動"
        elif links is None:
            s_lv, s_txt = "warn", f"服務執行中（PID {st.pid}），連線狀態不明"
        elif st.pid in links.connected_pids:
            s_lv, s_txt = "ok", f"服務已連上 broker（PID {st.pid}）"
        else:
            s_lv, s_txt = "bad", f"服務執行中（PID {st.pid}）但沒連上 broker"
        if links is None:
            d_lv, d_txt = "warn", "裝置狀態不明"
        elif links.devices is None:
            d_lv, d_txt = "warn", "broker 不在本機，看不到裝置"
        elif links.ips_for(svc.key):
            d_lv, d_txt = "ok", "裝置 " + "、".join(links.ips_for(svc.key))
        else:
            d_lv, d_txt = "bad", "裝置沒有連線（ESP32 沒開、WiFi／broker IP 錯，或還沒燒 cat- 主機名稱）"
        lv = _worst(s_lv, d_lv)
        mark = {"ok": "●", "warn": "△", "bad": "✕"}[lv]
        out.append((lv, f"{mark} {svc.title}：{s_txt}｜{d_txt}"))
    for d in (links.unknown() if links is not None else []):
        out.append(("warn", f"△ 未辨識裝置 {d.ip}（{d.hostname or '查不到名稱'}）：主機名稱不是 cat- 開頭"))
    n = {lv: sum(1 for x, _ in out if x == lv) for lv in ("ok", "warn", "bad")}
    out.append(("bad" if n["bad"] else "warn" if n["warn"] else "ok",
                f"── 結果：正常 {n['ok']}、警告 {n['warn']}、錯誤 {n['bad']} ──"))
    return out


def check_connections() -> list[tuple[str, str]]:
    """實際量一次（掃行程、連 broker、讀連線表、反查 DNS），可能要一兩秒，只在背景執行緒呼叫。"""
    statuses = scan()
    local_ifaces = {}
    if psutil is not None:
        try:
            local_ifaces = {a.address: name for name, addrs in psutil.net_if_addrs().items() for a in addrs}
        except Exception:  # noqa: BLE001
            pass
    return connection_report(statuses, broker_reachable(), mqtt_links(statuses), local_ifaces)
