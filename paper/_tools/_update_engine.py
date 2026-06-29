# -*- coding: utf-8 -*-
"""更新 健康引擎分析：四維算法 + v2 global sync + 讀用戶設定"""
import json

MAIN_PATH  = r'c:\ai_project\paper\貓咪主控.json'
ENGINE_ID  = '257d6083ac06afed'

NEW_FUNC = """let data    = msg.payload;
let stats   = (data && data.today_stats) ? data.today_stats : {};

// ── 讀取用戶設定（與 v2 P4 共用 global context）─────────────
let cfg         = global.get('v2_user_settings','file') || {};
let LICK_THR    = cfg.lick_pct    != null ? +cfg.lick_pct    : 20;
let SCRATCH_THR = cfg.scratch_pct != null ? +cfg.scratch_pct : 15;
let SHAKE_THR   = cfg.shake_pct   != null ? +cfg.shake_pct   : 10;
let STOP_THR    = cfg.stop_pct    != null ? +cfg.stop_pct    : 55;
let LSC_THR     = cfg.lsc_count   != null ? +cfg.lsc_count   : 10;

// ── 基礎時間變數 ──────────────────────────────────────────────
let walk_t    = Number(stats.walk_time    || 0);
let lick_t    = Number(stats.lick_time    || 0);
let scratch_t = Number(stats.scratch_time || 0);
let shake_t   = Number(stats.shake_time   || 0);
let stop_t    = Number(stats.stop_time    || 0);
let td_active = Number(stats.active_time  || 0);

// ── 1. 行為佔比 distribution ──────────────────────────────────
let totalBeh = walk_t + lick_t + scratch_t + shake_t + stop_t;
let dist = totalBeh > 0 ? {
    walk:    parseFloat((walk_t    / totalBeh * 100).toFixed(2)),
    lick:    parseFloat((lick_t    / totalBeh * 100).toFixed(2)),
    scratch: parseFloat((scratch_t / totalBeh * 100).toFixed(2)),
    shake:   parseFloat((shake_t   / totalBeh * 100).toFixed(2)),
    stop:    parseFloat((stop_t    / totalBeh * 100).toFixed(2))
} : {walk:0, lick:0, scratch:0, shake:0, stop:0};
data.distribution = dist;

// ── 2. 節律分析 rhythm（from hourly_distribution in payload）──
let hourly = stats.hourly_distribution || {};
let _pds = {
    '00-06':{walk:0,lick:0,scratch:0,shake:0,stop:0,total:0},
    '06-12':{walk:0,lick:0,scratch:0,shake:0,stop:0,total:0},
    '12-18':{walk:0,lick:0,scratch:0,shake:0,stop:0,total:0},
    '18-24':{walk:0,lick:0,scratch:0,shake:0,stop:0,total:0}
};
for (let h=0; h<24; h++) {
    let k=String(h).padStart(2,'0'), hd=hourly[k]||{};
    let pd=h<6?'00-06':h<12?'06-12':h<18?'12-18':'18-24';
    for (let b of ['walk','lick','scratch','shake','stop']) { _pds[pd][b]+=(hd[b]||0); _pds[pd].total+=(hd[b]||0); }
}
let rhythm = {};
for (let [pd,d] of Object.entries(_pds)) {
    if (d.total===0) { rhythm[pd]={dominant:'none',pct:{},total:0}; continue; }
    let bs=['walk','lick','scratch','shake','stop'];
    let dom=bs.reduce((a,b)=>d[a]>=d[b]?a:b,'stop');
    let pct={}; for (let b of bs) pct[b]=parseFloat((d[b]/d.total*100).toFixed(2));
    rhythm[pd]={dominant:dom, pct, total:parseFloat(d.total.toFixed(1))};
}

// ── 3. 同步寫入 v2 global context（讓 v2 定時任務基線引擎繼續正常運作）──
let _tdDate = new Date().toLocaleDateString('zh-TW');
global.set('v2_today', {
    date: _tdDate,
    walk:{count:stats.walk||0,time:walk_t}, lick:{count:stats.lick||0,time:lick_t},
    scratch:{count:stats.scratch||0,time:scratch_t}, shake:{count:stats.shake||0,time:shake_t},
    stop:{count:stats.stop||0,time:stop_t},
    active_time:stats.active_time||0, rest_time:stats.rest_time||0,
    not_detected_time:stats.not_detected_time||0, monitoring_seconds:stats.monitoring_seconds||0
}, 'file');
global.set('v2_hourly',            stats.hourly_distribution||{});
global.set('v2_transition_matrix', stats.transition_matrix  ||{});

// ── 4. dScore：行為占比偏離 (0-100) — 不需基線 ───────────────
let dScore = 0;
if (dist.lick    > LICK_THR)    dScore += Math.min(30, (dist.lick    - LICK_THR)    * 1.5);
if (dist.scratch > SCRATCH_THR) dScore += Math.min(35, (dist.scratch - SCRATCH_THR) * 2.0);
if (dist.shake   > SHAKE_THR)   dScore += Math.min(35, (dist.shake   - SHAKE_THR)   * 2.5);
if (dist.stop    > STOP_THR)    dScore += Math.min(20, (dist.stop    - STOP_THR)    * 1.0);
if (dist.walk    < 10 && td_active > 300) dScore += 15;
dScore = Math.min(100, Math.round(dScore));

// ── 5. fScore：頻率偏離（讀取 v2_baseline；無基線時 = 0）────────
let fScore = 0, hasBaseline = false;
try {
    let bl = global.get('v2_baseline', 'file') || global.get('v2_baseline');
    if (bl && bl.metrics) {
        hasBaseline = true;
        let m = bl.metrics;
        let pctDev = function(cur, key) {
            let b = m[key]; if (!b || b.mean === 0) return 0;
            return (cur - b.mean) / b.mean * 100;
        };
        let ld  = pctDev(lick_t,                         'lick_time');
        let sd  = pctDev(scratch_t,                      'scratch_time');
        let shd = pctDev(Number(stats.shake_count || 0), 'shake_count');
        let std = pctDev(stop_t,                         'stop_time');
        if (ld  > 150) fScore += 30; else if (ld  > 100) fScore += 20; else if (ld  >  50) fScore += 10;
        if (sd  > 200) fScore += 35; else if (sd  > 150) fScore += 25; else if (sd  >  80) fScore += 15;
        if (shd > 100) fScore += 35; else if (shd >  50) fScore += 20; else if (shd >  30) fScore += 10;
        if (std >  80) fScore += 20; else if (std >  50) fScore += 10;
        fScore = Math.min(100, Math.round(fScore));
    }
} catch(e) {}

// ── 6. rScore：節律偏離 (0-100) ───────────────────────────────
let rScore = 0;
let night = rhythm['00-06'] || {};
if (night.dominant==='scratch'||night.dominant==='lick') rScore += 30;
if (night.dominant==='shake')                             rScore += 25;
if (night.pct && night.pct.scratch > 20)                 rScore += 20;
let day06 = rhythm['06-12'] || {}, day12 = rhythm['12-18'] || {};
if (day06.pct && day12.pct && (day06.pct.walk||0)<15 && (day12.pct.walk||0)<15 && dist.walk<20) rScore += 20;
rScore = Math.min(100, rScore);

// ── 7. tScore：轉移偏離 (0-100) ───────────────────────────────
let mx = stats.transition_matrix || {};
let tScore = 0;
let lsc  = (mx['lick->scratch']||0) + (mx['scratch->lick']||0);
let shch = mx['shake->shake'] || 0;
let stst = mx['stop->stop']   || 0;
if (lsc>LSC_THR*2) tScore+=40; else if(lsc>LSC_THR) tScore+=20; else if(lsc>Math.max(2,Math.round(LSC_THR/2))) tScore+=10;
if (shch>15) tScore+=30; else if(shch>8) tScore+=15;
if (stst>50) tScore+=20; else if(stst>30) tScore+=10;
tScore = Math.min(100, tScore);

// ── 8. 綜合風險分（四維加權：占比30% 頻率35% 節律20% 轉移15%）──
let riskScore = Math.round(dScore*0.30 + fScore*0.35 + rScore*0.20 + tScore*0.15);
riskScore = Math.max(0, Math.min(100, riskScore));

let level, color, emoji;
if      (riskScore < 20) { level='Normal';    color='#4caf50'; emoji='✅'; }
else if (riskScore < 45) { level='Attention'; color='#ffa726'; emoji='⚠️'; }
else if (riskScore < 70) { level='Warning';   color='#ff7043'; emoji='🚨'; }
else                      { level='High Risk'; color='#f44336'; emoji='🆘'; }

// ── 警示列表 ──────────────────────────────────────────────────
let alerts = [];
if (dist.lick    > LICK_THR)    alerts.push({type:'lick',       msg:'舔舐佔比 ' + dist.lick.toFixed(1)    + '%，超過閾值 ' + LICK_THR    + '%'});
if (dist.scratch > SCRATCH_THR) alerts.push({type:'scratch',    msg:'搔抓佔比 ' + dist.scratch.toFixed(1) + '%，超過閾值 ' + SCRATCH_THR + '%'});
if (dist.shake   > SHAKE_THR)   alerts.push({type:'shake',      msg:'甩頭佔比 ' + dist.shake.toFixed(1)   + '%，超過閾值 ' + SHAKE_THR   + '%'});
if (dist.stop    > STOP_THR)    alerts.push({type:'inactivity', msg:'靜止佔比 ' + dist.stop.toFixed(1)    + '%，活動力偏低'});
if (lsc > LSC_THR) {
    let ls=mx['lick->scratch']||0, sl=mx['scratch->lick']||0, half=Math.max(2,Math.round(LSC_THR/2));
    let transMsg=(ls>half&&sl>half)?'lick↔scratch 循環 '+lsc+' 次，疑似皮膚不適':sl>ls?'scratch→lick '+sl+' 次（搔後舔舐），疑似皮膚不適':'lick→scratch '+ls+' 次（舔後搔抓），疑似皮膚不適';
    alerts.push({type:'transition', msg:transMsg});
}
if (hasBaseline && fScore >= 20) alerts.push({type:'dev', msg:'行為頻率與個體基線偏離（fScore: ' + fScore + '）'});

data.risk = {
    score: riskScore, level, color, emoji,
    components: { distribution:dScore, frequency:fScore, rhythm:rScore, transition:tScore },
    alerts,
    computed_at: (function(){ let d=new Date(),p=function(n){return String(n).padStart(2,'0')}; return d.getFullYear()+'-'+p(d.getMonth()+1)+'-'+p(d.getDate())+' '+p(d.getHours())+':'+p(d.getMinutes())+':'+p(d.getSeconds()); })()
};
data.alerts = alerts;
global.set('v2_risk', data.risk);
flow.set('v2_today_risk', {level, score: riskScore});
return msg;"""

with open(MAIN_PATH, 'r', encoding='utf-8') as f:
    nodes = json.load(f)

for n in nodes:
    if n.get('id') == ENGINE_ID:
        n['func'] = NEW_FUNC
        print('Fixed: 健康引擎分析 → 四維算法 (dScore/fScore/rScore/tScore) + v2 global sync')
        break

with open(MAIN_PATH, 'w', encoding='utf-8') as f:
    json.dump(nodes, f, ensure_ascii=False, indent=4)
print('Written: 貓咪主控.json')
