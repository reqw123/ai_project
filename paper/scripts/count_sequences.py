import os, json
from pathlib import Path
try:
    import yaml
except Exception:
    yaml=None
root=Path(r"C:/AI_Project/paper/skeletons")
conf_path=Path(r"c:/ai_project/paper/cat_monitoring_system/stgcn_config.yaml")
conf={}
if conf_path.exists() and yaml:
    conf=yaml.safe_load(conf_path.read_text(encoding='utf-8'))
SEQUENCE_LENGTH=int(conf.get('SEQUENCE_LENGTH',16))
STRICT_WINDOW_FILTER=bool(conf.get('STRICT_WINDOW_FILTER', False))
LABEL_PURITY_THRESHOLD=float(conf.get('LABEL_PURITY_THRESHOLD',0.8))
NUM_JOINTS=int(conf.get('NUM_JOINTS',17))
from collections import Counter
counts=Counter()
files=sorted([p for p in root.glob('*.json')])
for p in files:
    data=json.loads(p.read_text(encoding='utf-8'))
    frames=data.get('frames',[])
    if not frames or 'label' not in frames[0]:
        continue
    keypoint_frames=[]
    frame_labels=[]
    confs=[]
    for f in frames:
        kpts=f.get('keypoints',[])
        if len(kpts)==NUM_JOINTS:
            coords=[(k['x'],k['y']) for k in kpts]
            confs.append([k.get('conf',1.0) for k in kpts])
        else:
            coords=[(0.0,0.0)]*NUM_JOINTS
            confs.append([0.0]*NUM_JOINTS)
        keypoint_frames.append(coords)
        frame_labels.append(f.get('label','unannotated'))
    T=len(keypoint_frames)
    if T<SEQUENCE_LENGTH: continue
    stride=SEQUENCE_LENGTH//2
    for start in range(0, T-SEQUENCE_LENGTH+1, stride):
        window_labels=frame_labels[start:start+SEQUENCE_LENGTH]
        if STRICT_WINDOW_FILTER:
            if 'unannotated' in window_labels: continue
            # purity
            lab_counts=Counter(window_labels)
            best, cnt=lab_counts.most_common(1)[0]
            if cnt/SEQUENCE_LENGTH < LABEL_PURITY_THRESHOLD: continue
        else:
            annotated=[l for l in window_labels if l!='unannotated']
            if not annotated: continue
            lab_counts=Counter(annotated)
            best=lab_counts.most_common(1)[0][0]
        counts[best]+=1
print('SEQUENCE_LENGTH',SEQUENCE_LENGTH,'STRICT_WINDOW_FILTER',STRICT_WINDOW_FILTER)
print('Per-class sequence counts:')
for k in sorted(counts.keys()):
    print(f'  {k}: {counts[k]}')
print('Total sequences:', sum(counts.values()))
