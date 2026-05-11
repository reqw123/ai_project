import json
from pathlib import Path
from collections import Counter, defaultdict

root = Path(r"C:\cat_pose\gcn_pose\skeletons")
files = sorted(root.glob("*.json"))
seq_len = 32
stride = 16
classes = ["walk", "lick", "scratch", "shake"]

results = []
class_totals = Counter({c:0 for c in classes})
zero_files = []

def extract_frames(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("frames", "data", "sequence", "annotations"):
            v = obj.get(k)
            if isinstance(v, list):
                return v
    return []

for fp in files:
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        data = json.loads(fp.read_text(encoding="utf-8-sig"))
    frames = extract_frames(data)
    labels = []
    for fr in frames:
        label = "unannotated"
        if isinstance(fr, dict):
            label = fr.get("label", "unannotated")
        if label is None:
            label = "unannotated"
        label = str(label).strip().lower()
        if not label:
            label = "unannotated"
        labels.append(label)

    total = len(labels)
    counts = Counter(labels)
    annotated = total - counts.get("unannotated", 0)
    ann_ratio = (annotated / total) if total else 0.0

    per_file_valid = Counter({c:0 for c in classes})
    total_valid = 0
    total_windows = 0
    windows_with_unann = 0
    windows_fail_majority = 0

    if total >= seq_len:
        for s in range(0, total - seq_len + 1, stride):
            total_windows += 1
            w = labels[s:s+seq_len]
            if "unannotated" in w:
                windows_with_unann += 1
                continue
            wc = Counter(w)
            maj_label, maj_count = wc.most_common(1)[0]
            if maj_count / seq_len >= 0.8:
                if maj_label in classes:
                    per_file_valid[maj_label] += 1
                total_valid += 1
            else:
                windows_fail_majority += 1

    for c in classes:
        class_totals[c] += per_file_valid[c]

    reason = ""
    if total_valid == 0:
        if total == 0:
            reason = "empty file"
        elif total < seq_len:
            reason = f"too short (<{seq_len} frames)"
        elif counts.get("unannotated", 0) == total:
            reason = "all unannotated"
        elif total_windows > 0 and windows_with_unann == total_windows:
            reason = "all windows contain unannotated"
        elif total_windows > 0 and (total_windows - windows_with_unann) > 0:
            reason = "no window reaches >=0.8 majority"
        else:
            reason = "no valid windows"
        zero_files.append((fp.name, reason))

    results.append({
        "file": fp.name,
        "total_frames": total,
        "ann_ratio": ann_ratio,
        "counts": counts,
        "valid": per_file_valid,
        "valid_total": total_valid,
    })

# Print concise tables
print(f"Scanned files: {len(results)} from {root}")
print("\nPer-file summary")
print("file".ljust(36), "frames".rjust(7), "ann%".rjust(7), "walk".rjust(6), "lick".rjust(6), "scratch".rjust(8), "shake".rjust(7), "total".rjust(7), "labels")
for r in results:
    counts_short = ",".join(f"{k}:{v}" for k,v in sorted(r["counts"].items()))
    print(r["file"][:36].ljust(36),
          str(r["total_frames"]).rjust(7),
          f"{r['ann_ratio']*100:6.1f}",
          str(r["valid"]["walk"]).rjust(6),
          str(r["valid"]["lick"]).rjust(6),
          str(r["valid"]["scratch"]).rjust(8),
          str(r["valid"]["shake"]).rjust(7),
          str(r["valid_total"]).rjust(7),
          counts_short)

print("\nTotals by class (valid windows)")
print(" ".join(f"{c}:{class_totals[c]}" for c in classes), f"all:{sum(class_totals.values())}")

print("\nFiles with zero valid windows")
if zero_files:
    for n, reason in zero_files:
        print(f"- {n}: {reason}")
else:
    print("- none")
