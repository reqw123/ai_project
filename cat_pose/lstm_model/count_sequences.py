"""
Count sequences in saved JSON files
"""
import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(r"C:\cat_pose\lstm_model\behavior_data")
BEHAVIOR_NAMES = ['normal', 'licking', 'scratching', 'head_shaking']

def count_sequences():
    """Count all sequences in JSON files"""
    
    if not DATA_DIR.exists():
        print("❌ Data directory not found")
        return
    
    # Count by behavior
    counts = defaultdict(int)
    file_counts = defaultdict(int)
    
    # Scan all JSON files
    json_files = list(DATA_DIR.glob("*.json"))
    
    if not json_files:
        print("📂 No data files found")
        return
    
    print("="*70)
    print("📊 Sequence Statistics")
    print("="*70)
    
    for json_file in sorted(json_files):
        # Determine behavior from filename
        behavior = None
        for name in BEHAVIOR_NAMES:
            if json_file.name.startswith(name):
                behavior = name
                break
        
        if behavior:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    seq_count = len(data)
                    counts[behavior] += seq_count
                    file_counts[behavior] += 1
                    print(f"  📄 {json_file.name}: {seq_count} sequences")
            except Exception as e:
                print(f"  ⚠️ Error reading {json_file.name}: {e}")
    
    print("\n" + "="*70)
    print("📈 Summary by Behavior:")
    print("="*70)
    
    total = 0
    for behavior in BEHAVIOR_NAMES:
        count = counts[behavior]
        files = file_counts[behavior]
        total += count
        status = "✅" if count >= 100 else "⚠️" if count >= 50 else "❌"
        print(f"  {status} {behavior.upper():15s}: {count:4d} sequences ({files} files)")
    
    print("-"*70)
    print(f"  📊 TOTAL: {total} sequences")
    
    # Progress indicator
    print("\n" + "="*70)
    print("🎯 Collection Progress:")
    print("="*70)
    target_per_class = 100
    total_target = target_per_class * 4
    
    for behavior in BEHAVIOR_NAMES:
        count = counts[behavior]
        progress = min(count / target_per_class, 1.0)
        bar_len = 40
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  {behavior:15s} [{bar}] {count}/{target_per_class} ({progress*100:.0f}%)")
    
    overall_progress = total / total_target
    print(f"\n  Overall: {total}/{total_target} ({overall_progress*100:.0f}%)")
    
    if total < total_target:
        remaining = total_target - total
        print(f"\n  💡 Need {remaining} more sequences to reach target")
    else:
        print(f"\n  🎉 Target reached! Ready for training")
    
    print("="*70)

if __name__ == "__main__":
    count_sequences()
