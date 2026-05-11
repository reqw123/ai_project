"""
自动生成 labels.json 脚本
========================
根据骨骼JSON文件名称自动生成对应的标签

使用方式:
  python generate_labels.py    # 直接运行，使用预设路径

文件名格式示例:
  walk_1.json → walk (类别0)
  lying_2.json → lying (类别1)
  lip_3.json → lip (类别2)
  stop_4.json → stop (类别3)
"""

import json
from pathlib import Path
from collections import defaultdict

# ==================== 配置 (编辑这里改变路径) ====================
# 预设路径
SKELETON_DIR = r"C:\cat_pose\gcn_pose\skeletons"          # 骨骼JSON文件所在目录
OUTPUT_FILE = r"C:\cat_pose\gcn_pose\labels.json"         # 输出标签文件路径

# 類別映射（文件名前綴 → 類別ID）
CLASS_MAPPING = {
    'walk': 0,
    'lying': 1,
    'lick': 2,
    'shake': 3,
}

# 可選的其他別名映射（如果你用其他名字）
ALIAS_MAPPING = {
    'walking': 'walk',
    'lie': 'lying',
    'licking': 'lick',
    'lick': 'lick',
    'lip': 'lick',
    'standing': 'shake',
    'idle': 'shake',
    'rest': 'lying',
}


# ==================== 核心函数 ====================
def extract_class_from_filename(filename):
    """
    从文件名提取类别
    
    示例:
      walk_1.json → walk
      lying_2.json → lying
      lick_3.json → lick
      shake_4.json → shake                                      
      walk1.json → walk
      walking_cat_1.json → walk
    """
    stem = Path(filename).stem  # 移除 .json
    
    # 分割 - 或 _
    parts = stem.replace('-', '_').split('_')
    
    if not parts:
        return None
    
    # 第一个单词通常是类别
    first_part = parts[0].lower()
    
    # 检查第一个单词是否是类别
    if first_part in CLASS_MAPPING:
        return first_part
    
    # 检查别名
    if first_part in ALIAS_MAPPING:
        return ALIAS_MAPPING[first_part]
    
    return None


def generate_labels(skeleton_dir=SKELETON_DIR, output_file=OUTPUT_FILE, verbose=True):
    """
    主函数：生成标签文件
    
    Args:
        skeleton_dir: 骨骼文件目录 (默认使用预设路径)
        output_file: 输出标签文件名 (默认使用预设路径)
        verbose: 是否打印详细信息
    
    Returns:
        labels_dict: 生成的标签字典
    """
    # 确定路径
    skeleton_dir = Path(skeleton_dir)
    
    if not skeleton_dir.exists():
        print(f"✗ 目录不存在: {skeleton_dir}")
        return None
    
    if verbose:
        print("="*60)
        print("自动生成 labels.json")
        print("="*60)
        print(f"\n搜索目录: {skeleton_dir.absolute()}\n")
    
    # 找所有JSON文件
    skeleton_files = sorted(skeleton_dir.glob("*.json"))
    
    if not skeleton_files:
        print(f"✗ 未找到JSON文件在 {skeleton_dir}")
        return None
    
    if verbose:
        print(f"找到 {len(skeleton_files)} 个JSON文件\n")
    
    # 生成标签
    labels = {}
    class_stats = defaultdict(list)
    unmapped_files = []
    
    for json_file in skeleton_files:
        video_id = json_file.stem
        class_name = extract_class_from_filename(json_file.name)
        
        if class_name is None:
            if verbose:
                unmapped_files.append(video_id)
            continue
        
        class_id = CLASS_MAPPING[class_name]
        labels[video_id] = class_id
        class_stats[class_name].append(video_id)
    
    # 打印統計
    if verbose:
        print("生成的标签统计:")
        print("-" * 60)
        total = len(labels)
        for class_name in ['walk', 'lying', 'lick', 'shake']:
            count = len(class_stats[class_name])
            percentage = (count / total * 100) if total > 0 else 0
            class_id = CLASS_MAPPING[class_name]
            print(f"  {class_name:10s} (ID={class_id}): {count:3d} 个 ({percentage:5.1f}%)")
        print(f"\n  总计: {total} 个文件")
        if unmapped_files:
            print(f"\n⚠ {len(unmapped_files)} 个文件未能识别:")
            for fname in unmapped_files[:5]:  # 只显示前5个
                print(f"    - {fname}")
            if len(unmapped_files) > 5:
                print(f"    ... 还有 {len(unmapped_files)-5} 个")
    
    # 保存标签文件
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\n✓ 标签文件已保存: {output_path.absolute()}")
        print(f"  共 {len(labels)} 个条目\n")
    
    return labels


def verify_labels(labels_file=OUTPUT_FILE, skeleton_dir=SKELETON_DIR, verbose=True):
    """
    验证标签文件与骨骼文件的对应关系
    
    Args:
        labels_file: 标签JSON文件
        skeleton_dir: 骨骼文件目录
        verbose: 是否打印详细信息
    """
    skeleton_dir = Path(skeleton_dir)
    
    if verbose:
        print("="*60)
        print("验证标签文件")
        print("="*60 + "\n")
    
    # 加载标签
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    # 获取骨骼文件
    skeleton_files = {f.stem for f in skeleton_dir.glob("*.json")}
    
    # 检查匹配
    matched = 0
    missing = []
    extra = []
    
    for video_id, class_id in labels.items():
        if video_id in skeleton_files:
            matched += 1
        else:
            missing.append(video_id)
    
    for skeleton_id in skeleton_files:
        if skeleton_id not in labels:
            extra.append(skeleton_id)
    
    # 打印结果
    if verbose:
        print(f"标签文件: {labels_file}")
        print(f"骨骼目录: {skeleton_dir}\n")
        
        print(f"标签总数: {len(labels)}")
        print(f"骨骼文件: {len(skeleton_files)}")
        print(f"匹配: {matched}\n")
        
        if missing:
            print(f"✗ {len(missing)} 个标签的骨骼文件缺失:")
            for vid in missing[:5]:
                print(f"    - {vid}")
            if len(missing) > 5:
                print(f"    ... 还有 {len(missing)-5} 个")
            print()
        
        if extra:
            print(f"⚠ {len(extra)} 个骨骼文件没有对应标签:")
            for vid in extra[:5]:
                print(f"    - {vid}")
            if len(extra) > 5:
                print(f"    ... 还有 {len(extra)-5} 个")
            print()
        
        if not missing and not extra:
            print("✓ 所有文件完全匹配！\n")
        
        # 類別分布
        class_names = {0: 'walk', 1: 'lying', 2: 'lick', 3: 'shake'}
        class_counts = {cls: 0 for cls in range(4)}
        for class_id in labels.values():
            if class_id in class_counts:
                class_counts[class_id] += 1
        print("类别分布:")
        for class_id in range(4):
            class_name = class_names[class_id]
            count = class_counts[class_id]
            print(f"  {class_name:10s} (ID={class_id}): {count:3d} 个")
        print()
    
    return len(missing) == 0 and len(extra) == 0


def edit_labels_interactive(labels_file=OUTPUT_FILE, verbose=True):
    """
    交互式编辑标签文件
    
    Args:
        labels_file: 标签JSON文件
        verbose: 是否打印详细信息
    """
    # 加载标签
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    class_names = {0: 'walk', 1: 'lying', 2: 'lick', 3: 'shake'}
    reverse_mapping = {v: k for k, v in class_names.items()}
    
    if verbose:
        print("="*60)
        print("交互式编辑标签")
        print("="*60 + "\n")
        print("命令:")
        print("  list           - 列出所有标签")
        print("  set <id> <cls> - 设置标签 (set walk_1 lip)")
        print("  delete <id>    - 删除标签")
        print("  find <cls>     - 查找某类别")
        print("  stats          - 显示统计")
        print("  save           - 保存并退出")
        print("  quit           - 退出不保存\n")
    
    while True:
        cmd = input("> ").strip().split()
        
        if not cmd:
            continue
        
        if cmd[0] == 'list':
            for vid, cls_id in sorted(labels.items()):
                cls_name = class_names[cls_id]
                print(f"  {vid:20s} → {cls_name}")
        
        elif cmd[0] == 'set' and len(cmd) >= 3:
            video_id = cmd[1]
            class_name = cmd[2].lower()
            
            if class_name not in reverse_mapping:
                print(f"✗ 类别不存在: {class_name}")
                print(f"  可选: {', '.join(class_names.values())}")
                continue
            
            class_id = reverse_mapping[class_name]
            labels[video_id] = class_id
            print(f"✓ {video_id} → {class_name}")
        
        elif cmd[0] == 'delete' and len(cmd) >= 2:
            video_id = cmd[1]
            if video_id in labels:
                del labels[video_id]
                print(f"✓ 已删除 {video_id}")
            else:
                print(f"✗ 未找到 {video_id}")
        
        elif cmd[0] == 'find' and len(cmd) >= 2:
            class_name = cmd[1].lower()
            if class_name not in reverse_mapping:
                print(f"✗ 类别不存在")
                continue
            
            class_id = reverse_mapping[class_name]
            found = [vid for vid, cid in labels.items() if cid == class_id]
            print(f"找到 {len(found)} 个 {class_name}:")
            for vid in sorted(found):
                print(f"  - {vid}")
        
        elif cmd[0] == 'stats':
            class_counts = {i: 0 for i in range(4)}
            for cid in labels.values():
                class_counts[cid] += 1
            
            print("统计:")
            for cid in range(4):
                cname = class_names[cid]
                count = class_counts[cid]
                print(f"  {cname:10s}: {count}")
        
        elif cmd[0] == 'save':
            with open(labels_file, 'w', encoding='utf-8') as f:
                json.dump(labels, f, indent=2, ensure_ascii=False)
            print(f"✓ 已保存到 {labels_file}")
            break
        
        elif cmd[0] == 'quit':
            print("退出，不保存")
            break
        
        else:
            print("✗ 未知命令")


# ==================== 主函数 ====================
if __name__ == "__main__":
    import sys
    
    # 如果有命令行参数，执行对应操作
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == 'verify':
            # 验证标签
            verify_labels(OUTPUT_FILE, SKELETON_DIR, verbose=True)
        
        elif cmd == 'edit':
            # 编辑标签
            edit_labels_interactive(OUTPUT_FILE, verbose=True)
        
        else:
            print(f"未知命令: {cmd}")
            print("\n可用命令:")
            print("  python generate_labels.py          # 生成标签")
            print("  python generate_labels.py verify   # 验证标签")
            print("  python generate_labels.py edit     # 交互式编辑")
    
    else:
        # 默认: 生成标签
        labels = generate_labels(SKELETON_DIR, OUTPUT_FILE, verbose=True)
        
        if labels:
            # 自动验证
            verify_labels(OUTPUT_FILE, SKELETON_DIR, verbose=True)