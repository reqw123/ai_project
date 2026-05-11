"""
找到最优异常门槛
先扫描所有图像，分析分数分布，给出数据驱动的建议
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==================== 配置 ====================
MODEL_PATH = r"C:\cat_pose\no_aug.pt"
INPUT_IMAGE_DIR = r"C:\cat_pose\test\images"

# YOLO 推理参数
IMGSZ = 640
CONF_THRES = 0.5
KP_CONF_THRES = 0.5
TOTAL_KPTS = 17

# 关键点完整性检测参数
MIN_VISIBLE_KPT_RATIO = 0.75
MIN_BODY_SCALE = 1e-3


def compute_body_scale(kpts):
    """计算身体尺度（胸部到髋部）"""
    return float(np.linalg.norm(kpts[3] - kpts[5]))


def compute_quality_score(kpts, conf):
    """
    计算图像质量分数
    分数越高表示质量越差（异常度越高）
    """
    # 检查可见关键点比例
    visible_kpts = np.sum(conf > KP_CONF_THRES)
    visible_ratio = visible_kpts / TOTAL_KPTS
    
    if visible_ratio < MIN_VISIBLE_KPT_RATIO:
        return 99.0  # 关键点不完整
    
    # 计算身体尺度
    body_scale = compute_body_scale(kpts)
    if body_scale < MIN_BODY_SCALE:
        return 99.0  # 身体尺度太小
    
    # 计算关键点的平均信心度
    valid_conf = conf[conf > KP_CONF_THRES]
    mean_conf = float(np.mean(valid_conf))
    
    # 计算关键点分布
    body_center = (kpts[3] + kpts[5]) / 2
    distances = []
    for i, (x, y) in enumerate(kpts):
        if conf[i] > KP_CONF_THRES:
            dist = np.linalg.norm(np.array([x, y]) - body_center)
            norm_dist = dist / body_scale
            distances.append(norm_dist)
    
    if distances:
        kpt_spread_std = float(np.std(distances))
    else:
        kpt_spread_std = 0.0
    
    # 综合评分
    score = (1.0 - mean_conf) * 5.0 + kpt_spread_std
    
    return score


def scan_all_images():
    """扫描所有图像并计算分数"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 模型文件不存在: {MODEL_PATH}")
        return []
    
    if not os.path.exists(INPUT_IMAGE_DIR):
        print(f"❌ 错误: 输入图像文件夹不存在: {INPUT_IMAGE_DIR}")
        return []
    
    print(f"正在加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    model.to("cuda")
    print("✓ 模型加载成功\n")
    
    # 获取所有图像
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in os.listdir(INPUT_IMAGE_DIR)
                   if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        print(f"❌ 错误: 在 {INPUT_IMAGE_DIR} 中未找到图像文件")
        return []
    
    print(f"找到 {len(image_files)} 个图像文件\n")
    
    scores = []
    error_count = 0
    
    print("正在扫描所有图像...")
    for image_file in tqdm(image_files):
        image_path = os.path.join(INPUT_IMAGE_DIR, image_file)
        
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                continue
            
            # YOLO 推理
            result = model.predict(
                frame,
                imgsz=IMGSZ,
                conf=CONF_THRES,
                half=True,
                verbose=False
            )[0]
            
            if result.keypoints is None or len(result.keypoints.xy) == 0:
                continue
            
            kpts = result.keypoints.xy[0].cpu().numpy()
            conf = result.keypoints.conf[0].cpu().numpy()
            
            score = compute_quality_score(kpts, conf)
            scores.append(score)
        
        except Exception as e:
            error_count += 1
    
    print(f"\n✓ 扫描完成")
    print(f"  成功: {len(scores)} 个")
    print(f"  错误: {error_count} 个\n")
    
    return np.array(scores)


def analyze_distribution(scores):
    """分析分数分布并给出门槛建议"""
    if len(scores) == 0:
        print("❌ 没有有效的分数数据")
        return
    
    print("=" * 80)
    print("📊 分数分布分析")
    print("=" * 80)
    
    # 基本统计
    print(f"\n📈 基本统计:")
    print(f"  Mean:       {np.mean(scores):.3f}")
    print(f"  Median:     {np.median(scores):.3f}")
    print(f"  Std Dev:    {np.std(scores):.3f}")
    print(f"  Min:        {np.min(scores):.3f}")
    print(f"  Max:        {np.max(scores):.3f}")
    
    # 百分位数
    print(f"\n📍 百分位数（用于门槛参考）:")
    percentiles = [50, 75, 80, 85, 90, 95, 97, 99]
    for p in percentiles:
        val = np.percentile(scores, p)
        count = np.sum(scores > val)
        pct = 100 * count / len(scores)
        print(f"  {p}th: {val:.3f}  (高于该值: {count:4d} 图像, {pct:5.1f}%)")
    
    # 建议的门槛
    print(f"\n💡 门槛建议:")
    print(f"\n  根据不同的业务需求：")
    
    # 建议 1: 保守（低假正例率，可能遗漏异常）
    thres_conservative = np.percentile(scores, 95)
    count_conservative = np.sum(scores > thres_conservative)
    print(f"\n  1️⃣  保守（捕捉 5% 最异常）")
    print(f"     门槛 = {thres_conservative:.3f}")
    print(f"     异常图像: {count_conservative} ({100*count_conservative/len(scores):.1f}%)")
    
    # 建议 2: 平衡（常用设置）
    thres_balanced = np.percentile(scores, 90)
    count_balanced = np.sum(scores > thres_balanced)
    print(f"\n  2️⃣  平衡（捕捉 10% 最异常）")
    print(f"     门槛 = {thres_balanced:.3f}")
    print(f"     异常图像: {count_balanced} ({100*count_balanced/len(scores):.1f}%)")
    
    # 建议 3: 激进（高召回率，可能有假正例）
    thres_aggressive = np.percentile(scores, 85)
    count_aggressive = np.sum(scores > thres_aggressive)
    print(f"\n  3️⃣  激进（捕捉 15% 最异常）")
    print(f"     门槛 = {thres_aggressive:.3f}")
    print(f"     异常图像: {count_aggressive} ({100*count_aggressive/len(scores):.1f}%)")
    
    # 建议 4: 非常激进
    thres_very_aggressive = np.percentile(scores, 75)
    count_very_aggressive = np.sum(scores > thres_very_aggressive)
    print(f"\n  4️⃣  非常激进（捕捉 25% 最异常）")
    print(f"     门槛 = {thres_very_aggressive:.3f}")
    print(f"     异常图像: {count_very_aggressive} ({100*count_very_aggressive/len(scores):.1f}%)")
    
    # 极值
    print(f"\n  5️⃣  仅捕捉极端异常（Mean + 2 * Std）")
    thres_extreme = np.mean(scores) + 2 * np.std(scores)
    count_extreme = np.sum(scores > thres_extreme)
    print(f"     门槛 = {thres_extreme:.3f}")
    print(f"     异常图像: {count_extreme} ({100*count_extreme/len(scores):.1f}%)")
    
    print(f"\n" + "=" * 80)
    
    # 分布可视化
    print(f"\n📊 生成可视化图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 直方图
    axes[0, 0].hist(scores, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 0].axvline(np.median(scores), color='green', linestyle='--', linewidth=2, label='Median')
    axes[0, 0].set_xlabel('Quality Score', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Score Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 箱线图
    axes[0, 1].boxplot(scores, vert=True)
    axes[0, 1].set_ylabel('Quality Score', fontsize=11)
    axes[0, 1].set_title('Score Box Plot', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 门槛对比
    thresholds = {
        'Conservative\n(95th percentile)': thres_conservative,
        'Balanced\n(90th percentile)': thres_balanced,
        'Aggressive\n(85th percentile)': thres_aggressive,
        'Very Aggressive\n(75th percentile)': thres_very_aggressive,
    }
    
    abnormal_counts = {name: np.sum(scores > thres) for name, thres in thresholds.items()}
    colors_bar = ['#FF6B6B', '#FFA500', '#FFD700', '#90EE90']
    
    axes[1, 0].bar(thresholds.keys(), abnormal_counts.values(), color=colors_bar, edgecolor='black')
    axes[1, 0].set_ylabel('Number of Abnormal Images', fontsize=11)
    axes[1, 0].set_title('Threshold Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    for i, (name, count) in enumerate(abnormal_counts.items()):
        axes[1, 0].text(i, count + 5, str(count), ha='center', fontsize=10, fontweight='bold')
    
    # 累积分布
    sorted_scores = np.sort(scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
    axes[1, 1].plot(sorted_scores, cumulative, linewidth=2.5, color='darkblue')
    
    # 标注百分位数
    for p in [75, 85, 90, 95]:
        val = np.percentile(scores, p)
        axes[1, 1].axvline(val, color='gray', linestyle=':', alpha=0.5)
        axes[1, 1].text(val, 5, f'{p}%', fontsize=9, rotation=0)
    
    axes[1, 1].set_xlabel('Quality Score', fontsize=11)
    axes[1, 1].set_ylabel('Cumulative Percentage (%)', fontsize=11)
    axes[1, 1].set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('threshold_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ 已保存到: threshold_analysis.png")
    plt.show()
    
    # 建议总结
    print(f"\n" + "=" * 80)
    print("🎯 选择建议:")
    print("=" * 80)
    print(f"""
根据你的数据，我建议：

✅ 推荐使用 {thres_balanced:.3f}（90th percentile）
   - 这是保险和有效的中间值
   - 会筛出约 {count_balanced} 个异常图像（{100*count_balanced/len(scores):.1f}%）
   
💡 调整建议：
   - 如果想更严格（捕捉更多问题）→ 使用 {thres_aggressive:.3f}
   - 如果想更宽松（只捕捉明显问题）→ 使用 {thres_conservative:.3f}
   - 可以根据后续的标注结果反复微调

💾 修改方法：
   在 classify_pose_quality.py 中修改这行：
   MIN_ABNORMAL_THRES = {thres_balanced:.3f}
    """)
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🔍 异常门槛优化工具")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  模型: {MODEL_PATH}")
    print(f"  输入: {INPUT_IMAGE_DIR}\n")
    
    scores = scan_all_images()
    
    if len(scores) > 0:
        analyze_distribution(scores)
    
    print("\n✓ 分析完成！\n")
