"""
Cat Pose Quality Classifier
使用 YOLO Pose 模型推理图片，根据关键点稳定性分类到正常/异常文件夹
"""
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import shutil
from tqdm import tqdm

# ==================== 配置 ====================
MODEL_PATH = r"C:\cat_pose\no_aug.pt"
INPUT_IMAGE_DIR = r"C:\cat_pose\test\images"
OUTPUT_BASE_DIR = r"C:\Users\homec\Downloads\10"
NORMAL_DIR = os.path.join(OUTPUT_BASE_DIR, "normal")
ABNORMAL_DIR = os.path.join(OUTPUT_BASE_DIR, "abnormal")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_BASE_DIR, "visualizations")

# YOLO 推理参数
IMGSZ = 640
CONF_THRES = 0.5
KP_CONF_THRES = 0.5
TOTAL_KPTS = 17

# 异常检测参数（基于 EDA2.py）
ABNORMAL_PERCENTILE = 0.95     # 95% 百分位数作为自动门槛
MIN_ABNORMAL_THRES = 0.42       # 最小异常分数门槛
ACCEL_WEIGHT = 0.3              # 加速度权重
EMA_ALPHA = 0.7                 # EMA 平滑系数

# 关键点完整性检测参数（基于 EDA.py）
MIN_VISIBLE_KPT_RATIO = 0.75    # 最少可见关键点比例
MIN_BODY_SCALE = 1e-3           # 最小身体尺度

# ==================== 关键点名称 ====================
KPT_NAMES = [
    "nose", "left_ear_tip", "right_ear_tip",
    "chest", "mid_back", "hip",
    "left_front_elbow", "left_front_paw",
    "right_front_elbow", "right_front_paw",
    "left_hind_knee", "left_hind_paw",
    "right_hind_knee", "right_hind_paw",
    "tail_base", "tail_mid", "tail_tip"
]

# ==================== 颜色定义 ====================
COLOR_HEAD = (255, 255, 0)
COLOR_BODY = (0, 255, 0)
COLOR_LIMB = (255, 0, 0)
COLOR_TAIL = (255, 0, 255)
COLOR_LEFT_FRONT = (255, 0, 255)   # 品红
COLOR_RIGHT_FRONT = (0, 255, 255)  # 青
COLOR_LEFT_HIND = (255, 165, 0)    # 橙
COLOR_RIGHT_HIND = (0, 255, 0)     # 绿

# ==================== 骨架链接 ====================
HEAD_LINKS = [(0, 1), (0, 2), (1, 2)]
BODY_LINKS = [(0, 3), (3, 4), (4, 5)]
FRONT_LIMBS = [(3, 6), (6, 7), (3, 8), (8, 9)]
HIND_LIMBS = [(5, 10), (10, 11), (5, 12), (12, 13)]
TAIL_LINKS = [(5, 14), (14, 15), (15, 16)]


def draw_links(frame, kpts, conf, links, color):
    """绘制骨架链接"""
    for a, b in links:
        if conf[a] > KP_CONF_THRES and conf[b] > KP_CONF_THRES:
            cv2.line(frame,
                     (int(kpts[a][0]), int(kpts[a][1])),
                     (int(kpts[b][0]), int(kpts[b][1])),
                     color, 2)


def draw_skeleton(frame, kpts, conf):
    """绘制完整骨架"""
    draw_links(frame, kpts, conf, HEAD_LINKS, COLOR_HEAD)
    draw_links(frame, kpts, conf, BODY_LINKS, COLOR_BODY)
    draw_links(frame, kpts, conf, [(3, 6), (6, 7)], COLOR_LEFT_FRONT)
    draw_links(frame, kpts, conf, [(3, 8), (8, 9)], COLOR_RIGHT_FRONT)
    draw_links(frame, kpts, conf, [(5, 10), (10, 11)], COLOR_LEFT_HIND)
    draw_links(frame, kpts, conf, [(5, 12), (12, 13)], COLOR_RIGHT_HIND)
    draw_links(frame, kpts, conf, TAIL_LINKS, COLOR_TAIL)
    
    # 绘制关键点
    for i, (x, y) in enumerate(kpts):
        if conf[i] > KP_CONF_THRES:
            if i in [6, 7]:
                color = COLOR_LEFT_FRONT
            elif i in [8, 9]:
                color = COLOR_RIGHT_FRONT
            elif i in [10, 11]:
                color = COLOR_LEFT_HIND
            elif i in [12, 13]:
                color = COLOR_RIGHT_HIND
            else:
                color = (0, 0, 255)
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)


def compute_body_scale(kpts):
    """计算身体尺度（胸部到髋部）"""
    return float(np.linalg.norm(kpts[3] - kpts[5]))


def compute_quality_score(kpts, conf):
    """
    计算图像质量分数（基于关键点完整性和分布）
    分数越高表示质量越差（异常度越高）
    """
    # 检查可见关键点比例
    visible_kpts = np.sum(conf > KP_CONF_THRES)
    visible_ratio = visible_kpts / TOTAL_KPTS
    
    if visible_ratio < MIN_VISIBLE_KPT_RATIO:
        return 99.0  # 关键点不完整，视为异常
    
    # 计算身体尺度
    body_scale = compute_body_scale(kpts)
    if body_scale < MIN_BODY_SCALE:
        return 99.0  # 身体尺度太小，视为异常
    
    # 计算关键点的平均信心度
    valid_conf = conf[conf > KP_CONF_THRES]
    mean_conf = float(np.mean(valid_conf))
    
    # 计算关键点分布（相对于身体尺度的标准差）
    # 身体中心（胸部到髋部的中点）
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
    
    # 综合评分：低信心度 + 高离散度 = 高分（异常）
    # 反转信心度（1 - mean_conf）使其作为异常指标
    score = (1.0 - mean_conf) * 5.0 + kpt_spread_std
    
    return score


def plot_analysis_charts(scores, results, threshold):
    """绘制异常情况分析表"""
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib import font_manager

    # 強制設置中文字型，確保資訊類別正確顯示
    font_path = 'C:\\Windows\\Fonts\\msyh.ttc'
    if os.path.exists(font_path):
        my_font = font_manager.FontProperties(fname=font_path)
        rcParams['font.sans-serif'] = [font_path]
        rcParams['axes.unicode_minus'] = False
    else:
        my_font = None
        rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False
    
    scores_array = np.array(scores)
    normal_count = len(results['normal'])
    abnormal_count = len(results['abnormal'])
    error_count = len(results['error'])
    total_count = len(scores)
    
    # 创建4个子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cat Pose Quality Classification Analysis', fontsize=16, fontweight='bold')
    
    # ========== 1. 直方图 + 门槛线 ==========
    axes[0, 0].hist(scores_array, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(scores_array), color='green', linestyle='--', linewidth=2.5, label=f'Mean: {np.mean(scores_array):.3f}')
    axes[0, 0].axvline(np.median(scores_array), color='orange', linestyle='--', linewidth=2.5, label=f'Median: {np.median(scores_array):.3f}')
    axes[0, 0].axvline(threshold, color='red', linestyle='-', linewidth=3, label=f'Threshold: {threshold:.3f}')
    axes[0, 0].set_xlabel('Quality Score', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Score Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # ========== 2. 饼图 ==========
    sizes = [normal_count, abnormal_count, error_count]
    labels = [f'Normal\n({normal_count})', f'Abnormal\n({abnormal_count})', f'Error\n({error_count})']
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    explode = (0.05, 0.1, 0.05)

    if my_font:
        wedges, texts, autotexts = axes[0, 1].pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            explode=explode, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold', 'fontproperties': my_font})
        axes[0, 1].set_title('Classification Result', fontsize=12, fontweight='bold', fontproperties=my_font)
    else:
        wedges, texts, autotexts = axes[0, 1].pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            explode=explode, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        axes[0, 1].set_title('Classification Result', fontsize=12, fontweight='bold')

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
        if my_font:
            autotext.set_fontproperties(my_font)
    
    # ========== 3. 箱线图 + 统计 ==========
    box = axes[1, 0].boxplot(scores_array, vert=True, patch_artist=True, widths=0.5)
    box['boxes'][0].set_facecolor('lightblue')
    box['boxes'][0].set_edgecolor('darkblue')
    for whisker in box['whiskers']:
        whisker.set(linewidth=2)
    for cap in box['caps']:
        cap.set(linewidth=2)
    axes[1, 0].axhline(threshold, color='red', linestyle='--', linewidth=2.5, label='Threshold')
    axes[1, 0].set_ylabel('Quality Score', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Score Box Plot', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_xticklabels(['All Scores'])
    
    # 添加统计信息文本
    stats_text = f"""
统计信息：
Mean:           {np.mean(scores_array):.4f}
Median:         {np.median(scores_array):.4f}
Std Dev:        {np.std(scores_array):.4f}
Min:            {np.min(scores_array):.4f}
Max:            {np.max(scores_array):.4f}

总样本数：{total_count}
正常图像：{normal_count} ({100*normal_count/total_count:.1f}%)
异常图像：{abnormal_count} ({100*abnormal_count/total_count:.1f}%)
处理错误：{error_count} ({100*error_count/total_count:.1f}%)
    """
    if my_font:
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        fontproperties=my_font)
    else:
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def create_output_directories():
    """创建输出目录（先清空再创建）"""
    # 清空目录
    for dir_path in [NORMAL_DIR, ABNORMAL_DIR, VISUALIZATIONS_DIR]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    
    # 重新创建
    os.makedirs(NORMAL_DIR, exist_ok=True)
    os.makedirs(ABNORMAL_DIR, exist_ok=True)
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    print(f"✓ 输出目录已清空并重建:")
    print(f"  - {OUTPUT_BASE_DIR}/")
    print(f"    - normal/")
    print(f"    - abnormal/")
    print(f"    - visualizations/")


def process_images():
    """处理所有图像"""
    # 检查文件
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 模型文件不存在: {MODEL_PATH}")
        return
    
    if not os.path.exists(INPUT_IMAGE_DIR):
        print(f"❌ 错误: 输入图像文件夹不存在: {INPUT_IMAGE_DIR}")
        return
    
    # 加载模型
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
        return
    
    print(f"找到 {len(image_files)} 个图像文件\n")
    
    # 创建输出目录
    create_output_directories()
    print()
    
    # 处理每个图像
    scores = []
    results = {
        'normal': [],
        'abnormal': [],
        'error': []
    }
    
    print("开始处理图像...")
    print("=" * 80)
    
    for idx, image_file in enumerate(tqdm(image_files, desc="处理进度"), 1):
        image_path = os.path.join(INPUT_IMAGE_DIR, image_file)
        
        try:
            # 读取图像
            frame = cv2.imread(image_path)
            if frame is None:
                results['error'].append((image_file, "无法读取图像"))
                continue
            
            h, w = frame.shape[:2]
            
            # YOLO 推理
            result = model.predict(
                frame,
                imgsz=IMGSZ,
                conf=CONF_THRES,
                half=True,
                verbose=False
            )[0]
            
            # 检查是否检测到关键点
            if result.keypoints is None or len(result.keypoints.xy) == 0:
                results['error'].append((image_file, "未检测到关键点"))
                continue
            
            kpts = result.keypoints.xy[0].cpu().numpy()
            conf = result.keypoints.conf[0].cpu().numpy()
            
            # 计算质量分数
            score = compute_quality_score(kpts, conf)
            scores.append(score)
            
            # 分类
            is_abnormal = score > MIN_ABNORMAL_THRES
            dest_dir = ABNORMAL_DIR if is_abnormal else NORMAL_DIR
            
            # 复制图像
            dest_path = os.path.join(dest_dir, image_file)
            shutil.copy2(image_path, dest_path)
            
            category = "异常" if is_abnormal else "正常"
            if is_abnormal:
                results['abnormal'].append((image_file, score))
            else:
                results['normal'].append((image_file, score))
            
            # 绘制可视化（保存到 visualizations 文件夹）
            vis_frame = frame.copy()
            draw_skeleton(vis_frame, kpts, conf)
            
            # 添加分数标注
            status_text = f"{'ABNORMAL' if is_abnormal else 'NORMAL'} (Score: {score:.3f})"
            status_color = (0, 0, 255) if is_abnormal else (0, 255, 0)
            cv2.putText(vis_frame, status_text, (15, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
            cv2.putText(vis_frame, status_text, (15, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
            
            # 添加详细信息
            visible_kpts = np.sum(conf > KP_CONF_THRES)
            info_texts = [
                f"Visible KPTs: {visible_kpts}/{TOTAL_KPTS}",
                f"Mean Conf: {np.mean(conf[conf > KP_CONF_THRES]):.3f}",
                f"Image Size: {w}x{h}"
            ]
            
            for i, txt in enumerate(info_texts):
                cv2.putText(vis_frame, txt, (15, 70 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            vis_path = os.path.join(VISUALIZATIONS_DIR, f"vis_{image_file}")
            cv2.imwrite(vis_path, vis_frame)
        
        except Exception as e:
            results['error'].append((image_file, str(e)))
    
    # 绘制异常情况分析表
    if scores:
        plot_analysis_charts(scores, results, MIN_ABNORMAL_THRES)
    
    # 计算统计信息
    print("\n" + "=" * 80)
    print("\n📊 处理完成！统计信息:")
    print("-" * 80)
    print(f"✓ 正常图像: {len(results['normal'])}")
    print(f"⚠ 异常图像: {len(results['abnormal'])}")
    print(f"❌ 错误: {len(results['error'])}")
    print(f"📈 总计: {len(image_files)}")
    
    if scores:
        print(f"\n📉 质量分数统计:")
        print(f"  Mean: {np.mean(scores):.3f}")
        print(f"  Min: {np.min(scores):.3f}")
        print(f"  Max: {np.max(scores):.3f}")
        print(f"  Median: {np.median(scores):.3f}")
        print(f"  Threshold: {MIN_ABNORMAL_THRES:.3f}")
    
    print(f"\n📁 输出路径:")
    print(f"  正常: {NORMAL_DIR}/")
    print(f"  异常: {ABNORMAL_DIR}/")
    print(f"  可视化: {VISUALIZATIONS_DIR}/")
    
    # 列出异常图像（前 10 个）
    if results['abnormal']:
        print(f"\n⚠ 异常图像排名（Top 10）:")
        sorted_abnormal = sorted(results['abnormal'], key=lambda x: x[1], reverse=True)
        for rank, (fname, score) in enumerate(sorted_abnormal[:10], 1):
            print(f"  #{rank} {fname}: {score:.3f}")
    
    # 列出错误图像
    if results['error']:
        print(f"\n❌ 处理错误（Top 5）:")
        for fname, error in results['error'][:5]:
            print(f"  {fname}: {error}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Cat Pose Quality Classifier")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  模型: {MODEL_PATH}")
    print(f"  输入: {INPUT_IMAGE_DIR}")
    print(f"  输出: {OUTPUT_BASE_DIR}")
    print(f"  异常门槛: {MIN_ABNORMAL_THRES}")
    print()
    
    process_images()
    
    print("\n" + "=" * 80)
    print("✓ 处理完成！")
    print("=" * 80 + "\n")
