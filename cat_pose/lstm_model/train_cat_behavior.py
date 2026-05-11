"""
训练猫咪行为LSTM分类模型
4类：一般、舔拭、搔抓、甩头
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import matplotlib
from matplotlib import font_manager

# 配置中文字体
font_path = 'C:\\Windows\\Fonts\\msyh.ttc'
font_manager.fontManager.addfont(font_path)
matplotlib.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
matplotlib.rcParams['axes.unicode_minus'] = False

from cat_behavior_lstm import CatBehaviorLSTM, count_parameters

# ==================== 配置 ====================
class Config:
    # 数据
    data_dir = Path(r"C:\cat_pose\lstm_model\behavior_data")
    
    # 模型
    input_size = 51
    hidden_size = 128
    num_layers = 2
    num_classes = 4
    dropout = 0.3
    
    # 训练
    batch_size = 32
    num_epochs = 80
    learning_rate = 0.001
    weight_decay = 1e-4
    
    # 数据分割
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    # 其他
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = Path(r"C:\cat_pose\lstm_model\checkpoints")
    patience = 12
    
    # 类别权重
    use_class_weight = True
    
    # 类别名称
    class_names = ['一般', '舔拭', '搔抓', '甩头']
    class_names_en = ['normal', 'licking', 'scratching', 'head_shaking']

config = Config()
config.save_dir.mkdir(exist_ok=True)

# ==================== 数据集 ====================
class CatBehaviorDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.sequences = []
        self.labels = []
        self.augment = augment
        
        # 加载所有JSON文件
        json_files = sorted(list(Path(data_dir).glob("*.json")))
        if not json_files:
            print(f"⚠️ 未找到数据文件: {data_dir}")
            return
        
        print(f"\n📂 加载数据文件:")
        file_seq_counts = {}
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            seq_count = 0
            for item in data:
                seq = np.array(item['sequence'], dtype=np.float32)
                label = int(item['label'])
                self.sequences.append(seq)
                self.labels.append(label)
                seq_count += 1
            
            file_seq_counts[json_file.name] = seq_count
            print(f"  ✅ {json_file.name}: {seq_count} sequences")
        
        # 统计类别
        self.class_counts = [self.labels.count(i) for i in range(4)]
        print(f"\n📊 总计: {sum(self.class_counts)} 个序列 (来自 {len(json_files)} 个文件)")
        
        # 检查空类别
        empty_classes = []
        for i, count in enumerate(self.class_counts):
            status = "✅" if count > 0 else "❌"
            print(f"  {status} {config.class_names[i]}: {count}")
            if count == 0:
                empty_classes.append(config.class_names[i])
        
        if empty_classes:
            print(f"\n⚠️ 警告: 以下类别没有数据: {', '.join(empty_classes)}")
            print("   建议: 收集所有4个类别的数据以获得最佳效果")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        if self.augment:
            seq = self.augment_sequence(seq)
        
        return torch.FloatTensor(seq), torch.LongTensor([label])[0]
    
    def augment_sequence(self, seq):
        """数据增强"""
        # 高斯噪声
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.02, seq.shape)
            seq = seq + noise
        
        # 随机遮挡
        if np.random.rand() < 0.2:
            mask_len = np.random.randint(1, 5)
            start = np.random.randint(0, len(seq) - mask_len)
            seq[start:start+mask_len] = 0
        
        return seq
    
    def get_class_weights(self):
        """计算类别权重（处理零样本）"""
        total = sum(self.class_counts)
        weights = []
        for c in self.class_counts:
            if c == 0:
                weights.append(0.0)  # 没有数据的类别权重为0
            else:
                weights.append(total / (len(self.class_counts) * c))
        return torch.FloatTensor(weights)

# ==================== 训练函数 ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels in tqdm(dataloader, desc='Training'):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(sequences)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    # 检查是否有数据
    if len(dataloader) == 0:
        return 0.0, 0.0, [], []
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc='Validation'):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            logits, _ = model(sequences)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if total == 0:
        return 0.0, 0.0, [], []
    
    return total_loss / len(dataloader), 100. * correct / total, all_preds, all_labels

# ==================== 可视化 ====================
def plot_training_history(history, save_path):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='训练损失', linewidth=2)
    axes[0].plot(history['val_loss'], label='验证损失', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练和验证损失')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='训练准确率', linewidth=2)
    axes[1].plot(history['val_acc'], label='验证准确率', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率 (%)')
    axes[1].set_title('训练和验证准确率')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 训练历史已保存: {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': '数量'})
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 混淆矩阵已保存: {save_path}")

# ==================== 主程式 ====================
def main():
    print("="*70)
    print("猫咪行为LSTM训练")
    print("="*70)
    
    # 检查数据
    if not config.data_dir.exists():
        print(f"❌ 数据目录不存在: {config.data_dir}")
        print("请先运行 collect_cat_behavior.py 收集数据")
        return
    
    data_files = list(config.data_dir.glob("*.json"))
    if not data_files:
        print(f"❌ 未找到数据文件")
        return
    
    print(f"\n找到 {len(data_files)} 个数据文件")
    
    # 加载数据
    full_dataset = CatBehaviorDataset(config.data_dir, augment=False)
    
    # 检查空类别
    empty_classes = [i for i, count in enumerate(full_dataset.class_counts) if count == 0]
    if empty_classes:
        empty_names = [config.class_names[i] for i in empty_classes]
        print(f"\n⚠️⚠️⚠️ 严重警告 ⚠️⚠️⚠️")
        print(f"以下类别完全没有数据: {', '.join(empty_names)}")
        print("\n问题:")
        print("  ❌ 模型无法学习这些类别")
        print("  ❌ 推理时会误判为其他类别")
        print("  ❌ 准确率会严重下降")
        print("\n建议:")
        print("  1. 收集所有4个类别的数据（每类至少50个）")
        print("  2. 或者修改模型只训练有数据的类别")
        print("\n如果继续训练，模型质量会很差！")
        response = input("\n确定要继续? (yes/no): ")
        if response.lower() != 'yes':
            print("训练已取消。请先收集完整数据。")
            return
    
    if len(full_dataset) < 40:
        print(f"⚠️ 数据量过少({len(full_dataset)}个), 建议至少200个")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            return
    
    # 数据分割
    total_size = len(full_dataset)
    train_size = int(total_size * config.train_ratio)
    val_size = int(total_size * config.val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    train_dataset.dataset.augment = True
    
    print(f"\n数据分割:")
    print(f"  训练集: {train_size}")
    print(f"  验证集: {val_size}")
    print(f"  测试集: {test_size}")
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                            shuffle=False, num_workers=0)
    
    # 创建模型
    print(f"\n创建模型...")
    model = CatBehaviorLSTM(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout
    ).to(config.device)
    
    print(f"模型参数量: {count_parameters(model):,}")
    print(f"设备: {config.device}")
    
    # 损失函数
    if config.use_class_weight:
        class_weights = full_dataset.get_class_weights().to(config.device)
        print(f"类别权重: {class_weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                          weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 训练历史
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    patience_counter = 0
    
    print("\n开始训练...")
    print("="*70)
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.device
        )
        
        # 如果验证集为空，跳过验证
        if val_size > 0:
            val_loss, val_acc, _, _ = validate(
                model, val_loader, criterion, config.device
            )
            scheduler.step(val_loss)
        else:
            val_loss, val_acc = train_loss, train_acc
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\n结果:")
        print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        if val_size > 0:
            print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        else:
            print(f"  (无验证集,使用训练指标)")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            save_path = config.save_dir / "best_cat_behavior_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': vars(config)
            }, save_path)
            print(f"  ✅ 保存最佳模型 (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            print(f"\n⏹️ Early stopping (patience={config.patience})")
            break
    
    # 绘制训练历史
    plot_training_history(history, config.save_dir / "training_history.png")
    
    # 测试集评估
    print("\n" + "="*70)
    print("测试集评估")
    print("="*70)
    
    checkpoint = torch.load(config.save_dir / "best_cat_behavior_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, config.device
    )
    
    print(f"\n测试集准确率: {test_acc:.2f}%")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(
        test_labels, test_preds,
        labels=list(range(len(config.class_names))),
        target_names=config.class_names,
        zero_division=0
    ))
    
    # 混淆矩阵
    plot_confusion_matrix(
        test_labels, test_preds,
        config.class_names,
        config.save_dir / "confusion_matrix.png"
    )
    
    print("\n" + "="*70)
    print("✅ 训练完成!")
    print(f"最佳模型: {config.save_dir / 'best_cat_behavior_model.pth'}")
    print("="*70)

if __name__ == "__main__":
    main()
