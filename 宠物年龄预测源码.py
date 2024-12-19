# 导入操作系统接口模块
import os
# 导入数值计算库
import numpy as np
# 导入数据处理库
import pandas as pd
# 导入图像处理库
from PIL import Image
# 导入进度条库
from tqdm import tqdm

# 导入PyTorch库及其子模块
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# 导入预训练模型库
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B4_Weights

# 导入数据增强库
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 导入机器学习评估指标和数据分割工具
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 导入TensorBoard日志记录库
from torch.utils.tensorboard import SummaryWriter
# 导入绘图库
import matplotlib.pyplot as plt

# 导入作业库，用于保存和加载缩放器
import joblib  # 用于保存和加载 scaler，这个是因为我后面希望用我训练的模型去预测其他的图片加上的

# 设置中文字体和负号显示，确保Matplotlib能够正确显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义一个函数用于设置随机种子，确保结果的可复现性
def set_seed(seed=42):
    import random
    random.seed(seed)  # 设置Python随机数生成器的种子
    np.random.seed(seed)  # 设置NumPy随机数生成器的种子
    torch.manual_seed(seed)  # 设置PyTorch CPU随机数生成器的种子
    torch.cuda.manual_seed_all(seed)  # 设置PyTorch GPU随机数生成器的种子

# 调用设置随机种子的函数
set_seed()

# 设置设备为GPU（如果可用）否则为CPU，确保模型在GPU上训练加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 确保自己是用gpu训练的
print(f'使用设备: {device}')

# 数据路径设定，指定训练集和验证集的图片目录及对应的标签文件路径
train_img_dir = "C:\\Users\\24942\\Desktop\\深度学习竞赛\\trainset"
val_img_dir = "C:\\Users\\24942\\Desktop\\深度学习竞赛\\valset"
train_label_path = "C:\\Users\\24942\\Desktop\\深度学习竞赛\\annotations\\annotations\\train.txt"
val_label_path = "C:\\Users\\24942\\Desktop\\深度学习竞赛\\annotations\\annotations\\val.txt"

# 定义一个函数用于加载标签文件，处理可能的编码问题
def load_labels(annotation_path):
    try:
        # 尝试以UTF-8编码读取标签文件
        df = pd.read_csv(annotation_path, sep='\t', header=None, names=['image_name', 'age_month'], encoding='utf-8')
    except UnicodeDecodeError:
        # 如果遇到编码错误，使用GBK编码重新读取
        df = pd.read_csv(annotation_path, sep='\t', header=None, names=['image_name', 'age_month'], encoding='gbk')
    return df

# 定义一个函数用于列出指定目录下所有有效格式的图片文件
def list_images(img_dir):
    # 定义有效的图片扩展名，当时解压的时候出问题了，再想是不是我自己没有获取全部格式的图片
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
    return images

# 定义一个函数将图片文件与其对应的标签进行匹配
def get_labels_for_images(images, label_df):
    # 清理标签数据中的图片名称，去除空格并转换为小写
    label_df['image_name_clean'] = label_df['image_name'].str.strip().str.lower()
    # 创建一个字典，将图片名称映射到年龄标签
    label_dict = dict(zip(label_df['image_name_clean'], label_df['age_month']))

    filtered_labels = []  # 存储有标签的图片信息
    missing_labels = []   # 存储缺少标签的图片名称
    for img in images:
        img_clean = img.strip().lower()
        if img_clean in label_dict:
            age = label_dict[img_clean]
            filtered_labels.append({'image_name': img, 'age_month': age})
        else:
            missing_labels.append(img)

    # 如果有缺少标签的图片，打印警告信息
    if missing_labels:
        print(f'警告: 以下图片没有对应的标签: {missing_labels}')

    # 返回包含有标签的图片信息的DataFrame
    return pd.DataFrame(filtered_labels)

# 加载训练集和验证集的标签
train_df = load_labels(train_label_path)
val_df = load_labels(val_label_path)

print(f'原始训练集标签数: {len(train_df)}')
print(f'原始验证集标签数: {len(val_df)}')

# 列出训练集和验证集目录中的所有图片
train_images = list_images(train_img_dir)
val_images = list_images(val_img_dir)

print(f'训练集目录中的图片数: {len(train_images)}')
print(f'验证集目录中的图片数: {len(val_images)}')

# 将图片与标签进行匹配，过滤出有对应标签的图片
train_labels_filtered = get_labels_for_images(train_images, train_df)
val_labels_filtered = get_labels_for_images(val_images, val_df)

print(f'过滤后训练集标签数: {len(train_labels_filtered)}')
print(f'过滤后验证集标签数: {len(val_labels_filtered)}')

# 检查训练集是否有图片缺少标签
missing_train_labels = set([img.lower() for img in train_images]) - set(
    [img.lower() for img in train_labels_filtered['image_name']])
if missing_train_labels:
    print(f'警告: 以下训练集图片没有对应的标签: {missing_train_labels}')

# 检查验证集是否有图片缺少标签
missing_val_labels = set([img.lower() for img in val_images]) - set(
    [img.lower() for img in val_labels_filtered['image_name']])
if missing_val_labels:
    print(f'警告: 以下验证集图片没有对应的标签: {missing_val_labels}')

# 确保验证集的标签数量与图片数量一致
assert len(val_labels_filtered) == len(val_images), "验证集标签数与验证集图片数不一致！"

# 将训练集进一步划分为训练集和内部验证集（80%训练，20%验证）
train_filtered, inner_val_filtered = train_test_split(
    train_labels_filtered,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f'划分后训练集样本数: {len(train_filtered)}')
print(f'划分后内部验证集样本数: {len(inner_val_filtered)}')

# 定义训练集的数据增强策略，使用Albumentations库
train_transforms = A.Compose([
    A.Resize(224, 224),  # 调整图片大小到224x224
    A.HorizontalFlip(p=0.5),  # 以50%的概率水平翻转
    A.VerticalFlip(p=0.2),    # 以20%的概率垂直翻转
    A.RandomBrightnessContrast(p=0.3),  # 以30%的概率随机调整亮度和对比度
    A.Rotate(limit=20, p=0.7),  # 以70%的概率随机旋转图片，旋转角度限制在±20度
    A.RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0), p=0.7),  # 以70%的概率随机裁剪并调整大小
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=0.4),  # 以40%的概率随机调整颜色
    A.CoarseDropout(max_holes=2, max_height=32, max_width=32, p=0.5),  # 以50%的概率随机遮挡
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),  # 以50%的概率随机平移、缩放和旋转
    A.GaussNoise(p=0.2),  # 以20%的概率添加高斯噪声
    A.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化图片
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),  # 将图片转换为PyTorch的Tensor格式
])

# 定义验证集的数据处理流程，通常不进行数据增强，只进行必要的预处理
val_transforms = A.Compose([
    A.Resize(224, 224),  # 调整图片大小到224x224
    A.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化图片
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),  # 将图片转换为PyTorch的Tensor格式
])

# 定义测试集的数据处理流程，与验证集一致
test_transforms = A.Compose([
    A.Resize(224, 224),  # 调整图片大小到224x224
    A.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化图片
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),  # 将图片转换为PyTorch的Tensor格式
])

# 定义一个自定义的数据集类，支持目标变量的归一化
class PetAgeDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, target_scaler=None):
        """
        初始化数据集

        参数:
            df (DataFrame): 包含图片名称和标签的数据框
            img_dir (str): 图片所在的目录路径
            transform (albumentations.Compose): 图片预处理和数据增强的组合
            target_scaler (bool or scaler object): 是否对目标变量进行缩放，如果是True则使用传入的缩放器
        """
        self.df = df.reset_index(drop=True)  # 重置DataFrame的索引
        self.img_dir = img_dir  # 图片目录
        self.transform = transform  # 数据增强和预处理
        self.target_scaler = target_scaler  # 目标变量缩放器

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        获取指定索引的数据项

        参数:
            idx (int): 数据项的索引

        返回:
            tuple: (图片Tensor, 年龄标签Tensor)
        """
        img_name = self.df.loc[idx, 'image_name']  # 获取图片名称
        if self.target_scaler:
            age = self.df.loc[idx, 'age_month_scaled']  # 获取缩放后的年龄标签
        else:
            age = self.df.loc[idx, 'age_month']  # 获取原始年龄标签
        img_path = os.path.join(self.img_dir, img_name)  # 构建图片的完整路径
        try:
            # 打开并转换图片为RGB格式
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)  # 将图片转换为NumPy数组
            if self.transform:
                augmented = self.transform(image=image)  # 应用数据增强和预处理
                image = augmented['image']  # 获取处理后的图片
        except Exception as e:
            # 如果图片无法加载，打印警告并返回全黑图片
            print(f'警告: 无法加载图片 {img_path}，返回全黑图片。错误信息: {e}')
            image = torch.zeros(3, 224, 224, dtype=torch.float32)
        return image, torch.tensor(age, dtype=torch.float32)  # 返回图片和对应的年龄标签

# 对目标变量（年龄）进行标准化，减少不同量纲对模型训练的影响
target_scaler = StandardScaler()
train_filtered['age_month_scaled'] = target_scaler.fit_transform(train_filtered[['age_month']])  # 训练集缩放
inner_val_filtered['age_month_scaled'] = target_scaler.transform(inner_val_filtered[['age_month']])  # 内部验证集缩放
val_labels_filtered['age_month_scaled'] = target_scaler.transform(val_labels_filtered[['age_month']])  # 验证集缩放

# 保存缩放器，以便在预测时使用相同的缩放参数
joblib.dump(target_scaler, 'scaler.pkl')

# 创建训练集、内部验证集和测试集的数据集对象
train_dataset = PetAgeDataset(train_filtered, train_img_dir, transform=train_transforms, target_scaler=True)
inner_val_dataset = PetAgeDataset(inner_val_filtered, train_img_dir, transform=val_transforms, target_scaler=True)
test_dataset = PetAgeDataset(val_labels_filtered, val_img_dir, transform=test_transforms, target_scaler=True)

# 定义模型架构，这里使用EfficientNet作为回归模型
class EfficientNetRegressor(nn.Module):
    def __init__(self, pretrained=True, dropout_prob=0.5):
        """
        初始化EfficientNet回归模型

        参数:
            pretrained (bool): 是否使用在ImageNet上预训练的权重
            dropout_prob (float): dropout层的丢弃概率
        """
        super(EfficientNetRegressor, self).__init__()
        if pretrained:
            # 加载预训练的EfficientNet-B4模型
            self.model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        else:
            # 从头开始训练EfficientNet-B4模型
            self.model = models.efficientnet_b4(weights=None)
        num_features = self.model.classifier[1].in_features  # 获取分类器的输入特征数
        # 替换分类器的最后一层为自定义的回归层
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(dropout_prob),  # 添加Dropout层防止过拟合
            nn.Linear(num_features, 1)  # 输出单一的回归值
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x (Tensor): 输入的图片Tensor

        返回:
            Tensor: 预测的年龄
        """
        return self.model(x)

# 初始化模型，并将其移动到指定设备（GPU或CPU）
model = EfficientNetRegressor(pretrained=True).to(device)

# 解冻所有模型参数，使其在训练过程中可以更新
for param in model.parameters():
    param.requires_grad = True

# 定义损失函数和优化器
# 创建一个组合损失函数，结合L1损失（MAE）和L2损失（MSE）
class CombinedLoss(nn.Module):
    def __init__(self):
        """
        初始化组合损失函数
        """
        super(CombinedLoss, self).__init__()
        self.mae = nn.L1Loss()  # 定义L1损失
        self.mse = nn.MSELoss()  # 定义L2损失

    def forward(self, preds, targets):
        """
        计算组合损失

        参数:
            preds (Tensor): 模型的预测值
            targets (Tensor): 真实的标签值

        返回:
            Tensor: 组合后的损失值
        """
        return self.mae(preds, targets) + self.mse(preds, targets)

# 实例化组合损失函数
criterion = CombinedLoss()
# 定义优化器，使用AdamW优化器，并设置学习率和权重衰减
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
# 定义学习率调度器，使用余弦退火策略
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# 设置TensorBoard日志目录，用于记录训练过程中的各种指标（这里刚开始报错，搜索之后发现是路径不能有中文）
log_dir = "C:\\Users\\24942\\Desktop\\1\\pet_age_prediction_new"
if os.path.exists(log_dir):
    if os.path.isfile(log_dir):
        os.remove(log_dir)  # 如果日志目录存在且是文件，删除它
        print(f'已删除冲突文件: {log_dir}')
    else:
        print(f'日志目录已存在: {log_dir}')
else:
    os.makedirs(log_dir)  # 如果日志目录不存在，创建它
    print(f'已创建日志目录: {log_dir}')

# 初始化TensorBoard的SummaryWriter
writer = SummaryWriter(log_dir=log_dir)

# 定义训练一个epoch的函数
def train_epoch(model, loader, criterion, optimizer, amp_scaler, device):
    """
    训练模型一个epoch

    参数:
        model (nn.Module): 模型
        loader (DataLoader): 数据加载器
        criterion (nn.Module): 损失函数
        optimizer (Optimizer): 优化器
        amp_scaler (GradScaler): 混合精度缩放器
        device (torch.device): 设备

    返回:
        float: 该epoch的平均训练损失
    """
    model.train()  # 设置模型为训练模式
    running_loss = 0.0  # 初始化累计损失
    for images, ages in tqdm(loader, desc='Training', leave=False):
        images = images.to(device, non_blocking=True)  # 将图片移动到设备
        ages = ages.to(device).unsqueeze(1)  # 将年龄标签移动到设备并增加维度

        optimizer.zero_grad()  # 清零优化器的梯度
        with torch.cuda.amp.autocast():
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, ages)  # 计算损失

        amp_scaler.scale(loss).backward()  # 反向传播并缩放损失
        amp_scaler.unscale_(optimizer)  # 反缩放优化器的梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
        amp_scaler.step(optimizer)  # 优化器更新参数
        amp_scaler.update()  # 更新缩放器

        running_loss += loss.item() * images.size(0)  # 累加损失
    return running_loss / len(loader.dataset)  # 返回平均损失

# 定义验证一个epoch的函数
def validate_epoch(model, loader, criterion, device):
    """
    在验证集上评估模型一个epoch

    参数:
        model (nn.Module): 模型
        loader (DataLoader): 数据加载器
        criterion (nn.Module): 损失函数
        device (torch.device): 设备

    返回:
        float: 该epoch的平均验证损失
    """
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0  # 初始化累计损失
    with torch.no_grad():  # 禁用梯度计算
        for images, ages in tqdm(loader, desc='Validation', leave=False):
            images = images.to(device, non_blocking=True)  # 将图片移动到设备
            ages = ages.to(device).unsqueeze(1)  # 将年龄标签移动到设备并增加维度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, ages)  # 计算损失
            running_loss += loss.item() * images.size(0)  # 累加损失
    return running_loss / len(loader.dataset)  # 返回平均损失

# 定义评估模型的函数，用于获取预测值和真实值
def evaluate(model, loader, device):
    """
    评估模型，获取所有预测值和真实值

    参数:
        model (nn.Module): 模型
        loader (DataLoader): 数据加载器
        device (torch.device): 设备

    返回:
        tuple: (预测值列表, 真实值列表)
    """
    model.eval()  # 设置模型为评估模式
    preds = []  # 存储预测值
    true = []   # 存储真实值
    with torch.no_grad():  # 禁用梯度计算
        for images, ages in tqdm(loader, desc='Evaluating', leave=False):
            images = images.to(device, non_blocking=True)  # 将图片移动到设备
            outputs = model(images)  # 前向传播
            preds.extend(outputs.cpu().numpy().flatten())  # 获取预测值并添加到列表
            true.extend(ages.numpy())  # 获取真实值并添加到列表
    return preds, true  # 返回预测值和真实值

# 主训练循环，确保只有在直接运行脚本时才执行
if __name__ == '__main__':
    # 创建训练集、内部验证集和测试集的数据加载器（这里还发现一个问题就是， DataLoader必须在main函数中运行否则会报错）
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    inner_val_loader = DataLoader(inner_val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    num_epochs = 30  # 设定训练的总轮数
    patience = 10    # 早停的耐心值
    best_val_loss = float('inf')
    early_stop_counter = 0

    # 初始化混合精度训练的缩放器
    amp_scaler = torch.cuda.amp.GradScaler()

    train_losses = []  # 存储每轮的训练损失
    val_losses = []    # 存储每轮的验证损失

    # 开始训练循环
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')  # 打印当前轮数
        train_loss = train_epoch(model, train_loader, criterion, optimizer, amp_scaler, device)  # 训练一个epoch
        val_loss = validate_epoch(model, inner_val_loader, criterion, device)  # 验证一个epoch

        scheduler.step(val_loss)  # 更新学习率调度器
        current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | 当前学习率: {current_lr}')  # 打印损失和学习率

        # 将损失和学习率记录到TensorBoard
        writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch + 1)
        writer.add_scalar('Learning Rate', current_lr, epoch + 1)

        # 存储损失值以便后续绘图
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 早停逻辑：如果验证损失有提升，保存模型并重置计数器；否则增加计数器
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # 更新最佳验证损失
            torch.save(model.state_dict(), 'best_model.pth')  # 保存最佳模型的参数
            print('最佳模型已保存.')
            early_stop_counter = 0  # 重置早停计数器
        else:
            early_stop_counter += 1  # 增加早停计数器
            if early_stop_counter >= patience:
                print('早停触发。训练停止。')  # 打印早停信息
                break  # 退出训练循环

    writer.close()  # 关闭TensorBoard的写入器

    # 加载最佳模型的参数
    model.load_state_dict(torch.load('best_model.pth'))
    # 在内部验证集上进行评估，获取预测值和真实值
    val_preds_scaled, val_true_scaled = evaluate(model, inner_val_loader, device)
    # 逆缩放预测值和真实值，恢复到原始尺度
    val_preds = target_scaler.inverse_transform(np.array(val_preds_scaled).reshape(-1, 1)).flatten()
    val_true = target_scaler.inverse_transform(np.array(val_true_scaled).reshape(-1, 1)).flatten()
    # 计算MAE和MSE指标
    val_mae = mean_absolute_error(val_true, val_preds)
    val_mse = mean_squared_error(val_true, val_preds)
    print(f'\n内部验证集 MAE: {val_mae:.2f} 个月')
    print(f'内部验证集 MSE: {val_mse:.2f} 个月²')

    # 在测试集上进行预测
    test_preds_scaled, _ = evaluate(model, test_loader, device)
    # 逆缩放预测值，恢复到原始尺度
    test_preds = target_scaler.inverse_transform(np.array(test_preds_scaled).reshape(-1, 1)).flatten()
    # 将预测值限制在合理范围内，并取整
    test_preds = np.clip(test_preds, 0, 191)
    test_preds = np.round(test_preds).astype(int)

    # 确保预测结果数量与测试集图片数量一致
    assert len(test_preds) == len(val_labels_filtered), "预测结果数量与测试集图片数量不一致！"

    # 创建提交文件的数据框
    submission = pd.DataFrame({
        'image_name': val_labels_filtered['image_name'],
        'age_month': test_preds
    })

    # 检查是否有图片在验证集文件夹中未找到
    missing_in_valset = set(submission['image_name']) - set(val_images)
    # 如果存在缺失，打印警告并移除这些条目
    if missing_in_valset:
        print(f'警告: 以下图片在 valset 文件夹中未找到: {missing_in_valset}')
        submission = submission[~submission['image_name'].isin(missing_in_valset)]
    else:
        print('所有预测结果图片均存在于 valset 文件夹中。')

    # 将预测结果保存为文本文件，使用制表符分隔，不包含索引和表头
    submission.to_csv('pred_result.txt', sep='\t', index=False, header=False)
    print('\n预测结果已保存为 pred_result.txt')

    # 绘制训练和验证损失随Epoch变化的曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')  # 设置X轴标签
    plt.ylabel('Loss')   # 设置Y轴标签
    plt.title('Training and Validation Loss Over Epochs')  # 设置图表标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.savefig('loss_curve.png')  # 保存图像
    plt.show()  # 显示图像

    # 绘制预测值与真实值的散点图，评估模型的预测效果
    plt.figure(figsize=(6, 6))
    plt.scatter(val_true, val_preds, alpha=0.5)  # 绘制散点
    plt.xlabel('真实年龄 (个月)')  # 设置X轴标签
    plt.ylabel('预测年龄 (个月)')  # 设置Y轴标签
    plt.title('真实值 vs 预测值')  # 设置图表标题
    plt.plot([0, 200], [0, 200], 'r--')  # 绘制对角线参考线
    plt.grid(True)  # 显示网格
    plt.savefig('pred_vs_true.png')  # 保存图像
    plt.show()  # 显示图像
