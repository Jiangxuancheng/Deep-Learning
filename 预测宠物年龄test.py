import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B4_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2
import joblib



# 定义模型架构
class EfficientNetRegressor(nn.Module):
    def __init__(self, pretrained=True, dropout_prob=0.5):
        super(EfficientNetRegressor, self).__init__()
        if pretrained:
            self.model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        else:
            self.model = models.efficientnet_b4(weights=None)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(num_features, 1)
        )

    def forward(self, x):
        return self.model(x)

# 定义预处理函数
def get_prediction_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# 定义预测函数
def predict_image(image_path, model, scaler, transform, device):
    """
    对单张图片进行预测。

    参数:
        image_path (str): 图片路径。
        model (nn.Module): 训练好的模型。
        scaler (StandardScaler): 用于反归一化的缩放器。
        transform (albumentations.Compose): 预处理变换。
        device (torch.device): 设备（CPU或GPU）。

    返回:
        float: 预测的年龄（个月）。
    """
    try:
        # 加载图片
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        # 应用预处理
        augmented = transform(image=image)
        image = augmented['image'].unsqueeze(0).to(device)  # 添加批次维度

        with torch.no_grad():
            output = model(image)
            output = output.cpu().numpy().flatten()

        # 反归一化
        predicted_age_scaled = output[0]
        predicted_age = scaler.inverse_transform([[predicted_age_scaled]])[0][0]

        return predicted_age

    except Exception as e:
        print(f'预测时出错: {e}')
        return None

if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 定义模型架构并加载权重
    model = EfficientNetRegressor(pretrained=True).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    # 加载缩放器
    scaler = joblib.load('scaler.pkl')  # 确保在训练时保存了缩放器

    # 定义要预测的图片路径
    image_path = "C:\\Users\\24942\\Desktop\\fe875b5fa9cffabc8bbc24382aa351f.jpg"  # 测试图片路径

    # 获取预处理变换
    transform = get_prediction_transform()

    # 进行预测
    predicted_age = predict_image(image_path, model, scaler, transform, device)

    if predicted_age is not None:
        print(f'预测的年龄: {predicted_age:.2f} 个月')
    else:
        print('预测失败。')
