import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd

# 加载预训练的ResNet50模型，去掉分类层
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # 去掉最后的全连接层
model.eval()  # 将模型设置为评估模式（不进行训练）

# 定义图像预处理
preprocess = transforms.Compose([
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
])

# 读取并处理图像
def extract_dl_features(image_path):
    img = Image.open(image_path)  # 读取图像
    img_tensor = preprocess(img)  # 预处理图像
    img_tensor = img_tensor.unsqueeze(0)  # 增加batch维度

    # 使用ResNet50模型提取特征
    with torch.no_grad():  # 禁用梯度计算，节省内存
        features = model(img_tensor)  # 获取卷积特征
        features = features.flatten()  # 展平特征向量

    return features.numpy()  # 返回NumPy数组

# 提取T1和T2图像的深度学习特征
t1_dl_features = extract_dl_features('path_to_t1_image.jpg')
t2_dl_features = extract_dl_features('path_to_t2_image.jpg')

# 输出特征
print("T1深度学习特征:", t1_dl_features)
print("T2深度学习特征:", t2_dl_features)

# 保存特征
dl_features_df = pd.DataFrame([t1_dl_features, t2_dl_features])
dl_features_df.to_csv('dl_extracted_features.csv', index=False)
