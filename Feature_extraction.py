import SimpleITK as sitk
from radiomics import featureextractor
import os

# 加载MRI图像
def load_image(image_path):
    return sitk.ReadImage(image_path)

# 使用PyRadiomics提取特征
def extract_radiomics_features(image_path, mask_path):
    # 设置特征提取器
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()  # 启用所有特征类型

    # 提取特征
    result = extractor.execute(image_path, mask_path)
    return result

# 假设已经有图像和ROI的路径
image_t1_path = 'path_to_t1_image.nii'
mask_t1_path = 'path_to_t1_mask.nii'
image_t2_path = 'path_to_t2_image.nii'
mask_t2_path = 'path_to_t2_mask.nii'

# 提取T1和T2图像的特征
t1_features = extract_radiomics_features(image_t1_path, mask_t1_path)
t2_features = extract_radiomics_features(image_t2_path, mask_t2_path)

# 输出结果
print("T1特征:", t1_features)
print("T2特征:", t2_features)

# 保存特征为CSV文件
import pandas as pd
features_df = pd.DataFrame([t1_features, t2_features])  # 假设t1和t2是字典
features_df.to_csv('extracted_features.csv', index=False)
