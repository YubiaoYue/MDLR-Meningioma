import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# 加载数据（假设特征已经提取并存储为CSV文件）
data = pd.read_csv('extracted_features.csv')  # 特征数据

X = data.drop(columns=['label'])
y = data['label']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 计算Spearman相关系数
corr_matrix, _ = spearmanr(X_scaled)
redundant_features = np.where(abs(corr_matrix) > 0.9)

# 删除冗余特征
to_remove = set()
for i in range(len(redundant_features[0])):
    if redundant_features[0][i] < redundant_features[1][i]:
        to_remove.add(redundant_features[1][i])

X_reduced = np.delete(X_scaled, list(to_remove), axis=1)

# 使用LASSO回归进行特征选择
lasso = LassoCV(cv=5)
lasso.fit(X_reduced, y)

# 获取LASSO选择的非零特征
selected_features = np.where(lasso.coef_ != 0)[0]

# 保留LASSO选择的特征
X_final = X_reduced[:, selected_features]

# 保存最终选择的特征
import pandas as pd
final_features_df = pd.DataFrame(X_final)
final_features_df.to_csv('final_selected_features.csv', index=False)
