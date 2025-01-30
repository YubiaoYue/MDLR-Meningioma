import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =========================
# 1. 加载和准备数据
# =========================

# 定义文件路径
train_file = 'train.csv'
test_file = 'test.csv'

# 定义标签列名称
label_column = 'label'  # 根据你的CSV文件调整

# 加载训练数据
train_df = pd.read_csv(train_file)

# 加载测试数据
test_df = pd.read_csv(test_file)

# 检查数据
print("训练数据预览:")
print(train_df.head())
print("\n测试数据预览:")
print(test_df.head())

# 提取特征和标签
X_train = train_df.drop(columns=[label_column]).values
y_train = train_df[label_column].values

X_test = test_df.drop(columns=[label_column]).values
y_test = test_df[label_column].values

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 2. 训练SVM模型
# =========================

# 初始化SVM分类器
svm = SVC(kernel='rbf', random_state=42)

# 训练模型
svm.fit(X_train, y_train)

# =========================
# 3. 模型评估
# =========================

# 预测测试集
y_pred = svm.predict(X_test)

# 打印分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.ylabel('实际类别')
plt.xlabel('预测类别')
plt.title('混淆矩阵')
plt.show()

# =========================
# 4. 超参数调优（可选）
# =========================

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

# 初始化GridSearchCV
grid = GridSearchCV(SVC(random_state=42), param_grid, refit=True, verbose=2, cv=5)

# 执行网格搜索
grid.fit(X_train, y_train)

print(f"\n最佳参数: {grid.best_params_}")
print(f"最佳得分: {grid.best_score_}")

# 使用最佳参数进行预测
grid_predictions = grid.predict(X_test)

# 打印分类报告
print("\n网格搜索后的分类报告:")
print(classification_report(y_test, grid_predictions))

# =========================
# 5. 可视化决策边界（仅适用于二维数据）
# =========================

# 检查特征维度
if X_train.shape[1] == 2:
    # 训练SVM
    svm_vis = SVC(kernel='rbf', C=grid.best_params_.get('C', 1), 
                  gamma=grid.best_params_.get('gamma', 'scale'), random_state=42)
    svm_vis.fit(X_train, y_train)
    
    # 创建网格以绘制决策边界
    xx, yy = np.meshgrid(
        np.linspace(X_train[:,0].min()-1, X_train[:,0].max()+1, 500),
        np.linspace(X_train[:,1].min()-1, X_train[:,1].max()+1, 500)
    )
    
    Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘图
    plt.figure(figsize=(10,8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    sns.scatterplot(x=X_train[:,0], y=X_train[:,1], hue=y_train,
                    palette='coolwarm', edgecolor='k')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title('SVM 决策边界')
    plt.show()
else:
    print("\n特征维度大于2，跳过决策边界可视化。")

# =========================
# 6. 保存和加载模型（可选）
# =========================

# 保存模型和标准化器
model_filename = 'svm_model.joblib'
scaler_filename = 'scaler.joblib'
joblib.dump(svm, model_filename)
joblib.dump(scaler, scaler_filename)
print(f"\n模型已保存到 {model_filename}")
print(f"标准化器已保存到 {scaler_filename}")

# 加载模型和标准化器
loaded_svm = joblib.load(model_filename)
loaded_scaler = joblib.load(scaler_filename)

# 使用加载的模型进行预测
loaded_pred = loaded_svm.predict(X_test)
print("\n加载模型后的分类报告:")
print(classification_report(y_test, loaded_pred))
