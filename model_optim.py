import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import numpy as np

# 1. 加载数据
file_name =  "C:/Users/14711/Desktop/DM_proj/dataset/ScreenTime vs MentalWellness.csv"
df = pd.read_csv(file_name)

# 定义因变量 (Y)
Y = df['mental_wellness_index_0_100']

# 2. 定义自变量 (X) - 包含全部三个屏幕时间变量
X_to_include = [
    'age',
    'screen_time_hours',          # 故意保留以测试LASSO处理共线性的能力
    'work_screen_hours',          # 故意保留
    'leisure_screen_hours',       # 故意保留
    'sleep_hours',
    'sleep_quality_1_5',
    'stress_level_0_10',
    'productivity_0_100',
    'exercise_minutes_per_week',
    'social_hours_per_week'
]
X_continuous = df[X_to_include].copy()

# 名义变量独热编码
X_nominal = df[['gender', 'occupation', 'work_mode']].copy()
X_dummies = pd.get_dummies(X_nominal, drop_first=True, dtype=int)

# 合并所有自变量
X = pd.concat([X_continuous, X_dummies], axis=1)
feature_names = X.columns.tolist()

# 3. 数据清洗和标准化 (正则化模型的必需步骤)
# 处理潜在的非数值数据/NaNs
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
Y = pd.to_numeric(Y, errors='coerce').fillna(Y.mean()) 

# 对自变量进行标准化 (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 拟合 LASSO 回归模型
# Alpha=0.1 是一个常用值，用于观察系数收缩和特征选择效果
lasso_alpha = 0.1
lasso_model = Lasso(alpha=lasso_alpha, fit_intercept=True, max_iter=10000)
lasso_model.fit(X_scaled, Y)

# 5. 计算并打印结果
lasso_r2 = r2_score(Y, lasso_model.predict(X_scaled))

# 创建系数对比 DataFrame
comparison_df = pd.DataFrame({
    'Feature': ['(Intercept)'] + feature_names,
    f'LASSO_Coeff_a={lasso_alpha}': [lasso_model.intercept_] + list(lasso_model.coef_)
})

# 筛选并显示核心变量的系数
comparison_features = [
    'screen_time_hours', 'work_screen_hours', 'leisure_screen_hours',
    'stress_level_0_10', 'productivity_0_100', 'sleep_quality_1_5', 
    'social_hours_per_week', 'exercise_minutes_per_week'
]
comparison_output = comparison_df[comparison_df['Feature'].isin(comparison_features)].sort_values(by=f'LASSO_Coeff_a={lasso_alpha}', ascending=False)

print(f"R-squared (LASSO alpha={lasso_alpha}): {lasso_r2:.4f}")
print("\nLASSO 回归系数 (标准化数据):")
print(comparison_output.to_markdown(floatfmt=".4f"))
# 额外：打印所有系数（包含截距），方便完整检查
print("\n所有系数 (包含截距):")
print(comparison_df.to_markdown(index=False, floatfmt=".4f"))