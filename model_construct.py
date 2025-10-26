import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. 加载数据
file_name = "C:/Users/14711/Desktop/DM_proj/dataset/ScreenTime vs MentalWellness.csv"
df = pd.read_csv(file_name)

# 2. 定义因变量 (Y)
Y = pd.to_numeric(df['mental_wellness_index_0_100'], errors='coerce')

# 3. 定义自变量 (X)
# 连续变量和序数变量 - 排除 'work_screen_hours' 和 'leisure_screen_hours'
# 解决完全多重共线性问题 (VIF = inf)
X_to_include = [
    'age',
    'screen_time_hours',
    'work_screen_hours',
    'leisure_screen_hours',
    'sleep_hours',
    'sleep_quality_1_5',
    'stress_level_0_10',
    'productivity_0_100',
    'exercise_minutes_per_week',
    'social_hours_per_week'
]
X_continuous = df[X_to_include].apply(pd.to_numeric, errors='coerce')

# 需要进行独热编码的名义变量
X_nominal = df[['gender', 'occupation', 'work_mode']].copy()
# 创建哑变量，并丢弃第一个类别作为基准类别
X_dummies = pd.get_dummies(X_nominal, drop_first=True)

# 4. 合并所有自变量
X = pd.concat([X_continuous, X_dummies], axis=1)
X = X.astype(float)
Y = Y.fillna(Y.mean())  # Y 用均值填充
X = X.fillna(0)         # X 用 0 填充（保持您之前的逻辑）
# 5. 添加常数项（截距）
X = sm.add_constant(X)

# 6. 使用 OLS (最小二乘法) 拟合 MLR 模型
model_corrected = sm.OLS(Y, X).fit()

# 7. 计算 VIF 以诊断修正后的模型的多重共线性
vif_data_corrected = pd.DataFrame()
vif_data_corrected["Variable"] = X.columns
# 注意：在计算 VIF 之前，X 必须是数值型
vif_data_corrected["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# 8. 打印结果摘要和 VIF 结果
print("--- 修正后的多元线性回归模型摘要 (OLS) ---")
print(model_corrected.summary().as_text())
print("\n--- 修正后的 VIF 结果 ---")
print(vif_data_corrected.to_markdown(floatfmt=".2f"))