import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


file_name = "C:/Users/14711/Desktop/DM_proj/dataset/ScreenTime vs MentalWellness.csv"
df = pd.read_csv(file_name)


variables_to_analyze = [
    'screen_time_hours',
    'mental_wellness_index_0_100',
    'sleep_hours',
    'stress_level_0_10'
]

variable_labels = {
    'screen_time_hours': 'Screen Time (hours)',
    'mental_wellness_index_0_100': 'Mental Wellness Index (0-100)',
    'sleep_hours': 'Sleep Hours (hours)',
    'stress_level_0_10': 'Stress Level (0-10)'
}


for col in variables_to_analyze:
    label = variable_labels[col]
    file_name_output = f"C:/Users/14711/Desktop/DM_proj/{col}_distribution.png"

    # 创建一个新的 Figure，包含 1 行 2 列的子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(wspace=0.3)
    

    sns.histplot(df[col], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f'{label} Distribution', fontsize=14)
    axes[0].set_xlabel(label, fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)

    sns.boxplot(y=df[col], ax=axes[1], color='lightcoral')
    axes[1].set_title(f'{label} Distribution', fontsize=14)
    axes[1].set_ylabel(label, fontsize=12)
    axes[1].set_xlabel('')
    
    # 保存图像
    plt.savefig(file_name_output)
