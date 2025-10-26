import pandas as pd

file_name = "C:/Users/14711/Desktop/DM_proj/dataset/ScreenTime vs MentalWellness.csv"
df = pd.read_csv(file_name)

correlation_variables = [
    'mental_wellness_index_0_100',  # Dependent Variable
    'screen_time_hours',
    'work_screen_hours',
    'leisure_screen_hours',
    'sleep_hours',
    'sleep_quality_1_5',  
    'stress_level_0_10',  
    'productivity_0_100',
    'exercise_minutes_per_week',
    'social_hours_per_week',
    'age'
]


pearson_corr = df[correlation_variables].corr(method='pearson')
spearman_corr = df[correlation_variables].corr(method='spearman')

print(pearson_corr)
print(spearman_corr)