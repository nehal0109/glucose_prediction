import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv")
n_samples = len(df)

df['measurement_type'] = np.random.choice(['Fasting', 'Postprandial'], size=n_samples)

fasting_values = np.random.normal(loc=100, scale=10, size=n_samples)
postprandial_values = np.random.normal(loc=140, scale=20, size=n_samples)

df['fasting_glucose'] = np.where(df['measurement_type'] == 'Fasting', fasting_values, np.nan)
df['postprandial_glucose'] = np.where(df['measurement_type'] == 'Postprandial', postprandial_values, np.nan)

df['meal_carbs'] = np.random.normal(loc=50, scale=15, size=n_samples)
df['meal_sugar'] = np.random.normal(loc=25, scale=10, size=n_samples)

df['physical_activity_minutes'] = np.random.normal(loc=30, scale=10, size=n_samples)
df['sleep_quality'] = np.random.randint(1, 6, size=n_samples)
df['future_glucose'] = np.random.uniform(low=70, high=160, size=n_samples)

df.to_csv("dataset_augmented.csv", index=False)

print("Dataset augmented and saved to dataset_augmented.csv")
