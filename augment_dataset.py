import pandas as pd
import numpy as np

# Load the existing dataset
df = pd.read_csv("dataset.csv")
n_samples = len(df)

# Randomly assign a measurement type for each record: 'Fasting' or 'Postprandial'
df['measurement_type'] = np.random.choice(['Fasting', 'Postprandial'], size=n_samples)

# Generate synthetic glucose values based on measurement type:
# For fasting: normal distribution with mean=100 and std=10
# For postprandial: normal distribution with mean=140 and std=20
fasting_values = np.random.normal(loc=100, scale=10, size=n_samples)
postprandial_values = np.random.normal(loc=140, scale=20, size=n_samples)

# Create new columns for fasting and postprandial glucose:
df['fasting_glucose'] = np.where(df['measurement_type'] == 'Fasting', fasting_values, np.nan)
df['postprandial_glucose'] = np.where(df['measurement_type'] == 'Postprandial', postprandial_values, np.nan)

# Add synthetic meal intake data:
# Meal Carbs: normal distribution with mean=50 grams and std=15
# Meal Sugar: normal distribution with mean=25 grams and std=10
df['meal_carbs'] = np.random.normal(loc=50, scale=15, size=n_samples)
df['meal_sugar'] = np.random.normal(loc=25, scale=10, size=n_samples)

# Add synthetic physical activity minutes: normal distribution with mean=30 minutes and std=10
df['physical_activity_minutes'] = np.random.normal(loc=30, scale=10, size=n_samples)

# Add synthetic sleep quality: random integer between 1 and 5
df['sleep_quality'] = np.random.randint(1, 6, size=n_samples)

# Add synthetic future glucose values ranging from 70 to 160 mg/dL
df['future_glucose'] = np.random.uniform(low=70, high=160, size=n_samples)

# Save the augmented dataset to a new CSV file
df.to_csv("dataset_augmented.csv", index=False)

print("Dataset augmented and saved to dataset_augmented.csv")
