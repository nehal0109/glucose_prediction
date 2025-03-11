import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("dataset_augmented.csv")

columns_needed = [
    "AGE", "GENDER", "WEIGHT", "HEARTRATE", "LAST_EATEN", "DIABETIC", "GLUCOSE_LEVEL",
    "measurement_type", "fasting_glucose", "postprandial_glucose", "meal_carbs", "meal_sugar",
    "physical_activity_minutes", "sleep_quality", "future_glucose"
]
df_model = df[columns_needed].copy()

essential_cols = [
    "AGE", "GENDER", "WEIGHT", "HEARTRATE", "LAST_EATEN", "DIABETIC",
    "GLUCOSE_LEVEL", "measurement_type", "meal_carbs", "meal_sugar",
    "physical_activity_minutes", "sleep_quality", "future_glucose"
]
df_model.dropna(subset=essential_cols, inplace=True)
df_model.dropna(subset=["fasting_glucose", "postprandial_glucose"], how="all", inplace=True)
df_model["fasting_glucose"] = df_model["fasting_glucose"].fillna(0)
df_model["postprandial_glucose"] = df_model["postprandial_glucose"].fillna(0)

df_model["DIABETIC"] = df_model["DIABETIC"].map({"N": 0, "Y": 1})
df_model["GENDER"] = df_model["GENDER"].map({"M": 0, "F": 1})
df_model["measurement_type"] = df_model["measurement_type"].map({"Fasting": 0, "Postprandial": 1})
df_model["LAST_EATEN"] = pd.to_numeric(df_model["LAST_EATEN"], errors='coerce')
df_model.dropna(subset=["LAST_EATEN"], inplace=True)

numeric_cols = [
    "AGE", "WEIGHT", "HEARTRATE", "LAST_EATEN", "GLUCOSE_LEVEL",
    "meal_carbs", "meal_sugar", "physical_activity_minutes", "sleep_quality", "future_glucose"
]

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in numeric_cols:
    df_model = remove_outliers(df_model, col)

df_model["carbs_plus_sugar"] = df_model["meal_carbs"] + df_model["meal_sugar"]
df_model["glucose_diff"] = df_model["postprandial_glucose"] - df_model["fasting_glucose"]

features = [
    "AGE", "GENDER", "WEIGHT", "HEARTRATE", "LAST_EATEN", "DIABETIC",
    "measurement_type", "fasting_glucose", "postprandial_glucose",
    "meal_carbs", "meal_sugar", "physical_activity_minutes", "sleep_quality",
    "carbs_plus_sugar", "glucose_diff"
]
target = "future_glucose"

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_scaled, y_train)

y_pred = rf_regressor.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Future Glucose (mg/dL)")
plt.ylabel("Predicted Future Glucose (mg/dL)")
plt.title("Predicted vs. Actual Future Glucose Levels (Random Forest)")
plt.plot([70, 160], [70, 160], 'r--')
plt.xlim(70, 160)
plt.ylim(70, 160)
plt.show()

plt.figure(figsize=(12, 6))
sns.heatmap(df_model.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

df_model.hist(figsize=(12, 8), bins=20)
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_model["AGE"], y=df_model["GLUCOSE_LEVEL"], hue=df_model["DIABETIC"])
plt.title("Glucose Levels vs. Age")
plt.show()

print("\nRandom Forest Model Performance:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)
