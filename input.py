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

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

numeric_cols = [
    "AGE", "WEIGHT", "HEARTRATE", "LAST_EATEN", "GLUCOSE_LEVEL",
    "meal_carbs", "meal_sugar", "physical_activity_minutes", "sleep_quality", "future_glucose"
]

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

print("\nRandom Forest Model Performance:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Future Glucose (mg/dL)")
plt.ylabel("Predicted Future Glucose (mg/dL)")
plt.title("Predicted vs. Actual Future Glucose Levels (Random Forest)")
plt.plot([70, 160], [70, 160], 'r--')
plt.xlim(70, 160)
plt.ylim(70, 160)
plt.show()

print("\nEnter the following details to predict your future glucose level:")

try:
    age = float(input("AGE (years): "))
    gender_input = input("GENDER (M/F): ").strip().upper()
    gender = 0 if gender_input == "M" else 1
    weight = float(input("WEIGHT (kg): "))
    heartrate = float(input("HEARTRATE (bpm): "))
    last_eaten = float(input("LAST_EATEN (minutes since last meal): "))
    diabetic_input = input("DIABETIC (Y/N): ").strip().upper()
    diabetic = 1 if diabetic_input == "Y" else 0
    glucose_level = float(input("Current GLUCOSE_LEVEL (mg/dL): "))
    measurement_type_input = input("Measurement Type ('f' for fasting, 'p' for postprandial): ").strip().lower()

    if measurement_type_input == 'f':
        measurement_type = 0
        fasting_glucose = float(input("Enter Fasting Glucose (mg/dL): "))
        postprandial_glucose = 0.0
    elif measurement_type_input == 'p':
        measurement_type = 1
        postprandial_glucose = float(input("Enter Postprandial Glucose (mg/dL): "))
        fasting_glucose = 0.0
    else:
        measurement_type = 0
        fasting_glucose = float(input("Enter Fasting Glucose (mg/dL): "))
        postprandial_glucose = 0.0

    meal_carbs = float(input("Enter MEAL CARBS (grams): "))
    meal_sugar = float(input("Enter MEAL SUGAR (grams): "))
    physical_activity_minutes = float(input("Enter PHYSICAL ACTIVITY MINUTES: "))
    sleep_quality = int(input("Enter SLEEP QUALITY (1-5): "))
    carbs_plus_sugar = meal_carbs + meal_sugar
    glucose_diff = postprandial_glucose - fasting_glucose

    user_features = np.array([[age, gender, weight, heartrate, last_eaten, diabetic,
                               glucose_level, measurement_type, fasting_glucose, postprandial_glucose,
                               meal_carbs, meal_sugar, physical_activity_minutes, sleep_quality,
                               carbs_plus_sugar, glucose_diff]], dtype=float)

    user_features_scaled = scaler.transform(user_features)
    predicted_future_glucose = rf_regressor.predict(user_features_scaled)[0]
    print("\nPredicted Future Glucose Level (mg/dL): {:.2f}".format(predicted_future_glucose))

except Exception as e:
    print("Error in input:", e)
