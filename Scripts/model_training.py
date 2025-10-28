# ------------------------------
# model_training.py
# ------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# 1️⃣ Load your dataset
df = pd.read_csv("D:/Github/DSS_project/Data/synthetic_student_wellbeing.csv")  # adjust path if needed

# 2️⃣ Encode categorical variables
df['Residence'] = df['Residence'].map({'Home': 0, 'Hostel': 1, 'PG': 2})

# 3️⃣ Define features for each target
# Adjust these based on correlation/importance you determined
stress_features = [
    'Assignments_Per_Week', 'Backlogs', 'Gaming_Time_Total', 'Part_Time_Job',
    'Sleep_Time', 'Social_Media_Time', 'Study_Time_In_Screen', 'Lecture_Hours',
    'Active_Passive_Ratio', 'Late_Night_SM'
]

mental_features = [
    'Screen_Time_Total', 'Attendance_Percentage', 'Outdoor_Time',
    'Games_Installed', 'Notifications', 'Stress_Level', 'Residence'
]

productivity_features = [
    'GPA', 'Study_Time_Offline', 'Lecture_Hours', 'Assignments_Per_Week',
    'Stress_Level', 'Mental_Health_Rating'
]

# 4️⃣ Split data into X and y
X_stress = df[stress_features]
y_stress = df['Stress_Level']

X_mental = df[mental_features]
y_mental = df['Mental_Health_Rating']

X_prod = df[productivity_features]
y_prod = df['Productivity_Index']

# 5️⃣ Split into train and test
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train_s, X_test_s, y_train_s, y_test_s = split_data(X_stress, y_stress)
X_train_m, X_test_m, y_train_m, y_test_m = split_data(X_mental, y_mental)
X_train_p, X_test_p, y_train_p, y_test_p = split_data(X_prod, y_prod)

# 6️⃣ Initialize Random Forest Regressors
rf_stress = RandomForestRegressor(n_estimators=200, random_state=42)
rf_mental = RandomForestRegressor(n_estimators=200, random_state=42)
rf_prod = RandomForestRegressor(n_estimators=200, random_state=42)

# 7️⃣ Train the models
rf_stress.fit(X_train_s, y_train_s)
rf_mental.fit(X_train_m, y_train_m)
rf_prod.fit(X_train_p, y_train_p)

# 8️⃣ Evaluate the models
def evaluate_model(model, X_test, y_test, name="Model"):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    print(f"{name} -> R2: {r2:.3f}, MSE: {mse:.3f}")

evaluate_model(rf_stress, X_test_s, y_test_s, "Stress Model")
evaluate_model(rf_mental, X_test_m, y_test_m, "Mental Health Model")
evaluate_model(rf_prod, X_test_p, y_test_p, "Productivity Model")

import os
import joblib

# Make sure the folder exists
os.makedirs("D:/Github/DSS_project/clone/Models", exist_ok=True)

# Now save your models
joblib.dump(rf_stress, "D:/Github/DSS_project/clone/Models/stress_model.pkl")
joblib.dump(rf_mental, "D:/Github/DSS_project/clone/Models/mental_model.pkl")
joblib.dump(rf_prod, "D:/Github/DSS_project/clone/Models/productivity_model.pkl")

