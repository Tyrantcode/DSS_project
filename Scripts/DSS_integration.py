import pandas as pd
import joblib

# -----------------------------
# 1. Load trained models
# -----------------------------
stress_model = joblib.load("../Models/stress_model.pkl")
mental_model = joblib.load("../Models/mental_model.pkl")
productivity_model = joblib.load("../Models/productivity_model.pkl")

# -----------------------------
# 2. Example: Student Input
# -----------------------------
# Replace this dict with actual input from students
student_input = {
    "Screen_Time_Total": [13],
    "Social_Media_Time": [4],
    "Study_Time_In_Screen": [1],
    "Gaming_Time_Total": [8],
    "Sleep_Time": [8],
    "Study_Time_Offline": [0],
    "Outdoor_Time": [1],
    "Attendance_Percentage": [85],
    "Assignments_Per_Week": [2],
    "Backlogs": [0],
    "Notifications": [250],
    "Games_Installed": [6],
    "GPA": [8.0],
    "Lecture_Hours": [6.0],
    "Late_Night_SM": [1.0],
    "Active_Passive_Ratio": [0],
    "Gender": [0],
    "Residence": [0],
    "Part_Time_Job": [0]
}

# Create DataFrame
df_student = pd.DataFrame(student_input)

# -----------------------------
# 3. Ensure correct feature names for each model
# -----------------------------
# Use only the columns that the model was trained on
stress_features = stress_model.feature_names_in_
mental_features = mental_model.feature_names_in_
productivity_features = productivity_model.feature_names_in_

# -----------------------------
# 4. Make Predictions
# -----------------------------
df_student["Stress_Level"] = stress_model.predict(df_student[stress_features])

df_student["Mental_Health_Rating"] = mental_model.predict(df_student[mental_features])
df_student["Productivity_Index"] = productivity_model.predict(df_student[productivity_features])

# -----------------------------
# 5. Display Results
# -----------------------------
print("✅ DSS Predictions for Student Input:")
print(df_student[[
    "Stress_Level",
    "Mental_Health_Rating",
    "Productivity_Index"
]])

# --- 1. Define a function for recommendations based on 5 levels ---
def generate_recommendation(stress, mental, productivity):
    advice = []

    # --- Stress Level (0–10) ---
    if stress < 2:
        advice.append("Stress Level: Very Low - keep maintaining your routine.")
    elif stress < 4:
        advice.append("Stress Level: Low - good, monitor workload.")
    elif stress < 6:
        advice.append("Stress Level: Moderate - try relaxation techniques.")
    elif stress < 8:
        advice.append("Stress Level: High - take regular breaks and manage tasks.")
    else:
        advice.append("Stress Level: Very High - urgent action needed, consider counseling.")

    # --- Mental Health Rating (0–5) ---
    if mental < 1:
        advice.append("Mental Health: Very Poor - immediate attention needed.")
    elif mental < 2:
        advice.append("Mental Health: Poor - consider counseling or self-care activities.")
    elif mental < 3:
        advice.append("Mental Health: Moderate - maintain habits, monitor mental health.")
    elif mental < 4:
        advice.append("Mental Health: Good - keep healthy routines.")
    else:
        advice.append("Mental Health: Excellent - continue positive practices.")

    # --- Productivity Index (0–5) ---
    if productivity < 1:
        advice.append("Productivity: Very Low - improve time management and planning.")
    elif productivity < 2:
        advice.append("Productivity: Low - set achievable goals and prioritize tasks.")
    elif productivity < 3:
        advice.append("Productivity: Moderate - maintain consistency and monitor performance.")
    elif productivity < 4:
        advice.append("Productivity: High - good progress, keep optimizing workflow.")
    else:
        advice.append("Productivity: Very High - excellent performance, keep it up!")

    return advice

# --- 2. Apply on a student's predicted values ---
stress = df_student["Stress_Level"].iloc[0]
mental = df_student["Mental_Health_Rating"].iloc[0]
productivity = df_student["Productivity_Index"].iloc[0]

recommendations = generate_recommendation(stress, mental, productivity)

# --- 3. Print results ---
print("Predicted Values:")
print(f"Stress Level: {stress:.2f}")
print(f"Mental Health Rating: {mental:.2f}")
print(f"Productivity Index: {productivity:.2f}\n")

print("Recommendations:")
for rec in recommendations:
    print(f"- {rec}")

