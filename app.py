# ------------------------------
# app.py  |  Streamlit DSS App
# ------------------------------

import streamlit as st
import pandas as pd
import joblib
import os

# ------------------------------
# 1. Load trained models
# ------------------------------
stress_model = joblib.load("clone/Models/stress_model.pkl")
mental_model = joblib.load("clone/Models/mental_model.pkl")
productivity_model = joblib.load("clone/Models/productivity_model.pkl")

st.set_page_config(page_title="Student DSS - Wellbeing & Productivity", layout="centered")
st.title("🎓 Student Decision Support System (DSS)")
st.subheader("Predict Stress, Mental Health, and Productivity Levels")

# ------------------------------
# 2. Input Section
# ------------------------------
st.sidebar.header("🧠 Enter Student Details")

student_input = {
    "Screen_Time_Total": st.sidebar.number_input("Total Screen Time (hours/day)", 0.0, 24.0, 8.0),
    "Social_Media_Time": st.sidebar.number_input("Social Media Time (hours/day)", 0.0, 24.0, 3.0),
    "Study_Time_In_Screen": st.sidebar.number_input("Study Time (screen-based) (hours/day)", 0.0, 24.0, 2.0),
    "Gaming_Time_Total": st.sidebar.number_input("Gaming Time (hours/day)", 0.0, 24.0, 1.0),
    "Sleep_Time": st.sidebar.number_input("Sleep Time (hours/day)", 0.0, 24.0, 7.0),
    "Study_Time_Offline": st.sidebar.number_input("Offline Study Time (hours/day)", 0.0, 24.0, 2.0),
    "Outdoor_Time": st.sidebar.number_input("Outdoor Activity Time (hours/day)", 0.0, 24.0, 1.0),
    "Attendance_Percentage": st.sidebar.number_input("Attendance Percentage", 0, 100, 85),
    "Assignments_Per_Week": st.sidebar.number_input("Assignments per Week", 0, 10, 3),
    "Backlogs": st.sidebar.number_input("Backlogs", 0, 10, 0),
    "Notifications": st.sidebar.number_input("Avg Notifications per Day", 0, 1000, 200),
    "Games_Installed": st.sidebar.number_input("No. of Games Installed", 0, 50, 5),
    "GPA": st.sidebar.number_input("GPA", 0.0, 10.0, 8.0),
    "Lecture_Hours": st.sidebar.number_input("Lecture Hours per Day", 0.0, 12.0, 6.0),
    "Late_Night_SM": st.sidebar.number_input("Late Night Social Media (0-No, 1-Yes)", 0, 1, 1),
    "Active_Passive_Ratio": st.sidebar.number_input("Active/Passive Study Ratio (0-1)", 0.0, 1.0, 0.5),
    "Gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
    "Residence": st.sidebar.selectbox("Residence", ["Home", "Hostel", "PG"]),
    "Part_Time_Job": st.sidebar.selectbox("Part Time Job", ["No", "Yes"])
}

# Convert categorical inputs
student_input["Gender"] = 0 if student_input["Gender"] == "Male" else 1
student_input["Residence"] = {"Home": 0, "Hostel": 1, "PG": 2}[student_input["Residence"]]
student_input["Part_Time_Job"] = 1 if student_input["Part_Time_Job"] == "Yes" else 0

df_student = pd.DataFrame([student_input])

# ------------------------------
# 3. Prediction
# ------------------------------
if st.button("🔍 Predict Student Wellbeing"):
    # Use the model feature names exactly as during training
    stress_features = stress_model.feature_names_in_
    mental_features = mental_model.feature_names_in_
    productivity_features = productivity_model.feature_names_in_

    # Make predictions
    df_student["Stress_Level"] = stress_model.predict(df_student[stress_features])
    df_student["Mental_Health_Rating"] = mental_model.predict(df_student[mental_features])
    df_student["Productivity_Index"] = productivity_model.predict(df_student[productivity_features])

    stress = df_student["Stress_Level"].iloc[0]
    mental = df_student["Mental_Health_Rating"].iloc[0]
    productivity = df_student["Productivity_Index"].iloc[0]

    st.success("✅ Prediction Successful!")

    # Display metrics
    st.metric("🧩 Stress Level (0–10)", f"{stress:.2f}")
    st.metric("🩺 Mental Health Rating (0–5)", f"{mental:.2f}")
    st.metric("⚙️ Productivity Index (0–5)", f"{productivity:.2f}")

    # ------------------------------
    # 4. Decision/Recommendation Logic
    # ------------------------------
    def generate_recommendation(stress, mental, productivity):
        advice = []

        if stress < 2:
            advice.append("🧘‍♂️ Very Low Stress – Great! Keep a balanced routine.")
        elif stress < 4:
            advice.append("🙂 Low Stress – Stay consistent and manage workload.")
        elif stress < 6:
            advice.append("😐 Moderate Stress – Try relaxation or mindfulness.")
        elif stress < 8:
            advice.append("😣 High Stress – Prioritize rest and breaks.")
        else:
            advice.append("🚨 Very High Stress – Seek counseling or support ASAP.")

        if mental < 1:
            advice.append("💭 Very Poor Mental Health – Immediate attention needed.")
        elif mental < 2:
            advice.append("😔 Poor – Try self-care and talk to trusted peers.")
        elif mental < 3:
            advice.append("😌 Moderate – Maintain balance and awareness.")
        elif mental < 4:
            advice.append("😊 Good – Keep it up and stay active.")
        else:
            advice.append("🌈 Excellent – Strong mental health!")

        if productivity < 1:
            advice.append("📉 Very Low Productivity – Work on time management.")
        elif productivity < 2:
            advice.append("🕐 Low Productivity – Set achievable daily goals.")
        elif productivity < 3:
            advice.append("📈 Moderate – Stay consistent and review progress.")
        elif productivity < 4:
            advice.append("⚡ High – Excellent focus, keep refining your methods.")
        else:
            advice.append("🚀 Very High – Outstanding! Maintain your momentum.")

        return advice

    st.divider()
    st.subheader("📊 Recommendations")
    for rec in generate_recommendation(stress, mental, productivity):
        st.write(f"- {rec}")
