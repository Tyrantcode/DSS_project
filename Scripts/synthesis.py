import numpy as np
import pandas as pd
import os
# Number of students
n = 5000
# ---- 1. Screen Time (hrs/day)
# Majority values between 5–9, mean=7, std=2, clipped to 3–1
Screen_Time_Total = np.clip(np.random.normal(7, 2, n), 3, 16)

# ---- 2. Social Media Time (hrs/day)
# Majority between 1–4, mean=2.5, std=1, clipped to 0–5
Social_Media_Time = np.clip(np.random.normal(2.5, 1.5, n), 0, 8)

# ---- 3. Study Time (In Screen)
# Majority between 1–4, mean=2, std=1, clipped to 0–5
Study_Time_In_Screen = np.clip(np.random.normal(2, 1.5, n), 0, 5)

# ---- 4. Gaming Time (hrs/day)
# Majority between 1–4, mean=2, std=1.5, clipped to 0–6
Gaming_Time_Total = np.clip(np.random.normal(2, 1.5, n), 0, 6)
# ---- Adjust: Ensure sum of SM + Study + Game <= Screen Time
for i in range(n):
    total = Social_Media_Time[i] + Study_Time_In_Screen[i] + Gaming_Time_Total[i]
    if total > Screen_Time_Total[i]:
        scale = Screen_Time_Total[i] / total
        Social_Media_Time[i] *= scale
        Study_Time_In_Screen[i] *= scale
        Gaming_Time_Total[i] *= scale

# ---- 5. Sleep Time (hrs/day)
# Majority between 6–8, mean=7, std=1, clipped to 4–10
Sleep_Time = np.clip(np.random.normal(7, 1, n), 4, 10)

# ---- 6. Study Time (Offline)
# Majority between 2–4, mean=3, std=1, clipped to 0–6
Study_Time_Offline = np.clip(np.random.normal(3, 1, n), 0, 6)

# ---- 7. Outdoor Time (hrs/day)
# Majority between 1–4, mean=2.5, std=1, clipped to 0–6
Outdoor_Time = np.clip(np.random.normal(2.5, 1, n), 0, 6)

# ---- Adjust: Total daily hours <= 24
for i in range(n):
    total_day = Screen_Time_Total[i] + Sleep_Time[i] + Study_Time_Offline[i] + Outdoor_Time[i]
    if total_day > 24:
        factor = 24 / total_day
        Screen_Time_Total[i] *= factor
        Sleep_Time[i] *= factor
        Study_Time_Offline[i] *= factor
        Outdoor_Time[i] *= factor

# ---- 8. Attendance (%)
# Majority 60–90, mean=75, std=10
Attendance = np.clip(np.random.normal(75, 10, n), 40, 100)

# ---- 9. Assignments per Week
Assignments_Per_Week = np.random.choice([1, 2, 3], size=n, p=[0.3, 0.5, 0.2])

# ---- 10. Backlogs / Pending Subjects
Backlogs = np.random.choice([0, 1, 2, 3], size=n, p=[0.65, 0.25, 0.08, 0.02])

# ---- 11. Number of Notifications per Day
Notifications = np.clip(np.random.normal(120, 40, n), 30, 300).round().astype(int)

# ---- 12. Games Installed
Games_Installed = np.clip(np.random.normal(3, 1, n), 0, 8).round().astype(int)

# ---- If Games Installed == 0 → Game Time = 0
Gaming_Time_Total = np.where(Games_Installed == 0, 0, Gaming_Time_Total)

# ---- 13. GPA (0–10)
GPA = np.clip(np.random.normal(7.5, 1.2, n), 4, 10)

# ---- 14. Lecture Hours per Day
Lecture_Hours = np.clip(np.random.normal(6, 2, n), 2, 8)

# ---- 15. Late-Night SM Usage (After 11 PM, hrs/day)
Late_Night_SM = np.clip(np.random.normal(0.8, 0.5, n), 0, 3)

# ---- 16. Active vs Passive Ratio on SM (0–1)
Active_Passive_Ratio = np.clip(np.random.normal(0.4, 0.2, n), 0, 1)

# ---- 17. Gender (0=Male, 1=Female)
Gender = np.random.choice([0, 1], size=n, p=[0.5, 0.5])

# ---- 18. Residence (Home/Hostel/PG)
Residence = np.random.choice(['Home', 'Hostel', 'PG'], size=n, p=[0.45, 0.45, 0.1])

# ---- 19. Part-Time Job (0=No, 1=Yes)
Part_Time_Job = np.random.choice([0, 1], size=n, p=[0.8, 0.2])


# ---- Create DataFrame
df = pd.DataFrame({
    "Screen_Time_Total": Screen_Time_Total,
    "Social_Media_Time": Social_Media_Time,
    "Study_Time_In_Screen": Study_Time_In_Screen,
    "Gaming_Time_Total": Gaming_Time_Total,
    "Sleep_Time": Sleep_Time,
    "Study_Time_Offline": Study_Time_Offline,
    "Outdoor_Time": Outdoor_Time,
    "Attendance_Percentage": Attendance,
    "Assignments_Per_Week": Assignments_Per_Week,
    "Backlogs": Backlogs,
    "Notifications": Notifications,
    "Games_Installed": Games_Installed,
    "GPA": GPA,
    "Lecture_Hours": Lecture_Hours,
    "Late_Night_SM": Late_Night_SM,
    "Active_Passive_Ratio": Active_Passive_Ratio,
    "Gender": Gender,
    "Residence": Residence,
    "Part_Time_Job": Part_Time_Job,
})
# Rebalanced Stress_Level formula (controlled to stay within 1–10)
Stress_Level = (
    5 +                                       # baseline stress
    0.8 * df["Assignments_Per_Week"] +
    1.0 * df["Backlogs"] +
    0.5 * df["Gaming_Time_Total"] +
    0.5 * df["Part_Time_Job"] +
    - 1.0 * df["Sleep_Time"] +
    0.6 * df["Social_Media_Time"] +
    0.7 * df["Study_Time_In_Screen"] +
    0.3 * df["Lecture_Hours"] +
    0.3 * df["Active_Passive_Ratio"] +
    0.5 * df["Late_Night_SM"]
).clip(1,10).round(2)


# Mental_Health_Rating (scale 1–10)
# map residence to a small numeric benefit (higher = better mental health)
# tweak these values if you prefer a different ranking
residence_map = {"Home": 1.0, "Hostel": 0.8, "PG": 0.6}
df["Residence_Score"] = df["Residence"].map(residence_map).fillna(0.75)

# compute Mental_Health_Rating (direct weighted formula)
# note: Notifications scaled by 100 to bring numbers to similar scale as others
Mental_Health_Rating = (
      6.0                                 # baseline
    - 0.45 * Stress_Level           # strong negative effect of stress
    - 0.25 * (df["Screen_Time_Total"])          # higher screen time lowers MH
    - 0.20 * (df["Notifications"] / 100)  # many notifications hurt MH
    - 0.15 * (df["Games_Installed"])      # more installed games → slight negative
    + 0.35 * (df["Attendance_Percentage"] / 100)  # good attendance helps
    + 0.40 * df["Outdoor_Time"]           # outdoor time strongly helps
    + 0.60 * df["Residence_Score"]        # supportive living situation helps
).clip(1, 5).round(2)


# Productivity_Index (scale 0–2)
# ⚙️ Productivity Index Formula
Productivity_Index = (
    0.4 * (df["Study_Time_Offline"] + df["Study_Time_In_Screen"])     # more study → higher productivity
    + 0.2 * (df["Attendance_Percentage"] / 10)                        # higher attendance → better
    - 0.2 * (df["Screen_Time_Total"] / 10)                            # excessive screen time → lower
    - 0.2 * (df["Social_Media_Time"] / 10)                            # excessive SM → lower
    - 0.1 * (df["Gaming_Time_Total"] / 10)                            # gaming reduces productivity
    - 0.3 * (Stress_Level / 10)                                 # stress reduces productivity
    + np.random.uniform(-0.3, 0.3, len(df))                           # small random noise for variation
)

# ✅ Normalize between 0 and 5, and round

Productivity_Index = np.clip(Productivity_Index, 0, 5).round(2)

df["Stress_Level"] = Stress_Level
df["Mental_Health_Rating"] = Mental_Health_Rating
df["Productivity_Index"] = Productivity_Index
df=df.round(2)
# Display few rows
print(df.head())


# ✅ Define the CSV file path
file_path = os.path.join("DSS_Project","Data", "synthetic_student_wellbeing.csv")

# ✅ Save the DataFrame as a CSV file (rounded to 2 decimal places)
df = df.round(2)
df.to_csv(file_path, index=False)

# ✅ Print confirmation
print(f"✅ CSV file saved successfully at: {file_path}")