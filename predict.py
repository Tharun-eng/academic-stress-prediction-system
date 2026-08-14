import joblib
import pandas as pd

model = joblib.load("models/stress_model.pkl")
encoders = joblib.load("models/label_encoders.pkl")

sample = pd.DataFrame({
    "Age": [23],

    "Gender": [
        encoders["Gender"].transform(["Male"])[0]
    ],

    "Study_Hours": [3],

    "Class_Attendance": [60],

    "Tuition": [
        encoders["Tuition"].transform(["Yes"])[0]
    ],

    "Exam_Frequency": [10],

    "Assignment_Load": [10],

    "Sleep_Hours": [2],

    "Physical_Exercise": [
        encoders["Physical_Exercise"].transform(["No"])[0]
    ],

    "Social_Media_Use": [10],

    "Screen_Time": [12],

    "Family_Income_Level": [
        encoders["Family_Income_Level"].transform(["Low"])[0]
    ],

    "Peer_Pressure": [10],

    "Family_Support": [1],

    "Anxiety_Level": [10],

    "University_Type": [
        encoders["University_Type"].transform(
            ["Private University"]
        )[0]
    ]
})

prediction = model.predict(sample)

stress = encoders["Stress_Level"].inverse_transform(prediction)

print("Predicted Stress Level :", stress[0])