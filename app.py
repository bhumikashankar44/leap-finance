import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Title of the app
st.title("IELTS Student Improvement Predictor")

# Load dataset
df = pd.read_csv("IELTS_student_Performance_Dataset_50000.csv")

# Input from user
st.header("Enter Student Scores")
listening = st.number_input("Listening Score", min_value=0, max_value=9, value=5)
reading = st.number_input("Reading Score", min_value=0, max_value=9, value=5)
writing = st.number_input("Writing Score", min_value=0, max_value=9, value=5)
speaking = st.number_input("Speaking Score", min_value=0, max_value=9, value=5)

# Features and label encoding
X = df[['Listening_Score','Reading_Score','Writing_Score','Speaking_Score']]
le = LabelEncoder()
df['Area_Encoded'] = le.fit_transform(df['Area_To_Improve'])
y = df['Area_Encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Prepare user input as DataFrame to fix sklearn warning
student = pd.DataFrame([[listening, reading, writing, speaking]],
                       columns=['Listening_Score','Reading_Score','Writing_Score','Speaking_Score'])

# Predict on user input
pred_encoded = model.predict(student)[0]
pred_area = le.inverse_transform([pred_encoded])[0]

# Display prediction
st.subheader("Prediction Results")
st.write("**Predicted Area to Improve:**", pred_area)

# Improvement suggestion
suggestion_map = (
    df.groupby('Area_To_Improve')['Improvement_Suggestion']
    .first()
    .to_dict()
)
suggestion = suggestion_map[pred_area]
st.write("**Improvement Suggestion:**", suggestion)

# Average score
avg_score = round(sum(student.iloc[0]) / 4, 2)
st.write("**Average Score:**", avg_score)
