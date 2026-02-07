import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="IELTS Improvement Advisor", layout="centered")

st.title(" IELTS Performance Advisor")
st.write("Enter your IELTS section scores to get personalized guidance.")


@st.cache_data
def load_data():
    return pd.read_csv("IELTS_Student_Performance_Dataset_50000.csv")

df = load_data()


X = df[['Listening_Score','Reading_Score','Writing_Score','Speaking_Score']]

le = LabelEncoder()
df['Area_Encoded'] = le.fit_transform(df['Area_To_Improve'])
y = df['Area_Encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

suggestion_map = (
    df.groupby('Area_To_Improve')['Improvement_Suggestion']
    .first()
    .to_dict()
)


st.subheader(" Enter Scores (0 - 9)")

listening = st.number_input("Listening Score", 0.0, 9.0, step=0.5)
reading   = st.number_input("Reading Score", 0.0, 9.0, step=0.5)
writing   = st.number_input("Writing Score", 0.0, 9.0, step=0.5)
speaking  = st.number_input("Speaking Score", 0.0, 9.0, step=0.5)


if st.button(" Analyze Performance"):
    student = [[listening, reading, writing, speaking]]

    avg_score = round((listening + reading + writing + speaking) / 4, 2)

   
    pred_encoded = model.predict(student)[0]
    area = le.inverse_transform([pred_encoded])[0]

    suggestion = suggestion_map.get(
        area,
        "Focus on regular practice and take full IELTS mock tests."
    )


    st.success(" Analysis Complete")

    st.metric("Average Score", avg_score)
    st.metric(" Area To Improve", area)

    st.subheader(" What You Should Do")
    st.write(suggestion)

  
    if avg_score < 5.5:
        st.error(" High Risk: Immediate improvement required.")
    elif avg_score <= 6.5:
        st.warning(" Moderate Risk: Improvement recommended.")
    else:
        st.info(" Good Performance: Focus on refinement.")
