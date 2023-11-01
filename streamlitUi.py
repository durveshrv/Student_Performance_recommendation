import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import joblib

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the student dataset
sonar_data = pd.read_csv("student_dataset_95.csv")

# Split the data into features and target
X = sonar_data.drop(columns=["Student Name", "Course Needed"], axis=1)
Y = sonar_data["Course Needed"]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model with increased max_iter
model = GaussianNB()

model.fit(X_train_scaled,Y_train)

# Save the trained model to a file
joblib.dump(model, 'decision_tree_model.pkl')

# Hardcoded YouTube links
youtube_links = {
    "Operating System": "https://youtube.com/playlist?list=PLxCzCOWd7aiGz9donHRrE9I3Mwn6XdP8p&si=lOwVDdPgX-3E9x2D",
    "Computer Networks": "https://youtube.com/playlist?list=PLxCzCOWd7aiGFBD2-2joCpWOLUrDLvVV_&si=87qvZCkfeUNpA9AF",
    "DBMS": "https://youtube.com/playlist?list=PLDzeHZWIZsTpukecmA2p5rhHM14bl2dHU&si=2fygP_XKSJy3bs7K",
    "DSA": "https://youtube.com/playlist?list=PLgUwDviBIf0oF6QL8m22w1hIDC1vJ_BHz&si=C0Eq4y9cfKCBBh6M",
    "LR": "https://youtube.com/playlist?list=PLpyc33gOcbVADMKqylI__O_O_RMeHTyNK&si=2h6xoNplE0FjA3iV",
    "Maths": "https://youtube.com/playlist?list=PLXTyt_wUBqQ5Axt76rasrdgZ4SYE27SCe&si=qi291bTY-5ITkn-O",
    "English": "https://youtube.com/playlist?list=PLsXdBvuJ5ox7U8HipGaQoS95fFcIhKRRS&si=MhyRpcrtO7VT1w4J",
}

# Function to generate the study plan
def generate_study_plan(os_marks, cn_marks, dbms_marks, dsa_marks, lr_marks, maths_marks, study_time, english_score):
    # Convert input to numeric values
    features = [float(os_marks), float(cn_marks), float(dbms_marks), float(dsa_marks), float(lr_marks), float(maths_marks), float(study_time), float(english_score)]
    # Scale the input
    features_scaled = scaler.transform([features])
    # Make a prediction
    prediction = model.predict(features_scaled)[0]
    
    # Study plan and YouTube links
    study_plan = "Study Plan:\n"
    youtube_links_text = "YouTube Links:\n"

    if prediction == 1:  # Course Required
        if float(os_marks) < 50:
            study_plan += "\nOperating System: Study hard to improve your OS score.\nat least study 2hr a day\n"
            youtube_links_text += "\nOperating System: " + youtube_links["Operating System"] + "\n"

        if float(cn_marks) < 50:
            study_plan += "\nComputer Networks: Focus on improving your CN score.\nat least study 2hr a day\n"
            youtube_links_text += "\nComputer Networks: " + youtube_links["Computer Networks"] + "\n"

        if float(dbms_marks) < 50:
            study_plan += "\nDBMS: Work on enhancing your DBMS knowledge.\nat least study 2hr a day\n"
            youtube_links_text += "\nDBMS: " + youtube_links["DBMS"] + "\n"

        if float(dsa_marks) < 50:
            study_plan += "\nDSA: Pay attention to your DSA skills.\nat least study 2hr a day\n"
            youtube_links_text += "\nDSA: " + youtube_links["DSA"] + "\n"

        if float(lr_marks) < 50:
            study_plan += "\nLogistic Regression: Improve your LR understanding.\nat least study 2hr a day\n"
            youtube_links_text += "\nLogistic Regression: " + youtube_links["LR"] + "\n"

        if float(maths_marks) < 50:
            study_plan += "\nMaths: Dedicate more time to your Maths studies.\nat least study 2hr a day\n"
            youtube_links_text += "\nMaths: " + youtube_links["Maths"] + "\n"

        if float(english_score) < 50:
            study_plan += "\nEnglish: Enhance your English language skills.\nat least study 2hr a day\n"
            youtube_links_text += "\nEnglish: " + youtube_links["English"] + "\n"

        if float(study_time) < 4:
            study_plan += "Increase your study hours to at least 4 hours per day."
        
        study_plan += "\nCourse is required."
    else:
        study_plan += "Course is not required."

    return study_plan, youtube_links_text, prediction

# Streamlit UI
st.title("Course Requirement Predictor and Study Plan")

# Input fields
os_marks = st.text_input("Operating System Score")
cn_marks = st.text_input("Computer Networks Score")
dbms_marks = st.text_input("DBMS Score")
dsa_marks = st.text_input("DSA Score")
lr_marks = st.text_input("Logistic Regression Score")
maths_marks = st.text_input("Maths Score")
study_time = st.text_input("Study Time (in hours)")
english_score = st.text_input("English Score")

# Predict button
if st.button("Get Study Plan"):
    study_plan, youtube_links_text, prediction = generate_study_plan(os_marks, cn_marks, dbms_marks, dsa_marks, lr_marks, maths_marks, study_time, english_score)
    st.subheader("Study Plan:")
    st.write(study_plan)
    st.subheader("YouTube Links:")
    st.write(youtube_links_text)
    st.subheader("Course Required: " + ("Yes" if prediction == 1 else "No"))

# Visualize the dataset with different charts
st.subheader("Visualization of Student Data")
visualization_type = st.selectbox("Select a Visualization Type", ["Histogram", "Bar Chart", "Pie Chart"])

if visualization_type == "Histogram":
    st.text("Histogram")
    st.write(sonar_data)
    for column in sonar_data.columns:
        if column != "Student Name":
            st.subheader(column)
            fig, ax = plt.subplots()
            ax.hist(sonar_data[column])
            st.pyplot(fig)
            
elif visualization_type == "Bar Chart":
    st.text("Bar Chart")
    st.write(sonar_data)
    for column in sonar_data.columns:
        if column != "Student Name":
            st.subheader(column)
            value_counts = sonar_data[column].value_counts()
            st.bar_chart(value_counts)
            
elif visualization_type == "Pie Chart":
    st.text("Pie Chart")
    st.write(sonar_data)
    for column in sonar_data.columns:
        if column != "Student Name":
            st.subheader(column)
            value_counts = sonar_data[column].value_counts()
            st.write(value_counts)
            fig, ax = plt.subplots()
            ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
            st.pyplot(fig)

# Accuracy graph
st.subheader("Accuracy Graph")

# Create an empty list to store accuracy values
accuracy_values = []

# Train the model with different max_iter values and calculate accuracy
for i in range(1, 11):
    model = LogisticRegression(max_iter=i * 100)
    model.fit(X_train_scaled, Y_train)
    Y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(Y_test, Y_pred)
    accuracy_values.append(accuracy)

# Plot the accuracy graph
plt.figure(figsize=(8, 5))
plt.plot(range(100, 1001, 100), accuracy_values, marker='o')
plt.title("Accuracy vs. Max Iterations")
plt.xlabel("Max Iterations")
plt.ylabel("Accuracy")
st.pyplot()
