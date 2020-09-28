import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

st.write('''
  # Diabetes Detection
  Detect if someone has diabetes using Machine Learning and python
''')

image = Image.open('/Users/sungjbyun/ml_diabetes_prediction/ml_diabetes.png')
st.image(image, caption='ML', use_column_width=True)

# Get data
df = pd.read_csv('/Users/sungjbyun/ml_diabetes_prediction/diabetes.csv')

st.subheader('Data Info')
st.dataframe(df)
st.subheader('Summary')
st.write(df.describe())
chart = st.bar_chart(df)

# Split the data
X = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.25, random_state=0)

# Get future input from user(s)
# Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome


def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucode = st.sidebar.slider('glucose', 0, 199, 117)
    bp = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.5)
    bmi = st.sidebar.slider('bmi', 0.0, 67.1, 30.5)
    diabetes_pedigree = st.sidebar.slider(
        'diabetes_pedigree', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 82, 29)
    # Store dict
    user_data = {'Pregnancies': pregnancies, 'Glocose': glucode, 'Blood Pressure': bp, 'Skin Thickness': skin_thickness,
                 'Insulin': insulin, 'BMI': bmi, 'Diabetes Pedigree Fx': diabetes_pedigree, 'Age': age}
    # Transform to DataFrame
    features = pd.DataFrame(user_data, index=[0])
    return features


# Store the user intput to variables
user_input = get_user_input()

st.subheader('User Input')
st.write(user_input)

# Train the model
forest = RandomForestClassifier()
forest.fit(X_train, y_train)

st.subheader('Model Test Accuracy: ')
st.write(str(accuracy_score(y_test, forest.predict(X_test) * 100))+'%')

st.subheader('Classification: ')
prediction = forest.predict(user_input)
st.write(prediction)
