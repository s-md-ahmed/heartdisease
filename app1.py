import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import streamlit as st
from sklearn.metrics import classification_report
import numpy as np
# Load the dataset
np.random.seed(42)
data1 = pd.read_csv("heart_disease_uci.csv")
dropped_columns = ['id', 'dataset', 'ca', 'thal', 'slope']
data1.drop(dropped_columns, axis=1, inplace=True)

# Handling missing values
mean_trestbps = data1["trestbps"].mean()
data1["trestbps"].fillna(mean_trestbps, inplace=True)
mean_chol = data1["chol"].mean()
data1["chol"].fillna(mean_chol, inplace=True)
data1["fbs"].fillna(True, inplace=True)
mean_thalch = data1["thalch"].mean()
data1["thalch"].fillna(mean_thalch, inplace=True)
data1["restecg"].fillna("normal", inplace=True)
data1["exang"].fillna(True, inplace=True)
mean_oldpeak = data1["oldpeak"].mean()
data1["oldpeak"].fillna(mean_oldpeak, inplace=True)

# Label Encode categorical columns
le = LabelEncoder()
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang']
data1[categorical_columns] = data1[categorical_columns].apply(le.fit_transform)

X = data1.drop("num", axis=1)  # Features
y = data1["num"]  # Target variable

robust_scaler = RobustScaler()
X_scaled = robust_scaler.fit_transform(X)

# Apply MinMaxScaler to features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_scaled)

# Apply SMOTE to handle class imbalance
smt = SMOTE(sampling_strategy='not majority')
X_resampled, y_resampled = smt.fit_resample(X_scaled, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf_rf = RandomForestClassifier(n_estimators=30, max_depth=9)
clf_rf.fit(X_train, y_train)

# Streamlit UI
st.title("Welcome to my heart disease prediction App")

with st.form("user_input_form"):
    # Create input fields for features
    Age = st.slider("Age", int(X['age'].min()), int(X['age'].max()), 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("cp", ["atypical angina", "non-anginal", "typical angina", "asymptomatic"])
    trestbps = st.slider("trestbps", int(X['trestbps'].min()), int(X['trestbps'].max()), int(X['trestbps'].mean()))
    chol = st.slider("Cholestrol", int(X['chol'].min()), int(X['chol'].max()), int(X['chol'].mean()))
    fbs_options = [False, True]
    fbs = st.selectbox("fbs", fbs_options)
    restecg = st.selectbox("Restecg", ["lv hypertrophy", "normal", "st-t abnormality"])
    thalch = st.slider("Thalch", int(X['thalch'].min()), int(X['thalch'].max()), 65)
    exang_options = [False, True]
    exang = st.selectbox("exang", exang_options)
    oldpeak = st.slider("oldpeak", float(X['oldpeak'].min()), float(X['oldpeak'].max()), float(X['oldpeak'].mean()))
    submit_button = st.form_submit_button("Submit Prediction")

    # Label encode user input
    sex = le.transform([sex])[0]
    cp = le.transform([cp])[0]
    fbs = le.transform([fbs])[0]
    restecg = le.transform([restecg])[0]
    exang = le.transform([exang])[0]

    # Get the correct ordering of columns
    user_input_columns = X.columns

    # Create user_input DataFrame
    user_input = pd.DataFrame({'age': [Age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
                               'chol': [chol], 'fbs': [fbs], 'restecg': [restecg],
                               'thalch': [thalch], 'exang': [exang], 'oldpeak': oldpeak},
                              columns=user_input_columns)
    user_input_scaled = robust_scaler.transform(user_input)
    user_input_scaled = scaler.transform(user_input_scaled)

if submit_button:
    # Make predictions on user input
    prediction = clf_rf.predict(user_input_scaled)
    st.subheader("Prediction")

    if prediction[0] in [1, 2, 3, 4]:
        st.markdown(f'<p style="color:red;">Heart disease is present with severity level of {prediction[0]}</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:green;">Heart disease is not there</p>', unsafe_allow_html=True)
# Display classification report for the training set
# st.subheader("Classification Report (Training Set)")
# y_train_pred = clf_rf.predict(X_train)
# classification_rep_train = classification_report(y_train, y_train_pred, output_dict=True)
# st.table(pd.DataFrame(classification_rep_train).transpose())
# # Display classification report for the test set
# st.subheader("Classification Report (Test Set)")
# y_test_pred = clf_rf.predict(X_test)
# classification_rep_test = classification_report(y_test, y_test_pred, output_dict=True)
# st.table(pd.DataFrame(classification_rep_test).transpose())

# Import your analytics integration module


