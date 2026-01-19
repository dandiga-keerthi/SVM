import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from datetime import datetime
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error

# --------------------------------------------------
# Logger
# --------------------------------------------------
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"[{timestamp}] {message}")

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

# --------------------------------------------------
# Folder Setup
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

log("Application Started")

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config("End-to-End SVM Platform", layout="wide")
st.title("End-to-End SVM Platform")

# --------------------------------------------------
# Sidebar : Model Settings
# --------------------------------------------------
st.sidebar.header("Model Settings")

task_type = st.sidebar.selectbox("Task Type", ["Classification", "Regression"])
kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])

log(f"Task={task_type}, Kernel={kernel}, C={C}, Gamma={gamma}")

# --------------------------------------------------
# Step 1 : Data Ingestion
# --------------------------------------------------
st.header("Step 1 : Data Ingestion")

option = st.radio("Choose Data Source", ["Download Dataset", "Upload CSV"])

df = st.session_state.df
raw_path = None

if option == "Download Dataset":
    if st.button("Download Iris Dataset"):
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response = requests.get(url)

        raw_path = os.path.join(RAW_DIR, "iris.csv")
        with open(raw_path, "wb") as f:
            f.write(response.content)

        df = pd.read_csv(raw_path)
        st.session_state.df = df
        st.success("Dataset downloaded successfully")
        log("Iris dataset downloaded")

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        raw_path = os.path.join(RAW_DIR, uploaded_file.name)
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        df = pd.read_csv(raw_path)
        st.session_state.df = df
        st.success("Dataset uploaded successfully")
        log("CSV uploaded")

# --------------------------------------------------
# Step 2 : EDA
# --------------------------------------------------
if df is not None:
    st.header("Step 2 : Exploratory Data Analysis")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write("Missing Values:", df.isnull().sum())

    fig, ax = plt.subplots()
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    log("EDA completed")

# --------------------------------------------------
# Step 3 : Data Cleaning
# --------------------------------------------------
if df is not None:
    st.header("Step 3 : Data Cleaning")

    strategy = st.selectbox("Missing Value Strategy", ["Mean", "Median", "Drop Rows"])
    df_clean = df.copy()

    if strategy == "Drop Rows":
        df_clean.dropna(inplace=True)
    else:
        for col in df_clean.select_dtypes(include=np.number):
            if strategy == "Mean":
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

    st.session_state.df_clean = df_clean
    st.success("Data cleaned successfully")
    log("Data cleaning completed")

# --------------------------------------------------
# Step 4 : Save Cleaned Data
# --------------------------------------------------
if st.button("Save Cleaned Dataset"):
    if st.session_state.df_clean is None:
        st.error("No cleaned data found")
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"cleaned_{ts}.csv"
        path = os.path.join(CLEAN_DIR, filename)
        st.session_state.df_clean.to_csv(path, index=False)
        st.success("Cleaned dataset saved")
        log(f"Saved cleaned data: {path}")

# --------------------------------------------------
# Step 5 : Load Cleaned Dataset
# --------------------------------------------------
st.header("Step 5 : Load Cleaned Dataset")

files = os.listdir(CLEAN_DIR)

if files:
    selected = st.selectbox("Select Dataset", files)
    df_model = pd.read_csv(os.path.join(CLEAN_DIR, selected))
    st.dataframe(df_model.head())
    log("Cleaned dataset loaded")

# --------------------------------------------------
# Step 6 : Train SVM
# --------------------------------------------------
st.header("Step 6 : Train SVM")

target = st.selectbox("Select Target Column", df_model.columns)

X = df_model.drop(columns=[target]).select_dtypes(include=np.number)
y = df_model[target]

if task_type == "Classification" and y.dtype == "object":
    y = LabelEncoder().fit_transform(y)
    log("Target encoded")

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

if task_type == "Classification":
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.success(f"Accuracy: {acc:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    log("SVM Classification completed")

else:
    model = SVR(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.success(f"R² Score: {r2:.2f}")
    st.info(f"MSE: {mse:.2f}")

    # Display Actual vs Predicted values
    results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })

    st.subheader("Actual vs Predicted Values")
    st.dataframe(results.head(10))

    # Scatter Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted (Regression)")
    st.pyplot(fig)

    log("SVM Regression completed")