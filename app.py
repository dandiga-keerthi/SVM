import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import requests
from datetime import datetime

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

# logger
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# Session state
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

# Folder Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

st.set_page_config("End-to-End SVM", layout="wide")
st.title("End-to-End SVM Platform")

# Sidebar
st.sidebar.header("SVM Settings")
kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
C = st.sidebar.slider("C", 0.01, 10.0, 1.0)
gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])

# ============================================================================
# Step 1 : Data Ingestion
st.header("Step 1 : Data Ingestion")
option = st.radio("Choose Data Source", ["Download Dataset", "Upload CSV"])

df = None

if option == "Download Dataset":
    if st.button("Download Iris Dataset"):
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        df = pd.read_csv(url)
        st.success("Dataset Downloaded")

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File Uploaded")

# ============================================================================
# Step 2 : EDA
if df is not None:
    st.header("Step 2 : EDA")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write("Missing Values:", df.isnull().sum())

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ============================================================================
# Step 3 : Data Cleaning
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
    st.success("Data Cleaning Completed")

# ============================================================================
# Step 4 : Save Cleaned Data
if st.button("Save Cleaned Dataset"):
    if st.session_state.df_clean is None:
        st.error("No cleaned data available")
    else:
        path = os.path.join(CLEAN_DIR, "cleaned_data.csv")
        st.session_state.df_clean.to_csv(path, index=False)
        st.success("Cleaned Dataset Saved")

# ============================================================================
# Step 5 : Load Cleaned Dataset
st.header("Step 5 : Load Cleaned Dataset")
files = os.listdir(CLEAN_DIR)

if files:
    selected = st.selectbox("Select dataset", files)
    df_model = pd.read_csv(os.path.join(CLEAN_DIR, selected))
    st.dataframe(df_model.head())
else:
    st.stop()

# ============================================================================
# Step 6 : Train SVM
st.header("Step 6 : Train SVM")

target = st.selectbox("Select Target Column", df_model.columns)
y = df_model[target]

# 🚨 FIX: block continuous target
if y.dtype != "object" and y.nunique() > 20:
    st.error("Continuous target selected ❌\nSVM Classifier requires categorical labels.")
    st.stop()

if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)

x = df_model.drop(columns=[target])
x = x.select_dtypes(include=np.number)

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

model = SVC(kernel=kernel, C=C, gamma=gamma)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"Accuracy: {acc:.2f}")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)
