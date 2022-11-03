import streamlit as st
import pandas as pd
import os

# Import profiling capabalities
import pandas_profiling 
from streamlit_pandas_profiling import st_profile_report

# ML Stuff
from pycaret.classification import setup, compare_models, pull, save_model

# Navigation List
nav = ["Import data", "Data Profiling", "Auto Model", "Download Model"]

# Side Bar
with st.sidebar:
    st.title("Project : <AutoML>")
    st.image("https://149695847.v2.pressablecdn.com/wp-content/uploads/2020/08/ml-steps.png")
    choice = st.radio("Navigation", nav)

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

# Main Frame

# Step 01 - Import Data
if choice == nav[0]:
    st.header("Import the data for modeling :")
    file = st.file_uploader("Upload your file here!")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv",index=None)
        st.dataframe(df)

# Step 02 - Data Profiling
if choice == nav[1]:
    st.header("Automated EDA (Exploratory Data Analysis) :")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

# Step 03 - Auto Model
if choice == nav[2]:
    st.header("Auto ML Regression with PyCaret :")
    target = st.selectbox("Select the prediction variable 'target' :", df.columns)
    if st.button("Train Model"):
        setup(df, target=target)
        setup_df = pull()
        st.info("This is the ML Experiment Settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, "best_model")


# Step 04 - Download Model
if choice == nav[3]:
    st.header("Download the best performing model as \n(.pkl file)")
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the Model", f, "auto_model.pkl")