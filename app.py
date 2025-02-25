import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

# Set Streamlit page configuration
st.set_page_config(page_title="Employee Dashboard", layout="wide")

# Dummy user credentials
USER_CREDENTIALS = {
    "user": "password123",
    "admin": "admin123"
}

# Login function
def login():
    st.sidebar.header("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"): 
        if USER_CREDENTIALS.get(username) == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success(f"✅ Welcome, {username}!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

# ------------------- Check Login Status -------------------
def logout():
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

# Check login status
if not st.session_state.get("logged_in"):
    login()
else:
    logout()

    # ------------------- Load Models -------------------
    xgb_model = joblib.load("xgb_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    logreg_model = joblib.load("logreg_model.pkl")
    blended_model = joblib.load("blended_model.pkl")

    # ------------------- Load Data -------------------
    df = pd.read_csv(r"C:/Users/Lenovo/Downloads/Internship/Data_Science_Prj/Data/final_table.csv")

    # Handle Missing Values
    df.loc[:, "other_dept_name"] = df["other_dept_name"].fillna("None")

    # ------------------- Sidebar Filters -------------------
    st.sidebar.header("🔍 Employee Filters")
    
    # Define department list (including "All" as an option)
    departments = sorted(df["primary_dept_name"].unique())

    primary_departments = ["All"] + sorted(df["primary_dept_name"].unique().tolist())
    selected_primary_dept = st.sidebar.selectbox("Select Primary Department", primary_departments)

    other_departments = ["All"] + sorted(df["other_dept_name"].unique().tolist())
    selected_other_dept = st.sidebar.selectbox("Select Other Department", other_departments)

    # Title Filter
    titles = df["title"].unique()
    selected_title = st.sidebar.selectbox("Select Job Title", ["All"] + list(titles))

    genders = ["All", "M", "F"]
    selected_gender = st.sidebar.selectbox("Select Gender", genders)

    age_range = st.sidebar.slider("Select Age Range", int(df["age"].min()), int(df["age"].max()), (25, 50))
    tenure_range = st.sidebar.slider("Select Tenure Range (Years)", 0, 16, (2, 10))

    performance_ratings = ["All"] + sorted(df["last_performance_rating"].dropna().unique().tolist())
    selected_rating = st.sidebar.selectbox("Select Performance Rating", ["All"] + performance_ratings)

    salary_range = st.sidebar.slider("Salary Range", int(df["salary"].min()), int(df["salary"].max()), (40000, 120000))

        # Apply Filters
    filtered_df = df.copy()
    if selected_primary_dept != "All":
        filtered_df = filtered_df[filtered_df["primary_dept_name"] == selected_primary_dept]
    if selected_other_dept == "None":
        filtered_df = filtered_df[filtered_df["other_dept_name"].isna()]    
    if selected_other_dept != "All":
        filtered_df = filtered_df[filtered_df["other_dept_name"] == selected_other_dept]
    if selected_title != "All":
        filtered_df = filtered_df[filtered_df["title"] == selected_title]    
    if selected_rating != "All":
        filtered_df = filtered_df[filtered_df["last_performance_rating"] == selected_rating]
    if selected_gender != "All":
        filtered_df = filtered_df[filtered_df["sex"] == selected_gender]

    filtered_df = filtered_df[
        (filtered_df["age"] >= age_range[0]) & (filtered_df["age"] <= age_range[1]) &
        (filtered_df["tenure"] >= tenure_range[0]) & (filtered_df["tenure"] <= tenure_range[1]) &
        (filtered_df["salary"] >= salary_range[0]) & (filtered_df["salary"] <= salary_range[1])
    ]

    # ------------------- Display Filtered Data -------------------
    st.subheader(f"📄 Employee Data ({len(filtered_df)} Records)")
    st.dataframe(filtered_df)


    # ------------------- Attrition & Salary Analysis -------------------
    st.subheader("📊 Employee Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📉 Dynamic Attrition Rate by Department (Primary + Other)")

        # Combine Primary & Other Department Data
        dept_data = filtered_df.melt(
            id_vars=["left"], 
            value_vars=["primary_dept_name", "other_dept_name"], 
            var_name="Dept_Type", 
            value_name="Department"
        ).dropna()

        # Calculate Attrition Rate
        attrition_rates = (
            dept_data.groupby("Department")["left"]
            .mean()
            .sort_values()
        )

        # Plot the Attrition Rate
        fig, ax = plt.subplots()
        attrition_rates.plot(kind="bar", ax=ax, color="coral")
        ax.set_ylabel("Attrition Rate")
        ax.set_title("Attrition Rate by Department (Filtered)")
        st.pyplot(fig)

    with col2:
        st.subheader("💰 Salary Distribution by Primary Department")
        sorted_depts = filtered_df.groupby("primary_dept_name")["salary"].median().sort_values().index
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="primary_dept_name", y="salary", data=filtered_df, order=sorted_depts, ax=ax, palette="coolwarm")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("Department")
        ax.set_ylabel("Salary")
        st.pyplot(fig)

    # Additional Insights
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🎂 Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df["age"], bins=15, kde=True, color="blue", ax=ax)
        ax.set_xlabel("Age")
        ax.set_ylabel("Employee Count")
        st.pyplot(fig)

    with col4:
        st.subheader("👥 Gender Distribution")
        gender_counts = filtered_df["sex"].value_counts()
        fig, ax = plt.subplots()
        gender_counts.plot(kind="pie", autopct="%1.1f%%", colors=["lightblue", "pink"], ax=ax, startangle=90)
        ax.set_ylabel("")
        st.pyplot(fig)

    # Performance Rating Distribution
    st.subheader("📈 Performance Rating Distribution")
    rating_counts = filtered_df["last_performance_rating"].value_counts().sort_index()
    fig, ax = plt.subplots()
    sns.barplot(x=rating_counts.index, y=rating_counts.values, palette="viridis", ax=ax)
    ax.set_xlabel("Performance Rating")
    ax.set_ylabel("Number of Employees")
    st.pyplot(fig)

    # Tenure Analysis
    st.subheader("⌛ Tenure Analysis")
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("📊 Tenure Distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df["tenure"], bins=20, kde=True, color="green", ax=ax)
        ax.set_xlabel("Years at Company")
        ax.set_ylabel("Employee Count")
        st.pyplot(fig)

    with col6:
        st.subheader("💰 Salary vs Performance Rating")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="last_performance_rating", y="salary", data=filtered_df, palette="coolwarm", ax=ax)
        ax.set_xlabel("Performance Rating")
        ax.set_ylabel("Salary")
        ax.set_title("Salary Distribution Across Performance Ratings")
        st.pyplot(fig)



    # ------------------- Employee Attrition Prediction -------------------
    st.subheader("🔮 Predict Employee Attrition")

    # Sidebar Inputs
    st.sidebar.header("⚙️ Attrition Prediction Inputs")
    tenure = st.sidebar.slider("Years at Company", 0, 16, 5)
    age = st.sidebar.slider("Age", 20, 50, 30)
    salary = st.sidebar.number_input("Salary", min_value=40000, max_value=122000, value=50000)
    department_input = st.sidebar.selectbox("Department", departments)
    performance_rating = st.sidebar.selectbox("Last Performance Rating", performance_ratings)
    no_of_projects = st.sidebar.slider("Number of Projects", 1, 10, 5)

    def predict_attrition():
        # Create a dataframe with user input
        input_data = pd.DataFrame({
            "tenure": [tenure],
            "age": [age],
            "salary": [salary],
            "primary_dept_name": [department_input],
            "last_performance_rating": [performance_rating],
            "no_of_projects": [no_of_projects]
        })

        # One-hot encode categorical variables
        input_data = pd.get_dummies(input_data, columns=["primary_dept_name", "last_performance_rating"], dtype=bool)

        # Load training feature names
        train_features = joblib.load("training_features.pkl")

        for col in train_features:
            if col not in input_data.columns:
                input_data[col] = False  # Default missing one-hot encoded categories to 0

        # Ensure column order matches training data
        input_data = input_data[train_features]
        input_data = input_data.astype(float)  # Convert numerical columns to float

        # Predict probabilities using models
        xgb_prob = xgb_model.predict_proba(input_data)[:, 1]
        rf_prob = rf_model.predict_proba(input_data)[:, 1]
        logreg_prob = logreg_model.predict_proba(input_data)[:, 1]

        # Stack predictions for blended model
        stacked_features = pd.DataFrame({
            "XGB": xgb_prob,
            "RF": rf_prob,
            "LogReg": logreg_prob
        })

        # Final blended prediction
        final_pred = blended_model.predict_proba(stacked_features)[:, 1][0]

        # Display Prediction Result
        st.metric("Attrition Probability", f"{final_pred:.2%}")
        if final_pred > 0.5:
            st.error("⚠️ High chance of attrition!")
        else:
            st.success("✅ Low risk of attrition")

    st.button("Predict Attrition", on_click=predict_attrition)

    # ------------------- Model Performance Evaluation -------------------
    st.subheader("📈 Model Performance")

    def evaluate_model(model, model_name):
        X_test = joblib.load("X_test.pkl")
        y_test = joblib.load("y_test.pkl")

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        # Display Metrics
        st.subheader(f"🔹 {model_name}")
        st.write(f"**Accuracy:** {acc:.2f}")
        st.write(f"**Precision:** {prec:.2f}")
        st.write(f"**AUC Score:** {auc:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{model_name} - Confusion Matrix")
        st.pyplot(fig)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color="blue")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{model_name} - ROC Curve")
        ax.legend()
        st.pyplot(fig)

    evaluate_model(xgb_model, "XGBoost")
    evaluate_model(rf_model, "Random Forest")
    evaluate_model(logreg_model, "Logistic Regression")
