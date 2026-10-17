import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Interactive Analytics & ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-text {
        color: #4B5563;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
@st.cache_data
def load_sample_data():
    """Generates synthetic dataset for demonstration purposes."""
    np.random.seed(42)
    n_samples = 300
    age = np.random.randint(18, 65, size=n_samples)
    income = np.random.randint(20000, 120000, size=n_samples)
    spending_score = np.random.randint(1, 100, size=n_samples)
    purchased = np.where((income > 50000) & (spending_score > 40), 1, 0)
    
    df = pd.DataFrame({
        "Age": age,
        "Annual_Income": income,
        "Spending_Score": spending_score,
        "Purchased": purchased
    })
    return df

# ---------------------------------------------------------
# Sidebar Setup
# ---------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Module:",
    ["Overview & Dataset", "Exploratory Data Analysis", "ML Model Training", "Make Predictions"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Load dataset
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        data = load_sample_data()
else:
    st.sidebar.info("Using built-in sample dataset.")
    data = load_sample_data()

# ---------------------------------------------------------
# Module 1: Overview & Dataset
# ---------------------------------------------------------
if page == "Overview & Dataset":
    st.markdown('<p class="main-title">Project 3: Interactive Data Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Explore raw data, examine summary statistics, and evaluate schema properties.</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", data.shape[0])
    col2.metric("Total Columns", data.shape[1])
    col3.metric("Numerical Columns", len(data.select_dtypes(include=[np.number]).columns))
    col4.metric("Categorical Columns", len(data.select_dtypes(exclude=[np.number]).columns))

    st.subheader("Data Preview")
    rows_to_show = st.slider("Select number of rows to display:", min_value=5, max_value=50, value=10)
    st.dataframe(data.head(rows_to_show), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(data.describe(), use_container_width=True)

    st.subheader("Missing Values")
    missing_data = data.isnull().sum().reset_index()
    missing_data.columns = ["Column Name", "Missing Values Count"]
    st.dataframe(missing_data, use_container_width=True)

# ---------------------------------------------------------
# Module 2: Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
elif page == "Exploratory Data Analysis":
    st.markdown('<p class="main-title">Exploratory Visualizations</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Generate customizable, interactive charts using Plotly.</p>', unsafe_allow_html=True)

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) >= 2:
        tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Histogram / Distribution", "Correlation Heatmap"])

        with tab1:
            st.subheader("Bivariate Analysis: Scatter Plot")
            c1, c2, c3 = st.columns(3)
            with c1:
                x_axis = st.selectbox("Select X-axis:", numeric_cols, index=0)
            with c2:
                y_axis = st.selectbox("Select Y-axis:", numeric_cols, index=1)
            with c3:
                color_col = st.selectbox("Color By (Optional):", [None] + list(data.columns), index=0)

            scatter_fig = px.scatter(data, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} vs {x_axis}")
            st.plotly_chart(scatter_fig, use_container_width=True)

        with tab2:
            st.subheader("Univariate Analysis: Distribution")
            selected_dist_col = st.selectbox("Select Column:", numeric_cols)
            bins = st.slider("Select Number of Bins:", min_value=5, max_value=100, value=20)
            hist_fig = px.histogram(data, x=selected_dist_col, nbins=bins, marginal="box", title=f"Distribution of {selected_dist_col}")
            st.plotly_chart(hist_fig, use_container_width=True)

        with tab3:
            st.subheader("Correlation Matrix")
            corr_matrix = data[numeric_cols].corr()
            heatmap_fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="Viridis", title="Feature Correlation Heatmap")
            st.plotly_chart(heatmap_fig, use_container_width=True)
    else:
        st.warning("Insufficient numeric features available for exploratory charting.")

# ---------------------------------------------------------
# Module 3: Machine Learning Model Training
# ---------------------------------------------------------
elif page == "ML Model Training":
    st.markdown('<p class="main-title">Supervised Learning: Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Configure hyperparameters, train a Random Forest model, and evaluate performance.</p>', unsafe_allow_html=True)

    target_candidate = [col for col in data.columns if data[col].nunique() <= 10]
    
    if not target_candidate:
        st.error("No suitable classification target column found (requires 10 or fewer unique categories).")
    else:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            target_col = st.selectbox("Select Target Variable (Y):", target_candidate)
        with col_t2:
            feature_candidates = [col for col in data.columns if col != target_col and pd.api.types.is_numeric_dtype(data[col])]
            selected_features = st.multiselect("Select Feature Columns (X):", feature_candidates, default=feature_candidates)

        if len(selected_features) > 0:
            st.subheader("Model Hyperparameters")
            hp_col1, hp_col2, hp_col3 = st.columns(3)
            with hp_col1:
                n_estimators = st.slider("Number of Trees:", 10, 200, 100, step=10)
            with hp_col2:
                max_depth = st.slider("Max Tree Depth:", 2, 20, 5)
            with hp_col3:
                test_size = st.slider("Test Split Size (%):", 10, 50, 20, step=5) / 100.0

            if st.button("Train Model", type="primary"):
                # Clean missing rows for simplicity
                clean_df = data[selected_features + [target_col]].dropna()
                X = clean_df[selected_features]
                y = clean_df[target_col]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                # Store model in session state for testing module
                st.session_state["trained_model"] = model
                st.session_state["model_features"] = selected_features
                st.session_state["target_name"] = target_col

                st.success(f"Training Complete! Test Accuracy: {acc * 100:.2f}%")

                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted Label", y="True Label"))
                    st.plotly_chart(cm_fig, use_container_width=True)

                with res_col2:
                    st.subheader("Feature Importances")
                    importance_df = pd.DataFrame({
                        "Feature": selected_features,
                        "Importance": model.feature_importances_
                    }).sort_values(by="Importance", ascending=False)
                    fi_fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h")
                    st.plotly_chart(fi_fig, use_container_width=True)
        else:
            st.warning("Please select at least one numeric feature column.")

# ---------------------------------------------------------
# Module 4: Real-time Predictions
# ---------------------------------------------------------
elif page == "Make Predictions":
    st.markdown('<p class="main-title">Interactive Model Inference</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Supply inputs to run inference against the trained classification model.</p>', unsafe_allow_html=True)

    if "trained_model" not in st.session_state:
        st.warning("No active trained model found. Please train a model in the 'ML Model Training' tab first.")
    else:
        model = st.session_state["trained_model"]
        features = st.session_state["model_features"]
        target = st.session_state["target_name"]

        st.write(f"Predicting Target: **{target}**")
        st.write("Provide input feature values below:")

        user_inputs = {}
        cols = st.columns(min(len(features), 3))

        for idx, feat in enumerate(features):
            col_target = cols[idx % 3]
            min_val = float(data[feat].min())
            max_val = float(data[feat].max())
            mean_val = float(data[feat].mean())
            
            with col_target:
                user_inputs[feat] = st.number_input(
                    label=feat,
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val
                )

        if st.button("Generate Prediction", type="primary"):
            input_df = pd.DataFrame([user_inputs])
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]

            st.markdown("---")
            st.subheader(f"Predicted Class: **{prediction}**")

            prob_df = pd.DataFrame({
                "Class": model.classes_,
                "Probability": probabilities
            })
            prob_fig = px.bar(prob_df, x="Class", y="Probability", range_y=[0, 1], title="Prediction Probabilities")
            st.plotly_chart(prob_fig, use_container_width=True)