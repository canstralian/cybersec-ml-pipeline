import streamlit as st
import pandas as pd
import numpy as np
from data_processing import DataProcessor
from model_training import ModelTrainer
from visualizations import Visualizer
from utils import load_data, get_feature_names, save_model, load_saved_model, list_saved_models
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Pipeline for Purple Teaming",
    page_icon="🛡️",
    layout="wide"
)

def display_sidebar():
    st.sidebar.header("Pipeline Configuration")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset (CSV/JSON)",
        type=['csv', 'json']
    )
    return uploaded_file

def display_data_overview(df):
    st.header("1. Data Processing")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Overview")
        st.write(f"Shape: {df.shape}")
        st.write("Sample Data:")
        st.dataframe(df.head())

    with col2:
        st.subheader("Data Statistics")
        st.write(df.describe())

def display_feature_engineering(df):
    st.header("2. Feature Engineering")
    col3, col4 = st.columns(2)

    with col3:
        handling_strategy = st.selectbox(
            "Missing Values Strategy",
            ["mean", "median", "most_frequent", "constant"]
        )
        scaling_method = st.selectbox(
            "Scaling Method",
            ["standard", "minmax", "robust"]
        )

        st.subheader("Advanced Features")
        use_polynomial = st.checkbox("Use Polynomial Features")
        poly_degree = (
            st.slider("Polynomial Degree", 2, 5, 2) if use_polynomial else None
        )

        use_feature_selection = st.checkbox("Use Feature Selection")
        k_best_features = (
            st.slider("Number of Best Features", 5, 50, 10)
            if use_feature_selection else None
        )

    with col4:
        use_pca = st.checkbox("Use PCA")
        n_components = (
            st.slider("PCA Components (%)", 1, 100, 95) / 100.0
            if use_pca else None
        )

        add_cyber_features = st.checkbox("Add Cybersecurity Features")
        feature_cols = st.multiselect(
            "Select Features",
            get_feature_names(df),
            default=get_feature_names(df)
        )
        target_col = st.selectbox(
            "Select Target Column",
            df.columns.tolist()
        )

    feature_engineering_config = {
        'use_polynomial': use_polynomial,
        'poly_degree': poly_degree,
        'use_feature_selection': use_feature_selection,
        'k_best_features': k_best_features,
        'use_pca': use_pca,
        'n_components': n_components,
        'add_cyber_features': add_cyber_features
    }

    return (
        feature_engineering_config, feature_cols, target_col,
        handling_strategy, scaling_method
    )

def display_model_configuration():
    st.header("3. Model Configuration")
    col5, col6 = st.columns(2)

    with col5:
        n_estimators = st.slider(
            "Number of Trees", min_value=10, max_value=500, value=100
        )
        max_depth = st.slider(
            "Max Depth", min_value=1, max_value=50, value=10
        )

    with col6:
        min_samples_split = st.slider(
            "Min Samples Split", min_value=2, max_value=20, value=2
        )
        min_samples_leaf = st.slider(
            "Min Samples Leaf", min_value=1, max_value=10, value=1
        )

    return n_estimators, max_depth, min_samples_split, min_samples_leaf

def display_results(
    metrics, model, feature_cols, use_polynomial,
    X_train, y_test, X_test
):
    st.header("4. Results and Visualizations")
    col7, col8 = st.columns(2)

    with col7:
        st.subheader("Model Performance Metrics")
        for metric, value in metrics.items():
            st.metric(metric, f"{value:.4f}")

        model_name = st.text_input("Model Name (optional)")
        if st.button("Save Model"):
            try:
                preprocessing_params = {
                    'feature_engineering_config': feature_engineering_config,
                    'handling_strategy': handling_strategy,
                    'scaling_method': scaling_method
                }

                model_path, metadata_path = save_model(
                    model, feature_cols, preprocessing_params, metrics, model_name
                )
                st.success(
                    f"Model saved successfully! Files:\n- {model_path}\n- {metadata_path}"
                )
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")

    with col8:
        if not use_pca:
            st.subheader("Feature Importance")
            fig_importance = visualizer.plot_feature_importance(
                model,
                feature_cols if not use_polynomial else [
                    f"Feature_{i}" for i in range(X_train.shape[1])
                ]
            )
            st.pyplot(fig_importance)

        st.subheader("Confusion Matrix")
        fig_cm = visualizer.plot_confusion_matrix(y_test, model.predict(X_test))
        st.pyplot(fig_cm)

        st.subheader("ROC Curve")
        fig_roc = visualizer.plot_roc_curve(model, X_test, y_test)
        st.pyplot(fig_roc)

def display_saved_models():
    st.header("5. Saved Models")
    try:
        saved_models = list_saved_models()
        if saved_models:
            for model_info in saved_models:
                with st.expander(f"Model: {model_info['name']}"):
                    st.write(f"Type: {model_info['type']}")
                    st.write(f"Created: {model_info['created_at']}")
                    st.write("Performance Metrics:")
                    for metric, value in model_info['metrics'].items():
                        st.metric(metric, f"{value:.4f}")
        else:
            st.info("No saved models found.")
    except Exception as e:
        st.error(f"Error loading saved models: {str(e)}")

def main():
    st.title("🛡️ ML Pipeline for Cybersecurity Purple Teaming")

    uploaded_file = display_sidebar()

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)

            processor = DataProcessor()
            trainer = ModelTrainer()
            visualizer = Visualizer()

            display_data_overview(df)

            (
                feature_engineering_config, feature_cols, target_col,
                handling_strategy, scaling_method
            ) = display_feature_engineering(df)

            (
                n_estimators, max_depth, min_samples_split,
                min_samples_leaf
            ) = display_model_configuration()

            if st.button("Train Model"):
                with st.spinner("Processing data and training model..."):
                    (
                        X_train, X_test, y_train, y_test
                    ) = processor.process_data(
                        df, feature_cols, target_col, handling_strategy,
                        scaling_method, feature_engineering_config
                    )

                    model, metrics = trainer.train_model(
                        X_train, X_test, y_train, y_test,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf
                    )

                    display_results(
                        metrics, model, feature_cols, use_polynomial,
                        X_train, y_test, X_test
                    )
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("Please upload a dataset to begin.")

    display_saved_models()

if __name__ == "__main__":
    main()