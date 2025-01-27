import os
import json
from datetime import datetime
import pandas as pd
import joblib


def load_data(file):
    """
    Load data from an uploaded file.

    Args:
        file: Streamlit uploaded file object

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame

    Raises:
        ValueError: If the file format is unsupported
        Exception: If there is an error loading the data
    """
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.json'):
            df = pd.read_json(file)
        else:
            raise ValueError("Unsupported file format")
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {e}")


def get_feature_names(df):
    """
    Get a list of numeric columns suitable for features.

    Args:
        df (pd.DataFrame): DataFrame containing the data

    Returns:
        list: List of numeric column names

    Raises:
        Exception: If there is an error retrieving feature names
    """
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        return numeric_cols
    except Exception as e:
        raise Exception(f"Error getting feature names: {e}")


def save_model(model, feature_cols, preprocessing_params, metrics, model_name=None):
    """
    Save the trained model and its metadata.

    Args:
        model: Trained sklearn model
        feature_cols (list): List of feature column names
        preprocessing_params (dict): Dictionary of preprocessing parameters
        metrics (dict): Dictionary of model performance metrics
        model_name (str, optional): Custom name for the model. Defaults to None.

    Returns:
        tuple: Paths where the model and metadata were saved

    Raises:
        Exception: If there is an error saving the model or metadata
    """
    try:
        os.makedirs('models', exist_ok=True)

        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"model_{timestamp}"

        model_path = os.path.join('models', f"{model_name}.joblib")
        metadata_path = os.path.join('models', f"{model_name}_metadata.json")

        joblib.dump(model, model_path)

        metadata = {
            'feature_columns': feature_cols,
            'preprocessing_parameters': preprocessing_params,
            'performance_metrics': metrics,
            'created_at': datetime.now().isoformat(),
            'model_type': type(model).__name__
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        return model_path, metadata_path

    except Exception as e:
        raise Exception(f"Error saving model: {e}")


def load_saved_model(model_path, metadata_path):
    """
    Load a saved model and its metadata.

    Args:
        model_path (str): Path to the saved model file
        metadata_path (str): Path to the model metadata file

    Returns:
        tuple: Loaded model and metadata dictionary

    Raises:
        Exception: If there is an error loading the model or metadata
    """
    try:
        model = joblib.load(model_path)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return model, metadata

    except Exception as e:
        raise Exception(f"Error loading model: {e}")


def list_saved_models():
    """
    List all saved models in the 'models' directory.

    Returns:
        list: List of dictionaries containing model information

    Raises:
        Exception: If there is an error listing the models
    """
    try:
        models_info = []
        if not os.path.exists('models'):
            return models_info

        for filename in os.listdir('models'):
            if filename.endswith('_metadata.json'):
                metadata_path = os.path.join('models', filename)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    model_name = filename.replace('_metadata.json', '')
                    models_info.append({
                        'name': model_name,
                        'type': metadata['model_type'],
                        'created_at': metadata['created_at'],
                        'metrics': metadata['performance_metrics']
                    })

        return models_info

    except Exception as e:
        raise Exception(f"Error listing models: {e}")