import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime

def load_data(file):
    """
    Load data from uploaded file.

    Args:
        file: Streamlit uploaded file object

    Returns:
        pandas DataFrame
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
        raise Exception(f"Error loading data: {str(e)}")

def get_feature_names(df):
    """
    Get list of numeric columns suitable for features.

    Args:
        df: pandas DataFrame

    Returns:
        list of column names
    """
    try:
        # Select numeric columns
        numeric_cols = df.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()

        return numeric_cols
    except Exception as e:
        raise Exception(f"Error getting feature names: {str(e)}")

def save_model(model, feature_cols, preprocessing_params, metrics, model_name=None):
    """
    Save trained model and its metadata.

    Args:
        model: Trained sklearn model
        feature_cols: List of feature column names
        preprocessing_params: Dictionary of preprocessing parameters
        metrics: Dictionary of model performance metrics
        model_name: Optional custom name for the model

    Returns:
        saved_path: Path where model was saved
    """
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"model_{timestamp}"

        # Save paths
        model_path = f"models/{model_name}.joblib"
        metadata_path = f"models/{model_name}_metadata.json"

        # Save model using joblib
        joblib.dump(model, model_path)

        # Save metadata
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
        raise Exception(f"Error saving model: {str(e)}")

def load_saved_model(model_path, metadata_path):
    """
    Load a saved model and its metadata.

    Args:
        model_path: Path to the saved model file
        metadata_path: Path to the model metadata file

    Returns:
        model: Loaded model
        metadata: Dictionary containing model metadata
    """
    try:
        # Load model
        model = joblib.load(model_path)

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return model, metadata

    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def list_saved_models():
    """
    List all saved models in the models directory.

    Returns:
        list of dictionaries containing model info
    """
    try:
        models_info = []
        if not os.path.exists('models'):
            return models_info

        for filename in os.listdir('models'):
            if filename.endswith('_metadata.json'):
                with open(f"models/{filename}", 'r') as f:
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
        raise Exception(f"Error listing models: {str(e)}")