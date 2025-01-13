import pandas as pd
import numpy as np

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
