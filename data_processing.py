import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import dask.dataframe as dd

class DataProcessor:
    def __init__(self):
        self.scaler = None
        self.imputer = None
    
    def _get_scaler(self, method):
        """Returns the appropriate scaler based on method."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        return scalers.get(method, StandardScaler())
    
    def _get_imputer(self, strategy):
        """Returns the appropriate imputer based on strategy."""
        return SimpleImputer(strategy=strategy)
    
    def process_data(self, df, feature_cols, target_col, impute_strategy='mean', scaling_method='standard'):
        """
        Process the data using Dask for large datasets.
        
        Args:
            df: pandas DataFrame
            feature_cols: list of feature columns
            target_col: target column name
            impute_strategy: strategy for handling missing values
            scaling_method: method for scaling features
        
        Returns:
            X_train, X_test, y_train, y_test: processed and split data
        """
        try:
            # Convert to Dask DataFrame for large dataset handling
            ddf = dd.from_pandas(df, npartitions=4)
            
            # Select features and target
            X = ddf[feature_cols].compute()
            y = ddf[target_col].compute()
            
            # Handle missing values
            self.imputer = self._get_imputer(impute_strategy)
            X = self.imputer.fit_transform(X)
            
            # Scale features
            self.scaler = self._get_scaler(scaling_method)
            X = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            raise Exception(f"Error in data processing: {str(e)}")
