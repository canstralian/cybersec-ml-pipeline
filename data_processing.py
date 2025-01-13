import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import dask.dataframe as dd

class DataProcessor:
    def __init__(self):
        self.scaler = None
        self.imputer = None
        self.poly_features = None
        self.feature_selector = None
        self.pca = None

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

    def _engineer_features(self, X, feature_engineering_config):
        """Apply feature engineering transformations."""
        # Polynomial Features
        if feature_engineering_config.get('use_polynomial', False):
            degree = feature_engineering_config.get('poly_degree', 2)
            self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X = self.poly_features.fit_transform(X)

        # Feature Selection
        if feature_engineering_config.get('use_feature_selection', False):
            k = feature_engineering_config.get('k_best_features', 10)
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            X = self.feature_selector.fit_transform(X)

        # Dimensionality Reduction
        if feature_engineering_config.get('use_pca', False):
            n_components = feature_engineering_config.get('n_components', 0.95)
            self.pca = PCA(n_components=n_components)
            X = self.pca.fit_transform(X)

        # Add cybersecurity-specific features
        if feature_engineering_config.get('add_cyber_features', False):
            X = self._add_cyber_features(X)

        return X

    def _add_cyber_features(self, X):
        """Add cybersecurity-specific engineered features."""
        # Convert back to DataFrame for feature engineering
        X_df = pd.DataFrame(X)

        # Example cyber features (modify based on your specific needs):
        # - Entropy of numerical features
        # - Statistical moments (skewness, kurtosis)
        # - Rolling windows statistics

        for col in X_df.columns:
            if X_df[col].dtype in ['float64', 'int64']:
                # Calculate entropy for numerical columns
                X_df[f'{col}_entropy'] = X_df[col].apply(lambda x: -np.sum(x * np.log2(x)) if x != 0 else 0)

                # Add statistical moments
                X_df[f'{col}_skew'] = X_df[col].skew()
                X_df[f'{col}_kurt'] = X_df[col].kurtosis()

                # Add rolling statistics
                X_df[f'{col}_rolling_mean'] = X_df[col].rolling(window=3, min_periods=1).mean()
                X_df[f'{col}_rolling_std'] = X_df[col].rolling(window=3, min_periods=1).std()

        return X_df.values

    def process_data(self, df, feature_cols, target_col, impute_strategy='mean', 
                    scaling_method='standard', feature_engineering_config=None):
        """
        Process the data using Dask for large datasets.

        Args:
            df: pandas DataFrame
            feature_cols: list of feature columns
            target_col: target column name
            impute_strategy: strategy for handling missing values
            scaling_method: method for scaling features
            feature_engineering_config: dictionary of feature engineering parameters

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

            # Apply feature engineering if config is provided
            if feature_engineering_config:
                X = self._engineer_features(X, feature_engineering_config)

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