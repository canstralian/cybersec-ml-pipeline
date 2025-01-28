import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA


class EnhancedModelTrainer:
    def __init__(self):
        self.pipeline = None
        self.model = None

    def preprocess_data(self, X, y, categorical_features, numerical_features):
        """
        Preprocesses the dataset with advanced techniques.
        Args:
            X: Features dataframe
            y: Target variable
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names

        Returns:
            X_resampled, y_resampled: Preprocessed and balanced data
        """
        try:
            # Impute missing values
            num_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            cat_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Preprocess both numerical and categorical features
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_transformer, numerical_features),
                    ('cat', cat_transformer, categorical_features)
                ]
            )

            # Apply preprocessing
            X_preprocessed = preprocessor.fit_transform(X)

            # Handle class imbalance with SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

            return X_resampled, y_resampled, preprocessor
        except Exception as e:
            raise Exception(f"Error in preprocessing: {str(e)}")

    def train_model(self, X_train, y_train, **kwargs):
        """
        Train a Random Forest model with hyperparameter tuning.
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Hyperparameters for the Random Forest model

        Returns:
            pipeline: Fitted pipeline
        """
        try:
            # PCA for dimensionality reduction
            pca = PCA(n_components=0.95)

            # Random Forest model with hyperparameter tuning
            rf = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 2),
                min_samples_leaf=kwargs.get('min_samples_leaf', 1),
                random_state=42,
                n_jobs=-1
            )

            # Define pipeline
            self.pipeline = Pipeline(steps=[
                ('pca', pca),
                ('classifier', rf)
            ])

            # Fit the pipeline
            self.pipeline.fit(X_train, y_train)

            return self.pipeline
        except Exception as e:
            raise Exception(f"Error in model training: {str(e)}")

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model.
        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        try:
            y_pred = self.pipeline.predict(X_test)
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'F1 Score': f1_score(y_test, y_pred, average='weighted')
            }
            return metrics
        except Exception as e:
            raise Exception(f"Error in evaluation: {str(e)}")


# Example Usage
if __name__ == "__main__":
    # Mock dataset (replace with your actual dataset)
    data = pd.DataFrame({
        'Feature1': np.random.rand(100),
        'Feature2': np.random.rand(100),
        'Category': np.random.choice(['A', 'B', 'C'], 100),
        'Target': np.random.choice([0, 1], 100)
    })

    # Split data
    X = data[['Feature1', 'Feature2', 'Category']]
    y = data['Target']
    categorical_features = ['Category']
    numerical_features = ['Feature1', 'Feature2']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and preprocess
    trainer = EnhancedModelTrainer()
    X_train_processed, y_train_processed, preprocessor = trainer.preprocess_data(
        X_train, y_train, categorical_features, numerical_features
    )
    X_test_processed = preprocessor.transform(X_test)

    # Train and evaluate
    trained_pipeline = trainer.train_model(
        X_train_processed, y_train_processed,
        n_estimators=200, max_depth=15
    )
    metrics = trainer.evaluate_model(X_test_processed, y_test)

    print("Evaluation Metrics:", metrics)