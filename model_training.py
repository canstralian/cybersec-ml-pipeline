from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.model = None
    
    def train_model(self, X_train, X_test, y_train, y_test, **kwargs):
        """
        Train a Random Forest model with given parameters.
        
        Args:
            X_train, X_test, y_train, y_test: Training and test data
            **kwargs: Model parameters
        
        Returns:
            model: Trained model
            metrics: Dictionary of evaluation metrics
        """
        try:
            # Initialize and train model
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 2),
                min_samples_leaf=kwargs.get('min_samples_leaf', 1),
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'F1 Score': f1_score(y_test, y_pred, average='weighted')
            }
            
            return self.model, metrics
            
        except Exception as e:
            raise Exception(f"Error in model training: {str(e)}")
