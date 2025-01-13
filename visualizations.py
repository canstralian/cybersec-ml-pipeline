import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

class Visualizer:
    def __init__(self):
        plt.style.use('seaborn')
    
    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance from the trained model."""
        try:
            plt.figure(figsize=(10, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.title("Feature Importance")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(
                range(len(importances)),
                [feature_names[i] for i in indices],
                rotation=45,
                ha='right'
            )
            plt.tight_layout()
            return plt.gcf()
            
        except Exception as e:
            raise Exception(f"Error plotting feature importance: {str(e)}")
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        try:
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar=False
            )
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()
            return plt.gcf()
            
        except Exception as e:
            raise Exception(f"Error plotting confusion matrix: {str(e)}")
    
    def plot_roc_curve(self, model, X_test, y_test):
        """Plot ROC curve."""
        try:
            plt.figure(figsize=(8, 6))
            y_prob = model.predict_proba(X_test)
            
            # Handle multi-class case
            if y_prob.shape[1] > 2:
                # Plot ROC curve for each class
                for i in range(y_prob.shape[1]):
                    fpr, tpr, _ = roc_curve(
                        (y_test == i).astype(int),
                        y_prob[:, i]
                    )
                    auc_score = auc(fpr, tpr)
                    plt.plot(
                        fpr,
                        tpr,
                        label=f'Class {i} (AUC = {auc_score:.2f})'
                    )
            else:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                auc_score = auc(fpr, tpr)
                plt.plot(
                    fpr,
                    tpr,
                    label=f'ROC curve (AUC = {auc_score:.2f})'
                )
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.tight_layout()
            return plt.gcf()
            
        except Exception as e:
            raise Exception(f"Error plotting ROC curve: {str(e)}")
