"""
Credit Card Fraud Detection System
A complete machine learning pipeline for detecting fraudulent transactions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class FraudDetectionSystem:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        
    def generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic transaction data"""
        print("Generating synthetic transaction data...")
        
        # Normal transactions (95%)
        n_normal = int(n_samples * 0.95)
        normal_transactions = {
            'amount': np.random.exponential(scale=50, size=n_normal),
            'time_hour': np.random.normal(14, 4, n_normal) % 24,
            'transaction_frequency': np.random.poisson(5, n_normal),
            'days_since_last': np.random.exponential(2, n_normal),
            'merchant_category': np.random.choice([1, 2, 3, 4, 5], n_normal),
            'is_fraud': np.zeros(n_normal)
        }
        
        # Fraudulent transactions (5%)
        n_fraud = n_samples - n_normal
        fraud_transactions = {
            'amount': np.random.exponential(scale=200, size=n_fraud) + 100,
            'time_hour': np.random.choice([0, 1, 2, 3, 22, 23], n_fraud),
            'transaction_frequency': np.random.poisson(15, n_fraud),
            'days_since_last': np.random.exponential(0.5, n_fraud),
            'merchant_category': np.random.choice([1, 2, 3, 4, 5], n_fraud),
            'is_fraud': np.ones(n_fraud)
        }
        
        # Combine datasets
        df_normal = pd.DataFrame(normal_transactions)
        df_fraud = pd.DataFrame(fraud_transactions)
        df = pd.concat([df_normal, df_fraud], ignore_index=True)
        
        # Add derived features
        df['amount_log'] = np.log1p(df['amount'])
        df['is_night'] = ((df['time_hour'] >= 22) | (df['time_hour'] <= 6)).astype(int)
        df['high_frequency'] = (df['transaction_frequency'] > 10).astype(int)
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Dataset created: {len(df)} transactions")
        print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
        
        return df
    
    def explore_data(self, df):
        """Perform exploratory data analysis"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        print("\nDataset Info:")
        print(df.info())
        
        print("\nStatistical Summary:")
        print(df.describe())
        
        print("\nClass Distribution:")
        print(df['is_fraud'].value_counts())
        
        # Visualizations
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Fraud Detection - Exploratory Analysis', fontsize=16)
        
        # 1. Amount distribution
        axes[0, 0].hist(df[df['is_fraud']==0]['amount'], bins=50, alpha=0.7, label='Normal', color='green')
        axes[0, 0].hist(df[df['is_fraud']==1]['amount'], bins=50, alpha=0.7, label='Fraud', color='red')
        axes[0, 0].set_xlabel('Transaction Amount')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Transaction Amount Distribution')
        axes[0, 0].legend()
        
        # 2. Time of day
        axes[0, 1].hist(df[df['is_fraud']==0]['time_hour'], bins=24, alpha=0.7, label='Normal', color='green')
        axes[0, 1].hist(df[df['is_fraud']==1]['time_hour'], bins=24, alpha=0.7, label='Fraud', color='red')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Transaction Time Distribution')
        axes[0, 1].legend()
        
        # 3. Transaction frequency
        axes[0, 2].hist(df[df['is_fraud']==0]['transaction_frequency'], bins=30, alpha=0.7, label='Normal', color='green')
        axes[0, 2].hist(df[df['is_fraud']==1]['transaction_frequency'], bins=30, alpha=0.7, label='Fraud', color='red')
        axes[0, 2].set_xlabel('Transaction Frequency')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Transaction Frequency Distribution')
        axes[0, 2].legend()
        
        # 4. Correlation heatmap
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 0])
        axes[1, 0].set_title('Feature Correlation Matrix')
        
        # 5. Fraud by merchant category
        fraud_by_merchant = df.groupby('merchant_category')['is_fraud'].mean()
        axes[1, 1].bar(fraud_by_merchant.index, fraud_by_merchant.values, color='coral')
        axes[1, 1].set_xlabel('Merchant Category')
        axes[1, 1].set_ylabel('Fraud Rate')
        axes[1, 1].set_title('Fraud Rate by Merchant Category')
        
        # 6. Days since last transaction
        axes[1, 2].boxplot([df[df['is_fraud']==0]['days_since_last'], 
                            df[df['is_fraud']==1]['days_since_last']], 
                           labels=['Normal', 'Fraud'])
        axes[1, 2].set_ylabel('Days Since Last Transaction')
        axes[1, 2].set_title('Days Since Last Transaction')
        
        plt.tight_layout()
        plt.savefig('fraud_detection_eda.png', dpi=300, bbox_inches='tight')
        print("\nEDA plots saved as 'fraud_detection_eda.png'")
        plt.show()
    
    def prepare_data(self, df):
        """Prepare data for modeling"""
        print("\n" + "="*60)
        print("DATA PREPARATION")
        print("="*60)
        
        # Features and target
        feature_cols = ['amount', 'time_hour', 'transaction_frequency', 
                       'days_since_last', 'merchant_category', 'amount_log', 
                       'is_night', 'high_frequency']
        
        X = df[feature_cols]
        y = df['is_fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Number of features: {X_train.shape[1]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    
    def train_models(self, X_train, y_train):
        """Train multiple classification models"""
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        # Train models
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            print(f"{name} training complete!")
    
    def evaluate_models(self, X_test, y_test, feature_cols):
        """Evaluate all models and select the best one"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{name} Results:")
            print("-" * 40)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
        
        # Select best model based on F1-score
        best_model_name = max(results, key=lambda x: results[x]['f1_score'])
        self.best_model = self.models[best_model_name]
        
        print(f"\n{'='*60}")
        print(f"Best Model: {best_model_name}")
        print(f"{'='*60}")
        
        # Visualize results
        self.visualize_results(X_test, y_test, results, feature_cols)
        
        return results, best_model_name
    
    def visualize_results(self, X_test, y_test, results, feature_cols):
        """Create comprehensive visualization of results"""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. ROC Curves
        ax1 = plt.subplot(2, 3, 1)
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            ax1.plot(fpr, tpr, label=f"{name} (AUC={result['roc_auc']:.3f})")
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curves
        ax2 = plt.subplot(2, 3, 2)
        for name, result in results.items():
            precision, recall, _ = precision_recall_curve(y_test, result['y_pred_proba'])
            ax2.plot(recall, precision, label=name)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Model Performance Comparison
        ax3 = plt.subplot(2, 3, 3)
        metrics = ['accuracy', 'f1_score', 'roc_auc']
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, (name, result) in enumerate(results.items()):
            values = [result[m] for m in metrics]
            ax3.bar(x + i*width, values, width, label=name)
        
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Score')
        ax3.set_title('Model Performance Comparison')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(['Accuracy', 'F1-Score', 'ROC-AUC'])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Confusion Matrix for best model
        best_name = max(results, key=lambda x: results[x]['f1_score'])
        cm = confusion_matrix(y_test, results[best_name]['y_pred'])
        ax4 = plt.subplot(2, 3, 4)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        ax4.set_title(f'Confusion Matrix - {best_name}')
        ax4.set_xticklabels(['Normal', 'Fraud'])
        ax4.set_yticklabels(['Normal', 'Fraud'])
        
        # 5. Feature Importance (for Random Forest)
        if 'Random Forest' in self.models:
            ax5 = plt.subplot(2, 3, 5)
            importances = self.models['Random Forest'].feature_importances_
            indices = np.argsort(importances)[::-1]
            ax5.barh(range(len(importances)), importances[indices], color='skyblue')
            ax5.set_yticks(range(len(importances)))
            ax5.set_yticklabels([feature_cols[i] for i in indices])
            ax5.set_xlabel('Importance')
            ax5.set_title('Feature Importance (Random Forest)')
            ax5.grid(True, alpha=0.3, axis='x')
        
        # 6. Fraud Detection Rate vs Threshold
        ax6 = plt.subplot(2, 3, 6)
        thresholds = np.linspace(0, 1, 100)
        detection_rates = []
        false_positive_rates = []
        
        best_proba = results[best_name]['y_pred_proba']
        for thresh in thresholds:
            y_pred_thresh = (best_proba >= thresh).astype(int)
            detection_rate = np.sum((y_pred_thresh == 1) & (y_test == 1)) / np.sum(y_test == 1)
            false_positive_rate = np.sum((y_pred_thresh == 1) & (y_test == 0)) / np.sum(y_test == 0)
            detection_rates.append(detection_rate)
            false_positive_rates.append(false_positive_rate)
        
        ax6.plot(thresholds, detection_rates, label='Detection Rate', color='green')
        ax6.plot(thresholds, false_positive_rates, label='False Positive Rate', color='red')
        ax6.set_xlabel('Classification Threshold')
        ax6.set_ylabel('Rate')
        ax6.set_title('Detection vs False Positive Rate')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fraud_detection_results.png', dpi=300, bbox_inches='tight')
        print("\nResults visualization saved as 'fraud_detection_results.png'")
        plt.show()
    
    def predict_transaction(self, transaction_data):
        """Predict if a new transaction is fraudulent"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        
        transaction_scaled = self.scaler.transform([transaction_data])
        prediction = self.best_model.predict(transaction_scaled)[0]
        probability = self.best_model.predict_proba(transaction_scaled)[0][1]
        
        return prediction, probability


def main():
    """Main execution function"""
    print("="*60)
    print("FRAUD DETECTION SYSTEM")
    print("="*60)
    
    # Initialize system
    fraud_detector = FraudDetectionSystem()
    
    # Generate data
    df = fraud_detector.generate_synthetic_data(n_samples=10000)
    
    # Save dataset
    df.to_csv('fraud_transactions.csv', index=False)
    print("\nDataset saved as 'fraud_transactions.csv'")
    
    # Explore data
    fraud_detector.explore_data(df)
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_cols = fraud_detector.prepare_data(df)
    
    # Train models
    fraud_detector.train_models(X_train, y_train)
    
    # Evaluate models
    results, best_model_name = fraud_detector.evaluate_models(X_test, y_test, feature_cols)
    
    # Test prediction on a new transaction
    print("\n" + "="*60)
    print("TESTING NEW TRANSACTION PREDICTION")
    print("="*60)
    
    # Example: Suspicious transaction (high amount, late night, high frequency)
    new_transaction = [250.0, 2.0, 18, 0.3, 3, np.log1p(250.0), 1, 1]
    
    prediction, probability = fraud_detector.predict_transaction(new_transaction)
    
    print("\nTest Transaction Features:")
    print(f"Amount: $250.00")
    print(f"Time: 2 AM")
    print(f"Transaction Frequency: 18")
    print(f"Days Since Last: 0.3")
    print(f"\nPrediction: {'FRAUD' if prediction == 1 else 'NORMAL'}")
    print(f"Fraud Probability: {probability*100:.2f}%")
    
    print("\n" + "="*60)
    print("PROJECT COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print("1. fraud_transactions.csv - Dataset")
    print("2. fraud_detection_eda.png - Exploratory Analysis")
    print("3. fraud_detection_results.png - Model Results")


if __name__ == "__main__":
    main()