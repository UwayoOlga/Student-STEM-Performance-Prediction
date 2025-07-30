#!/usr/bin/env python3
"""
Machine Learning Models for STEM Performance Prediction
Predicts which students are more likely to excel in STEM subjects
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class STEMPerformancePredictor:
    def __init__(self, data_path="oulad_cleaned"):
        self.data_path = data_path
        self.datasets = {}
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
        self.load_data()
        self.define_stem_modules()
        
    def define_stem_modules(self):
        """Define STEM modules based on known OULAD mappings"""
        self.stem_modules = {
            'AAA': 'Computing and IT',
            'FFF': 'Science', 
            'GGG': 'Engineering and Technology',
            'HHH': 'Mathematics and Statistics'
        }
        
        print("STEM Modules Identified:")
        for code, subject in self.stem_modules.items():
            print(f"  {code}: {subject}")
    
    def load_data(self):
        """Load cleaned datasets"""
        print("Loading cleaned datasets...")
        
        try:
            self.datasets['studentInfo'] = pd.read_csv(f"{self.data_path}/studentInfo_cleaned.csv")
            self.datasets['studentAssessment'] = pd.read_csv(f"{self.data_path}/studentAssessment_cleaned.csv")
            self.datasets['merged'] = pd.read_csv(f"{self.data_path}/merged_cleaned.csv")
            print("Datasets loaded successfully!")
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def prepare_stem_dataset(self):
        """Prepare STEM-specific dataset for machine learning"""
        print("Preparing STEM dataset for machine learning...")
        
        # Get student info
        student_info = self.datasets['studentInfo'].copy()
        
        # Add STEM labels
        student_info['subject_area'] = student_info['code_module'].map(self.stem_modules)
        student_info['is_stem'] = student_info['code_module'].isin(self.stem_modules.keys())
        
        # Filter for STEM students only
        stem_students = student_info[student_info['is_stem'] == True].copy()
        
        print(f"Total STEM students: {len(stem_students):,}")
        
        # Create target variables
        # STEM Excellence: Distinction only
        stem_students['stem_excellence'] = np.where(
            stem_students['final_result'] == -2.0289591117016377,  # Distinction
            1,  # STEM Excellence
            0   # Not Excellence
        )
        
        # STEM Success: Distinction or Pass
        stem_students['stem_success'] = np.where(
            (stem_students['final_result'] == -2.0289591117016377) |  # Distinction
            (stem_students['final_result'] == 0.08374058178886125),   # Pass
            1,  # Success
            0   # Failure
        )
        
        # Print target distributions
        excellence_dist = stem_students['stem_excellence'].value_counts()
        success_dist = stem_students['stem_success'].value_counts()
        
        print("STEM Excellence Distribution:")
        print(f"  Excellence (1): {excellence_dist.get(1, 0):,} ({excellence_dist.get(1, 0)/len(stem_students)*100:.1f}%)")
        print(f"  Not Excellence (0): {excellence_dist.get(0, 0):,} ({excellence_dist.get(0, 0)/len(stem_students)*100:.1f}%)")
        
        print("STEM Success Distribution:")
        print(f"  Success (1): {success_dist.get(1, 0):,} ({success_dist.get(1, 0)/len(stem_students)*100:.1f}%)")
        print(f"  Failure (0): {success_dist.get(0, 0):,} ({success_dist.get(0, 0)/len(stem_students)*100:.1f}%)")
        
        return stem_students
    
    def select_features(self, df):
        """Select relevant features for STEM performance prediction"""
        print("Selecting features for STEM performance prediction...")
        
        # Define feature columns (excluding identifiers and targets)
        feature_columns = [
            'gender', 'region', 'highest_education', 'imd_band', 'age_band',
            'num_of_prev_attempts', 'studied_credits', 'disability'
        ]
        
        # Filter available features
        available_features = [col for col in feature_columns if col in df.columns]
        
        print(f"Selected features: {available_features}")
        print(f"Number of features: {len(available_features)}")
        
        return available_features
    
    def prepare_features_and_targets(self, df, target_column):
        """Prepare features and target for machine learning"""
        print(f"Preparing features and target: {target_column}")
        
        # Select features
        feature_columns = self.select_features(df)
        
        # Prepare X (features) and y (target)
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Ensure all features are numerical
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_models(self, X, y, target_name):
        """Train multiple machine learning models"""
        print(f"Training models for {target_name} prediction...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            if name == 'SVM':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation score
            if name == 'SVM':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  {name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
        
        # Store results
        self.results[target_name] = results
        
        return results
    
    def hyperparameter_tuning(self, X, y, target_name):
        """Perform hyperparameter tuning for the best model"""
        print(f"Performing hyperparameter tuning for {target_name}...")
        
        # Get the best model based on F1 score
        best_model_name = max(self.results[target_name].keys(), 
                            key=lambda x: self.results[target_name][x]['f1_score'])
        
        print(f"Best model for tuning: {best_model_name}")
        
        # Define parameter grids for different models
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if best_model_name in param_grids:
            # Perform grid search
            grid_search = GridSearchCV(
                self.results[target_name][best_model_name]['model'],
                param_grids[best_model_name],
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best F1 score: {grid_search.best_score_:.3f}")
            
            # Evaluate tuned model on test set
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            tuned_model = grid_search.best_estimator_
            tuned_model.fit(X_train, y_train)
            y_pred = tuned_model.predict(X_test)
            y_pred_proba = tuned_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics for tuned model
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Store results with full metrics
            self.results[target_name][f"{best_model_name} (Tuned)"] = {
                'model': tuned_model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
    
    def create_visualizations(self, target_name):
        """Create visualizations for model performance"""
        print(f"Creating visualizations for {target_name}...")
        
        results = self.results[target_name]
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'STEM {target_name.title()} Prediction - Model Comparison', fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        
        axes[0, 0].bar(model_names, accuracies)
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim(0, 1)
        
        # 2. F1 Score comparison
        f1_scores = [results[name]['f1_score'] for name in model_names]
        
        axes[0, 1].bar(model_names, f1_scores)
        axes[0, 1].set_title('Model F1 Score Comparison')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylim(0, 1)
        
        # 3. ROC Curves
        for name in model_names:
            if 'y_pred_proba' in results[name]:
                fpr, tpr, _ = roc_curve(results[name]['y_test'], results[name]['y_pred_proba'])
                auc = results[name]['auc']
                axes[1, 0].plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1, 0].set_title('ROC Curves')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. Confusion Matrix for best model
        best_model_name = max(model_names, key=lambda x: results[x]['f1_score'])
        cm = confusion_matrix(results[best_model_name]['y_test'], results[best_model_name]['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'stem_{target_name.lower()}_prediction_results.png', dpi=300, bbox_inches='tight')
        print(f"Visualizations saved as 'stem_{target_name.lower()}_prediction_results.png'")
    
    def feature_importance_analysis(self, X, target_name):
        """Analyze feature importance for the best model"""
        print(f"Analyzing feature importance for {target_name}...")
        
        # Get the best model
        best_model_name = max(self.results[target_name].keys(), 
                            key=lambda x: self.results[target_name][x]['f1_score'])
        
        model = self.results[target_name][best_model_name]['model']
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print("Model doesn't support feature importance analysis")
            return
        
        # Create feature importance plot
        feature_names = X.columns
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Feature Importance - {best_model_name} ({target_name})')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(f'stem_{target_name.lower()}_feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"Feature importance saved as 'stem_{target_name.lower()}_feature_importance.png'")
        
        # Print top features
        print("Top 5 Most Important Features:")
        for i in range(min(5, len(indices))):
            print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.3f}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("STEM PERFORMANCE PREDICTION SUMMARY REPORT")
        print("=" * 60)
        
        for target_name, results in self.results.items():
            print(f"\n{target_name.upper()} PREDICTION RESULTS")
            print("-" * 40)
            
            # Find best model
            best_model_name = max(results.keys(), 
                                key=lambda x: results[x]['f1_score'])
            
            print(f"Best Model: {best_model_name}")
            print(f"Accuracy: {results[best_model_name]['accuracy']:.3f}")
            print(f"Precision: {results[best_model_name]['precision']:.3f}")
            print(f"Recall: {results[best_model_name]['recall']:.3f}")
            print(f"F1 Score: {results[best_model_name]['f1_score']:.3f}")
            print(f"AUC: {results[best_model_name]['auc']:.3f}")
            
            # Model comparison
            print(f"\nModel Comparison:")
            for name, metrics in results.items():
                print(f"  {name}: F1={metrics['f1_score']:.3f}, AUC={metrics['auc']:.3f}")
        
        print(f"\nFiles Generated:")
        for target_name in self.results.keys():
            print(f"  stem_{target_name.lower()}_prediction_results.png - Model performance")
            print(f"  stem_{target_name.lower()}_feature_importance.png - Feature importance")
        
        print("\nSTEM performance prediction completed successfully!")

def main():
    """Main function to run STEM performance prediction"""
    print("STEM PERFORMANCE PREDICTION USING MACHINE LEARNING")
    print("=" * 60)
    
    try:
        # Initialize predictor
        predictor = STEMPerformancePredictor()
        
        # Prepare STEM dataset
        stem_data = predictor.prepare_stem_dataset()
        
        # Train models for STEM Excellence prediction
        print("\n" + "="*50)
        print("PREDICTING STEM EXCELLENCE")
        print("="*50)
        
        X_excellence, y_excellence = predictor.prepare_features_and_targets(stem_data, 'stem_excellence')
        excellence_results = predictor.train_models(X_excellence, y_excellence, 'Excellence')
        
        # Train models for STEM Success prediction
        print("\n" + "="*50)
        print("PREDICTING STEM SUCCESS")
        print("="*50)
        
        X_success, y_success = predictor.prepare_features_and_targets(stem_data, 'stem_success')
        success_results = predictor.train_models(X_success, y_success, 'Success')
        
        # Hyperparameter tuning
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING")
        print("="*50)
        
        predictor.hyperparameter_tuning(X_excellence, y_excellence, 'Excellence')
        predictor.hyperparameter_tuning(X_success, y_success, 'Success')
        
        # Create visualizations
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        predictor.create_visualizations('Excellence')
        predictor.create_visualizations('Success')
        
        # Feature importance analysis
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        predictor.feature_importance_analysis(X_excellence, 'Excellence')
        predictor.feature_importance_analysis(X_success, 'Success')
        
        # Generate summary report
        predictor.generate_summary_report()
        
    except Exception as e:
        print(f"Error during STEM prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 