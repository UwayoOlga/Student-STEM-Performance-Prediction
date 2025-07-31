#!/usr/bin/env python3
"""
Custom Ensemble Techniques for STEM Performance Prediction
Advanced ensemble methods with dynamic weighting and stacking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class CustomEnsemble:
    def __init__(self, base_models=None, meta_model=None):
        """
        Custom Ensemble with dynamic weighting and stacking
        
        Args:
            base_models: Dictionary of base models
            meta_model: Meta-learner for stacking
        """
        self.base_models = base_models or self._get_default_base_models()
        self.meta_model = meta_model or LogisticRegression(random_state=42)
        self.base_predictions = {}
        self.weights = {}
        self.is_fitted = False
        
    def _get_default_base_models(self):
        """Get default base models with diverse characteristics"""
        return {
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Ridge': RidgeClassifier(random_state=42)
        }
    
    def calculate_dynamic_weights(self, X, y):
        """
        INNOVATIVE FEATURE: Calculate dynamic weights based on model performance
        """
        print("Calculating dynamic weights for ensemble...")
        
        # Use cross-validation to estimate model performance
        cv_scores = {}
        for name, model in self.base_models.items():
            try:
                scores = cross_val_score(model, X, y, cv=5, scoring='f1')
                cv_scores[name] = scores.mean()
                print(f"  {name}: CV F1 = {scores.mean():.3f} (+/- {scores.std():.3f})")
            except:
                cv_scores[name] = 0.0
                print(f"  {name}: Failed to calculate CV score")
        
        # Convert scores to weights (softmax)
        scores_array = np.array(list(cv_scores.values()))
        weights = np.exp(scores_array) / np.sum(np.exp(scores_array))
        
        # Store weights
        for i, name in enumerate(self.base_models.keys()):
            self.weights[name] = weights[i]
        
        print("Dynamic weights calculated:")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.3f}")
        
        return self.weights
    
    def fit(self, X, y):
        """Fit the custom ensemble"""
        print("Fitting custom ensemble...")
        
        # Calculate dynamic weights
        self.calculate_dynamic_weights(X, y)
        
        # Fit base models
        for name, model in self.base_models.items():
            print(f"Fitting {name}...")
            model.fit(X, y)
        
        # Generate base predictions for stacking
        self.base_predictions = {}
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                self.base_predictions[name] = model.predict_proba(X)[:, 1]
            else:
                self.base_predictions[name] = model.predict(X)
        
        # Fit meta-model for stacking
        meta_features = np.column_stack(list(self.base_predictions.values()))
        self.meta_model.fit(meta_features, y)
        
        self.is_fitted = True
        print("Custom ensemble fitted successfully!")
        
        return self
    
    def predict(self, X):
        """Make predictions using the ensemble"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get base predictions
        base_preds = {}
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                base_preds[name] = model.predict_proba(X)[:, 1]
            else:
                base_preds[name] = model.predict(X)
        
        # Weighted voting
        weighted_pred = np.zeros(len(X))
        for name, pred in base_preds.items():
            weighted_pred += self.weights[name] * pred
        
        # Convert to binary predictions
        return (weighted_pred > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Get probability predictions"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get base predictions
        base_preds = {}
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                base_preds[name] = model.predict_proba(X)[:, 1]
            else:
                base_preds[name] = model.predict(X)
        
        # Weighted voting for probabilities
        weighted_proba = np.zeros(len(X))
        for name, pred in base_preds.items():
            weighted_proba += self.weights[name] * pred
        
        # Return as 2D array
        return np.column_stack([1 - weighted_proba, weighted_proba])
    
    def get_feature_importance(self):
        """Get feature importance from ensemble"""
        importance_dict = {}
        
        for name, model in self.base_models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_dict[name] = np.abs(model.coef_[0])
            else:
                importance_dict[name] = None
        
        return importance_dict

class CreativeModelApproach:
    def __init__(self, data_path="oulad_cleaned"):
        self.data_path = data_path
        self.datasets = {}
        self.scaler = StandardScaler()
        self.results = {}
        self.load_data()
        self.define_stem_modules()
        
    def define_stem_modules(self):
        """Define STEM modules"""
        self.stem_modules = {
            'AAA': 'Computing and IT',
            'FFF': 'Science', 
            'GGG': 'Engineering and Technology',
            'HHH': 'Mathematics and Statistics'
        }
        
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
    
    def create_creative_features(self, df):
        """
        CREATIVE FEATURE: Create innovative features based on domain knowledge
        """
        print("Creating creative features...")
        
        df_creative = df.copy()
        
        # 1. STEM Subject Difficulty Index
        subject_difficulty = {
            'AAA': 0.7,  # Computing - moderate difficulty
            'FFF': 0.8,  # Science - high difficulty
            'GGG': 0.9,  # Engineering - very high difficulty
            'HHH': 0.85  # Mathematics - very high difficulty
        }
        
        df_creative['subject_difficulty'] = df_creative['code_module'].map(subject_difficulty)
        
        # 2. Student Resilience Score
        df_creative['resilience_score'] = (
            (1 - df_creative['num_of_prev_attempts'] / df_creative['num_of_prev_attempts'].max()) * 0.6 +
            (df_creative['studied_credits'] / df_creative['studied_credits'].max()) * 0.4
        )
        
        # 3. Learning Efficiency Ratio
        df_creative['learning_efficiency'] = (
            df_creative['studied_credits'] / (df_creative['num_of_prev_attempts'] + 1)
        )
        
        # 4. Academic Momentum
        df_creative['academic_momentum'] = (
            df_creative['studied_credits'] * (1 - df_creative['num_of_prev_attempts'] / 10)
        )
        
        # 5. STEM Aptitude Score
        education_stem_aptitude = {
            'HE Qualification': 0.9,
            'A Level or Equivalent': 0.8,
            'Lower Than A Level': 0.6,
            'Post Graduate Qualification': 1.0
        }
        
        df_creative['stem_aptitude'] = (
            df_creative['highest_education'].map(education_stem_aptitude).fillna(0.7) * 0.7 +
            (1 - df_creative['subject_difficulty']) * 0.3
        )
        
        print("Creative features created:")
        print("  - STEM Subject Difficulty Index")
        print("  - Student Resilience Score")
        print("  - Learning Efficiency Ratio")
        print("  - Academic Momentum")
        print("  - STEM Aptitude Score")
        
        return df_creative
    
    def prepare_stem_dataset(self):
        """Prepare STEM dataset with creative features"""
        print("Preparing STEM dataset with creative features...")
        
        # Get student info
        student_info = self.datasets['studentInfo'].copy()
        
        # Add STEM labels
        student_info['subject_area'] = student_info['code_module'].map(self.stem_modules)
        student_info['is_stem'] = student_info['code_module'].isin(self.stem_modules.keys())
        
        # Filter for STEM students only
        stem_students = student_info[student_info['is_stem'] == True].copy()
        
        print(f"Total STEM students: {len(stem_students):,}")
        
        # Create target variables
        stem_students['stem_excellence'] = np.where(
            stem_students['final_result'] == -2.0289591117016377,  # Distinction
            1,  # STEM Excellence
            0   # Not Excellence
        )
        
        stem_students['stem_success'] = np.where(
            (stem_students['final_result'] == -2.0289591117016377) |  # Distinction
            (stem_students['final_result'] == 0.08374058178886125),   # Pass
            1,  # Success
            0   # Failure
        )
        
        # Add creative features
        stem_students = self.create_creative_features(stem_students)
        
        return stem_students
    
    def select_creative_features(self, df):
        """Select features including creative features"""
        print("Selecting creative features...")
        
        # Original features
        original_features = [
            'gender', 'region', 'highest_education', 'imd_band', 
            'age_band', 'num_of_prev_attempts', 'studied_credits', 'disability'
        ]
        
        # Creative features
        creative_features = [
            'subject_difficulty', 'resilience_score', 'learning_efficiency',
            'academic_momentum', 'stem_aptitude'
        ]
        
        # Combine all features
        all_features = original_features + creative_features
        
        # Ensure all features exist
        available_features = [col for col in all_features if col in df.columns]
        print(f"Selected features: {available_features}")
        print(f"Number of features: {len(available_features)}")
        print(f"  - Original features: {len([f for f in original_features if f in df.columns])}")
        print(f"  - Creative features: {len([f for f in creative_features if f in df.columns])}")
        
        return available_features
    
    def prepare_features_and_targets(self, df, target_column):
        """Prepare features and target variables"""
        print(f"Preparing features and target: {target_column}")
        
        # Select features
        feature_columns = self.select_creative_features(df)
        
        # Prepare feature matrix
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Ensure all features are numerical
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def evaluate_custom_ensemble(self, X, y, target_name):
        """Evaluate custom ensemble approach"""
        print(f"Evaluating custom ensemble for {target_name} prediction...")
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and fit custom ensemble
        custom_ensemble = CustomEnsemble()
        custom_ensemble.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = custom_ensemble.predict(X_test_scaled)
        y_pred_proba = custom_ensemble.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(custom_ensemble, X_train_scaled, y_train, cv=5, scoring='f1')
        
        # Store results
        results = {
            'Custom Ensemble': {
                'model': custom_ensemble,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'weights': custom_ensemble.weights
            }
        }
        
        print(f"Custom Ensemble - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
        
        # Store results
        self.results[target_name] = results
        
        return results
    
    def create_creative_visualizations(self, target_name):
        """Create creative visualizations"""
        print(f"Creating creative visualizations for {target_name}...")
        
        results = self.results[target_name]
        ensemble = results['Custom Ensemble']
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Creative Model Approach - STEM {target_name.title()}', fontsize=16, fontweight='bold')
        
        # 1. Model weights visualization
        weights = ensemble['weights']
        model_names = list(weights.keys())
        weight_values = list(weights.values())
        
        bars = axes[0, 0].bar(model_names, weight_values, alpha=0.8, color='skyblue')
        axes[0, 0].set_title('Dynamic Model Weights')
        axes[0, 0].set_ylabel('Weight')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, weight in zip(bars, weight_values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Performance metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        metric_values = [ensemble[metric] for metric in metrics]
        
        bars = axes[0, 1].bar(metric_names, metric_values, alpha=0.8, color='lightgreen')
        axes[0, 1].set_title('Performance Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. ROC Curve
        fpr, tpr, _ = roc_curve(ensemble['y_test'], ensemble['y_pred_proba'])
        auc = ensemble['auc']
        axes[1, 0].plot(fpr, tpr, color='red', linewidth=2, label=f'Custom Ensemble (AUC={auc:.3f})')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Confusion Matrix
        cm = confusion_matrix(ensemble['y_test'], ensemble['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title('Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'creative_approach_{target_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Creative approach visualizations saved as 'creative_approach_{target_name.lower()}.png'")
    
    def generate_creative_report(self):
        """Generate creative approach report"""
        print("\n" + "="*80)
        print("CREATIVE MODEL APPROACH REPORT")
        print("="*80)
        
        for target_name, results in self.results.items():
            print(f"\n{target_name.upper()} PREDICTION - CREATIVE APPROACH")
            print("-" * 60)
            
            ensemble = results['Custom Ensemble']
            
            print(f"Custom Ensemble Performance:")
            print(f"  F1 Score: {ensemble['f1_score']:.3f}")
            print(f"  AUC: {ensemble['auc']:.3f}")
            print(f"  Accuracy: {ensemble['accuracy']:.3f}")
            print(f"  Precision: {ensemble['precision']:.3f}")
            print(f"  Recall: {ensemble['recall']:.3f}")
            print(f"  Cross-Validation F1: {ensemble['cv_mean']:.3f} (+/- {ensemble['cv_std']:.3f})")
            
            print(f"\nCreative Features Implemented:")
            print("  1. STEM Subject Difficulty Index")
            print("  2. Student Resilience Score")
            print("  3. Learning Efficiency Ratio")
            print("  4. Academic Momentum")
            print("  5. STEM Aptitude Score")
            
            print(f"\nDynamic Model Weights:")
            for name, weight in ensemble['weights'].items():
                print(f"  {name}: {weight:.3f}")
    
    def main(self):
        """Main creative approach pipeline"""
        print("CREATIVE MODEL APPROACH")
        print("="*50)
        
        # Prepare dataset with creative features
        stem_students = self.prepare_stem_dataset()
        
        # Evaluate Excellence prediction with creative approach
        print("\n" + "="*50)
        print("EVALUATING STEM EXCELLENCE PREDICTION - CREATIVE APPROACH")
        print("="*50)
        X_excellence, y_excellence = self.prepare_features_and_targets(stem_students, 'stem_excellence')
        self.evaluate_custom_ensemble(X_excellence, y_excellence, 'Excellence')
        self.create_creative_visualizations('Excellence')
        
        # Evaluate Success prediction with creative approach
        print("\n" + "="*50)
        print("EVALUATING STEM SUCCESS PREDICTION - CREATIVE APPROACH")
        print("="*50)
        X_success, y_success = self.prepare_features_and_targets(stem_students, 'stem_success')
        self.evaluate_custom_ensemble(X_success, y_success, 'Success')
        self.create_creative_visualizations('Success')
        
        # Generate creative report
        self.generate_creative_report()
        
        print("\n" + "="*50)
        print("CREATIVE APPROACH COMPLETED")
        print("="*50)

if __name__ == "__main__":
    creative_approach = CreativeModelApproach()
    creative_approach.main() 