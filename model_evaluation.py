#!/usr/bin/env python3
"""
Comprehensive Model Evaluation for STEM Performance Prediction
Evaluates models using multiple metrics and provides detailed analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score, log_loss,
    brier_score_loss, cohen_kappa_score, matthews_corrcoef
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelEvaluator:
    def __init__(self, data_path="oulad_cleaned"):
        self.data_path = data_path
        self.datasets = {}
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
        """Prepare STEM-specific dataset for evaluation"""
        print("Preparing STEM dataset for evaluation...")
        
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
        
        print("\nSTEM Success Distribution:")
        print(f"  Success (1): {success_dist.get(1, 0):,} ({success_dist.get(1, 0)/len(stem_students)*100:.1f}%)")
        print(f"  Failure (0): {success_dist.get(0, 0):,} ({success_dist.get(0, 0)/len(stem_students)*100:.1f}%)")
        
        return stem_students
    
    def select_features(self, df):
        """Select relevant features for STEM performance prediction"""
        print("Selecting features for STEM performance prediction...")
        
        # Select features based on domain knowledge and EDA
        feature_columns = [
            'gender', 'region', 'highest_education', 'imd_band', 
            'age_band', 'num_of_prev_attempts', 'studied_credits', 'disability'
        ]
        
        # Ensure all features exist
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"Selected features: {available_features}")
        print(f"Number of features: {len(available_features)}")
        
        return available_features
    
    def prepare_features_and_targets(self, df, target_column):
        """Prepare features and target variables"""
        print(f"Preparing features and target: {target_column}")
        
        # Select features
        feature_columns = self.select_features(df)
        
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
    
    def handle_class_imbalance(self, X, y, method='smote'):
        """Handle class imbalance using SMOTE"""
        print(f"Handling class imbalance using {method.upper()}...")
        
        if method == 'smote':
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
        else:
            X_balanced, y_balanced = X, y
        
        print(f"Original distribution: {y.value_counts().to_dict()}")
        print(f"Balanced distribution: {pd.Series(y_balanced).value_counts().to_dict()}")
        
        return X_balanced, y_balanced
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Advanced metrics
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['true_negatives'] = cm[0, 0]
        metrics['false_positives'] = cm[0, 1]
        metrics['false_negatives'] = cm[1, 0]
        metrics['true_positives'] = cm[1, 1]
        
        # Calculate rates
        metrics['specificity'] = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        metrics['sensitivity'] = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        
        return metrics
    
    def evaluate_models(self, X, y, target_name):
        """Evaluate multiple models with comprehensive metrics"""
        print(f"Evaluating models for {target_name} prediction...")
        
        # Handle class imbalance
        X_balanced, y_balanced = self.handle_class_imbalance(X, y, method='smote')
        
        # Split the balanced data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
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
        
        # Evaluate each model
        results = {}
        
        for name, model in models.items():
            print(f"Evaluating {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate comprehensive metrics
            metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
            metrics['cv_f1_mean'] = cv_scores.mean()
            metrics['cv_f1_std'] = cv_scores.std()
            
            # Store results
            results[name] = {
                'model': model,
                'metrics': metrics,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  {name} - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}, AUC: {metrics['auc_roc']:.3f}")
        
        # Store results
        self.results[target_name] = results
        
        return results
    
    def create_comprehensive_visualizations(self, target_name):
        """Create comprehensive evaluation visualizations"""
        print(f"Creating comprehensive visualizations for {target_name}...")
        
        results = self.results[target_name]
        
        # Create comprehensive evaluation dashboard
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle(f'Comprehensive Model Evaluation - STEM {target_name.title()}', fontsize=16, fontweight='bold')
        
        model_names = list(results.keys())
        
        # 1. Accuracy comparison
        accuracies = [results[name]['metrics']['accuracy'] for name in model_names]
        axes[0, 0].bar(model_names, accuracies, color='skyblue')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim(0, 1)
        
        # 2. F1 Score comparison
        f1_scores = [results[name]['metrics']['f1_score'] for name in model_names]
        axes[0, 1].bar(model_names, f1_scores, color='lightgreen')
        axes[0, 1].set_title('Model F1 Score Comparison')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylim(0, 1)
        
        # 3. AUC comparison
        auc_scores = [results[name]['metrics']['auc_roc'] for name in model_names]
        axes[0, 2].bar(model_names, auc_scores, color='orange')
        axes[0, 2].set_title('Model AUC Comparison')
        axes[0, 2].set_ylabel('AUC')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].set_ylim(0, 1)
        
        # 4. Precision vs Recall
        precisions = [results[name]['metrics']['precision'] for name in model_names]
        recalls = [results[name]['metrics']['recall'] for name in model_names]
        axes[1, 0].scatter(precisions, recalls, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            axes[1, 0].annotate(name, (precisions[i], recalls[i]), xytext=(5, 5), textcoords='offset points')
        axes[1, 0].set_xlabel('Precision')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Precision vs Recall')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. ROC Curves
        for name in model_names:
            fpr, tpr, _ = roc_curve(results[name]['y_test'], results[name]['y_pred_proba'])
            auc = results[name]['metrics']['auc_roc']
            axes[1, 1].plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].set_title('ROC Curves')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Precision-Recall Curves
        for name in model_names:
            precision, recall, _ = precision_recall_curve(results[name]['y_test'], results[name]['y_pred_proba'])
            ap = results[name]['metrics']['average_precision']
            axes[1, 2].plot(recall, precision, label=f'{name} (AP = {ap:.3f})')
        axes[1, 2].set_xlabel('Recall')
        axes[1, 2].set_ylabel('Precision')
        axes[1, 2].set_title('Precision-Recall Curves')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Confusion Matrix for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['f1_score'])
        cm = confusion_matrix(results[best_model_name]['y_test'], results[best_model_name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2, 0])
        axes[2, 0].set_title(f'Confusion Matrix - {best_model_name}')
        axes[2, 0].set_xlabel('Predicted')
        axes[2, 0].set_ylabel('Actual')
        
        # 8. Cross-validation scores
        cv_means = [results[name]['metrics']['cv_f1_mean'] for name in model_names]
        cv_stds = [results[name]['metrics']['cv_f1_std'] for name in model_names]
        axes[2, 1].bar(model_names, cv_means, yerr=cv_stds, capsize=5, color='lightcoral')
        axes[2, 1].set_title('Cross-Validation F1 Scores')
        axes[2, 1].set_ylabel('CV F1 Score')
        axes[2, 1].tick_params(axis='x', rotation=45)
        axes[2, 1].set_ylim(0, 1)
        
        # 9. Advanced metrics comparison
        advanced_metrics = ['cohen_kappa', 'matthews_corrcoef', 'brier_score']
        metric_names = ['Cohen Kappa', 'Matthews CorrCoef', 'Brier Score']
        
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, (metric, metric_name) in enumerate(zip(advanced_metrics, metric_names)):
            values = [results[name]['metrics'][metric] for name in model_names]
            axes[2, 2].bar(x + i*width, values, width, label=metric_name, alpha=0.8)
        
        axes[2, 2].set_xlabel('Models')
        axes[2, 2].set_ylabel('Score')
        axes[2, 2].set_title('Advanced Metrics Comparison')
        axes[2, 2].set_xticks(x + width)
        axes[2, 2].set_xticklabels(model_names, rotation=45)
        axes[2, 2].legend()
        
        plt.tight_layout()
        plt.savefig(f'comprehensive_evaluation_{target_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comprehensive evaluation visualizations saved as 'comprehensive_evaluation_{target_name.lower()}.png'")
    
    def generate_detailed_report(self):
        """Generate detailed evaluation report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION REPORT")
        print("="*80)
        
        for target_name, results in self.results.items():
            print(f"\n{target_name.upper()} PREDICTION EVALUATION")
            print("-" * 60)
            
            # Find best model
            best_model = max(results.keys(), key=lambda x: results[x]['metrics']['f1_score'])
            best_metrics = results[best_model]['metrics']
            
            print(f"Best Model: {best_model}")
            print(f"F1 Score: {best_metrics['f1_score']:.3f}")
            print(f"AUC: {best_metrics['auc_roc']:.3f}")
            print(f"Accuracy: {best_metrics['accuracy']:.3f}")
            
            print(f"\nDetailed Metrics for {best_model}:")
            print(f"  Precision: {best_metrics['precision']:.3f}")
            print(f"  Recall: {best_metrics['recall']:.3f}")
            print(f"  Specificity: {best_metrics['specificity']:.3f}")
            print(f"  Sensitivity: {best_metrics['sensitivity']:.3f}")
            print(f"  Cohen Kappa: {best_metrics['cohen_kappa']:.3f}")
            print(f"  Matthews Correlation: {best_metrics['matthews_corrcoef']:.3f}")
            print(f"  Brier Score: {best_metrics['brier_score']:.3f}")
            print(f"  Log Loss: {best_metrics['log_loss']:.3f}")
            print(f"  Average Precision: {best_metrics['average_precision']:.3f}")
            print(f"  Cross-Validation F1: {best_metrics['cv_f1_mean']:.3f} (+/- {best_metrics['cv_f1_std']:.3f})")
            
            print(f"\nConfusion Matrix for {best_model}:")
            print(f"  True Negatives: {best_metrics['true_negatives']}")
            print(f"  False Positives: {best_metrics['false_positives']}")
            print(f"  False Negatives: {best_metrics['false_negatives']}")
            print(f"  True Positives: {best_metrics['true_positives']}")
            
            print(f"\nModel Comparison:")
            for name, result in results.items():
                metrics = result['metrics']
                print(f"  {name}: F1={metrics['f1_score']:.3f}, AUC={metrics['auc_roc']:.3f}, Acc={metrics['accuracy']:.3f}")
    
    def main(self):
        """Main evaluation pipeline"""
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*50)
        
        # Prepare dataset
        stem_students = self.prepare_stem_dataset()
        
        # Evaluate Excellence prediction
        print("\n" + "="*50)
        print("EVALUATING STEM EXCELLENCE PREDICTION")
        print("="*50)
        X_excellence, y_excellence = self.prepare_features_and_targets(stem_students, 'stem_excellence')
        self.evaluate_models(X_excellence, y_excellence, 'Excellence')
        self.create_comprehensive_visualizations('Excellence')
        
        # Evaluate Success prediction
        print("\n" + "="*50)
        print("EVALUATING STEM SUCCESS PREDICTION")
        print("="*50)
        X_success, y_success = self.prepare_features_and_targets(stem_students, 'stem_success')
        self.evaluate_models(X_success, y_success, 'Success')
        self.create_comprehensive_visualizations('Success')
        
        # Generate detailed report
        self.generate_detailed_report()
        
        print("\n" + "="*50)
        print("COMPREHENSIVE EVALUATION COMPLETED")
        print("="*50)

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.main() 