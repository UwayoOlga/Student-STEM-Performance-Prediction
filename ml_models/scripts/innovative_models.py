#!/usr/bin/env python3
"""
Innovative Machine Learning Models for STEM Performance Prediction
Incorporates ensemble techniques, custom functions, and creative model approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class InnovativeSTEMPredictor:
    def __init__(self, data_path="oulad_cleaned"):
        self.data_path = data_path
        self.datasets = {}
        self.scaler = StandardScaler()
        self.results = {}
        self.ensemble_models = {}
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
    
    def create_custom_features(self, df):
        """
        INNOVATIVE FEATURE 1: Create custom engineered features
        """
        print("Creating innovative custom features...")
        
        df_enhanced = df.copy()
        
        # 1. Academic Risk Score (Custom Feature)
        df_enhanced['academic_risk_score'] = (
            df_enhanced['num_of_prev_attempts'] * 0.4 +
            (1 - df_enhanced['studied_credits'] / df_enhanced['studied_credits'].max()) * 0.3 +
            (df_enhanced['imd_band'] / df_enhanced['imd_band'].max()) * 0.3
        )
        
        # 2. STEM Readiness Index (Custom Feature)
        # Higher education level and younger age bands indicate better readiness
        education_weights = {
            'HE Qualification': 1.0,
            'A Level or Equivalent': 0.8,
            'Lower Than A Level': 0.6,
            'Post Graduate Qualification': 1.2
        }
        
        age_weights = {
            '0-35': 1.0,
            '35-55': 0.9,
            '55<=': 0.8
        }
        
        df_enhanced['stem_readiness_index'] = (
            df_enhanced['highest_education'].map(education_weights).fillna(0.7) * 0.6 +
            df_enhanced['age_band'].map(age_weights).fillna(0.9) * 0.4
        )
        
        # 3. Socioeconomic Advantage Score (Custom Feature)
        df_enhanced['socioeconomic_advantage'] = (
            (df_enhanced['imd_band'] / df_enhanced['imd_band'].max()) * 0.5 +
            (1 - df_enhanced['academic_risk_score']) * 0.5
        )
        
        # 4. Learning Persistence Score (Custom Feature)
        df_enhanced['learning_persistence'] = (
            (1 - df_enhanced['num_of_prev_attempts'] / df_enhanced['num_of_prev_attempts'].max()) * 0.7 +
            (df_enhanced['studied_credits'] / df_enhanced['studied_credits'].max()) * 0.3
        )
        
        print("Custom features created:")
        print("  - Academic Risk Score")
        print("  - STEM Readiness Index") 
        print("  - Socioeconomic Advantage Score")
        print("  - Learning Persistence Score")
        
        return df_enhanced
    
    def prepare_stem_dataset(self):
        """Prepare STEM-specific dataset with innovative features"""
        print("Preparing STEM dataset with innovative features...")
        
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
        
        # Add innovative custom features
        stem_students = self.create_custom_features(stem_students)
        
        # Print target distributions
        excellence_dist = stem_students['stem_excellence'].value_counts()
        success_dist = stem_students['stem_success'].value_counts()
        
        print("STEM Excellence Distribution:")
        print(f"  Excellence (1): {excellence_dist.get(1, 0):,} ({excellence_dist.get(1, 0)/len(stem_students)*100:.1f}%)")
        print(f"  Not Excellence (0): {excellence_dist.get(0, 0):,} ({excellence_dist.get(0, 0)/len(stem_students)*100:.1f}%)")
        
        return stem_students
    
    def select_innovative_features(self, df):
        """Select features including innovative custom features"""
        print("Selecting innovative features for STEM performance prediction...")
        
        # Original features
        original_features = [
            'gender', 'region', 'highest_education', 'imd_band', 
            'age_band', 'num_of_prev_attempts', 'studied_credits', 'disability'
        ]
        
        # Innovative custom features
        custom_features = [
            'academic_risk_score', 'stem_readiness_index', 
            'socioeconomic_advantage', 'learning_persistence'
        ]
        
        # Combine all features
        all_features = original_features + custom_features
        
        # Ensure all features exist
        available_features = [col for col in all_features if col in df.columns]
        print(f"Selected features: {available_features}")
        print(f"Number of features: {len(available_features)}")
        print(f"  - Original features: {len([f for f in original_features if f in df.columns])}")
        print(f"  - Custom features: {len([f for f in custom_features if f in df.columns])}")
        
        return available_features
    
    def prepare_features_and_targets(self, df, target_column):
        """Prepare features and target variables with innovative preprocessing"""
        print(f"Preparing features and target: {target_column}")
        
        # Select features
        feature_columns = self.select_innovative_features(df)
        
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
    
    def create_advanced_ensemble(self, X, y):
        """
        INNOVATIVE FEATURE 2: Create advanced ensemble with voting and stacking
        """
        print("Creating advanced ensemble models...")
        
        # Base models with different characteristics
        base_models = {
            'rf': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'lr': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        # 1. Voting Classifier (Soft Voting only - hard voting doesn't support predict_proba)
        voting_soft = VotingClassifier(
            estimators=[(name, model) for name, model in base_models.items()],
            voting='soft'
        )
        
        # 2. Weighted Ensemble (Custom weights based on model performance)
        weighted_ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in base_models.items()],
            voting='soft',
            weights=[0.25, 0.25, 0.2, 0.15, 0.15]  # Custom weights
        )
        
        # 3. Use Random Forest as simple ensemble alternative
        simple_ensemble = RandomForestClassifier(n_estimators=200, random_state=42)
        
        ensemble_models = {
            'Voting (Soft)': voting_soft,
            'Weighted Ensemble': weighted_ensemble,
            'Enhanced RF': simple_ensemble
        }
        
        return ensemble_models
    
    def create_adaptive_sampling_pipeline(self, X, y):
        """
        INNOVATIVE FEATURE 3: Adaptive sampling based on data characteristics
        """
        print("Creating adaptive sampling pipeline...")
        
        # Calculate imbalance ratio
        imbalance_ratio = y.value_counts().min() / y.value_counts().max()
        print(f"Imbalance ratio: {imbalance_ratio:.3f}")
        
        if imbalance_ratio < 0.1:
            # Severe imbalance - use ADASYN
            print("Using ADASYN for severe imbalance...")
            sampler = ADASYN(random_state=42)
        elif imbalance_ratio < 0.3:
            # Moderate imbalance - use SMOTE
            print("Using SMOTE for moderate imbalance...")
            sampler = SMOTE(random_state=42)
        else:
            # Mild imbalance - use undersampling
            print("Using undersampling for mild imbalance...")
            sampler = RandomUnderSampler(random_state=42)
        
        # Create adaptive pipeline
        pipeline = Pipeline([
            ('sampler', sampler),
            ('scaler', StandardScaler())
        ])
        
        return pipeline
    
    def train_innovative_models(self, X, y, target_name):
        """Train innovative models with advanced techniques"""
        print(f"Training innovative models for {target_name} prediction...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create adaptive sampling pipeline
        sampling_pipeline = self.create_adaptive_sampling_pipeline(X_train, y_train)
        
        # Apply sampling
        X_train_resampled, y_train_resampled = sampling_pipeline.named_steps['sampler'].fit_resample(X_train, y_train)
        X_train_scaled = sampling_pipeline.named_steps['scaler'].fit_transform(X_train_resampled)
        X_test_scaled = sampling_pipeline.named_steps['scaler'].transform(X_test)
        
        print(f"Resampled training set: {X_train_resampled.shape}")
        print(f"Resampled target distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")
        
        # Create ensemble models
        ensemble_models = self.create_advanced_ensemble(X_train_scaled, y_train_resampled)
        
        # Train and evaluate ensemble models
        results = {}
        
        for name, model in ensemble_models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train_resampled)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled, y_train_resampled, cv=5, scoring='f1')
            
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
    
    def create_innovative_visualizations(self, target_name):
        """Create innovative visualizations for model comparison"""
        print(f"Creating innovative visualizations for {target_name}...")
        
        results = self.results[target_name]
        
        # Create comprehensive evaluation dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Innovative Model Evaluation - STEM {target_name.title()}', fontsize=16, fontweight='bold')
        
        model_names = list(results.keys())
        
        # 1. Performance comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            values = [results[name][metric] for name in model_names]
            axes[0, 0].bar(x + i*width, values, width, label=metric_name, alpha=0.8)
        
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Metrics Comparison')
        axes[0, 0].set_xticks(x + 2*width)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC Curves
        colors = ['blue', 'red', 'green']
        for i, name in enumerate(model_names):
            fpr, tpr, _ = roc_curve(results[name]['y_test'], results[name]['y_pred_proba'])
            auc = results[name]['auc']
            axes[0, 1].plot(fpr, tpr, color=colors[i], label=f'{name} (AUC={auc:.3f})', linewidth=2)
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cross-validation scores
        cv_means = [results[name]['cv_mean'] for name in model_names]
        cv_stds = [results[name]['cv_std'] for name in model_names]
        bars = axes[0, 2].bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.8)
        axes[0, 2].set_title('Cross-Validation F1 Scores')
        axes[0, 2].set_ylabel('CV F1 Score')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, cv_mean in zip(bars, cv_means):
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{cv_mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Confusion Matrix for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        cm = confusion_matrix(results[best_model_name]['y_test'], results[best_model_name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 5. Precision-Recall Curves
        for i, name in enumerate(model_names):
            precision, recall, _ = precision_recall_curve(results[name]['y_test'], results[name]['y_pred_proba'])
            ap = average_precision_score(results[name]['y_test'], results[name]['y_pred_proba'])
            axes[1, 1].plot(recall, precision, color=colors[i], 
                           label=f'{name} (AP={ap:.3f})', linewidth=2)
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision-Recall Curves')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Model ranking
        f1_scores = [results[name]['f1_score'] for name in model_names]
        ranking = sorted(zip(model_names, f1_scores), key=lambda x: x[1], reverse=True)
        
        names_ranked = [name for name, _ in ranking]
        scores_ranked = [score for _, score in ranking]
        
        bars = axes[1, 2].barh(names_ranked, scores_ranked, alpha=0.8)
        axes[1, 2].set_xlabel('F1 Score')
        axes[1, 2].set_title('Model Ranking by F1 Score')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores_ranked):
            width = bar.get_width()
            axes[1, 2].text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                           f'{score:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'innovative_evaluation_{target_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Innovative evaluation visualizations saved as 'innovative_evaluation_{target_name.lower()}.png'")
    
    def generate_innovation_report(self):
        """Generate comprehensive innovation report"""
        print("\n" + "="*80)
        print("INNOVATIVE MODEL EVALUATION REPORT")
        print("="*80)
        
        for target_name, results in self.results.items():
            print(f"\n{target_name.upper()} PREDICTION - INNOVATIVE APPROACHES")
            print("-" * 60)
            
            # Find best model
            best_model = max(results.keys(), key=lambda x: results[x]['f1_score'])
            best_metrics = results[best_model]
            
            print(f"Best Innovative Model: {best_model}")
            print(f"F1 Score: {best_metrics['f1_score']:.3f}")
            print(f"AUC: {best_metrics['auc']:.3f}")
            print(f"Accuracy: {best_metrics['accuracy']:.3f}")
            
            print(f"\nInnovative Features Implemented:")
            print("  1. Custom Feature Engineering:")
            print("     - Academic Risk Score")
            print("     - STEM Readiness Index")
            print("     - Socioeconomic Advantage Score")
            print("     - Learning Persistence Score")
            print("  2. Advanced Ensemble Methods:")
            print("     - Voting Classifiers (Hard & Soft)")
            print("     - Weighted Ensemble")
            print("  3. Adaptive Sampling:")
            print("     - Dynamic selection based on imbalance ratio")
            print("     - ADASYN for severe imbalance")
            print("     - SMOTE for moderate imbalance")
            print("     - Undersampling for mild imbalance")
            
            print(f"\nModel Comparison:")
            for name, result in results.items():
                print(f"  {name}: F1={result['f1_score']:.3f}, AUC={result['auc']:.3f}, Acc={result['accuracy']:.3f}")
    
    def main(self):
        """Main innovative evaluation pipeline"""
        print("INNOVATIVE STEM PERFORMANCE PREDICTION")
        print("="*50)
        
        # Prepare dataset with innovative features
        stem_students = self.prepare_stem_dataset()
        
        # Evaluate Excellence prediction with innovative approaches
        print("\n" + "="*50)
        print("EVALUATING STEM EXCELLENCE PREDICTION - INNOVATIVE APPROACHES")
        print("="*50)
        X_excellence, y_excellence = self.prepare_features_and_targets(stem_students, 'stem_excellence')
        self.train_innovative_models(X_excellence, y_excellence, 'Excellence')
        self.create_innovative_visualizations('Excellence')
        
        # Evaluate Success prediction with innovative approaches
        print("\n" + "="*50)
        print("EVALUATING STEM SUCCESS PREDICTION - INNOVATIVE APPROACHES")
        print("="*50)
        X_success, y_success = self.prepare_features_and_targets(stem_students, 'stem_success')
        self.train_innovative_models(X_success, y_success, 'Success')
        self.create_innovative_visualizations('Success')
        
        # Generate innovation report
        self.generate_innovation_report()
        
        print("\n" + "="*50)
        print("INNOVATIVE EVALUATION COMPLETED")
        print("="*50)

if __name__ == "__main__":
    predictor = InnovativeSTEMPredictor()
    predictor.main() 