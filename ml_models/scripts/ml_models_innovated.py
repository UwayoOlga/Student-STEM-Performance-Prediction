#!/usr/bin/env python3
"""
Machine Learning Models for STEM Performance Prediction
Ensemble methods with custom feature engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Clean styling for plots
plt.style.use('default')
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

class STEMPerformancePredictor:
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
    
    def create_custom_features(self, df):
        """Create custom engineered features for enhanced prediction"""
        print("Creating custom engineered features...")
        
        df_enhanced = df.copy()
        
        # 1. Academic Risk Score
        df_enhanced['academic_risk_score'] = (
            df_enhanced['num_of_prev_attempts'] * 0.4 +
            (1 - df_enhanced['studied_credits'] / df_enhanced['studied_credits'].max()) * 0.3 +
            (df_enhanced['imd_band'] / df_enhanced['imd_band'].max()) * 0.3
        )
        
        # 2. STEM Readiness Index
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
        
        # 3. Socioeconomic Advantage Score
        df_enhanced['socioeconomic_advantage'] = (
            (df_enhanced['imd_band'] / df_enhanced['imd_band'].max()) * 0.5 +
            (1 - df_enhanced['academic_risk_score']) * 0.5
        )
        
        # 4. Learning Persistence Score
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
        """Prepare STEM-specific dataset with custom features"""
        print("Preparing STEM dataset for machine learning...")
        
        # Get student info
        student_info = self.datasets['studentInfo'].copy()
        
        # Add STEM labels
        student_info['subject_area'] = student_info['code_module'].map(self.stem_modules)
        student_info['is_stem'] = student_info['code_module'].isin(self.stem_modules.keys())
        
        # Filter for STEM students only
        stem_students = student_info[student_info['is_stem'] == True].copy()
        
        print(f"Total STEM students: {len(stem_students):,}")
        
        # Check the actual values in final_result
        print("Unique final_result values:")
        print(stem_students['final_result'].value_counts().head(10))
        
        # Create target variables based on actual scaled values
        # -2.0289591117016377 = Distinction
        # 0.08374058178886125 = Pass
        # 1.1964402752793602 = Fail
        # 2.309139968769859 = Withdrawn
        
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
        
        # Add custom features
        stem_students = self.create_custom_features(stem_students)
        
        # Print target distributions
        excellence_dist = stem_students['stem_excellence'].value_counts()
        success_dist = stem_students['stem_success'].value_counts()
        
        print("STEM Excellence Distribution:")
        print(f"  Excellence (1): {excellence_dist.get(1, 0):,} ({excellence_dist.get(1, 0)/len(stem_students)*100:.1f}%)")
        print(f"  Not Excellence (0): {excellence_dist.get(0, 0):,} ({excellence_dist.get(0, 0)/len(stem_students)*100:.1f}%)")
        
        return stem_students
    
    def select_features(self, df):
        """Select features including custom engineered features"""
        print("Selecting features for STEM performance prediction...")
        
        # Original features
        original_features = [
            'gender', 'region', 'highest_education', 'imd_band', 
            'age_band', 'num_of_prev_attempts', 'studied_credits', 'disability'
        ]
        
        # Custom features
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
    
    def create_simple_ensemble_models(self, X, y):
        """Create simple ensemble models"""
        print("Creating simple ensemble models...")
        
        # Base models with reasonable complexity
        base_models = {
            'rf': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'lr': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        # 1. Simple Voting Classifier
        voting_classifier = VotingClassifier(
            estimators=[(name, model) for name, model in base_models.items()],
            voting='soft'
        )
        
        # 2. Random Forest (as baseline)
        random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # 3. Gradient Boosting (as baseline)
        gradient_boosting = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        ensemble_models = {
            'Voting Classifier': voting_classifier,
            'Random Forest': random_forest,
            'Gradient Boosting': gradient_boosting
        }
        
        return ensemble_models
    
    def handle_class_imbalance(self, X, y):
        """Handle class imbalance using SMOTE"""
        print("Handling class imbalance using SMOTE...")
        
        # Use SMOTE for balancing
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        print(f"Original distribution: {pd.Series(y).value_counts().to_dict()}")
        print(f"Balanced distribution: {pd.Series(y_balanced).value_counts().to_dict()}")
        
        return X_balanced, y_balanced
    
    def train_models(self, X, y, target_name):
        """Train ensemble models"""
        print(f"Training models for {target_name} prediction...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Balanced training set: {X_train_balanced.shape}")
        print(f"Balanced target distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
        
        # Create ensemble models
        ensemble_models = self.create_simple_ensemble_models(X_train_scaled, y_train_balanced)
        
        # Train and evaluate ensemble models
        results = {}
        
        for name, model in ensemble_models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train_balanced)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation score (3-fold for speed)
            cv_scores = cross_val_score(model, X_train_scaled, y_train_balanced, cv=3, scoring='f1')
            
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
    
    def create_visualizations(self, target_name):
        """Create clean, readable visualizations"""
        print(f"Creating visualizations for {target_name}...")
        
        results = self.results[target_name]
        
        # Create clean 2x2 layout
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'STEM {target_name} Prediction Results', fontsize=14, fontweight='bold')
        
        # Set colors
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        model_names = list(results.keys())
        
        # 1. Performance Metrics (Top Left)
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            values = [results[name][metric] for name in model_names]
            axes[0, 0].bar(x + i*width, values, width, label=metric_name, alpha=0.8, color=colors[i % len(colors)])
        
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Metrics')
        axes[0, 0].set_xticks(x + 1.5*width)
        axes[0, 0].set_xticklabels(model_names, rotation=0)
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC Curves (Top Right)
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
        
        # 3. Cross-Validation Scores (Bottom Left)
        cv_means = [results[name]['cv_mean'] for name in model_names]
        cv_stds = [results[name]['cv_std'] for name in model_names]
        bars = axes[1, 0].bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.8, color=colors[:len(model_names)])
        axes[1, 0].set_title('Cross-Validation F1 Scores')
        axes[1, 0].set_ylabel('CV F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=0)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, cv_mean in zip(bars, cv_means):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{cv_mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Confusion Matrix for best model (Bottom Right)
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        cm = confusion_matrix(results[best_model_name]['y_test'], results[best_model_name]['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'stem_{target_name.lower()}_results.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"Visualizations saved as 'stem_{target_name.lower()}_results.png'")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*70)
        print("STEM PERFORMANCE PREDICTION - FINAL RESULTS")
        print("="*70)
        
        for target_name, results in self.results.items():
            print(f"\n{target_name.upper()} PREDICTION RESULTS")
            print("-" * 50)
            
            # Find best model
            best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
            best_metrics = results[best_model_name]
            
            print(f"Best Model: {best_model_name}")
            print(f"F1 Score: {best_metrics['f1_score']:.3f}")
            print(f"AUC: {best_metrics['auc']:.3f}")
            print(f"Accuracy: {best_metrics['accuracy']:.3f}")
            print(f"Precision: {best_metrics['precision']:.3f}")
            print(f"Recall: {best_metrics['recall']:.3f}")
            print(f"CV F1 Score: {best_metrics['cv_mean']:.3f} (±{best_metrics['cv_std']*2:.3f})")
            
            print(f"\nModel Comparison:")
            for name, result in results.items():
                print(f"  {name}: F1={result['f1_score']:.3f}, AUC={result['auc']:.3f}, Acc={result['accuracy']:.3f}")
        
        print(f"\nGenerated Files:")
        for target_name in self.results.keys():
            print(f"  stem_{target_name.lower()}_results.png - Model evaluation")
        
        print("\n" + "="*70)
        print("STEM Performance Prediction Completed Successfully!")
        print("="*70)

def main():
    """Main function to run STEM performance prediction"""
    print("STEM PERFORMANCE PREDICTION USING MACHINE LEARNING")
    print("=" * 50)
    
    try:
        # Initialize predictor
        predictor = STEMPerformancePredictor()
        
        # Prepare STEM dataset
        stem_data = predictor.prepare_stem_dataset()
        
        # Train models for STEM Excellence prediction
        print("\n" + "="*40)
        print("PREDICTING STEM EXCELLENCE")
        print("="*40)
        
        X_excellence, y_excellence = predictor.prepare_features_and_targets(stem_data, 'stem_excellence')
        excellence_results = predictor.train_models(X_excellence, y_excellence, 'Excellence')
        
        # Train models for STEM Success prediction
        print("\n" + "="*40)
        print("PREDICTING STEM SUCCESS")
        print("="*40)
        
        X_success, y_success = predictor.prepare_features_and_targets(stem_data, 'stem_success')
        success_results = predictor.train_models(X_success, y_success, 'Success')
        
        # Create visualizations
        print("\n" + "="*40)
        print("CREATING VISUALIZATIONS")
        print("="*40)
        
        predictor.create_visualizations('Excellence')
        predictor.create_visualizations('Success')
        
        # Generate summary report
        predictor.generate_summary_report()
        
    except Exception as e:
        print(f"Error during STEM prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 