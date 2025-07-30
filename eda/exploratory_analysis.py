#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for OULAD Dataset
Includes comprehensive data cleaning and preprocessing steps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class OULADEDA:
    def __init__(self, data_path="oulad_sampled"):
        self.data_path = data_path
        self.datasets = {}
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}
        self.load_data()
        
    def load_data(self):
        """Load all OULAD datasets"""
        print("Loading OULAD datasets...")
        
        try:
            self.datasets['studentInfo'] = pd.read_csv(f"{self.data_path}/studentInfo.csv")
            self.datasets['studentAssessment'] = pd.read_csv(f"{self.data_path}/studentAssessment.csv")
            self.datasets['assessments'] = pd.read_csv(f"{self.data_path}/assessments.csv")
            self.datasets['courses'] = pd.read_csv(f"{self.data_path}/courses.csv")
            self.datasets['studentRegistration'] = pd.read_csv(f"{self.data_path}/studentRegistration.csv")
            
            # Try to load merged dataset if available
            try:
                self.datasets['merged'] = pd.read_csv(f"{self.data_path}/oulad_merged.csv")
                print("Merged dataset loaded")
            except:
                print("Merged dataset not found")
            
            print("All datasets loaded successfully!")
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def analyze_missing_values(self, df, dataset_name):
        """Analyze missing values in a dataset"""
        print(f"Missing Values Analysis - {dataset_name}")
        print("-" * 50)
        
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        
        missing_summary = pd.DataFrame({
            'Column': missing_values.index,
            'Missing_Count': missing_values.values,
            'Missing_Percentage': missing_percentage.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
        
        if len(missing_summary) > 0:
            print("Columns with missing values:")
            for _, row in missing_summary.iterrows():
                print(f"   {row['Column']}: {row['Missing_Count']:,} ({row['Missing_Percentage']:.1f}%)")
        else:
            print("No missing values found!")
        
        return missing_summary
    
    def handle_missing_values(self, df, dataset_name):
        """Handle missing values in the dataset"""
        print(f"Handling Missing Values - {dataset_name}")
        print("-" * 50)
        
        df_cleaned = df.copy()
        missing_summary = self.analyze_missing_values(df_cleaned, dataset_name)
        
        if len(missing_summary) == 0:
            print("No missing values to handle")
            return df_cleaned
        
        for column in df_cleaned.columns:
            if df_cleaned[column].isnull().sum() > 0:
                if df_cleaned[column].dtype in ['object', 'category']:
                    # For categorical variables, use mode
                    mode_value = df_cleaned[column].mode()[0]
                    df_cleaned[column].fillna(mode_value, inplace=True)
                    print(f"   {column}: Filled with mode '{mode_value}'")
                else:
                    # For numerical variables, use median
                    median_value = df_cleaned[column].median()
                    df_cleaned[column].fillna(median_value, inplace=True)
                    print(f"   {column}: Filled with median {median_value:.2f}")
        
        return df_cleaned
    
    def detect_and_handle_outliers(self, df, dataset_name, columns=None):
        """Detect and handle outliers in numerical columns"""
        print(f"Outlier Detection - {dataset_name}")
        print("-" * 50)
        
        df_cleaned = df.copy()
        outlier_summary = {}
        
        if columns is None:
            numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        else:
            numerical_cols = [col for col in columns if col in df_cleaned.columns]
        
        for column in numerical_cols:
            Q1 = df_cleaned[column].quantile(0.25)
            Q3 = df_cleaned[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_cleaned[(df_cleaned[column] < lower_bound) | (df_cleaned[column] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                outlier_percentage = (outlier_count / len(df_cleaned)) * 100
                print(f"   {column}: {outlier_count} outliers ({outlier_percentage:.1f}%)")
                print(f"      Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
                
                # Handle outliers based on the column
                if column in ['date_submitted', 'is_banked']:
                    # Cap outliers to bounds
                    df_cleaned[column] = df_cleaned[column].clip(lower=lower_bound, upper=upper_bound)
                    print(f"      Capped outliers to bounds")
                else:
                    # Replace outliers with median
                    median_value = df_cleaned[column].median()
                    df_cleaned.loc[df_cleaned[column] < lower_bound, column] = median_value
                    df_cleaned.loc[df_cleaned[column] > upper_bound, column] = median_value
                    print(f"      Replaced outliers with median {median_value:.2f}")
                
                outlier_summary[column] = {
                    'count': outlier_count,
                    'percentage': outlier_percentage,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        return df_cleaned, outlier_summary
    
    def encode_categorical_features(self, df, dataset_name):
        """Encode categorical features using LabelEncoder"""
        print(f"Encoding Categorical Features - {dataset_name}")
        print("-" * 50)
        
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        for column in categorical_cols:
            if column in ['code_module', 'code_presentation']:
                continue  # Skip identifier columns
                
            encoder = LabelEncoder()
            df_encoded[column] = encoder.fit_transform(df_encoded[column].astype(str))
            
            self.encoders[f"{dataset_name}_{column}"] = encoder
            print(f"   {column}: Encoded {len(encoder.classes_)} categories")
        
        return df_encoded
    
    def scale_numerical_features(self, df, dataset_name, method='standard'):
        """Scale numerical features"""
        print(f"Scaling Numerical Features - {dataset_name}")
        print("-" * 50)
        
        df_scaled = df.copy()
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if 'id_' not in col.lower()]
        
        if len(numerical_cols) > 0:
            if method == 'standard':
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            
            df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
            self.scalers[f"{dataset_name}_{method}"] = scaler
            
            print(f"   Scaled {len(numerical_cols)} numerical columns using {method} scaling")
            print(f"   Columns: {', '.join(numerical_cols)}")
        
        return df_scaled
    
    def create_target_variable(self, df):
        """Create target variable for success prediction"""
        print("Creating Target Variable")
        print("-" * 50)
        
        df_with_target = df.copy()
        
        # Create binary target: Success (Distinction/Pass) vs Failure (Fail/Withdrawn)
        df_with_target['success'] = df_with_target['final_result'].map({
            'Distinction': 1,
            'Pass': 1,
            'Fail': 0,
            'Withdrawn': 0
        })
        
        # Create multi-class target
        df_with_target['final_result_encoded'] = df_with_target['final_result'].map({
            'Distinction': 3,
            'Pass': 2,
            'Fail': 1,
            'Withdrawn': 0
        })
        
        # Print target distribution
        target_dist = df_with_target['success'].value_counts()
        print("Target Variable Distribution:")
        success_count = target_dist.get(1, 0)
        failure_count = target_dist.get(0, 0)
        print(f"   Success (1): {success_count:,} ({success_count/len(df_with_target)*100:.1f}%)")
        print(f"   Failure (0): {failure_count:,} ({failure_count/len(df_with_target)*100:.1f}%)")
        
        return df_with_target
    
    def clean_all_datasets(self):
        """Clean all datasets with comprehensive preprocessing"""
        print("Starting Comprehensive Data Cleaning Process")
        print("=" * 60)
        
        cleaned_datasets = {}
        
        for name, df in self.datasets.items():
            print(f"Cleaning {name}")
            print("=" * 20)
            
            # Step 1: Handle missing values
            df_cleaned = self.handle_missing_values(df, name)
            
            # Step 2: Detect and handle outliers (for numerical datasets)
            if name in ['studentAssessment', 'merged']:
                df_cleaned, outliers = self.detect_and_handle_outliers(df_cleaned, name)
            
            # Step 3: Encode categorical features
            df_encoded = self.encode_categorical_features(df_cleaned, name)
            
            # Step 4: Scale numerical features (for main datasets)
            if name in ['studentInfo', 'studentAssessment', 'merged']:
                df_scaled = self.scale_numerical_features(df_encoded, name, method='standard')
            else:
                df_scaled = df_encoded
            
            # Step 5: Create target variable for main dataset
            if name == 'studentInfo':
                df_final = self.create_target_variable(df_scaled)
            else:
                df_final = df_scaled
            
            cleaned_datasets[name] = df_final
            print(f"{name} cleaned successfully!")
        
        return cleaned_datasets
    
    def generate_descriptive_statistics(self, cleaned_datasets):
        """Generate comprehensive descriptive statistics"""
        print("DESCRIPTIVE STATISTICS")
        print("=" * 60)
        
        for name, df in cleaned_datasets.items():
            print(f"{name.upper()} DATASET")
            print("-" * 40)
            print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Basic info
            print("Data Types:")
            print(df.dtypes.value_counts())
            
            # Descriptive stats for numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                print("Numerical Columns Statistics:")
                print(df[numerical_cols].describe())
            
            # Missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                print("Missing Values:")
                print(missing_values[missing_values > 0])
            else:
                print("No missing values")
                
            print("-" * 40)
    
    def analyze_target_variable(self, cleaned_datasets):
        """Analyze the target variable (success/failure)"""
        print("TARGET VARIABLE ANALYSIS")
        print("=" * 60)
        
        df = cleaned_datasets['studentInfo']
        
        if 'success' in df.columns:
            success_dist = df['success'].value_counts()
            print("Success Distribution:")
            print(f"  Success (1): {success_dist.get(1, 0):,} students")
            print(f"  Failure (0): {success_dist.get(0, 0):,} students")
            
            if success_dist.sum() > 0:
                success_rate = success_dist.get(1, 0) / success_dist.sum() * 100
                print(f"  Success Rate: {success_rate:.1f}%")
        
        if 'final_result' in df.columns:
            result_dist = df['final_result'].value_counts()
            print("Final Result Distribution:")
            for result, count in result_dist.items():
                percentage = count / len(df) * 100
                print(f"  {result}: {count:,} students ({percentage:.1f}%)")
    
    def create_visualizations(self, cleaned_datasets):
        """Create comprehensive visualizations"""
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('OULAD Dataset Analysis (After Cleaning)', fontsize=16, fontweight='bold')
        
        df = cleaned_datasets['studentInfo']
        
        # 1. Final Result Distribution
        if 'final_result' in df.columns:
            result_counts = df['final_result'].value_counts()
            axes[0, 0].pie(result_counts.values, labels=result_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Final Result Distribution')
        
        # 2. Gender Distribution
        if 'gender' in df.columns:
            gender_counts = df['gender'].value_counts()
            axes[0, 1].bar(gender_counts.index, gender_counts.values)
            axes[0, 1].set_title('Gender Distribution')
            axes[0, 1].set_ylabel('Number of Students')
        
        # 3. Age Band Distribution
        if 'age_band' in df.columns:
            age_counts = df['age_band'].value_counts()
            axes[0, 2].bar(range(len(age_counts)), age_counts.values)
            axes[0, 2].set_title('Age Band Distribution')
            axes[0, 2].set_ylabel('Number of Students')
            axes[0, 2].set_xticks(range(len(age_counts)))
            axes[0, 2].set_xticklabels(age_counts.index, rotation=45)
        
        # 4. Education Level Distribution
        if 'highest_education' in df.columns:
            edu_counts = df['highest_education'].value_counts()
            axes[1, 0].bar(range(len(edu_counts)), edu_counts.values)
            axes[1, 0].set_title('Highest Education Level')
            axes[1, 0].set_ylabel('Number of Students')
            axes[1, 0].set_xticks(range(len(edu_counts)))
            axes[1, 0].set_xticklabels(edu_counts.index, rotation=45)
        
        # 5. Previous Attempts Distribution
        if 'num_of_prev_attempts' in df.columns:
            attempt_counts = df['num_of_prev_attempts'].value_counts().sort_index()
            axes[1, 1].bar(attempt_counts.index, attempt_counts.values)
            axes[1, 1].set_title('Number of Previous Attempts')
            axes[1, 1].set_xlabel('Previous Attempts')
            axes[1, 1].set_ylabel('Number of Students')
        
        # 6. Studied Credits Distribution
        if 'studied_credits' in df.columns:
            axes[1, 2].hist(df['studied_credits'], bins=20, alpha=0.7, edgecolor='black')
            axes[1, 2].set_title('Studied Credits Distribution')
            axes[1, 2].set_xlabel('Credits')
            axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('oulad_eda_cleaned.png', dpi=300, bbox_inches='tight')
        print("Cleaned data visualizations saved as 'oulad_eda_cleaned.png'")
        
        # Create correlation heatmap
        self.create_correlation_heatmap(cleaned_datasets)
        
        # Create detailed analysis plots
        self.create_detailed_analysis(cleaned_datasets)
    
    def create_correlation_heatmap(self, cleaned_datasets):
        """Create correlation heatmap for numerical variables"""
        print("Creating correlation heatmap...")
        
        df = cleaned_datasets['studentInfo']
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = df[numerical_cols].corr()
            
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5)
            plt.title('Correlation Heatmap - Student Information (Cleaned)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('oulad_correlation_heatmap_cleaned.png', dpi=300, bbox_inches='tight')
            print("Correlation heatmap saved as 'oulad_correlation_heatmap_cleaned.png'")
    
    def create_detailed_analysis(self, cleaned_datasets):
        """Create detailed analysis plots"""
        print("Creating detailed analysis plots...")
        
        df = cleaned_datasets['studentInfo']
        
        # Create figure for detailed analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Analysis - Student Performance Factors (Cleaned)', fontsize=16, fontweight='bold')
        
        # 1. Success Rate by Gender
        if 'gender' in df.columns and 'success' in df.columns:
            success_by_gender = df.groupby('gender')['success'].mean()
            axes[0, 0].bar(success_by_gender.index, success_by_gender.values)
            axes[0, 0].set_title('Success Rate by Gender')
            axes[0, 0].set_ylabel('Success Rate')
            axes[0, 0].set_ylim(0, 1)
        
        # 2. Success Rate by Age Band
        if 'age_band' in df.columns and 'success' in df.columns:
            success_by_age = df.groupby('age_band')['success'].mean()
            axes[0, 1].bar(range(len(success_by_age)), success_by_age.values)
            axes[0, 1].set_title('Success Rate by Age Band')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_xticks(range(len(success_by_age)))
            axes[0, 1].set_xticklabels(success_by_age.index, rotation=45)
            axes[0, 1].set_ylim(0, 1)
        
        # 3. Success Rate by Education Level
        if 'highest_education' in df.columns and 'success' in df.columns:
            success_by_edu = df.groupby('highest_education')['success'].mean()
            axes[1, 0].bar(range(len(success_by_edu)), success_by_edu.values)
            axes[1, 0].set_title('Success Rate by Education Level')
            axes[1, 0].set_ylabel('Success Rate')
            axes[1, 0].set_xticks(range(len(success_by_edu)))
            axes[1, 0].set_xticklabels(success_by_edu.index, rotation=45)
            axes[1, 0].set_ylim(0, 1)
        
        # 4. Studied Credits vs Success
        if 'studied_credits' in df.columns and 'success' in df.columns:
            axes[1, 1].scatter(df['studied_credits'], df['success'], alpha=0.6)
            axes[1, 1].set_title('Studied Credits vs Success')
            axes[1, 1].set_xlabel('Studied Credits')
            axes[1, 1].set_ylabel('Success (0/1)')
        
        plt.tight_layout()
        plt.savefig('oulad_detailed_analysis_cleaned.png', dpi=300, bbox_inches='tight')
        print("Detailed analysis plots saved as 'oulad_detailed_analysis_cleaned.png'")
    
    def save_cleaned_data(self, cleaned_datasets, output_path="oulad_cleaned"):
        """Save cleaned datasets"""
        print(f"Saving Cleaned Data to '{output_path}'...")
        
        import os
        from pathlib import Path
        
        # Create output directory
        Path(output_path).mkdir(exist_ok=True)
        
        # Save each dataset
        for name, df in cleaned_datasets.items():
            filename = f"{name}_cleaned.csv"
            filepath = os.path.join(output_path, filename)
            df.to_csv(filepath, index=False)
            print(f"   {filename}: {len(df):,} rows, {len(df.columns)} columns")
        
        # Save cleaning metadata
        metadata = {
            'encoders': list(self.encoders.keys()),
            'scalers': list(self.scalers.keys()),
            'cleaning_summary': {
                'datasets_cleaned': len(cleaned_datasets),
                'total_encoders': len(self.encoders),
                'total_scalers': len(self.scalers)
            }
        }
        
        import json
        metadata_file = os.path.join(output_path, 'cleaning_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Cleaning metadata saved: {metadata_file}")
        print(f"Output directory: {output_path}")
    
    def generate_summary_report(self, cleaned_datasets):
        """Generate a summary report"""
        print("SUMMARY REPORT")
        print("=" * 60)
        
        total_students = len(cleaned_datasets['studentInfo'])
        total_assessments = len(cleaned_datasets['studentAssessment'])
        
        print("Dataset Overview:")
        print(f"  Total Students: {total_students:,}")
        print(f"  Total Assessments: {total_assessments:,}")
        print(f"  Assessment per Student: {total_assessments/total_students:.1f}")
        
        # Success rate
        df = cleaned_datasets['studentInfo']
        if 'success' in df.columns:
            success_rate = df['success'].mean() * 100
            print(f"  Overall Success Rate: {success_rate:.1f}%")
        
        print("Cleaning Summary:")
        print(f"  Datasets Cleaned: {len(cleaned_datasets)}")
        print(f"  Encoders Applied: {len(self.encoders)}")
        print(f"  Scalers Applied: {len(self.scalers)}")
        
        print("Files Generated:")
        print(f"  oulad_eda_cleaned.png - Cleaned data visualizations")
        print(f"  oulad_correlation_heatmap_cleaned.png - Correlation analysis")
        print(f"  oulad_detailed_analysis_cleaned.png - Detailed analysis")
        
        print("EDA with cleaning completed successfully!")

def main():
    """Main function to run EDA with cleaning"""
    print("OULAD EXPLORATORY DATA ANALYSIS WITH CLEANING")
    print("=" * 60)
    
    try:
        # Initialize EDA
        eda = OULADEDA()
        
        # Step 1: Clean all datasets
        cleaned_datasets = eda.clean_all_datasets()
        
        # Step 2: Save cleaned data
        eda.save_cleaned_data(cleaned_datasets, output_path="oulad_cleaned")
        
        # Step 3: Generate descriptive statistics
        eda.generate_descriptive_statistics(cleaned_datasets)
        
        # Step 4: Analyze target variable
        eda.analyze_target_variable(cleaned_datasets)
        
        # Step 5: Create visualizations
        eda.create_visualizations(cleaned_datasets)
        
        # Step 6: Generate summary report
        eda.generate_summary_report(cleaned_datasets)
        
    except Exception as e:
        print(f"Error during EDA: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 