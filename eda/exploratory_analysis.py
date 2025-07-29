#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for OULAD Dataset
Generates descriptive statistics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class OULADEDA:
    def __init__(self, data_path="oulad_cleaned"):
        self.data_path = data_path
        self.datasets = {}
        self.load_data()
        
    def load_data(self):
        """Load all cleaned OULAD datasets"""
        print("Loading cleaned OULAD datasets...")
        
        try:
            self.datasets['studentInfo'] = pd.read_csv(f"{self.data_path}/studentInfo_cleaned.csv")
            self.datasets['studentAssessment'] = pd.read_csv(f"{self.data_path}/studentAssessment_cleaned.csv")
            self.datasets['assessments'] = pd.read_csv(f"{self.data_path}/assessments_cleaned.csv")
            self.datasets['courses'] = pd.read_csv(f"{self.data_path}/courses_cleaned.csv")
            self.datasets['studentRegistration'] = pd.read_csv(f"{self.data_path}/studentRegistration_cleaned.csv")
            self.datasets['merged'] = pd.read_csv(f"{self.data_path}/merged_cleaned.csv")
            
            print("All datasets loaded successfully!")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            
    def generate_descriptive_statistics(self):
        """Generate comprehensive descriptive statistics"""
        print("DESCRIPTIVE STATISTICS")
        print("=" * 60)
        
        for name, df in self.datasets.items():
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
    
    def analyze_target_variable(self):
        """Analyze the target variable (success/failure)"""
        print("TARGET VARIABLE ANALYSIS")
        print("=" * 60)
        
        df = self.datasets['studentInfo']
        
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
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('OULAD Dataset Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        df = self.datasets['studentInfo']
        
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
        plt.savefig('oulad_eda_basic.png', dpi=300, bbox_inches='tight')
        print("Basic visualizations saved as 'oulad_eda_basic.png'")
        
        # Create correlation heatmap
        self.create_correlation_heatmap()
        
        # Create detailed analysis plots
        self.create_detailed_analysis()
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap for numerical variables"""
        print("Creating correlation heatmap...")
        
        df = self.datasets['studentInfo']
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = df[numerical_cols].corr()
            
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5)
            plt.title('Correlation Heatmap - Student Information', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('oulad_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            print("Correlation heatmap saved as 'oulad_correlation_heatmap.png'")
    
    def create_detailed_analysis(self):
        """Create detailed analysis plots"""
        print("Creating detailed analysis plots...")
        
        df = self.datasets['studentInfo']
        
        # Create figure for detailed analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Analysis - Student Performance Factors', fontsize=16, fontweight='bold')
        
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
        plt.savefig('oulad_detailed_analysis.png', dpi=300, bbox_inches='tight')
        print("Detailed analysis plots saved as 'oulad_detailed_analysis.png'")
    
    def generate_summary_report(self):
        """Generate a summary report"""
        print("SUMMARY REPORT")
        print("=" * 60)
        
        total_students = len(self.datasets['studentInfo'])
        total_assessments = len(self.datasets['studentAssessment'])
        
        print("Dataset Overview:")
        print(f"  Total Students: {total_students:,}")
        print(f"  Total Assessments: {total_assessments:,}")
        print(f"  Assessment per Student: {total_assessments/total_students:.1f}")
        
        # Success rate
        df = self.datasets['studentInfo']
        if 'success' in df.columns:
            success_rate = df['success'].mean() * 100
            print(f"  Overall Success Rate: {success_rate:.1f}%")
        
        print("Files Generated:")
        print(f"  oulad_eda_basic.png - Basic visualizations")
        print(f"  oulad_correlation_heatmap.png - Correlation analysis")
        print(f"  oulad_detailed_analysis.png - Detailed analysis")
        
        print("EDA completed successfully!")

def main():
    """Main function to run EDA"""
    print("OULAD EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    try:
        # Initialize EDA
        eda = OULADEDA()
        
        # Generate descriptive statistics
        eda.generate_descriptive_statistics()
        
        # Analyze target variable
        eda.analyze_target_variable()
        
        # Create visualizations
        eda.create_visualizations()
        
        # Generate summary report
        eda.generate_summary_report()
        
    except Exception as e:
        print(f"Error during EDA: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 