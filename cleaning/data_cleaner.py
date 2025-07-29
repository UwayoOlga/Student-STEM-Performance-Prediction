#!/usr/bin/env python3
"""
OULAD Data Cleaning Module
Handles missing values, inconsistent formats, outliers, and data transformations
for the Open University Learning Analytics Dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class OULADDataCleaner:
    """
    Class for cleaning and preprocessing OULAD dataset
    """
    
    def __init__(self, data_path="oulad_sampled"):
        self.data_path = data_path
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}
        
    def load_data(self):
        """
        Load the sampled OULAD datasets
        Returns:
            dict: Dictionary containing all datasets
        """
        print("Loading OULAD datasets...")
        
        datasets = {}
        try:
            # Load main datasets
            datasets['studentInfo'] = pd.read_csv(f"{self.data_path}/studentInfo.csv")
            datasets['studentAssessment'] = pd.read_csv(f"{self.data_path}/studentAssessment.csv")
            datasets['assessments'] = pd.read_csv(f"{self.data_path}/assessments.csv")
            datasets['courses'] = pd.read_csv(f"{self.data_path}/courses.csv")
            datasets['studentRegistration'] = pd.read_csv(f"{self.data_path}/studentRegistration.csv")
            
            # Load merged dataset if available
            try:
                datasets['merged'] = pd.read_csv(f"{self.data_path}/oulad_merged.csv")
                print("Merged dataset loaded")
            except:
                print("Merged dataset not found, will create it")
            
            print("All datasets loaded successfully!")
            return datasets
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def analyze_missing_values(self, df, dataset_name):
        """
        Analyze missing values in a dataset
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset
        """
        print(f"Missing Values Analysis - {dataset_name}")
        print("-" * 50)
        
        # Calculate missing values
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        
        # Create missing values summary
        missing_summary = pd.DataFrame({
            'Column': missing_values.index,
            'Missing_Count': missing_values.values,
            'Missing_Percentage': missing_percentage.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        # Filter only columns with missing values
        missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
        
        if len(missing_summary) > 0:
            print("Columns with missing values:")
            for _, row in missing_summary.iterrows():
                print(f"   {row['Column']}: {row['Missing_Count']:,} ({row['Missing_Percentage']:.1f}%)")
        else:
            print("No missing values found!")
        
        return missing_summary
    
    def handle_missing_values(self, df, dataset_name):
        """
        Handle missing values in the dataset
        Args:
            df: DataFrame to clean
            dataset_name: Name of the dataset
        Returns:
            DataFrame: Cleaned DataFrame
        """
        print(f"Handling Missing Values - {dataset_name}")
        print("-" * 50)
        
        df_cleaned = df.copy()
        
        # Analyze missing values
        missing_summary = self.analyze_missing_values(df_cleaned, dataset_name)
        
        if len(missing_summary) == 0:
            print("No missing values to handle")
            return df_cleaned
        
        # Handle missing values based on data type
        for column in df_cleaned.columns:
            if df_cleaned[column].isnull().sum() > 0:
                if df_cleaned[column].dtype in ['object', 'category']:
                    # For categorical variables, use mode
                    mode_value = df_cleaned[column].mode()[0]
                    df_cleaned[column].fillna(mode_value, inplace=True)
                    print(f"   Filled missing values in {column} with mode: {mode_value}")
                else:
                    # For numerical variables, use median
                    median_value = df_cleaned[column].median()
                    df_cleaned[column].fillna(median_value, inplace=True)
                    print(f"   Filled missing values in {column} with median: {median_value}")
        
        return df_cleaned
    
    def detect_and_handle_outliers(self, df, dataset_name, columns=None):
        """
        Detect and handle outliers in numerical columns
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset
            columns: Specific columns to check for outliers
        Returns:
            tuple: (cleaned DataFrame, outlier summary)
        """
        print(f"Outlier Detection - {dataset_name}")
        print("-" * 50)
        
        df_cleaned = df.copy()
        outlier_summary = {}
        
        # Select numerical columns
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
        """
        Encode categorical features using LabelEncoder
        Args:
            df: DataFrame to encode
            dataset_name: Name of the dataset
        Returns:
            DataFrame: Encoded DataFrame
        """
        print(f"Encoding Categorical Features - {dataset_name}")
        print("-" * 50)
        
        df_encoded = df.copy()
        
        # Select categorical columns
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        for column in categorical_cols:
            if column in ['code_module', 'code_presentation']:
                # Skip these columns as they are identifiers
                continue
                
            # Create encoder for this column
            encoder = LabelEncoder()
            df_encoded[column] = encoder.fit_transform(df_encoded[column].astype(str))
            
            # Store encoder for later use
            self.encoders[f"{dataset_name}_{column}"] = encoder
            
            print(f"   {column}: Encoded {len(encoder.classes_)} categories")
        
        return df_encoded
    
    def scale_numerical_features(self, df, dataset_name, method='standard'):
        """
        Scale numerical features
        Args:
            df: DataFrame to scale
            dataset_name: Name of the dataset
            method: Scaling method ('standard' or 'minmax')
        Returns:
            DataFrame: Scaled DataFrame
        """
        print(f"Scaling Numerical Features - {dataset_name}")
        print("-" * 50)
        
        df_scaled = df.copy()
        
        # Select numerical columns (excluding IDs)
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if 'id_' not in col.lower()]
        
        if len(numerical_cols) > 0:
            if method == 'standard':
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            
            df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
            
            # Store scaler for later use
            self.scalers[f"{dataset_name}_{method}"] = scaler
            
            print(f"   Scaled {len(numerical_cols)} numerical columns using {method} scaling")
            print(f"   Columns: {', '.join(numerical_cols)}")
        
        return df_scaled
    
    def create_target_variable(self, df):
        """
        Create target variable for STEM success prediction
        Args:
            df: DataFrame with final_result column
        Returns:
            DataFrame: DataFrame with target variable
        """
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
        """
        Clean all OULAD datasets
        Returns:
            dict: Dictionary of cleaned datasets
        """
        print("Starting OULAD Dataset Cleaning Process")
        print("=" * 60)
        
        # Load data
        datasets = self.load_data()
        if datasets is None:
            return None
        
        cleaned_datasets = {}
        
        # Clean each dataset
        for name, df in datasets.items():
            print(f"Cleaning {name}")
            print("=" * 20)
            
            # Handle missing values
            df_cleaned = self.handle_missing_values(df, name)
            
            # Detect and handle outliers (for numerical datasets)
            if name in ['studentAssessment', 'merged']:
                df_cleaned, outliers = self.detect_and_handle_outliers(df_cleaned, name)
            
            # Encode categorical features
            df_encoded = self.encode_categorical_features(df_cleaned, name)
            
            # Scale numerical features (for main datasets)
            if name in ['studentInfo', 'studentAssessment', 'merged']:
                df_scaled = self.scale_numerical_features(df_encoded, name, method='standard')
            else:
                df_scaled = df_encoded
            
            # Create target variable for main dataset
            if name == 'studentInfo':
                df_final = self.create_target_variable(df_scaled)
            else:
                df_final = df_scaled
            
            cleaned_datasets[name] = df_final
            
            print(f"{name} cleaned successfully!")
        
        return cleaned_datasets
    
    def save_cleaned_data(self, cleaned_datasets, output_path="oulad_cleaned"):
        """
        Save cleaned datasets
        Args:
            cleaned_datasets: Dictionary of cleaned datasets
            output_path: Directory to save cleaned data
        """
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

def main():
    """
    Main function to run the OULAD data cleaning process
    """
    print("Starting OULAD Data Cleaning Process...")
    print("=" * 50)
    
    try:
        # Initialize the cleaner
        cleaner = OULADDataCleaner(data_path="oulad_sampled")
        
        # Run the complete cleaning process
        cleaned_datasets = cleaner.clean_all_datasets()
        
        # Save the cleaned data
        cleaner.save_cleaned_data(cleaned_datasets, output_path="oulad_cleaned")
        
        print("Data cleaning completed successfully!")
        print("Cleaned data saved to: oulad_cleaned/")
        
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code) 