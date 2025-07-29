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
        print("📊 Loading OULAD datasets...")
        
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
                print("✅ Merged dataset loaded")
            except:
                print("⚠️ Merged dataset not found, will create it")
            
            print("✅ All datasets loaded successfully!")
            return datasets
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def analyze_missing_values(self, df, dataset_name):
        """
        Analyze missing values in a dataset
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset
        """
        print(f"\n🔍 Missing Values Analysis - {dataset_name}")
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
            print("📊 Columns with missing values:")
            for _, row in missing_summary.iterrows():
                print(f"   {row['Column']}: {row['Missing_Count']:,} ({row['Missing_Percentage']:.1f}%)")
        else:
            print("✅ No missing values found!")
        
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
        print(f"\n🧹 Handling Missing Values - {dataset_name}")
        print("-" * 50)
        
        df_cleaned = df.copy()
        
        # Analyze missing values first
        missing_summary = self.analyze_missing_values(df, dataset_name)
        
        if len(missing_summary) == 0:
            print("✅ No missing values to handle")
            return df_cleaned
        
        # Handle missing values based on column type and context
        for _, row in missing_summary.iterrows():
            column = row['Column']
            missing_pct = row['Missing_Percentage']
            
            # Categorical columns
            if df[column].dtype == 'object' or df[column].dtype.name == 'category':
                if missing_pct < 10:  # Small amount of missing data
                    # Use mode (most frequent value)
                    mode_value = df[column].mode()[0] if len(df[column].mode()) > 0 else 'Unknown'
                    df_cleaned[column].fillna(mode_value, inplace=True)
                    print(f"   ✅ {column}: Filled with mode '{mode_value}'")
                else:  # Large amount of missing data
                    # Create 'Missing' category
                    df_cleaned[column].fillna('Missing', inplace=True)
                    print(f"   ✅ {column}: Filled with 'Missing' category")
            
            # Numerical columns
            else:
                if missing_pct < 10:  # Small amount of missing data
                    # Use median for numerical data
                    median_value = df[column].median()
                    df_cleaned[column].fillna(median_value, inplace=True)
                    print(f"   ✅ {column}: Filled with median {median_value:.2f}")
                else:  # Large amount of missing data
                    # Use mean or create a separate indicator
                    mean_value = df[column].mean()
                    df_cleaned[column].fillna(mean_value, inplace=True)
                    print(f"   ✅ {column}: Filled with mean {mean_value:.2f}")
        
        return df_cleaned
    
    def detect_and_handle_outliers(self, df, dataset_name, columns=None):
        """
        Detect and handle outliers in numerical columns
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset
            columns: Specific columns to check (if None, check all numerical)
        Returns:
            DataFrame: DataFrame with outliers handled
        """
        print(f"\n📊 Outlier Detection - {dataset_name}")
        print("-" * 50)
        
        df_cleaned = df.copy()
        
        # Select numerical columns
        if columns is None:
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numerical_columns = [col for col in columns if col in df.columns and df[col].dtype in [np.number]]
        
        outliers_summary = {}
        
        for column in numerical_columns:
            # Skip if column has too many missing values or is an ID
            if column in ['id_student', 'id_assessment', 'id_site']:
                continue
                
            # Calculate Q1, Q3, and IQR
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(df)) * 100
            
            if outlier_count > 0:
                print(f"   📈 {column}: {outlier_count:,} outliers ({outlier_percentage:.1f}%)")
                print(f"      Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
                
                # Handle outliers based on percentage
                if outlier_percentage < 5:  # Small number of outliers
                    # Cap outliers to bounds
                    df_cleaned[column] = df_cleaned[column].clip(lower=lower_bound, upper=upper_bound)
                    print(f"      ✅ Capped outliers to bounds")
                elif outlier_percentage < 15:  # Moderate number of outliers
                    # Replace with median
                    median_value = df[column].median()
                    df_cleaned.loc[df_cleaned[column] < lower_bound, column] = median_value
                    df_cleaned.loc[df_cleaned[column] > upper_bound, column] = median_value
                    print(f"      ✅ Replaced outliers with median {median_value:.2f}")
                else:  # Large number of outliers
                    # Keep as is but log the information
                    print(f"      ⚠️ Large number of outliers - keeping as is")
                
                outliers_summary[column] = {
                    'count': outlier_count,
                    'percentage': outlier_percentage,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            else:
                print(f"   ✅ {column}: No outliers detected")
        
        return df_cleaned, outliers_summary
    
    def encode_categorical_features(self, df, dataset_name):
        """
        Encode categorical features
        Args:
            df: DataFrame to encode
            dataset_name: Name of the dataset
        Returns:
            DataFrame: DataFrame with encoded features
        """
        print(f"\n🔤 Encoding Categorical Features - {dataset_name}")
        print("-" * 50)
        
        df_encoded = df.copy()
        
        # Identify categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Skip ID columns and columns that should remain categorical
        skip_columns = ['id_student', 'id_assessment', 'id_site', 'code_module', 'code_presentation']
        categorical_columns = [col for col in categorical_columns if col not in skip_columns]
        
        for column in categorical_columns:
            if column in df_encoded.columns:
                # Create label encoder
                le = LabelEncoder()
                
                # Handle missing values before encoding
                if df_encoded[column].isnull().any():
                    df_encoded[column].fillna('Missing', inplace=True)
                
                # Fit and transform
                df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
                
                # Store encoder for later use
                self.encoders[f"{dataset_name}_{column}"] = le
                
                print(f"   ✅ {column}: Encoded {len(le.classes_)} categories")
        
        return df_encoded
    
    def scale_numerical_features(self, df, dataset_name, method='standard'):
        """
        Scale numerical features
        Args:
            df: DataFrame to scale
            dataset_name: Name of the dataset
            method: 'standard' or 'minmax'
        Returns:
            DataFrame: DataFrame with scaled features
        """
        print(f"\n📏 Scaling Numerical Features - {dataset_name}")
        print("-" * 50)
        
        df_scaled = df.copy()
        
        # Identify numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Skip ID columns and target variables
        skip_columns = ['id_student', 'id_assessment', 'id_site', 'final_result_encoded']
        numerical_columns = [col for col in numerical_columns if col not in skip_columns]
        
        if len(numerical_columns) > 0:
            # Choose scaler
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError("Method must be 'standard' or 'minmax'")
            
            # Fit and transform
            df_scaled[numerical_columns] = scaler.fit_transform(df_scaled[numerical_columns])
            
            # Store scaler for later use
            self.scalers[f"{dataset_name}_{method}"] = scaler
            
            print(f"   ✅ Scaled {len(numerical_columns)} numerical columns using {method} scaling")
            print(f"   📊 Columns: {', '.join(numerical_columns)}")
        else:
            print("   ⚠️ No numerical columns to scale")
        
        return df_scaled
    
    def create_target_variable(self, df):
        """
        Create target variable for STEM success prediction
        Args:
            df: DataFrame with final_result column
        Returns:
            DataFrame: DataFrame with target variable
        """
        print(f"\n🎯 Creating Target Variable")
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
        print(f"📊 Target Variable Distribution:")
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
        print("🚀 Starting OULAD Dataset Cleaning Process")
        print("=" * 60)
        
        # Load data
        datasets = self.load_data()
        if datasets is None:
            return None
        
        cleaned_datasets = {}
        
        # Clean each dataset
        for name, df in datasets.items():
            print(f"\n{'='*20} Cleaning {name} {'='*20}")
            
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
            
            print(f"✅ {name} cleaned successfully!")
        
        return cleaned_datasets
    
    def save_cleaned_data(self, cleaned_datasets, output_path="oulad_cleaned"):
        """
        Save cleaned datasets
        Args:
            cleaned_datasets: Dictionary of cleaned datasets
            output_path: Directory to save cleaned data
        """
        print(f"\n💾 Saving Cleaned Data to '{output_path}'...")
        
        import os
        from pathlib import Path
        
        # Create output directory
        Path(output_path).mkdir(exist_ok=True)
        
        # Save each cleaned dataset
        for name, data in cleaned_datasets.items():
            file_path = os.path.join(output_path, f"{name}_cleaned.csv")
            data.to_csv(file_path, index=False)
            print(f"   ✅ {name}_cleaned.csv: {len(data):,} rows, {len(data.columns)} columns")
        
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
        metadata_path = os.path.join(output_path, "cleaning_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        
        print(f"✅ Cleaning metadata saved: {metadata_path}")
        print(f"📁 Output directory: {output_path}")

def main():
    """
    Main function to clean OULAD datasets
    """
    # Initialize cleaner
    cleaner = OULADDataCleaner()
    
    # Clean all datasets
    cleaned_datasets = cleaner.clean_all_datasets()
    
    if cleaned_datasets is not None:
        # Save cleaned data
        cleaner.save_cleaned_data(cleaned_datasets)
        
        print("\n🎉 OULAD Dataset Cleaning Completed Successfully!")
        print("=" * 60)
        
        # Summary
        print(f"\n📊 Cleaning Summary:")
        print(f"   Datasets cleaned: {len(cleaned_datasets)}")
        print(f"   Encoders created: {len(cleaner.encoders)}")
        print(f"   Scalers created: {len(cleaner.scalers)}")
        
        return cleaned_datasets
    else:
        print("❌ Cleaning failed!")
        return None

if __name__ == "__main__":
    main() 