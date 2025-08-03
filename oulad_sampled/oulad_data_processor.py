#!/usr/bin/env python3
"""
OULAD Data Processor
Handles the sampling and processing of the Open University Learning Analytics Dataset (OULAD)
Creates a 25% sample of the original dataset for manageable analysis
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class OULADDataProcessor:
    """
    Class for processing and sampling the OULAD dataset
    """
    
    def __init__(self, original_data_path="open+university+learning+analytics+dataset.zip", 
                 output_path="oulad_sampled", sampling_ratio=0.25, random_seed=42):
        """
        Initialize the OULAD data processor
        
        Args:
            original_data_path (str): Path to the original OULAD dataset
            output_path (str): Path to save the sampled dataset
            sampling_ratio (float): Ratio of students to sample (default: 0.25 for 25%)
            random_seed (int): Random seed for reproducibility
        """
        self.original_data_path = original_data_path
        self.output_path = output_path
        self.sampling_ratio = sampling_ratio
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
    def load_original_data(self):
        """
        Load the original OULAD dataset files
        
        Returns:
            dict: Dictionary containing all original datasets
        """
        print("Loading original OULAD dataset...")
        
        try:
            # Check if we have the original dataset
            if os.path.exists(self.original_data_path):
                # Extract from zip file if needed
                import zipfile
                with zipfile.ZipFile(self.original_data_path, 'r') as zip_ref:
                    zip_ref.extractall('temp_oulad')
                data_path = 'temp_oulad'
            else:
                # Assume data is already extracted
                data_path = '.'
            
            datasets = {}
            
            # Load main datasets
            datasets['studentInfo'] = pd.read_csv(f"{data_path}/studentInfo.csv")
            datasets['studentAssessment'] = pd.read_csv(f"{data_path}/studentAssessment.csv")
            datasets['studentRegistration'] = pd.read_csv(f"{data_path}/studentRegistration.csv")
            datasets['studentVle'] = pd.read_csv(f"{data_path}/studentVle.csv")
            datasets['assessments'] = pd.read_csv(f"{data_path}/assessments.csv")
            datasets['courses'] = pd.read_csv(f"{data_path}/courses.csv")
            datasets['vle'] = pd.read_csv(f"{data_path}/vle.csv")
            
            print(f" Loaded {len(datasets)} datasets successfully!")
            print(f"  Original students: {len(datasets['studentInfo']):,}")
            
            return datasets
            
        except Exception as e:
            print(f" Error loading original data: {e}")
            return None
    
    def create_student_sample(self, student_info_df):
        """
        Create a random sample of students
        
        Args:
            student_info_df (DataFrame): Original student information
            
        Returns:
            list: List of sampled student IDs
        """
        print(f"Creating {self.sampling_ratio*100}% sample of students...")
        
        # Get unique student IDs
        unique_students = student_info_df['id_student'].unique()
        total_students = len(unique_students)
        
        # Calculate sample size
        sample_size = int(total_students * self.sampling_ratio)
        
        # Random sampling without replacement
        sampled_student_ids = np.random.choice(
            unique_students, 
            size=sample_size, 
            replace=False
        )
        
        print(f"   Total students: {total_students:,}")
        print(f"   Sample size: {sample_size:,}")
        print(f"   Sampling ratio: {self.sampling_ratio}")
        
        return sampled_student_ids.tolist()
    
    def filter_dataset_by_students(self, df, student_ids, student_id_column='id_student'):
        """
        Filter a dataset to include only the sampled students
        
        Args:
            df (DataFrame): Dataset to filter
            student_ids (list): List of student IDs to include
            student_id_column (str): Name of the student ID column
            
        Returns:
            DataFrame: Filtered dataset
        """
        if student_id_column in df.columns:
            filtered_df = df[df[student_id_column].isin(student_ids)].copy()
            return filtered_df
        else:
            # If no student ID column, return the original dataset
            return df.copy()
    
    def create_sampled_datasets(self, original_datasets, sampled_student_ids):
        """
        Create sampled versions of all datasets
        
        Args:
            original_datasets (dict): Dictionary of original datasets
            sampled_student_ids (list): List of sampled student IDs
            
        Returns:
            dict: Dictionary of sampled datasets
        """
        print("Creating sampled datasets...")
        
        sampled_datasets = {}
        
        # Define which datasets to filter by student ID
        student_based_datasets = [
            'studentInfo', 'studentAssessment', 'studentRegistration', 'studentVle'
        ]
        
        # Filter student-based datasets
        for dataset_name in student_based_datasets:
            if dataset_name in original_datasets:
                print(f"   Processing {dataset_name}...")
                sampled_datasets[dataset_name] = self.filter_dataset_by_students(
                    original_datasets[dataset_name], 
                    sampled_student_ids
                )
                print(f"     Original: {len(original_datasets[dataset_name]):,} records")
                print(f"     Sampled: {len(sampled_datasets[dataset_name]):,} records")
        
        # Copy reference datasets (no filtering needed)
        reference_datasets = ['assessments', 'courses', 'vle']
        for dataset_name in reference_datasets:
            if dataset_name in original_datasets:
                print(f"   Copying {dataset_name}...")
                sampled_datasets[dataset_name] = original_datasets[dataset_name].copy()
        
        return sampled_datasets
    
    def create_merged_dataset(self, sampled_datasets):
        """
        Create a merged dataset for easier analysis
        
        Args:
            sampled_datasets (dict): Dictionary of sampled datasets
            
        Returns:
            DataFrame: Merged dataset
        """
        print("Creating merged dataset...")
        
        # Start with student information
        merged_df = sampled_datasets['studentInfo'].copy()
        
        # Add assessment information
        if 'studentAssessment' in sampled_datasets:
            # Merge with assessments to get assessment details
            assessment_info = sampled_datasets['assessments'].copy()
            student_assessment = sampled_datasets['studentAssessment'].copy()
            
            # Merge student assessment with assessment info
            assessment_merged = student_assessment.merge(
                assessment_info, 
                on=['code_module', 'code_presentation', 'id_assessment'], 
                how='left'
            )
            
            # Aggregate assessment scores by student
            assessment_summary = assessment_merged.groupby('id_student').agg({
                'score': ['mean', 'std', 'count'],
                'final_result': 'first'
            }).reset_index()
            
            # Flatten column names
            assessment_summary.columns = [
                'id_student', 'avg_score', 'score_std', 'assessment_count', 'final_result'
            ]
            
            # Merge with main dataset
            merged_df = merged_df.merge(assessment_summary, on='id_student', how='left')
        
        print(f"   Merged dataset shape: {merged_df.shape}")
        return merged_df
    
    def save_sampled_datasets(self, sampled_datasets, merged_df):
        """
        Save all sampled datasets to files
        
        Args:
            sampled_datasets (dict): Dictionary of sampled datasets
            merged_df (DataFrame): Merged dataset
        """
        print("Saving sampled datasets...")
        
        # Save individual datasets
        for dataset_name, df in sampled_datasets.items():
            file_path = os.path.join(self.output_path, f"{dataset_name}.csv")
            df.to_csv(file_path, index=False)
            print(f"   Saved {dataset_name}.csv ({len(df):,} records)")
        
        # Save merged dataset
        merged_file_path = os.path.join(self.output_path, "oulad_merged.csv")
        merged_df.to_csv(merged_file_path, index=False)
        print(f"   Saved oulad_merged.csv ({len(merged_df):,} records)")
        
        # Save dataset information
        self.save_dataset_info(sampled_datasets, merged_df)
    
    def save_dataset_info(self, sampled_datasets, merged_df):
        """
        Save information about the sampled dataset
        
        Args:
            sampled_datasets (dict): Dictionary of sampled datasets
            merged_df (DataFrame): Merged dataset
        """
        info_file_path = os.path.join(self.output_path, "dataset_info.txt")
        
        with open(info_file_path, 'w') as f:
            f.write("OULAD Sampled Dataset Information\n")
            f.write("=" * 40 + "\n")
            f.write(f"sampling_ratio: {self.sampling_ratio}\n")
            f.write(f"random_seed: {self.random_seed}\n")
            f.write(f"total_students: {len(merged_df)}\n")
            
            if 'studentAssessment' in sampled_datasets:
                f.write(f"total_assessments: {len(sampled_datasets['studentAssessment'])}\n")
            
            if 'studentVle' in sampled_datasets:
                f.write(f"total_vle_interactions: {len(sampled_datasets['studentVle'])}\n")
            
            if 'courses' in sampled_datasets:
                f.write(f"courses: {len(sampled_datasets['courses'])}\n")
            
            # Count unique presentations
            if 'studentInfo' in sampled_datasets:
                presentations = sampled_datasets['studentInfo']['code_presentation'].nunique()
                f.write(f"presentations: {presentations}\n")
        
        print(f"   Saved dataset_info.txt")
    
    def get_dataset_summary(self, sampled_datasets, merged_df):
        """
        Get a summary of the sampled dataset
        
        Returns:
            dict: Summary information
        """
        summary = {
            'info': {
                'total_students': len(merged_df),
                'total_assessments': len(sampled_datasets.get('studentAssessment', [])),
                'total_vle_interactions': len(sampled_datasets.get('studentVle', [])),
                'courses': len(sampled_datasets.get('courses', [])),
                'presentations': sampled_datasets.get('studentInfo', pd.DataFrame())['code_presentation'].nunique() if 'studentInfo' in sampled_datasets else 0
            }
        }
        
        return summary

def create_oulad_sample():
    """
    Main function to create the sampled OULAD dataset
    
    Returns:
        dict: Summary information about the created dataset
    """
    print("🚀 OULAD Data Sampling Process")
    print("=" * 50)
    
    # Initialize processor
    processor = OULADDataProcessor()
    
    # Load original data
    original_datasets = processor.load_original_data()
    if original_datasets is None:
        return None
    
    # Create student sample
    sampled_student_ids = processor.create_student_sample(original_datasets['studentInfo'])
    
    # Create sampled datasets
    sampled_datasets = processor.create_sampled_datasets(original_datasets, sampled_student_ids)
    
    # Create merged dataset
    merged_df = processor.create_merged_dataset(sampled_datasets)
    
    # Save all datasets
    processor.save_sampled_datasets(sampled_datasets, merged_df)
    
    # Get summary
    summary = processor.get_dataset_summary(sampled_datasets, merged_df)
    
    print("\nSampling completed successfully!")
    print(f"📁 Output directory: {processor.output_path}")
    print(f"📊 Sample size: {summary['info']['total_students']:,} students")
    
    return summary

if __name__ == "__main__":
    # Run the sampling process
    result = create_oulad_sample()
    
    if result:
        print("\n📋 Dataset Summary:")
        for key, value in result['info'].items():
            print(f"   {key}: {value:,}")
    else:
        print(" Sampling failed!") 