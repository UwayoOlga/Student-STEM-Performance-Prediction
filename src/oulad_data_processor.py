"""
OULAD Data Processor for Student Performance Analysis
Reduces the large OULAD dataset to a manageable size (25% of students)
while maintaining all data relationships and structure
"""

import pandas as pd
import numpy as np
import os
import random
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class OULADDataProcessor:
    """
    Class for processing and reducing the OULAD dataset
    """
    
    def __init__(self, data_path="open+university+learning+analytics+dataset (1)"):
        self.data_path = data_path
        self.sample_ratio = 0.25  # 25% of students
        self.random_seed = 42
        
    def load_and_sample_data(self):
        """
        Load OULAD data and sample 25% of students
        Returns:
            dict: Dictionary containing all sampled datasets
        """
        print("🔄 Loading OULAD dataset...")
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Load all datasets
        datasets = {}
        
        try:
            # Load student info (main dataset)
            print("📊 Loading studentInfo.csv...")
            student_info = pd.read_csv(os.path.join(self.data_path, "studentInfo.csv"))
            print(f"   Original students: {len(student_info)}")
            
            # Sample 25% of unique students
            unique_students = student_info['id_student'].unique()
            sampled_students = np.random.choice(unique_students, 
                                              size=int(len(unique_students) * self.sample_ratio), 
                                              replace=False)
            
            print(f"   Sampled students: {len(sampled_students)}")
            
            # Filter student info to sampled students
            student_info_sampled = student_info[student_info['id_student'].isin(sampled_students)]
            datasets['studentInfo'] = student_info_sampled
            
            # Load and filter student assessment data
            print("📊 Loading studentAssessment.csv...")
            student_assessment = pd.read_csv(os.path.join(self.data_path, "studentAssessment.csv"))
            student_assessment_sampled = student_assessment[student_assessment['id_student'].isin(sampled_students)]
            datasets['studentAssessment'] = student_assessment_sampled
            
            # Load and filter student registration data
            print("📊 Loading studentRegistration.csv...")
            student_registration = pd.read_csv(os.path.join(self.data_path, "studentRegistration.csv"))
            student_registration_sampled = student_registration[student_registration['id_student'].isin(sampled_students)]
            datasets['studentRegistration'] = student_registration_sampled
            
            # Load other datasets (no filtering needed as they're small)
            print("📊 Loading other datasets...")
            datasets['assessments'] = pd.read_csv(os.path.join(self.data_path, "assessments.csv"))
            datasets['courses'] = pd.read_csv(os.path.join(self.data_path, "courses.csv"))
            datasets['vle'] = pd.read_csv(os.path.join(self.data_path, "vle.csv"))
            
            # Load and filter student VLE data (this is the largest file)
            print("📊 Loading studentVle.csv (this may take a moment)...")
            student_vle = pd.read_csv(os.path.join(self.data_path, "studentVle.csv"))
            student_vle_sampled = student_vle[student_vle['id_student'].isin(sampled_students)]
            datasets['studentVle'] = student_vle_sampled
            
            print("✅ All datasets loaded and sampled successfully!")
            
            return datasets
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def save_sampled_data(self, datasets, output_path="oulad_sampled"):
        """
        Save the sampled datasets to a new directory
        Args:
            datasets: Dictionary of sampled datasets
            output_path: Directory to save the sampled data
        """
        print(f"\n💾 Saving sampled data to '{output_path}'...")
        
        # Create output directory
        Path(output_path).mkdir(exist_ok=True)
        
        # Save each dataset
        for name, data in datasets.items():
            file_path = os.path.join(output_path, f"{name}.csv")
            data.to_csv(file_path, index=False)
            print(f"   ✅ {name}.csv: {len(data):,} rows")
        
        # Save dataset info
        info = {
            'sampling_ratio': self.sample_ratio,
            'random_seed': self.random_seed,
            'total_students': len(datasets['studentInfo']['id_student'].unique()),
            'total_assessments': len(datasets['studentAssessment']),
            'total_vle_interactions': len(datasets['studentVle']),
            'courses': len(datasets['studentInfo']['code_module'].unique()),
            'presentations': len(datasets['studentInfo']['code_presentation'].unique())
        }
        
        info_path = os.path.join(output_path, "dataset_info.txt")
        with open(info_path, 'w') as f:
            f.write("OULAD Sampled Dataset Information\n")
            f.write("="*40 + "\n")
            for key, value in info.items():
                f.write(f"{key}: {value}\n")
        
        print(f"✅ Sampled data saved successfully!")
        print(f"📁 Output directory: {output_path}")
        
        return info
    
    def create_merged_dataset(self, datasets):
        """
        Create a merged dataset combining student info with assessment data
        Args:
            datasets: Dictionary of sampled datasets
        Returns:
            DataFrame: Merged dataset ready for analysis
        """
        print("\n🔗 Creating merged dataset...")
        
        try:
            # Get the datasets
            student_info = datasets['studentInfo']
            student_assessment = datasets['studentAssessment']
            assessments = datasets['assessments']
            
            # Check column names for debugging
            print(f"Student assessment columns: {list(student_assessment.columns)}")
            print(f"Assessments columns: {list(assessments.columns)}")
            print(f"Student info columns: {list(student_info.columns)}")
            
            # Merge assessment data with assessment metadata
            # First, let's check if the merge keys exist
            common_cols_assessment = set(student_assessment.columns) & set(assessments.columns)
            print(f"Common columns between student_assessment and assessments: {common_cols_assessment}")
            
            # Merge with student info
            common_cols_student = set(student_assessment.columns) & set(student_info.columns)
            print(f"Common columns between student_assessment and student_info: {common_cols_student}")
            
            # Create a simple merged dataset focusing on key features
            merged_data = student_assessment.merge(
                student_info[['id_student', 'code_module', 'code_presentation', 'gender', 'region', 
                             'highest_education', 'imd_band', 'age_band', 'num_of_prev_attempts', 
                             'studied_credits', 'disability', 'final_result']], 
                on=['id_student', 'code_module', 'code_presentation'], 
                how='left'
            )
            
            # Add assessment metadata
            merged_data = merged_data.merge(
                assessments[['id_assessment', 'code_module', 'code_presentation', 'assessment_type', 'date', 'weight']], 
                on=['id_assessment', 'code_module', 'code_presentation'], 
                how='left'
            )
            
            print(f"✅ Merged dataset created: {len(merged_data):,} rows")
            print(f"   Features: {len(merged_data.columns)}")
            
            return merged_data
            
        except Exception as e:
            print(f"❌ Error creating merged dataset: {e}")
            # Return a simpler merged dataset
            student_info = datasets['studentInfo']
            student_assessment = datasets['studentAssessment']
            
            # Simple merge on student ID only
            merged_data = student_assessment.merge(
                student_info, 
                on='id_student', 
                how='left'
            )
            
            print(f"✅ Simple merged dataset created: {len(merged_data):,} rows")
            return merged_data
    
    def analyze_sampled_data(self, datasets):
        """
        Analyze the sampled dataset
        Args:
            datasets: Dictionary of sampled datasets
        """
        print("\n📊 Analyzing sampled dataset...")
        
        student_info = datasets['studentInfo']
        
        # Basic statistics
        print(f"📈 Dataset Statistics:")
        print(f"   Total students: {len(student_info['id_student'].unique()):,}")
        print(f"   Total courses: {len(student_info['code_module'].unique())}")
        print(f"   Total presentations: {len(student_info['code_presentation'].unique())}")
        
        # Course distribution
        print(f"\n📚 Course Distribution:")
        course_counts = student_info['code_module'].value_counts()
        for course, count in course_counts.items():
            print(f"   {course}: {count:,} students")
        
        # Final result distribution
        print(f"\n🎯 Final Result Distribution:")
        result_counts = student_info['final_result'].value_counts()
        for result, count in result_counts.items():
            percentage = (count / len(student_info)) * 100
            print(f"   {result}: {count:,} students ({percentage:.1f}%)")
        
        # Gender distribution
        print(f"\n👥 Gender Distribution:")
        gender_counts = student_info['gender'].value_counts()
        for gender, count in gender_counts.items():
            percentage = (count / len(student_info)) * 100
            print(f"   {gender}: {count:,} students ({percentage:.1f}%)")
        
        # Age distribution
        print(f"\n📅 Age Distribution:")
        age_counts = student_info['age_band'].value_counts()
        for age, count in age_counts.items():
            percentage = (count / len(student_info)) * 100
            print(f"   {age}: {count:,} students ({percentage:.1f}%)")
        
        return {
            'course_distribution': course_counts,
            'result_distribution': result_counts,
            'gender_distribution': gender_counts,
            'age_distribution': age_counts
        }

def create_oulad_sample():
    """
    Main function to create a sampled OULAD dataset
    """
    print("🚀 OULAD Dataset Sampling Tool")
    print("="*50)
    
    # Initialize processor
    processor = OULADDataProcessor()
    
    # Load and sample data
    datasets = processor.load_and_sample_data()
    
    if datasets is None:
        print("❌ Failed to load data")
        return None
    
    # Analyze sampled data
    analysis = processor.analyze_sampled_data(datasets)
    
    # Save sampled data
    info = processor.save_sampled_data(datasets)
    
    # Create merged dataset
    merged_data = processor.create_merged_dataset(datasets)
    
    # Save merged dataset
    merged_data.to_csv("oulad_sampled/oulad_merged.csv", index=False)
    print(f"✅ Merged dataset saved: oulad_sampled/oulad_merged.csv")
    
    print("\n🎉 OULAD sampling completed successfully!")
    print(f"📊 Dataset reduced to {info['total_students']:,} students (25% of original)")
    
    return {
        'datasets': datasets,
        'merged_data': merged_data,
        'analysis': analysis,
        'info': info
    }

if __name__ == "__main__":
    # Run the sampling process
    result = create_oulad_sample() 