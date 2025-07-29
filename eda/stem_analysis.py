#!/usr/bin/env python3
"""
STEM Performance Prediction Analysis
Uses known OULAD module codes to identify and analyze STEM courses only
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

class STEMAnalysis:
    def __init__(self, data_path="oulad_cleaned"):
        self.data_path = data_path
        self.datasets = {}
        self.stem_modules = {}
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
        """Load all cleaned datasets"""
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
    
    def add_subject_labels(self):
        """Add subject area labels to datasets"""
        print("Adding subject area labels...")
        
        # Add subject labels to studentInfo
        self.datasets['studentInfo']['subject_area'] = self.datasets['studentInfo']['code_module'].map(self.stem_modules)
        self.datasets['studentInfo']['is_stem'] = self.datasets['studentInfo']['code_module'].isin(self.stem_modules.keys())
        
        # Add subject labels to merged dataset
        self.datasets['merged']['subject_area'] = self.datasets['merged']['code_module'].map(self.stem_modules)
        self.datasets['merged']['is_stem'] = self.datasets['merged']['code_module'].isin(self.stem_modules.keys())
        
        print("Subject labels added successfully!")
    
    def create_stem_target_variable(self):
        """Create target variable for STEM courses only"""
        print("Creating STEM target variable...")
        
        df = self.datasets['studentInfo'].copy()
        
        # Filter for STEM courses only
        stem_df = df[df['is_stem'] == True].copy()
        
        # Create binary target: Success (Distinction/Pass) vs Failure (Fail/Withdrawn)
        # Based on the encoded values we saw earlier
        stem_df['success'] = np.where(
            (stem_df['final_result'] == 0.08374058178886125) | (stem_df['final_result'] == -2.0289591117016377),
            1,  # Success (Pass or Distinction)
            0   # Failure (Fail or Withdrawn)
        )
        
        # Create STEM excellence target (Distinction only)
        stem_df['stem_excellence'] = np.where(
            stem_df['final_result'] == -2.0289591117016377,
            1,  # STEM Excellence (Distinction)
            0   # Not Excellence
        )
        
        # Print STEM target distribution
        success_dist = stem_df['success'].value_counts()
        excellence_dist = stem_df['stem_excellence'].value_counts()
        
        print("STEM Success Distribution:")
        print(f"  Success (1): {success_dist.get(1, 0):,} students ({success_dist.get(1, 0)/len(stem_df)*100:.1f}%)")
        print(f"  Failure (0): {success_dist.get(0, 0):,} students ({success_dist.get(0, 0)/len(stem_df)*100:.1f}%)")
        
        print("STEM Excellence Distribution:")
        print(f"  Excellence (1): {excellence_dist.get(1, 0):,} students ({excellence_dist.get(1, 0)/len(stem_df)*100:.1f}%)")
        print(f"  Not Excellence (0): {excellence_dist.get(0, 0):,} students ({excellence_dist.get(0, 0)/len(stem_df)*100:.1f}%)")
        
        # Update the dataset
        self.datasets['studentInfo'] = df
        self.datasets['stem_students'] = stem_df
        
        return stem_df
    
    def analyze_stem_performance(self):
        """Analyze STEM performance patterns"""
        print("STEM Performance Analysis")
        print("=" * 60)
        
        stem_df = self.datasets['stem_students']
        
        print(f"Total STEM Students: {len(stem_df):,}")
        print(f"STEM Courses Available: {len(stem_df['code_module'].unique())}")
        
        # STEM course distribution
        print("STEM Course Distribution:")
        course_dist = stem_df['code_module'].value_counts()
        for course, count in course_dist.items():
            subject = self.stem_modules.get(course, 'Unknown')
            percentage = count / len(stem_df) * 100
            print(f"  {course} ({subject}): {count:,} students ({percentage:.1f}%)")
        
        # Success rate by STEM subject
        if 'success' in stem_df.columns:
            print("STEM Success Rate by Subject:")
            success_by_subject = stem_df.groupby('subject_area')['success'].agg(['count', 'mean'])
            for subject, row in success_by_subject.iterrows():
                success_rate = row['mean'] * 100
                print(f"  {subject}: {row['count']:,} students, {success_rate:.1f}% success rate")
        
        # Excellence rate by STEM subject
        if 'stem_excellence' in stem_df.columns:
            print("STEM Excellence Rate by Subject:")
            excellence_by_subject = stem_df.groupby('subject_area')['stem_excellence'].agg(['count', 'mean'])
            for subject, row in excellence_by_subject.iterrows():
                excellence_rate = row['mean'] * 100
                print(f"  {subject}: {row['count']:,} students, {excellence_rate:.1f}% excellence rate")
    
    def create_stem_visualizations(self):
        """Create STEM-specific visualizations"""
        print("Creating STEM visualizations...")
        
        stem_df = self.datasets['stem_students']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('STEM Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. STEM Success vs Failure
        if 'success' in stem_df.columns:
            success_counts = stem_df['success'].value_counts()
            axes[0, 0].pie(success_counts.values, labels=['Failure', 'Success'], autopct='%1.1f%%')
            axes[0, 0].set_title('STEM Success vs Failure')
        
        # 2. STEM Excellence Distribution
        if 'stem_excellence' in stem_df.columns:
            excellence_counts = stem_df['stem_excellence'].value_counts()
            axes[0, 1].pie(excellence_counts.values, labels=['Not Excellence', 'Excellence'], autopct='%1.1f%%')
            axes[0, 1].set_title('STEM Excellence Distribution')
        
        # 3. STEM Subject Distribution
        subject_counts = stem_df['subject_area'].value_counts()
        axes[0, 2].bar(range(len(subject_counts)), subject_counts.values)
        axes[0, 2].set_title('STEM Subject Distribution')
        axes[0, 2].set_ylabel('Number of Students')
        axes[0, 2].set_xticks(range(len(subject_counts)))
        axes[0, 2].set_xticklabels(subject_counts.index, rotation=45)
        
        # 4. Success Rate by STEM Subject
        if 'success' in stem_df.columns:
            success_by_subject = stem_df.groupby('subject_area')['success'].mean()
            axes[1, 0].bar(range(len(success_by_subject)), success_by_subject.values)
            axes[1, 0].set_title('Success Rate by STEM Subject')
            axes[1, 0].set_ylabel('Success Rate')
            axes[1, 0].set_xticks(range(len(success_by_subject)))
            axes[1, 0].set_xticklabels(success_by_subject.index, rotation=45)
            axes[1, 0].set_ylim(0, 1)
        
        # 5. Excellence Rate by STEM Subject
        if 'stem_excellence' in stem_df.columns:
            excellence_by_subject = stem_df.groupby('subject_area')['stem_excellence'].mean()
            axes[1, 1].bar(range(len(excellence_by_subject)), excellence_by_subject.values)
            axes[1, 1].set_title('Excellence Rate by STEM Subject')
            axes[1, 1].set_ylabel('Excellence Rate')
            axes[1, 1].set_xticks(range(len(excellence_by_subject)))
            axes[1, 1].set_xticklabels(excellence_by_subject.index, rotation=45)
            axes[1, 1].set_ylim(0, 1)
        
        # 6. Gender Distribution in STEM
        if 'gender' in stem_df.columns:
            gender_counts = stem_df['gender'].value_counts()
            axes[1, 2].bar(['Female', 'Male'], gender_counts.values)
            axes[1, 2].set_title('Gender Distribution in STEM')
            axes[1, 2].set_ylabel('Number of Students')
        
        plt.tight_layout()
        plt.savefig('stem_performance_analysis.png', dpi=300, bbox_inches='tight')
        print("STEM visualizations saved as 'stem_performance_analysis.png'")
    
    def generate_stem_summary(self):
        """Generate STEM-specific summary report"""
        print("STEM Summary Report")
        print("=" * 60)
        
        stem_df = self.datasets['stem_students']
        total_students = len(stem_df)
        
        print("STEM Dataset Overview:")
        print(f"  Total STEM Students: {total_students:,}")
        print(f"  STEM Courses: {len(stem_df['code_module'].unique())}")
        
        if 'success' in stem_df.columns:
            success_rate = stem_df['success'].mean() * 100
            print(f"  Overall STEM Success Rate: {success_rate:.1f}%")
        
        if 'stem_excellence' in stem_df.columns:
            excellence_rate = stem_df['stem_excellence'].mean() * 100
            print(f"  Overall STEM Excellence Rate: {excellence_rate:.1f}%")
        
        print("STEM Courses Available:")
        for code, subject in self.stem_modules.items():
            if code in stem_df['code_module'].unique():
                count = len(stem_df[stem_df['code_module'] == code])
                print(f"  {code} ({subject}): {count:,} students")
        
        print("Files Generated:")
        print(f"  stem_performance_analysis.png - STEM-specific visualizations")
        
        print("STEM analysis completed successfully!")

def main():
    """Main function to run STEM analysis"""
    print("STEM Performance Prediction Analysis")
    print("=" * 60)
    
    try:
        # Initialize STEM analysis
        stem_analysis = STEMAnalysis()
        
        # Add subject labels
        stem_analysis.add_subject_labels()
        
        # Create STEM target variable
        stem_analysis.create_stem_target_variable()
        
        # Analyze STEM performance
        stem_analysis.analyze_stem_performance()
        
        # Create STEM visualizations
        stem_analysis.create_stem_visualizations()
        
        # Generate STEM summary
        stem_analysis.generate_stem_summary()
        
    except Exception as e:
        print(f"Error during STEM analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 