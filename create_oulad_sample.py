#!/usr/bin/env python3
"""
Script to create a sampled OULAD dataset (25% of students)
Run this script to reduce the large OULAD dataset to a manageable size
"""

import sys
import os

# Add src directory to path
sys.path.append('src')

def main():
    """
    Main function to create the sampled OULAD dataset
    """
    print(" Creating Sampled OULAD Dataset")
    print("="*50)
    
    try:
        # Import the OULAD processor
        from oulad_data_processor import create_oulad_sample
        
        # Run the sampling process
        result = create_oulad_sample()
        
        if result is not None:
            print("\Success! Your sampled dataset is ready.")
            print("\n📁 Files created:")
            print("   oulad_sampled/")
            print("   ├── studentInfo.csv")
            print("   ├── studentAssessment.csv")
            print("   ├── studentRegistration.csv")
            print("   ├── studentVle.csv")
            print("   ├── assessments.csv")
            print("   ├── courses.csv")
            print("   ├── vle.csv")
            print("   ├── oulad_merged.csv")
            print("   └── dataset_info.txt")
            
            print(f"\n📊 Dataset Summary:")
            print(f"   Students: {result['info']['total_students']:,}")
            print(f"   Assessments: {result['info']['total_assessments']:,}")
            print(f"   VLE Interactions: {result['info']['total_vle_interactions']:,}")
            print(f"   Courses: {result['info']['courses']}")
            print(f"   Presentations: {result['info']['presentations']}")
            
            print("\n Next steps:")
            print("1. Run 'python test_oulad_analysis.py' to test the sampled data")
            print("2. Run 'python main_oulad_analysis.py' for complete analysis")
            print("3. Open 'notebooks/oulad_exploration.ipynb' for detailed exploration")
            
        else:
            print("Failed to create sampled dataset")
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you have all required packages installed:")
        print("pip install -r requirements.txt")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check that the OULAD dataset files are in the correct location")

if __name__ == "__main__":
    main() 