#!/usr/bin/env python3
"""
Main script for OULAD STEM Performance Prediction Project
Runs the complete data processing pipeline
"""

import sys
import os

def main():
    """
    Main function to run the complete OULAD analysis pipeline
    """
    print("OULAD STEM Performance Prediction Project")
    print("=" * 60)
    
    # Step 1: Data Sampling
    print("Step 1: Creating sampled dataset...")
    try:
        from create_oulad_sample import main as sample_main
        sample_main()
        print("Data sampling completed successfully!")
    except Exception as e:
        print(f"Error in data sampling: {e}")
        return 1
    
    # Step 2: Data Cleaning
    print("\nStep 2: Cleaning data...")
    try:
        sys.path.append('cleaning')
        from data_cleaner import main as clean_main
        clean_main()
        print("Data cleaning completed successfully!")
    except Exception as e:
        print(f"Error in data cleaning: {e}")
        return 1
    
    # Step 3: Exploratory Data Analysis
    print("\nStep 3: Running exploratory data analysis...")
    try:
        sys.path.append('eda')
        from exploratory_analysis import main as eda_main
        eda_main()
        print("EDA completed successfully!")
    except Exception as e:
        print(f"Error in EDA: {e}")
        return 1
    
    # Step 4: STEM Analysis
    print("\nStep 4: Running STEM-specific analysis...")
    try:
        from stem_analysis import main as stem_main
        stem_main()
        print("STEM analysis completed successfully!")
    except Exception as e:
        print(f"Error in STEM analysis: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("All analysis steps completed successfully!")
    print("Check the 'eda' folder for generated visualizations and reports.")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 