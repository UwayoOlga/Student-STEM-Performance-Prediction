#!/usr/bin/env python3
"""
OULAD Data Cleaning Script
Runs the complete data cleaning pipeline for the sampled OULAD dataset
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from data_cleaner import OULADDataCleaner

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
    exit_code = main()
    sys.exit(exit_code) 