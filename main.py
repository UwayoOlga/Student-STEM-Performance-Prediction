#!/usr/bin/env python3
"""
Comprehensive Pipeline for OULAD STEM Performance Prediction Project
Orchestrates the complete data processing, analysis, and model training workflow
"""

import sys
import os
import subprocess
import time
from datetime import datetime

def print_step_header(step_number, step_name):
    """Print formatted step header"""
    print(f"\n{'='*80}")
    print(f"STEP {step_number}: {step_name}")
    print(f"{'='*80}")

def print_step_completion(step_name, success=True):
    """Print step completion message"""
    status = "COMPLETED SUCCESSFULLY" if success else "FAILED"
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] {step_name}: {status}")
    print("-" * 60)

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print(f"Running {description}...")
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        execution_time = time.time() - start_time
        print(f"Output: {result.stdout}")
        print(f"Execution time: {execution_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in {description}:")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Script not found: {script_path}")
        return False

def run_module_script(module_path, script_name, description):
    """Run a script from a module directory"""
    script_path = os.path.join(module_path, script_name)
    return run_script(script_path, description)

def main():
    """
    Main function to orchestrate the complete OULAD analysis pipeline
    """
    print("OULAD STEM Performance Prediction Project")
    print("Comprehensive Pipeline Orchestration")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    pipeline_steps = [
        {
            "number": 1,
            "name": "Data Sampling and Preparation",
            "script": "create_oulad_sample.py",
            "description": "Creating sampled dataset from OULAD"
        },
        {
            "number": 2,
            "name": "Data Cleaning and Preprocessing",
            "script": "cleaning/data_cleaner.py",
            "description": "Cleaning and preprocessing datasets"
        },
        {
            "number": 3,
            "name": "Exploratory Data Analysis",
            "script": "eda/exploratory_analysis.py",
            "description": "Comprehensive exploratory data analysis"
        },
        {
            "number": 4,
            "name": "STEM-Specific Analysis",
            "script": "eda/stem_analysis.py",
            "description": "STEM-focused analysis and insights"
        },
        {
            "number": 5,
            "name": "Machine Learning Model Training",
            "script": "ml_models/scripts/ml_models_innovated.py",
            "description": "Training ensemble ML models for STEM prediction"
        },
        {
            "number": 6,
            "name": "Model Evaluation and Visualization",
            "script": "ml_models/scripts/model_evaluation.py",
            "description": "Comprehensive model evaluation and performance analysis"
        },
        {
            "number": 7,
            "name": "Power BI Data Preparation",
            "script": "create_powerbi_files.py",
            "description": "Generating Power BI dashboard data files"
        }
    ]
    
    successful_steps = 0
    total_steps = len(pipeline_steps)
    
    for step in pipeline_steps:
        print_step_header(step["number"], step["name"])
        
        # Run the script
        if step["number"] == 2 or step["number"] == 3 or step["number"] == 4:
            # These are module scripts
            module_path = step["script"].split('/')[0]
            script_name = step["script"].split('/')[1]
            success = run_module_script(module_path, script_name, step["description"])
        elif step["number"] == 5 or step["number"] == 6:
            # These are ML model scripts
            module_path = step["script"].split('/')[0] + '/' + step["script"].split('/')[1]
            script_name = step["script"].split('/')[2]
            success = run_module_script(module_path, script_name, step["description"])
        else:
            # These are root-level scripts
            success = run_script(step["script"], step["description"])
        
        print_step_completion(step["name"], success)
        
        if success:
            successful_steps += 1
        else:
            print(f"\nPipeline stopped at Step {step['number']} due to error.")
            print("Please fix the issue and restart the pipeline.")
            return 1
    
    # Pipeline completion summary
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Steps: {total_steps}")
    print(f"Successful Steps: {successful_steps}")
    print(f"Failed Steps: {total_steps - successful_steps}")
    print(f"Success Rate: {(successful_steps/total_steps)*100:.1f}%")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if successful_steps == total_steps:
        print("\nAll pipeline steps completed successfully!")
        print("\nGenerated Outputs:")
        print("- Sampled dataset: oulad_sampled/")
        print("- Cleaned data: oulad_cleaned/")
        print("- EDA visualizations: eda/")
        print("- ML model results: ml_models/results/")
        print("- Model visualizations: ml_models/visualizations/")
        print("- Power BI files: powerbi/")
        print("\nNext Steps:")
        print("1. Review generated visualizations and reports")
        print("2. Import Power BI files into your dashboard")
        print("3. Analyze model performance and insights")
        return 0
    else:
        print(f"\nPipeline completed with {total_steps - successful_steps} failed steps.")
        print("Please review error messages and re-run failed steps.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 