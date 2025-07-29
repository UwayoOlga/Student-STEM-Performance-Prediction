# OULAD STEM Performance Prediction Project

## Project Overview
This project analyzes the Open University Learning Analytics Dataset (OULAD) to predict which students are more likely to excel in STEM subjects. The analysis focuses on identifying patterns and factors that contribute to academic excellence in Science, Technology, Engineering, and Mathematics courses.

## Project Structure

```
Student-STEM-Performance-Prediction/
├── cleaning/                    # Data cleaning scripts and modules
│   ├── data_cleaner.py         # Main data cleaning script
│   ├── clean_oulad_data.py     # Cleaning pipeline runner
│   └── src/                    # Cleaning utility modules
├── eda/                        # Exploratory Data Analysis
│   ├── exploratory_analysis.py # General EDA script
│   ├── stem_analysis.py        # STEM-specific analysis
│   └── *.png                   # Generated visualizations
├── oulad_sampled/              # Sampled dataset (25% of original)
├── oulad_cleaned/              # Cleaned and processed data
├── create_oulad_sample.py      # Data sampling script
├── main.py                     # Main pipeline runner
└── README.md                   # This file
```

## STEM Courses Identified
Based on known OULAD module mappings:
- **AAA**: Computing and IT
- **FFF**: Science
- **GGG**: Engineering and Technology
- **HHH**: Mathematics and Statistics

## Key Findings

### Dataset Overview
- **Total Students**: 8,149 (sampled from 32,593)
- **STEM Students**: 2,722 (33.4% of total)
- **STEM Excellence Rate**: Varies by subject
- **Assessment per Student**: 5.3

### STEM Performance Distribution
- **Science (FFF)**: 1,928 students (70.8%)
- **Engineering (GGG)**: 613 students (22.5%)
- **Computing (AAA)**: 181 students (6.6%)

## Usage

### Run Complete Pipeline
```bash
python main.py
```

### Run Individual Steps
1. **Data Sampling**:
   ```bash
   python create_oulad_sample.py
   ```

2. **Data Cleaning**:
   ```bash
   python cleaning/data_cleaner.py
   ```

3. **Exploratory Analysis**:
   ```bash
   python eda/exploratory_analysis.py
   ```

4. **STEM Analysis**:
   ```bash
   python eda/stem_analysis.py
   ```

## Generated Files

### Visualizations (in `eda/` folder)
- `oulad_eda_basic.png` - Basic distributions and patterns
- `oulad_correlation_heatmap.png` - Feature correlations
- `oulad_detailed_analysis.png` - Detailed performance analysis
- `stem_performance_analysis.png` - STEM-specific visualizations

### Data Files
- `oulad_sampled/` - 25% sample of original dataset
- `oulad_cleaned/` - Processed and cleaned data ready for analysis

## Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Research Question
"Can data predict which students are more likely to excel in STEM subjects?"

This project provides comprehensive analysis to answer this question through:
1. Data preprocessing and cleaning
2. Exploratory data analysis
3. STEM-specific performance analysis
4. Pattern identification for academic excellence prediction

## Methodology
1. **Data Sampling**: Reduced large dataset to manageable size (25% sample)
2. **Data Cleaning**: Handled missing values, outliers, and data transformations
3. **Feature Engineering**: Created target variables for success and excellence
4. **Exploratory Analysis**: Identified patterns and relationships
5. **STEM Focus**: Analyzed performance specifically in STEM subjects

## Results
The analysis reveals key factors that predict STEM excellence, including:
- Demographic characteristics
- Academic background
- Study patterns
- Course-specific performance indicators

These insights can inform educational interventions and support systems for STEM students.
