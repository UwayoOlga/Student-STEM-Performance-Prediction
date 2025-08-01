# Student STEM Performance Prediction Project

## Project Overview
This project analyzes student performance in STEM (Science, Technology, Engineering, Mathematics) subjects using the Open University Learning Analytics Dataset (OULAD). The goal is to predict student success and excellence in STEM education using machine learning techniques.

## Dataset Overview
- **Original Dataset**: 28,784 unique students
- **Sampling Method**: Simple Random Sampling (SRS) without replacement
- **Sampled Dataset**: 7,196 students
- **STEM Students**: 2,722 students (37.8% of sampled data)
- **Source**: Open University Learning Analytics Dataset (OULAD)

## Key Findings

### STEM Success vs Risk Paradox
A notable finding reveals an apparent contradiction in STEM performance analysis:
- **STEM subjects show high success rates** (80-85% for Computing, 60-65% for Engineering)
- **Individual STEM students are classified as "High Risk"** by the prediction model
- **Explanation**: This suggests STEM subjects are selective - only confident, high-performing students choose them, leading to high overall success rates despite individual risk factors
- **Implication**: The risk prediction model may need STEM-specific adjustments, as success rate and individual risk measure different aspects of performance 