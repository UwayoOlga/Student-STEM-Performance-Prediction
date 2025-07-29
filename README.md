# Predicting Student Success in STEM Subjects
## Big Data Analytics Capstone Project

### Project Overview
This project analyzes student performance data to predict which students are more likely to excel in STEM subjects (Mathematics) based on academic performance and socio-demographic features.

### Problem Statement
**Can we predict which students are likely to succeed in STEM subjects (like Math) based on academic performance and socio-demographic features?**

This analysis aims to:
- Identify key factors that influence student success in mathematics
- Build predictive models to classify students as likely to succeed or struggle
- Provide insights for educational interventions and support programs

### Dataset Information
- **Dataset Title:** Student Performance Dataset (Math and Portuguese)
- **Source:** UCI Machine Learning Repository
- **Structure:** 
  - `student-mat.csv`: 395 students, 33 features (Mathematics performance)
  - `student-por.csv`: 649 students, 33 features (Portuguese performance)
- **Data Type:** Structured (CSV)
- **Status:** Requires preprocessing (encoding, scaling, feature engineering)

### Key Features
- **Demographic:** Age, Gender, Address
- **Family Background:** Parental education, jobs, family size, support
- **Academic:** Study time, previous failures, absences
- **Social:** Activities, internet access, romantic relationships
- **Performance:** G1 (first period), G2 (second period), G3 (final grade)

### Project Structure
```
├── data/                    # Raw datasets
│   ├── student-mat.csv
│   └── student-por.csv
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/                    # Python modules
│   ├── data_processing.py
│   ├── visualization.py
│   └── modeling.py
├── dashboards/            # Power BI files
│   └── student_performance_dashboard.pbix
├── reports/               # Generated reports
│   └── project_presentation.pptx
└── README.md
```

### Methodology

#### 1. Data Preprocessing
- Handle missing values and outliers
- Encode categorical variables
- Scale numerical features
- Feature engineering (create success indicators)

#### 2. Exploratory Data Analysis
- Distribution analysis of grades
- Correlation analysis between features
- Gender-based performance comparison
- Family background impact analysis

#### 3. Machine Learning Models
- **Classification Models:**
  - Logistic Regression
  - Random Forest
  - Support Vector Machine
  - Gradient Boosting
- **Target Variable:** Success in STEM (G3 ≥ 12)

#### 4. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Feature Importance Analysis
- Cross-validation

#### 5. Power BI Dashboard
- Interactive visualizations
- Performance trends by demographics
- Success prediction insights
- Filtering and drill-down capabilities

### Innovation Features
- **Ensemble Methods:** Combining multiple models for better prediction
- **SHAP Values:** Explainable AI for model interpretability
- **Custom Success Score:** Weighted combination of academic and social factors
- **Advanced Visualizations:** Interactive charts and dynamic filtering

### Results & Insights
[To be populated after analysis]

### Recommendations
[To be populated after analysis]

### Future Work
- Expand to other STEM subjects
- Real-time prediction system
- Integration with school management systems
- Longitudinal study tracking

### Technologies Used
- **Python:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Power BI:** Data visualization and dashboard creation
- **Jupyter Notebooks:** Interactive analysis and documentation

### Installation & Setup
```bash
# Clone the repository
git clone [repository-url]

# Install required packages
pip install -r requirements.txt

# Run Jupyter notebooks
jupyter notebook notebooks/
```

### Contact
- **Student:** [Your Name]
- **Course:** INSY 8413 - Introduction to Big Data Analytics
- **Instructor:** Eric Maniraguha

---
*This project is part of the Big Data Analytics Capstone Project for the Faculty of Information Technology, Academic Year 2024-2025, Semester III.* 
