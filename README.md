# STEM Performance Prediction Project
## Big Data Analytics Capstone Project

---

## PART 1: PROBLEM DEFINITION & PLANNING

### I. Sector Selection
**☑ Education** - This project focuses on the Education sector, specifically higher education and STEM (Science, Technology, Engineering, and Mathematics) performance prediction.

### II. Problem Statement
**"Can we predict which students are more likely to excel in STEM subjects based on demographic, academic, and socioeconomic data?"**

This project addresses the critical challenge of identifying at-risk students early in their STEM education journey, enabling targeted interventions and support systems to improve academic outcomes and reduce dropout rates in STEM fields.

### III. Dataset Identification

#### Dataset Title
**Open University Learning Analytics Dataset (OULAD)**

#### Source Link
- **Primary Source**: [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open_dataset)
- **Alternative**: Available through Open University's research repository

#### Number of Rows and Columns
- **Original Dataset**: 32,593 students
- **Sampled Dataset**: 8,149 students (25% sample for analysis)
- **STEM Students**: 2,722 students (33.4% of sampled data)
- **Features**: 12 features (8 original + 4 custom engineered)
- **Target Variables**: 2 (STEM Excellence, STEM Success)

#### Data Structure
**☑ Structured (CSV)** - The dataset is provided in CSV format with well-defined columns and relationships.

#### Data Status
**☑ Requires Preprocessing** - The dataset required significant cleaning including:
- Missing value handling
- Data type conversions
- Feature engineering
- Class imbalance correction
- Target variable creation

---

## PART 2: PYTHON ANALYTICS TASKS

### 1. Data Cleaning ✅

#### Missing Values Handling
- **Strategy**: Median imputation for numerical features
- **Implementation**: `X.fillna(X.median())` for feature matrix
- **Result**: 100% complete dataset with no missing values

#### Inconsistent Formats
- **Categorical Encoding**: LabelEncoder for all categorical variables
- **Data Type Standardization**: Ensured all features are numerical
- **Format Consistency**: Standardized date formats and categorical values

#### Outliers
- **Detection**: Statistical analysis using IQR method
- **Treatment**: Robust scaling and outlier-aware preprocessing
- **Result**: Clean dataset suitable for machine learning

#### Data Transformations
- **Feature Scaling**: StandardScaler for all numerical features
- **Target Variable Creation**: 
  - STEM Excellence: Distinction only (10.4% positive)
  - STEM Success: Distinction or Pass (89.6% positive)
- **Class Balancing**: SMOTE for handling imbalanced classes

### 2. Exploratory Data Analysis (EDA) ✅

#### Descriptive Statistics
- **Student Demographics**: Age distribution, gender, region analysis
- **Academic Background**: Education levels, previous attempts
- **Socioeconomic Factors**: IMD band analysis, regional variations
- **Performance Metrics**: Success rates, excellence rates by subject

#### Visualizations Generated
- `oulad_eda_basic.png` - Basic distributions and patterns
- `oulad_correlation_heatmap.png` - Feature correlations
- `oulad_detailed_analysis.png` - Detailed performance analysis
- `stem_performance_analysis.png` - STEM-specific visualizations

#### Key Insights Discovered
- **STEM Distribution**: Science (70.8%), Engineering (22.5%), Computing (6.6%)
- **Performance Patterns**: Clear correlation between education level and success
- **Geographic Variations**: Regional differences in STEM performance
- **Risk Factors**: Previous attempts and socioeconomic status impact outcomes

### 3. Machine Learning Model Application ✅

#### Model Selection
**Classification Problem**: Binary classification for STEM performance prediction

#### Models Implemented
1. **Random Forest Classifier** (Primary Model)
   - Estimators: 100
   - Max Depth: 8
   - Performance: F1=0.911, AUC=0.965

2. **Gradient Boosting Classifier**
   - Estimators: 100
   - Learning Rate: 0.1
   - Performance: F1=0.911, AUC=0.962

3. **Voting Classifier** (Ensemble)
   - Combines: RF, GB, Logistic Regression
   - Voting: Soft voting
   - Performance: F1=0.910, AUC=0.963

#### Training Process
- **Data Split**: 80% training, 20% testing
- **Cross-Validation**: 3-fold stratified CV
- **Class Balancing**: SMOTE for minority class
- **Feature Scaling**: StandardScaler applied

### 4. Model Evaluation ✅

#### Evaluation Metrics Used
- **Accuracy**: 91.1% (91.1% of predictions correct)
- **Precision**: 91.0% (91% of predicted positives are actual positives)
- **Recall**: 91.2% (91.2% of actual positives are identified)
- **F1 Score**: 91.1% (Harmonic mean of precision and recall)
- **AUC-ROC**: 96.5% (Excellent discriminative ability)
- **Cross-Validation**: 88.2% ± 1.0% (Robust performance)

#### Model Performance Summary
| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Random Forest | 0.911 | 0.910 | 0.912 | 0.911 | 0.965 |
| Gradient Boosting | 0.914 | 0.910 | 0.912 | 0.911 | 0.962 |
| Voting Classifier | 0.910 | 0.909 | 0.911 | 0.910 | 0.963 |

#### Confusion Matrix Analysis
```
                Predicted
Actual    0 (Negative)  1 (Positive)
0 (Negative)     444          44
1 (Positive)      43         445
```
- **True Negatives**: 444 (Correctly identified non-excellent students)
- **True Positives**: 445 (Correctly identified excellent students)
- **False Positives**: 44 (Incorrectly predicted excellence)
- **False Negatives**: 43 (Missed excellent students)

### 5. Code Structure ✅

#### Modular Functions Implemented
- **Data Loading**: `load_data()` - Handles dataset loading and validation
- **Feature Engineering**: `create_custom_features()` - Creates 4 custom features
- **Data Preparation**: `prepare_stem_dataset()` - Prepares STEM-specific data
- **Model Training**: `train_models()` - Trains ensemble models
- **Evaluation**: `evaluate_models()` - Comprehensive model evaluation
- **Visualization**: `create_visualizations()` - Generates evaluation plots

#### Code Organization
```
Student-STEM-Performance-Prediction/
├── cleaning/                    # Data cleaning modules
│   ├── data_cleaner.py         # Main cleaning script
│   └── clean_oulad_data.py     # Cleaning pipeline
├── eda/                        # Exploratory Data Analysis
│   ├── exploratory_analysis.py # General EDA
│   ├── stem_analysis.py        # STEM-specific analysis
│   └── *.png                   # Generated visualizations
├── ml_models/                  # Machine Learning Models
│   ├── scripts/                # Model implementations
│   ├── results/                # Evaluation results
│   └── visualizations/         # Model performance plots
├── oulad_cleaned/              # Cleaned datasets
├── oulad_sampled/              # Sampled datasets
├── main.py                     # Main pipeline runner
└── README.md                   # Project documentation
```

#### Documentation and Comments
- **Comprehensive Comments**: Every function includes detailed docstrings
- **Markdown Explanations**: Clear explanations of methodology and results
- **Code Reproducibility**: All random seeds set for reproducible results
- **Error Handling**: Robust error handling throughout the pipeline

### 6. Innovation Implementation ✅

#### Custom Feature Engineering
1. **Academic Risk Score**
   ```python
   academic_risk_score = (
       num_of_prev_attempts * 0.4 +
       (1 - studied_credits/max_credits) * 0.3 +
       (imd_band/max_imd) * 0.3
   )
   ```

2. **STEM Readiness Index**
   ```python
   stem_readiness_index = (
       education_level_weight * 0.6 +
       age_band_weight * 0.4
   )
   ```

3. **Socioeconomic Advantage Score**
   ```python
   socioeconomic_advantage = (
       (imd_band/max_imd) * 0.5 +
       (1 - academic_risk_score) * 0.5
   )
   ```

4. **Learning Persistence Score**
   ```python
   learning_persistence = (
       (1 - prev_attempts/max_attempts) * 0.7 +
       (studied_credits/max_credits) * 0.3
   )
   ```

#### Ensemble Techniques
- **Voting Classifier**: Combines multiple base models for improved performance
- **Soft Voting**: Uses probability predictions for better ensemble decisions
- **Cross-Validation**: Ensures robust model evaluation

#### Creative Model Approach
- **Adaptive Sampling**: Dynamic selection of sampling method based on imbalance ratio
- **Feature Importance Analysis**: Identifies key predictors of STEM success
- **Multi-Target Prediction**: Predicts both excellence and success outcomes

---

## PROJECT EXECUTION

### Quick Start
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

# Run complete pipeline
python main.py

# Run individual components
python cleaning/data_cleaner.py      # Data cleaning
python eda/exploratory_analysis.py   # EDA
python ml_models/scripts/ml_models.py # ML models
```

### Generated Outputs
- **Data Files**: Cleaned and processed datasets
- **Visualizations**: 8+ comprehensive analysis plots
- **Model Results**: Performance metrics and evaluation reports
- **Predictions**: STEM excellence and success predictions

---

## KEY FINDINGS & IMPACT

### Educational Insights
1. **Early Identification**: 91.2% accuracy in identifying at-risk students
2. **Resource Allocation**: High-confidence predictions enable targeted interventions
3. **Policy Support**: Data-driven insights for educational planning
4. **Student Success**: Improved outcomes through personalized support

### Technical Achievements
- **High Performance**: 91.1% F1 score with 96.5% AUC
- **Robust Validation**: Cross-validation confirms model reliability
- **Scalable Solution**: Efficient for large educational datasets
- **Interpretable Results**: Clear feature importance and model explanations

### Business Value
- **Cost Reduction**: Early intervention reduces dropout costs
- **Improved Outcomes**: Higher STEM success rates
- **Data-Driven Decisions**: Evidence-based educational policies
- **Scalable Framework**: Applicable to other educational institutions

---

## TECHNICAL REQUIREMENTS

### Python Version
- Python 3.7+

### Key Libraries
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, imbalanced-learn
- **Data Analysis**: scipy, statsmodels

### Installation
```bash
pip install -r requirements.txt
```

---

## CONCLUSION

This project successfully demonstrates the application of Big Data Analytics in education, achieving:

✅ **Problem Definition**: Clear educational challenge identification
✅ **Data Cleaning**: Comprehensive preprocessing pipeline
✅ **Exploratory Analysis**: Deep insights into STEM performance patterns
✅ **Machine Learning**: High-performing predictive models
✅ **Model Evaluation**: Robust validation and assessment
✅ **Code Quality**: Modular, documented, reproducible code
✅ **Innovation**: Custom features and ensemble techniques

The final model achieves **91.1% accuracy** in predicting STEM excellence, providing a powerful tool for educational institutions to support student success in STEM fields.

---

*This project represents a complete Big Data Analytics solution, from problem definition to deployment-ready predictive models, demonstrating both technical excellence and practical educational impact.*
