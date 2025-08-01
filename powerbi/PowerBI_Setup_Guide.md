# Power BI Dashboard Setup Guide
## STEM Performance Prediction Analysis

### **Dataset Information**
- **Total Students**: 7,196 unique students (25% sample of 28,784)
- **Total Records**: 8,149 (some students take multiple courses)
- **Subjects**: 8 subjects (4 STEM + 4 Non-STEM)
- **Success Rate**: 47.4%
- **Excellence Rate**: 9.4%

---

## **Step 1: Import Data**

### **Files to Import:**
1. `comprehensive_data.csv` - **ALL DATA IN ONE TABLE** (No relationships needed!)
2. `summary_stats.csv` - KPI metrics
3. `performance_by_subject.csv` - Subject-wise performance
4. `feature_importance.csv` - Feature importance rankings

### **Import Steps:**
1. Open Power BI Desktop
2. Click **Get Data** → **Text/CSV**
3. Select each CSV file from the `powerbi/` folder
4. Click **Load** for each file

---

## **Step 2: Create Data Model**

### **No Relationships Needed! 🎉**
Since we have all data in one comprehensive table, you don't need to create any relationships.

### **Create Calculated Columns (DAX) in comprehensive_data table:**

#### **In comprehensive_data table:**
```dax
STEM Subject = 
SWITCH(
    comprehensive_data[code_module],
    "AAA", "Computing and IT",
    "BBB", "Business and Management", 
    "CCC", "Creative Arts and Design",
    "DDD", "Education and Teaching",
    "EEE", "Health and Social Care",
    "FFF", "Science",
    "GGG", "Engineering and Technology",
    "HHH", "Mathematics and Statistics"
)
```

```dax
Risk Level = 
SWITCH(
    comprehensive_data[risk_level],
    "High Risk", "High Risk",
    "Medium Risk", "Medium Risk", 
    "Low Risk", "Low Risk",
    "Unknown"
)
```

```dax
Success Status = 
SWITCH(
    comprehensive_data[final_result],
    "Distinction", "Distinction",
    "Pass", "Pass",
    "Fail", "Fail",
    "Withdrawn", "Withdrawn"
)
```

---

## **Step 3: Create Multi-Page Dashboard**

### **Page 1: Executive Summary (Home)**

#### **Layout:**
```
┌─────────────────────────────────────────────────────────┐
│  STEM Performance Prediction Dashboard                  │
├─────────────────────────────────────────────────────────┤
│  [KPI Cards Row]                                        │
│  Total Students | Success Rate | Excellence Rate | Model│
│  [7,196]        | [47.4%]      | [9.4%]         | [91%]│
├─────────────────────────────────────────────────────────┤
│  [Charts Row]                                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │ Risk Dist.  │ │ Success by  │ │ Model Perf. │        │
│  │ (Donut)     │ │ Subject     │ │ (Gauge)     │        │
│  │             │ │ (Bar)       │ │             │        │
│  └─────────────┘ └─────────────┘ └─────────────┘        │
├─────────────────────────────────────────────────────────┤
│  [Navigation Buttons]                                   │
│  [← Previous] [Next →]                                  │
└─────────────────────────────────────────────────────────┘
```

#### **Visuals to Create:**

**1. KPI Cards:**
- **Total Students**: Card visual from `summary_stats[value]` where `summary_stats[metric] = "Total Students"`
- **Success Rate**: Card visual from `summary_stats[value]` where `summary_stats[metric] = "Success Rate (%)"`
- **Excellence Rate**: Card visual from `summary_stats[value]` where `summary_stats[metric] = "Excellence Rate (%)"`
- **Model Accuracy**: Card visual from `summary_stats[value]` where `summary_stats[metric] = "Model Accuracy (%)"`

**2. Risk Distribution (Donut Chart):**
- **Values**: `comprehensive_data[risk_level]` (Count)
- **Legend**: `comprehensive_data[risk_level]`
- **Colors**: Red (High Risk), Yellow (Medium Risk), Green (Low Risk)

**3. Success Rate by Subject (Bar Chart):**
- **Axis**: `performance_by_subject[subject]`
- **Values**: `performance_by_subject[success_rate]`
- **Color**: `performance_by_subject[is_stem]` (True = Blue, False = Gray)

**4. Model Performance (Gauge Chart):**
- **Value**: `summary_stats[value]` where `summary_stats[metric] = "Model Accuracy (%)"`
- **Min**: 0, **Max**: 100, **Target**: 90

**5. Global Filters (Slicers):**
- **Subject**: `comprehensive_data[subject_name]`
- **Risk Level**: `comprehensive_data[risk_level]`
- **Gender**: `comprehensive_data[gender]`

---

### **Page 2: Student Demographics**

#### **Layout:**
```
┌─────────────────────────────────────────────────────────┐
│  Student Demographics Analysis                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │ Age Dist.   │ │ Gender Dist.│ │ Region Dist.│        │
│  │ (Pie)       │ │ (Donut)     │ │ (Map)       │        │
│  └─────────────┘ └─────────────┘ └─────────────┘        │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │ Education   │ │ IMD Band    │ │ Disability  │        │
│  │ Level       │ │ (Bar)       │ │ (Bar)       │        │
│  │ (Bar)       │ │             │ │             │        │
│  └─────────────┘ └─────────────┘ └─────────────┘        │
├─────────────────────────────────────────────────────────┤
│  [Navigation Buttons]                                   │
└─────────────────────────────────────────────────────────┘
```

#### **Visuals to Create:**

**1. Age Distribution (Pie Chart):**
- **Values**: `studentInfo_for_powerbi[age_band]` (Count)
- **Legend**: `studentInfo_for_powerbi[age_band]`

**2. Gender Distribution (Donut Chart):**
- **Values**: `studentInfo_for_powerbi[gender]` (Count)
- **Legend**: `studentInfo_for_powerbi[gender]`

**3. Regional Distribution (Map):**
- **Location**: `studentInfo_for_powerbi[region]`
- **Size**: Count of students
- **Color**: `studentInfo_for_powerbi[is_stem]`

**4. Education Level (Bar Chart):**
- **Axis**: `studentInfo_for_powerbi[highest_education]`
- **Values**: Count of students
- **Color**: `studentInfo_for_powerbi[is_stem]`

**5. IMD Band Distribution (Bar Chart):**
- **Axis**: `studentInfo_for_powerbi[imd_band]`
- **Values**: Count of students
- **Color**: `studentInfo_for_powerbi[is_stem]`

**6. Disability Status (Bar Chart):**
- **Axis**: `studentInfo_for_powerbi[disability]`
- **Values**: Count of students
- **Color**: `studentInfo_for_powerbi[is_stem]`

---

### **Page 3: Academic Performance**

#### **Layout:**
```
┌─────────────────────────────────────────────────────────┐
│  Academic Performance Analysis                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │ Performance │ │ Success by  │ │ Excellence  │        │
│  │ by Subject  │ │ Education   │ │ by Age      │        │
│  │ (Bar)       │ │ (Bar)       │ │ (Bar)       │        │
│  └─────────────┘ └─────────────┘ └─────────────┘        │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │ Performance │ │ Credits vs  │ │ Previous    │        │
│  │ by Gender   │ │ Success     │ │ Attempts    │        │
│  │ (Bar)       │ │ (Scatter)   │ │ (Histogram) │        │
│  └─────────────┘ └─────────────┘ └─────────────┘        │
├─────────────────────────────────────────────────────────┤
│  [Navigation Buttons]                                   │
└─────────────────────────────────────────────────────────┘
```

#### **Visuals to Create:**

**1. Performance by Subject (Bar Chart):**
- **Axis**: `performance_by_subject[subject]`
- **Values**: `performance_by_subject[success_rate]`
- **Color**: `performance_by_subject[is_stem]`

**2. Success by Education Level (Bar Chart):**
- **Axis**: `studentInfo_for_powerbi[highest_education]`
- **Values**: Average of `studentInfo_for_powerbi[stem_success]`
- **Color**: `studentInfo_for_powerbi[is_stem]`

**3. Excellence by Age (Bar Chart):**
- **Axis**: `studentInfo_for_powerbi[age_band]`
- **Values**: Average of `studentInfo_for_powerbi[stem_excellence]`
- **Color**: `studentInfo_for_powerbi[is_stem]`

**4. Performance by Gender (Bar Chart):**
- **Axis**: `studentInfo_for_powerbi[gender]`
- **Values**: Average of `studentInfo_for_powerbi[stem_success]`
- **Color**: `studentInfo_for_powerbi[is_stem]`

**5. Credits vs Success (Scatter Plot):**
- **X-Axis**: `studentInfo_for_powerbi[studied_credits]`
- **Y-Axis**: `studentInfo_for_powerbi[stem_success]`
- **Color**: `studentInfo_for_powerbi[is_stem]`

**6. Previous Attempts (Histogram):**
- **Values**: `studentInfo_for_powerbi[num_of_prev_attempts]`
- **Color**: `studentInfo_for_powerbi[is_stem]`

---

### **Page 4: Risk Analysis**

#### **Layout:**
```
┌─────────────────────────────────────────────────────────┐
│  Risk Analysis & Intervention                           │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │ Risk Level  │ │ Risk by     │ │ Risk by     │        │
│  │ Distribution│ │ Subject     │ │ Education   │        │
│  │ (Donut)     │ │ (Bar)       │ │ (Bar)       │        │
│  └─────────────┘ └─────────────┘ └─────────────┘        │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │ Academic    │ │ STEM        │ │ Learning    │        │
│  │ Risk Score  │ │ Readiness   │ │ Persistence │        │
│  │ (Histogram) │ │ (Histogram) │ │ (Histogram) │        │
│  └─────────────┘ └─────────────┘ └─────────────┘        │
├─────────────────────────────────────────────────────────┤
│  [Navigation Buttons]                                   │
└─────────────────────────────────────────────────────────┘
```

#### **Visuals to Create:**

**1. Risk Level Distribution (Donut Chart):**
- **Values**: `model_results[risk_level]` (Count)
- **Legend**: `model_results[risk_level]`

**2. Risk by Subject (Bar Chart):**
- **Axis**: `studentInfo_for_powerbi[subject_name]`
- **Values**: Count of students
- **Color**: `model_results[risk_level]`

**3. Risk by Education (Bar Chart):**
- **Axis**: `studentInfo_for_powerbi[highest_education]`
- **Values**: Count of students
- **Color**: `model_results[risk_level]`

**4. Academic Risk Score (Histogram):**
- **Values**: `custom_features[academic_risk_score]`
- **Color**: `model_results[risk_level]`

**5. STEM Readiness (Histogram):**
- **Values**: `custom_features[stem_readiness_index]`
- **Color**: `model_results[risk_level]`

**6. Learning Persistence (Histogram):**
- **Values**: `custom_features[learning_persistence]`
- **Color**: `model_results[risk_level]`

---

### **Page 5: Model Predictions**

#### **Layout:**
```
┌─────────────────────────────────────────────────────────┐
│  Machine Learning Model Predictions                     │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │ Prediction  │ │ Excellence  │ │ Success     │        │
│  │ Accuracy    │ │ Probability │ │ Probability │        │
│  │ (Gauge)     │ │ (Histogram) │ │ (Histogram) │        │
│  └─────────────┘ └─────────────┘ └─────────────┘        │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │ Actual vs   │ │ Feature     │ │ Prediction  │        │
│  │ Predicted   │ │ Importance  │ │ Confidence  │        │
│  │ (Matrix)    │ │ (Bar)       │ │ (Scatter)   │        │
│  └─────────────┘ └─────────────┘ └─────────────┘        │
├─────────────────────────────────────────────────────────┤
│  [Navigation Buttons]                                   │
└─────────────────────────────────────────────────────────┘
```

#### **Visuals to Create:**

**1. Prediction Accuracy (Gauge Chart):**
- **Value**: `summary_stats[value]` where `summary_stats[metric] = "Model Accuracy (%)"`
- **Min**: 0, **Max**: 100, **Target**: 90

**2. Excellence Probability (Histogram):**
- **Values**: `model_results[excellence_probability]`
- **Color**: `model_results[predicted_excellence]`

**3. Success Probability (Histogram):**
- **Values**: `model_results[success_probability]`
- **Color**: `model_results[predicted_success]`

**4. Actual vs Predicted (Matrix):**
- **Rows**: `studentInfo_for_powerbi[stem_excellence]`
- **Columns**: `model_results[predicted_excellence]`
- **Values**: Count of students

**5. Feature Importance (Bar Chart):**
- **Axis**: `feature_importance[feature]`
- **Values**: `feature_importance[importance_score]`

**6. Prediction Confidence (Scatter Plot):**
- **X-Axis**: `model_results[excellence_probability]`
- **Y-Axis**: `model_results[success_probability]`
- **Color**: `model_results[risk_level]`

---

### **Page 6: Subject Comparison**

#### **Layout:**
```
┌─────────────────────────────────────────────────────────┐
│  STEM vs Non-STEM Subject Comparison                    │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │ STEM vs     │ │ Success     │ │ Excellence  │        │
│  │ Non-STEM    │ │ Rate        │ │ Rate        │        │
│  │ (Donut)     │ │ Comparison  │ │ Comparison  │        │
│  │             │ │ (Bar)       │ │ (Bar)       │        │
│  └─────────────┘ └─────────────┘ └─────────────┘        │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │ Risk Level  │ │ Average     │ │ Student     │        │
│  │ by Category │ │ Credits     │ │ Count       │        │
│  │ (Bar)       │ │ (Bar)       │ │ (Bar)       │        │
│  └─────────────┘ └─────────────┘ └─────────────┘        │
├─────────────────────────────────────────────────────────┤
│  [Navigation Buttons]                                   │
└─────────────────────────────────────────────────────────┘
```

#### **Visuals to Create:**

**1. STEM vs Non-STEM (Donut Chart):**
- **Values**: `studentInfo_for_powerbi[is_stem]` (Count)
- **Legend**: `studentInfo_for_powerbi[is_stem]`

**2. Success Rate Comparison (Bar Chart):**
- **Axis**: `performance_by_subject[subject]`
- **Values**: `performance_by_subject[success_rate]`
- **Color**: `performance_by_subject[is_stem]`

**3. Excellence Rate Comparison (Bar Chart):**
- **Axis**: `performance_by_subject[subject]`
- **Values**: `performance_by_subject[excellence_rate]`
- **Color**: `performance_by_subject[is_stem]`

**4. Risk Level by Category (Bar Chart):**
- **Axis**: `studentInfo_for_powerbi[is_stem]`
- **Values**: Count of students
- **Color**: `model_results[risk_level]`

**5. Average Credits (Bar Chart):**
- **Axis**: `performance_by_subject[subject]`
- **Values**: Average of `studentInfo_for_powerbi[studied_credits]`
- **Color**: `performance_by_subject[is_stem]`

**6. Student Count (Bar Chart):**
- **Axis**: `performance_by_subject[subject]`
- **Values**: `performance_by_subject[student_count]`
- **Color**: `performance_by_subject[is_stem]`

---

## **Step 4: Add Navigation**

### **Create Navigation Buttons:**
1. **Insert** → **Buttons** → **Blank**
2. Create buttons for each page:
   - "Executive Summary"
   - "Student Demographics"
   - "Academic Performance"
   - "Risk Analysis"
   - "Model Predictions"
   - "Subject Comparison"

### **Add Page Navigation:**
1. Select each button
2. **Format** → **Action** → **Page navigation**
3. Select the target page

### **Add Directional Buttons:**
1. Create "Previous" and "Next" buttons on each page
2. Set up page navigation for each direction

---

## **Step 5: Add Advanced Features**

### **DAX Measures:**
```dax
Success Rate = 
DIVIDE(
    COUNTROWS(FILTER(studentInfo_for_powerbi, studentInfo_for_powerbi[stem_success] = 1)),
    COUNTROWS(studentInfo_for_powerbi),
    0
)
```

```dax
Average Risk Score = 
AVERAGE(custom_features[academic_risk_score])
```

```dax
Prediction Accuracy = 
DIVIDE(
    COUNTROWS(FILTER(
        studentInfo_for_powerbi,
        studentInfo_for_powerbi[stem_excellence] = model_results[predicted_excellence]
    )),
    COUNTROWS(studentInfo_for_powerbi),
    0
)
```

### **Custom Tooltips:**
1. Create tooltip pages with detailed information
2. Add tooltip visuals to main charts
3. Configure tooltip content for each visual

### **Bookmarks:**
1. Create bookmarks for different filter states
2. Add bookmark buttons for quick navigation
3. Set up bookmark actions

---

## **Step 6: Final Design**

### **Color Theme:**
- **Primary**: Blue (#0078D4)
- **Secondary**: Orange (#FF8C00)
- **Success**: Green (#107C10)
- **Warning**: Yellow (#FFB900)
- **Error**: Red (#D13438)

### **Layout Guidelines:**
- Use consistent spacing (16px margins)
- Align visuals properly
- Use clear, readable fonts
- Add titles and subtitles
- Include data source information

### **Interactivity:**
- Enable cross-filtering between visuals
- Add drill-down capabilities
- Include hover effects
- Set up slicer interactions

---

## **Step 7: Publish & Share**

1. **Save** the Power BI file
2. **Publish** to Power BI Service
3. **Share** with stakeholders
4. **Set up** automatic refresh if needed

---

## **Expected Results:**

Your Power BI dashboard will provide:
- **Comprehensive view** of 7,196 students across 8 subjects
- **Clear insights** into STEM vs Non-STEM performance
- **Risk analysis** for intervention planning
- **Model predictions** for early warning systems
- **Interactive exploration** of student data
- **Professional presentation** suitable for stakeholders

This dashboard will effectively communicate the STEM performance prediction analysis and support data-driven decision making in education. 