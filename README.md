# **Salifort Motors: Predicting Employee Turnover and Improving Retention**  

This project leverages **Python** and **machine learning** to analyze employee turnover trends at **Salifort Motors**, a leading alternative energy vehicle manufacturer. By exploring factors like workload, satisfaction, tenure, and promotions, I built predictive models—including **logistic regression, decision trees, and random forests**—to identify employees at risk of leaving. The goal is to empower HR with data-driven insights to enhance retention strategies and reduce attrition costs.  

## **Project Navigation: Two-Part Structure**  

For clarity and optimal GitHub performance, the project is divided into **two logical parts**:  

- **[Part 1 – Plan and Analyze Stages](https://github.com/Cyberoctane29/Salifort-Motors-Predicting-Employee-Turnover-and-Improving-Retention/blob/main/Project%20Parts/Salifort_Motors_Turnover_Part1_Plan_and_Analyze_Includes_EDA.ipynb):** This notebook covers **stakeholder alignment, data cleaning, and exploratory data analysis (EDA)**, uncovering key trends in employee satisfaction, workload, and attrition.  

  [View Part 1 Notebook](https://github.com/Cyberoctane29/Salifort-Motors-Predicting-Employee-Turnover-and-Improving-Retention/blob/main/Project%20Parts/Salifort_Motors_Turnover_Part1_Plan_and_Analyze_Includes_EDA.ipynb)  
<br><br>
<div style="width:100%;text-align: center;"> <img align=middle src="https://i.ibb.co/hJSFddK6/stages-part1.png" alt="Phase4_6"> </div>
<br><br>

- **[Part 2 – Construct and Execute Stages](https://github.com/Cyberoctane29/Salifort-Motors-Predicting-Employee-Turnover-and-Improving-Retention/blob/main/Project%20Parts/Salifort_Motors_Turnover_Part2_Construct_and_Execute_Includes_Modeling.ipynb):** This notebook transitions to **predictive modeling**, comparing logistic regression with tree-based models (decision trees and random forests) to select the best-performing solution for turnover prediction.  

  [View Part 2 Notebook](https://github.com/Cyberoctane29/Salifort-Motors-Predicting-Employee-Turnover-and-Improving-Retention/blob/main/Project%20Parts/Salifort_Motors_Turnover_Part2_Construct_and_Execute_Includes_Modeling.ipynb)  
<br><br>
<div style="width:100%;text-align: center;"> <img align=middle src="https://i.ibb.co/pBTngP0w/stages-part2.png" alt="Phase4_6"> </div>
<br><br> 

For a complete and consolidated view of both parts in a single notebook, check out the project on [Kaggle](https://www.kaggle.com/code/saswatsethda/tiktok-claims-classification-analysis-modeling) or [Github](https://github.com/Cyberoctane29/TikTok-Claims-Classification-End-to-End-Analysis-and-Modeling/blob/main/TikTok%20_Claims_Classification_End_to_End_Analysis_and_Modeling.ipynb)

 [View Kaggle Notebook](https://www.kaggle.com/code/saswatsethda/tiktok-claims-classification-analysis-modeling)

[View Github Notebook](https://github.com/Cyberoctane29/TikTok-Claims-Classification-End-to-End-Analysis-and-Modeling/blob/main/TikTok%20_Claims_Classification_End_to_End_Analysis_and_Modeling.ipynb)


## **Project Overview**  

**Salifort Motors** faces rising employee turnover, prompting HR to seek data-driven retention strategies. This project:  

- **Analyzes attrition drivers**: Workload, satisfaction, tenure, and promotions  
- **Builds predictive models**: Identifies employees at risk of leaving  
- **Recommends interventions**: Targeted policies to improve retention  

## **Dataset Structure**  

The dataset includes **15,000 observations** with features such as:  

| Column               | Description                                  |  
|----------------------|----------------------------------------------|  
| `satisfaction_level` | Employee-reported satisfaction (0–1)         |  
| `last_evaluation`    | Performance review score (0–1)               |  
| `number_project`     | Number of projects assigned                  |  
| `average_monthly_hours` | Avg. monthly hours worked                 |  
| `tenure`            | Years at the company                         |  
| `work_accident`      | Whether the employee had a workplace accident|  
| `left`              | Whether the employee left (target variable)  |  
| `promotion_last_5years` | Promoted in the last 5 years?             |  
| `department`        | Employee’s department (e.g., Sales, HR)      |  
| `salary`            | Salary tier (low, medium, high)              | 

This dataset serves as the foundation for analyzing patterns in employee attrition at Salifort Motors and building predictive models to identify employees at risk of leaving the company.

## **Data Processing and Analysis Steps**  

### **Data Cleaning**  
- Removed **3,008 duplicate entries** (20% of dataset) to ensure data integrity  
- Handled outliers in `tenure` using **IQR method**, removing 824 extreme cases  
- Encoded categorical variables:  
  - Ordinal encoding for `salary` (low=0, medium=1, high=2)  
  - One-hot encoding for `department`  

### **Exploratory Data Analysis (EDA)**  
- **Workload Analysis**: Employees with **7 projects all left** (critical risk threshold)  
- **Tenure Trends**: 4-year tenure employees showed **lowest satisfaction**  
- **Hours vs Satisfaction**: Revealed two attrition clusters:  
  - **Overworked** (high hours, low satisfaction)  
  - **Disengaged** (low hours, low satisfaction)  
- **Departmental Attrition**: HR had **highest turnover rate**; Management had lowest  

### **Statistical Insights**  
- **Satisfaction Correlation**: Strong negative relationship with attrition (r=-0.35)  
- **Promotion Impact**: Only **2.1%** of employees promoted in last 5 years  
- **Evaluation Paradox**: High performers working long hours still left (burnout risk)  

### **Machine Learning Modeling**  

#### **Model Development Approach**
- Conducted two rounds of modeling to address potential data leakage:
  - **Round 1**: Initial models with all features including `satisfaction_level` and `average_monthly_hours`
  - **Round 2**: Refined models after removing leakage-prone features and creating the `overworked` indicator

#### **Decision Tree Performance**
- **Round 1 (All Features)**:
  - Best Parameters: `max_depth=4`, `min_samples_leaf=5`
  - CV ROC AUC: 0.9698
  - Test Recall: 91.4%
- **Round 2 (Leakage-Adjusted)**:
  - Best Parameters: `max_depth=6`, `min_samples_leaf=2` 
  - CV ROC AUC: 0.9587 (1.1% drop)
  - Test Recall: 91.8% (maintained strong performance)
  - Demonstrated robustness despite removing key features

#### **Random Forest Evolution**
- **Round 1 (All Features)**:
  - Best Parameters: 500 estimators, `max_depth=5`
  - CV ROC AUC: 0.9804 (top performer)
  - Test Recall: 92.0%
- **Round 2 (Leakage-Adjusted)**:
  - Best Parameters: 300 estimators, `max_depth=5`
  - CV ROC AUC: 0.9648 (1.6% drop)
  - Test Recall: 90.4% (still excellent)
  - Selected as **Champion Model** due to:
    - Minimal performance degradation (only 1.6% AUC drop)
    - Better generalization than Decision Tree
    - More stable feature importance rankings

#### **Key Modeling Insights**
- The 1-2% performance drop between rounds confirms some predictive power came from leakage-prone features
- Despite this, both tree-based models maintained >90% recall in Round 2
- Random Forest showed more consistent performance across rounds, making it more reliable for deployment

This two-round approach validated that our final models make predictions based on stable, non-leaky features while maintaining strong performance.

### **Key Workflow Decisions**  
- Addressed **data leakage** by removing post-attrition indicators  
- Prioritized **recall** to minimize false negatives (missed at-risk employees)  
- Used **stratified sampling** to maintain class balance in train/test splits  

## **Key Insights**  

### **Exploratory Data Analysis (EDA)**  
- **Overworked employees**: Those working **>175 hrs/month** had **3× higher attrition**  
- **Mid-tenure risk**: Employees with **4–6 years tenure** showed the **lowest satisfaction**  
- **Promotion gap**: Only **2.1%** of employees were promoted in the last 5 years  

### **Model Performance**  

| Model               | Accuracy | Recall (Leavers) | Precision (Leavers) |  
|---------------------|----------|------------------|---------------------|  
| Logistic Regression | 82%      | 26%              | 44%                 |  
| Decision Tree       | 94%      | 92%              | 78%                 |  
| **Random Forest**   | **96%**  | **90%**          | **87%**             |  

**Random Forest emerged as the champion model**, balancing high recall (minimizing false negatives) with strong precision.  

### **Top Attrition Drivers**  
1. **High workload (`overworked`)**  
2. **Low promotions (`promotion_last_5years`)**  
3. **Long tenure without growth (`tenure`)**  
4. **Departmental trends (HR highest attrition)**  

## **Project Highlights**  
<div style="width:100%;text-align: center;"> <img align=middle src="https://i.ibb.co/ycBZr552/stages.png" alt="AllPhases"> </div>
<br><br>

- **Comprehensive Workforce Analysis**: Uncovered critical attrition patterns including the "4-year slump" in employee satisfaction and the 7-project burnout threshold  
- **High-Performance Predictive Model**: Developed a Random Forest classifier with **90.4% recall** for at-risk employees while maintaining **87% precision**  
- **Actionable Risk Indicators**: Identified four key attrition drivers: performance scores, project load, tenure duration, and overwork status  
- **Ethical AI Implementation**:  
  - Addressed data leakage by removing post-attrition indicators  
  - Balanced model performance with interpretability for HR stakeholders  
  - Established protocols for fair application of predictive insights  
- **Strategic Visualization Suite**: Created intuitive dashboards showing:  
  - Departmental turnover hotspots  
  - Workload vs satisfaction tradeoffs  
  - Tenure-based retention opportunities  
- **Operational Readiness**: Designed model outputs to integrate directly with HRIS systems for real-time risk alerts  

<br><br>
<div style="width:100%;text-align: center;"> <img align=middle src="https://i.ibb.co/y2Rb2L6/Phases4-6.jpg" alt="Phase4_6"> </div>
<br><br>

This end-to-end solution transforms raw employee data into strategic retention opportunities, enabling Salifort Motors to proactively address turnover risks while maintaining ethical AI standards. The combination of robust analytics and actionable business intelligence creates a new capability for data-driven workforce management.

## **Future Work**  

- **Address class imbalance** with techniques like SMOTE or class weighting.  
- **Incorporate additional features** like manager feedback or team dynamics.  
- **Deploy model as an HR dashboard** for real-time attrition risk alerts.  

### **Tools & Technologies**  
- **Python**: pandas, NumPy, scikit-learn, Matplotlib, Seaborn  
- **Machine Learning**: Logistic Regression, Decision Trees, Random Forest  
- **Model Evaluation**: Precision-Recall AUC, ROC AUC, F1-score  
- **Workflow**: PACE (Plan, Analyze, Construct, Execute) framework

This project equips **Salifort Motors** with actionable insights to **reduce turnover, improve employee satisfaction, and optimize retention strategies**. By proactively identifying at-risk employees, HR can intervene early, fostering a more stable and engaged workforce.  
