# **Salifort Motors: Predicting Employee Turnover and Improving Retention**  

This project explores how employee-related factors such as workload, satisfaction, tenure, and promotions influence the likelihood of employee attrition at Salifort Motors, a leading alternative energy vehicle manufacturer, using machine learning in Python. By analyzing relationships between these factors and attrition outcomes, I built predictive models including logistic regression, decision trees, and random forests to identify employees at risk of leaving and support proactive HR interventions. The goal is to provide data-driven insights that enhance retention strategies and reduce attrition costs. The project leverages Python libraries like pandas, scikit-learn, and matplotlib for data analysis, predictive modeling, and visualization.

## **Project Navigation: Two-Part Structure**  

For clarity and easier navigation on GitHub, the project is divided into **two logical parts**, allowing viewers to focus on either Exploratory Data Analysis or Modeling without the notebook becoming too lengthy or cumbersome.

- **[Part 1 – Plan and Analyze Stages](https://github.com/Cyberoctane29/Salifort-Motors-Predicting-Employee-Turnover-and-Improving-Retention/blob/main/Project%20Parts/Salifort_Motors_Turnover_Part1_Plan_and_Analyze_Includes_EDA.ipynb):** This notebook covers **stakeholder alignment, data cleaning, and exploratory data analysis (EDA)**, uncovering key trends in employee satisfaction, workload, and attrition.  

  [View Part 1 Notebook](https://github.com/Cyberoctane29/Salifort-Motors-Predicting-Employee-Turnover-and-Improving-Retention/blob/main/Project%20Parts/Salifort_Motors_Turnover_Part1_Plan_and_Analyze_Includes_EDA.ipynb)  
<br><br>
<div style="width:100%;text-align: center;"> <img align=middle src="https://i.ibb.co/VYJfmS7j/stages-part1.png" alt="Phase4_6"> </div>
<br><br>

- **[Part 2 – Construct and Execute Stages](https://github.com/Cyberoctane29/Salifort-Motors-Predicting-Employee-Turnover-and-Improving-Retention/blob/main/Project%20Parts/Salifort_Motors_Turnover_Part2_Construct_and_Execute_Includes_Modeling.ipynb):** This notebook transitions to **predictive modeling**, comparing logistic regression with tree-based models (decision trees and random forests) to select the best-performing solution for turnover prediction.  

  [View Part 2 Notebook](https://github.com/Cyberoctane29/Salifort-Motors-Predicting-Employee-Turnover-and-Improving-Retention/blob/main/Project%20Parts/Salifort_Motors_Turnover_Part2_Construct_and_Execute_Includes_Modeling.ipynb)  
<br><br>
<div style="width:100%;text-align: center;"> <img align=middle src="https://i.ibb.co/6RGnnC6c/stages-part2.png" alt="Phase4_6"> </div>
<br><br> 

For a complete and consolidated view of both parts in a single notebook, check out the project on [Kaggle](https://www.kaggle.com/code/saswatsethda/tiktok-claims-classification-analysis-modeling) or [Github](https://github.com/Cyberoctane29/Salifort-Motors-Predicting-Employee-Turnover-and-Improving-Retention/blob/main/Salifort_Motors_Predicting_Employee_Turnover_and_Improving_Retention.ipynb)

 [View Kaggle Notebook](https://www.kaggle.com/code/saswatsethda/tiktok-claims-classification-analysis-modeling)

[View Github Notebook](https://github.com/Cyberoctane29/Salifort-Motors-Predicting-Employee-Turnover-and-Improving-Retention/blob/main/Salifort_Motors_Predicting_Employee_Turnover_and_Improving_Retention.ipynb)

## **Project Overview**

The **Salifort Motors Employee Attrition Analysis** project aims to:

- **Analyze Attrition Drivers**: Examine how factors like workload, job satisfaction, tenure, and promotions influence employee turnover  
- **Assess Data Patterns & Risks**: Identify trends, outliers, and key predictors associated with increased attrition risk  
- **Build Predictive Models**: Develop logistic regression, decision tree, and random forest models to predict which employees are likely to leave  
- **Support Proactive HR Strategy**: Provide actionable, data-driven recommendations to reduce attrition rates and improve employee retention  
- **Ensure Ethical, Bias-Aware Analysis**: Address data quality issues and avoid post-attrition information leakage to maintain fair, reliable insights 

## **Dataset Structure**  

The dataset contains **15,000 employee records** from **Salifort Motors**, each representing a unique employee and their work-related attributes. Key features include:

- **satisfaction_level**: Employee’s self-reported job satisfaction score (continuous: 0–1)  
- **last_evaluation**: Most recent performance review score (continuous: 0–1)  
- **number_project**: Number of projects assigned to the employee (discrete integer)  
- **average_monthly_hours**: Average hours worked per month (continuous)  
- **tenure**: Total number of years the employee has spent at the company (discrete integer)  
- **work_accident**: Indicates whether the employee experienced a workplace accident (binary: 0 = No, 1 = Yes)  
- **promotion_last_5years**: Whether the employee was promoted in the last five years (binary: 0 = No, 1 = Yes)  
- **department**: Employee’s functional department (categorical: e.g., Sales, IT, HR)  
- **salary**: Salary tier categorized as low, medium, or high (categorical)  
- **left**: Target variable indicating if the employee left the company (binary: 0 = Stayed, 1 = Left)  

This dataset serves as the foundation for analyzing employee attrition patterns at **Salifort Motors** and building machine learning models to identify employees at risk of leaving, enabling proactive, data-driven HR retention strategies.

## **Data Processing and Analysis Steps**  

### **Data Cleaning**  

- Confirmed **no missing values** across all features  
- Removed **3,008 duplicate entries** (20% of dataset) to ensure data integrity  
- Handled outliers in `tenure` using the **IQR method**, removing 824 extreme cases  
- Encoded categorical variables:  
  - **Ordinal encoding** for `salary` (low=0, medium=1, high=2)  
  - **One-hot encoding** for `department`

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

#### **Logistic Regression Performance**  
- Served as the baseline model for employee attrition prediction.  
- **Assumptions Evaluation**: All assumptions for logistic regression were evaluated, confirming the model’s suitability.  
- **Performance Evaluation**: Assessed using both AUC ROC and AUC PR to account for class imbalance:  
  - **Overall Accuracy**: 82%  
  - **AUC ROC**: 0.86  
  - **AUC PR**: 0.46 (indicating limited performance for the minority class)  
- **Precision-Recall Trade-off**: The model showed low recall for employees who left, highlighting the class imbalance issue.

#### **Model Development Approach**  
Two rounds of modeling were conducted to address potential data leakage:  
- **Round 1**: Initial models including all features, such as `satisfaction_level` and `average_monthly_hours`  
- **Round 2**: Refined models after removing leakage-prone features and creating the `overworked` indicator  

#### **Decision Tree Performance**  
- **Round 1 (All Features)**:  
  - **Best Parameters**: `max_depth=4`, `min_samples_leaf=5`  
  - **CV ROC AUC**: 0.9698  
  - **Test Recall**: 91.4%  
- **Round 2 (Leakage-Adjusted)**:  
  - **Best Parameters**: `max_depth=6`, `min_samples_leaf=2`  
  - **CV ROC AUC**: 0.9587 (1.1% drop)  
  - **Test Recall**: 91.8% (maintained strong performance)  
  - Demonstrated robustness despite removing key features  

#### **Random Forest Evolution**  
- **Round 1 (All Features)**:  
  - **Best Parameters**: 500 estimators, `max_depth=5`  
  - **CV ROC AUC**: 0.9804 (top performer)  
  - **Test Recall**: 92.0%  
- **Round 2 (Leakage-Adjusted)**:  
  - **Best Parameters**: 300 estimators, `max_depth=5`  
  - **CV ROC AUC**: 0.9648 (1.6% drop)  
  - **Test Recall**: 90.4% (still excellent)  
  - **Champion Model**: Selected due to:  
    - Minimal performance degradation (1.6% AUC drop)  
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
<br><br>
<div style="width:100%;text-align: center;"> <img align=middle src="https://i.ibb.co/9m97dQF5/stages.png" alt="AllStages"> </div>
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
