### Problem Statement
* Understanding Problem, Problem formalization
* Scope of the model - Model predicts 6hrs before onset of a disease, Before making the decision of discharge at a hospital
* What is time period that is looked at while creating observations for Target & Non-Target population

### Data Cleaning
* Cohort of interest
* Identify & resolves Discrepancies in data

### Data-preprocessing
* Univariate analysis - Data cleaning. Look for bad data (Numeric value in categorical var), Duplicates (Visit ID might not be unique when we combine data from diff clients); Duplicates(In online data, a user might have submitted entries multiple times); Typos in Categorical data (observe Freq table)
* Outlier treatment (Temp_F might have values less than 35 - one assumption can be that the temp value is actually Celsius and convert it into Fahrenheit)
* Missing value treatment
* Data Transformation - Right skewed/Left skewed
* Feature engineering - Numeric/Categorical/DateTime Vars - Transactional data/Temporal data
* Feature selection
* Split data into - Train, Validation & Test

### Model Building
* Overfitting - Cross Validation, Regularization
* Model Interpretation - LIME, shap

### Evaluation & Comparision - Regression, classification, Unsupervised
* Precision-Recall Curves, AUC Curves - To determine the cut-off for an effective model



