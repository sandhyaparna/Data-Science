#### Data understanding is fundamental to solving any problem


#### Creating Hpothesis/Assumptions are testing them
* Healthcare
  * Is PatientID a unique identifier of a visit or within a single visit do Patients get multiple Patient IDs - How do you combine them? - Create a Arrival date for the combinedPatientIDs 
  * Is PatientID a unique indentifier among different clients? UniqueID is created by concatenating ClientID and PatientID
  * If there is gap in vitals/labs of patients for more than certain amount of time - Why do you think data is missing?
  * Time difference between Collection date Time and Creation date time
  * How is discharge date time actually created? What do you know about the time of making the discharge decision for a patient? Can we assume Last collection date time is time at which the decision making of discharge is done
  

#### Exploration
* Variable Identification
  * Type of Variable: Target, Unique Idnetifier, Predictor Vars
  * Data Type: Numeric, Character - How can you encode?
* Univariate, Bivariate analysis
* Look for bad data - for eg in Gender variable if there is something like 63.9 - we can assume it is bad data and set to missing. Or sometimes missing data might be represented by a different number like 99 all together
* Duplicate values - For eg: Visit ID may be unique to a hospital, but when we use diff hospitals there might be duplicates in Visit ID. So, create a new Unique Identifier - Concatenation of Client/Hospital ID and Visit ID. Duplicate rows may actually be present.
* Missing value treatment
* Outlier treatmnet
* Variable transformation, creation, feature engineering





