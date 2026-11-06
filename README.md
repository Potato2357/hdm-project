# GLP-1 Initiation and Changes in BMI and HbA1c

## Research Question: 
- Among patients who start a GLP-1 receptor agonist, how much do BMI and HbA1c change from baseline to 6 months after initiation?

## Additional Insight: 
- Does the change differ by patient characteristics?
- Does the change differ between patients with and without type 2 diabetes dx at baseline?

## Data: 
Data from the de-identified EHR dataset in Databricks
- Prod_deid_clinical.ehr_dataset
- Medication: Identify GLP-1 agonist orders 
- Flow_Sheet: BMI measurements before and after starting a GLP-1
- Lab_result: HbA1c results before and after starting GLP-1
- Patient: Demographics 
- Diagnosis: Type 2 Diabetes status before and after starting medication

## Analysis Plan: 
### Steps:
1. Identify adults who are on a GLP-1
2. Pull each patients BMI and HbA1c from the 6 months before starting and 6 months after 
3. Compare the changes over time
4. Summarize the overall change, considering patient demographics (sex, age, race, etc)
5. Analyze the change in individual patients, e.g. regression analysis
### Goals: 
- A snapshot of how patients benefit or don’t from GLP 1 usage.
- Gain insight into how GLP 1 benefits might vary based on patient demographic 

