## MimicIII Project 
##### Predict the probability of death of a patient that is entering an ICU (Intensive Care Unit), using the machine learning models we have covered in class.

##### The dataset comes from MIMIC project (https://mimic.physionet.org/). MIMIC-III (Medical Information Mart for Intensive Care III) is a large, freely-available database comprising deidentified health-related data associated with over forty thousand patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012.

##### Each row of mimic_train_X.csv correponds to one ICU stay (hadm_id+icustay_id) of one patient (subject_id). In mimic_train_y.csv HOSPITAL_EXPIRE_FLAG is the indicator of death (=1) as a result of the current hospital stay; this is the outcome to predict in our modelling exercise. The columns of mimic_train_X.csv correspond to vitals of each patient (when entering the ICU), plus some general characteristics (age, gender, etc.), and their explanation can be found at mimic_patient_metadata.csv.

##### Main tasks are:
##### - Using mimic_train.csv file build a predictive model for HOSPITAL_EXPIRE_FLAG.
##### - For this analysis there is an extra test dataset, mimic_test_death.csv. Apply final model to this extra dataset and produce a prediction csv file in same format.
