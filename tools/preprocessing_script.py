import pandas as pd
import numpy as np
import joblib


medications = ['metformin', 'repaglinide', 'nateglinide',
       'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
       'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
       'miglitol', 'troglitazone', 'tolazamide', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone']

obj_cols = ['gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
                'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'max_glu_serum', 'diag_1']

num_cols = ['age',
            'num_procedures',
            'time_in_hospital',
            'num_lab_procedures',
            'services_used',
            'num_changes',
            'num_meds_use',
            'num_medications',
            'number_diagnoses']

feature_set = ['age', 'discharge_disposition_id', 'time_in_hospital',
       'num_lab_procedures', 'num_procedures', 'num_medications',
       'number_diagnoses', 'services_used',
       'num_changes', 'num_meds_use', 'race_AfricanAmerican', 'race_Asian',
       'race_Caucasian', 'race_Hispanic', 'race_Other', 'gender_1',
       'admission_type_id_1', 'admission_type_id_2', 'admission_type_id_3',
       'admission_type_id_4', 'admission_source_id_1', 'admission_source_id_4',
       'admission_source_id_7', 'admission_source_id_8',
       'admission_source_id_9', 'admission_source_id_11', 'max_glu_serum_-99',
       'max_glu_serum_0', 'max_glu_serum_1', 'A1Cresult_-99', 'A1Cresult_0',
       'A1Cresult_1', 'change_0', 'change_1', 'diabetesMed_0', 'diabetesMed_1',
       'diag_1_0.0', 'diag_1_1.0', 'diag_1_2.0', 'diag_1_3.0', 'diag_1_4.0',
       'diag_1_5.0', 'diag_1_6.0', 'diag_1_7.0', 'diag_1_8.0', 'diag_1_9.0',
       'diag_1_10.0', 'diag_1_11.0', 'diag_1_12.0', 'diag_1_13.0',
       'diag_1_14.0', 'diag_1_16.0', 'diag_1_17.0']

def get_num_changes(readmission_df):
    result = pd.Series(np.zeros(readmission_df.shape[0])).astype(int)
    for col in medications:
        result = (result.add(readmission_df[col].apply(lambda x: 0 if (x == 'No' or x == 'Steady') else 1)))
    return result

def get_num_meds_use(readmission_df):
    result = pd.Series(np.zeros(readmission_df.shape[0])).astype(int)
    for col in medications:
        result = (result.add(readmission_df[col].apply(lambda x: 0 if (x == 'No') else 1)))
    return result

def category_combine(column, combining_dict):
    try:
        for key in combining_dict.keys():
            column = column.replace(key, combining_dict[key])
    except KeyError:
        print("combining_dict does not match columns categories")
        
    return column

def categorizing_features(readmission_df):

    readmission_df.loc[readmission_df['diag_1'].str.contains('V',na=False,case=False), 'diag_1'] = 0
    readmission_df.loc[readmission_df['diag_1'].str.contains('E',na=False,case=False), 'diag_1'] = 0

    readmission_df['diag_1'] = readmission_df['diag_1'].astype(float)

    readmission_df['diag_1'].loc[(readmission_df['diag_1']>=1) & (readmission_df['diag_1']< 140)] = 1
    readmission_df['diag_1'].loc[(readmission_df['diag_1']>=140) & (readmission_df['diag_1']< 240)] = 2
    readmission_df['diag_1'].loc[(readmission_df['diag_1']>=240) & (readmission_df['diag_1']< 280)] = 3
    readmission_df['diag_1'].loc[(readmission_df['diag_1']>=280) & (readmission_df['diag_1']< 290)] = 4
    readmission_df['diag_1'].loc[(readmission_df['diag_1']>=290) & (readmission_df['diag_1']< 320)] = 5
    readmission_df['diag_1'].loc[(readmission_df['diag_1']>=320) & (readmission_df['diag_1']< 390)] = 6
    readmission_df['diag_1'].loc[(readmission_df['diag_1']>=390) & (readmission_df['diag_1']< 460)] = 7
    readmission_df['diag_1'].loc[(readmission_df['diag_1']>=460) & (readmission_df['diag_1']< 520)] = 8
    readmission_df['diag_1'].loc[(readmission_df['diag_1']>=520) & (readmission_df['diag_1']< 580)] = 9
    readmission_df['diag_1'].loc[(readmission_df['diag_1']>=580) & (readmission_df['diag_1']< 630)] = 10
    readmission_df['diag_1'].loc[(readmission_df['diag_1']>=630) & (readmission_df['diag_1']< 680)] = 11
    readmission_df['diag_1'].loc[(readmission_df['diag_1']>=680) & (readmission_df['diag_1']< 710)] = 12
    readmission_df['diag_1'].loc[(readmission_df['diag_1']>=710) & (readmission_df['diag_1']< 740)] = 13
    readmission_df['diag_1'].loc[(readmission_df['diag_1']>=740) & (readmission_df['diag_1']< 760)] = 14
    readmission_df['diag_1'].loc[(readmission_df['diag_1']>=760) & (readmission_df['diag_1']< 780)] = 15
    readmission_df['diag_1'].loc[(readmission_df['diag_1']>=780) & (readmission_df['diag_1']< 800)] = 16
    readmission_df['diag_1'].loc[(readmission_df['diag_1']>=800) & (readmission_df['diag_1']< 1000)] = 17
    readmission_df['diag_1'].loc[(readmission_df['diag_1']==-1)] = 0


    readmission_df['change'] = readmission_df['change'].replace('Ch', 1)
    readmission_df['change'] = readmission_df['change'].replace('No', 0)


    readmission_df['diabetesMed'] = readmission_df['diabetesMed'].replace('Yes', 1)
    readmission_df['diabetesMed'] = readmission_df['diabetesMed'].replace('No', 0)


    readmission_df['gender'] = readmission_df['gender'].replace('Male', 1)
    readmission_df['gender'] = readmission_df['gender'].replace('Female', 1)
    
    age_dict = {'[0-10)' : 5, '[10-20)' : 15, '[20-30)' : 25, '[30-40)' : 45, '[40-50)' : 45, '[50-60)' : 55, '[60-70)' : 65, '[70-80)' : 75,'[80-90)' : 85, '[90-100)' : 95}

    readmission_df['age'] = readmission_df['age'].map(age_dict)

    readmission_df['A1Cresult'] = readmission_df['A1Cresult'].replace('>7', 1)
    readmission_df['A1Cresult'] = readmission_df['A1Cresult'].replace('>8', 1)
    readmission_df['A1Cresult'] = readmission_df['A1Cresult'].replace('Norm', 0)
    readmission_df['A1Cresult'] = readmission_df['A1Cresult'].replace('None', -99)

    readmission_df['max_glu_serum'] = readmission_df['max_glu_serum'].replace('>200', 1)
    readmission_df['max_glu_serum'] = readmission_df['max_glu_serum'].replace('>300', 1)
    readmission_df['max_glu_serum'] = readmission_df['max_glu_serum'].replace('Norm', 0)
    readmission_df['max_glu_serum'] = readmission_df['max_glu_serum'].replace('None', -99)

    return readmission_df

def Scale(df, scaler_path):
    scaler = joblib.load(scaler_path) 

    scaled_df = scaler.transform(df)
    return scaled_df


def preprocessing_data(readmission_df):

    readmission_df = readmission_df.replace('?', np.NaN)

    readmission_df.race = readmission_df.race.fillna(readmission_df.race.mode()[0])
    readmission_df.diag_1 = readmission_df.diag_1.fillna(readmission_df.diag_1.mode()[0])

    readmission_df['services_used'] = readmission_df['number_inpatient'] + readmission_df['number_emergency'] + readmission_df['number_outpatient']

    

    readmission_df['num_changes'] = get_num_changes(readmission_df).astype(int)
    readmission_df['num_meds_use'] = get_num_meds_use(readmission_df)

    combining_addmission_type = {
    2 : 1,
    3 : 2,
    4 : 3,
    5 : 4,
    6 : 4,
    7 : 1,
    8 : 4
    }
    readmission_df['admission_type_id'] = category_combine(readmission_df['admission_type_id'], combining_addmission_type)

    combining_addmission_source = {
    2:1,
    3:1,
    4:4,
    5:4,
    6:4,
    10:4,
    22:4,
    25:4,
    15:9,
    17:9,
    20:9,
    21:9,
    13:11,
    14:11
    }
    readmission_df['admission_source_id'] = category_combine(readmission_df['admission_source_id'], combining_addmission_source)

    readmission_df = categorizing_features(readmission_df)

    readmission_df[obj_cols] =readmission_df[obj_cols].astype(object)

    readmission_df[num_cols] = Scale(readmission_df[num_cols], '/home/pavlo/projects/readmission/models/scaler.save')

    df = pd.get_dummies(readmission_df, columns=['race', 'gender', 'admission_type_id', 'admission_source_id', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed', 'diag_1'])

    for col in feature_set:
        if col not in df.columns:
            df[col] = 0

    print(df[num_cols])

    return df[feature_set]