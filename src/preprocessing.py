'''
Derived from preprocessing-py.py to run in Spark for parallelism
'''
import os
from typing import List, Tuple

import pyspark.pandas as ps
import pyspark.sql.types as T
import pyspark.sql.functions as F
import os

# Set environment variables (both should point to same pythonpath)
os.environ['PYSPARK_PYTHON'] = '~/miniconda3/envs/hc_nlp_2/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '~/miniconda3/envs/hc_nlp_2/bin/python'

# Input files
RAW_BASE_PATH = '../data/raw/{fname}'
ADMISSIONS_FNAME =  'ADMISSIONS.csv.gz'
DIAGNOSES_FNAME = 'DIAGNOSES_ICD.csv.gz'
LABEVENTS_FNAME = 'LABEVENTS.csv.gz'
PRESCRIPTIONS_FNAME = 'PRESCRIPTIONS.csv.gz'
PATIENTS_FNAME = 'PATIENTS.csv.gz'
NOTES_FNAME = 'NOTEEVENTS.csv.gz'

PATH_PROCESSED = '../data/processed/'

DIAG_PATH = RAW_BASE_PATH.format(fname=DIAGNOSES_FNAME)
PATIENTS_PATH = RAW_BASE_PATH.format(fname=PATIENTS_FNAME)

PATIENT_SAMPLE_SIZE = 46520 # total is 46520
TRAIN_SIZE = 0.8
# We need to take into account only the events that happened during the observation window. The end of observation window is N days before death for deceased patients and date of last event for alive patients. We can have several sets of events (e.g. labs, diags, meds), so we need to choose the latest date out of those.
# OBSERVATION_WINDOW = 2000
OBSERVATION_WINDOW = 365*2
PREDICTION_WINDOW = 50

RANDOM_SEED = 1

# Respiratory illnesses ICD9 diag codes. Sources: 
# https://en.wikipedia.org/wiki/List_of_ICD-9_codes_460–519:_diseases_of_the_respiratory_system
# https://basicmedicalkey.com/diseases-of-the-respiratory-system-icd-9-cm-chapter-8-codes-460-519-and-icd-10-cm-chapter-10-codes-j00-j99/
acute_diag_codes = [
    460, 461, 462, 463, 464, 465, 466
]
other_resp_tract_diag_codes = [
    470, 471, 472, 473, 474, 475, 476, 477, 478
]
pneumonia_and_influenza_diag_codes = [
    480, 481, 482, 483, 484, 485, 486, 487, 488
]
grp4 = [
    490, 491, 492, 493, 494, 495, 496
]
grp5 = [
    500, 501, 502, 503, 504, 505, 506, 507, 508
]
grp6 = [
    510, 511, 512, 513, 514, 515, 516, 517, 518, 519
]

relevant_diag_codes: List[int] = [*acute_diag_codes, *other_resp_tract_diag_codes, *pneumonia_and_influenza_diag_codes, *grp4, *grp5, *grp6]
RELEVANT_DIAG_CODES = [str(e) for e in relevant_diag_codes]

# Feature tables column names
all_admissions_cols = [
	'ROW_ID',
	'SUBJECT_ID',
	'HADM_ID',
	'ADMITTIME',
	'DISCHTIME',
	'DEATHTIME',
	'ADMISSION_TYPE',
	'ADMISSION_LOCATION',
	'DISCHARGE_LOCATION',
	'INSURANCE',
	'LANGUAGE',
	'RELIGION',
	'MARITAL_STATUS',
	'ETHNICITY',
	'EDREGTIME',
	'EDOUTTIME',
	'DIAGNOSIS',
	'HOSPITAL_EXPIRE_FLAG',
	'HAS_CHARTEVENTS_DATA',
]

all_lab_results_cols = [
	'ROW_ID',
	'SUBJECT_ID',
	'HADM_ID',
	'ITEMID',
	'VALUE',
	'VALUENUM',
	'VALUEUOM',
	'FLAG',
]

all_meds_cols = [
	'ROW_ID',
	'SUBJECT_ID',
	'HADM_ID',
	'ICUSTAY_ID',
	'STARTDATE',
	'ENDDATE',
	'DRUG_TYPE',
	'DRUG',
	'DRUG_NAME_POE',
	'DRUG_NAME_GENERIC',
	'FORMULARY_DRUG_CD',
	'GSN',
	'NDC',
	'PROD_STRENGTH',
	'DOSE_VAL_RX',
	'DOSE_UNIT_RX',
	'FORM_VAL_DISP',
	'FORM_UNIT_DISP',
	'ROUTE'
]

all_notes_cols = [
	'ROW_ID',
	'SUBJECT_ID',
	'HADM_ID',
	'CHARTDATE',
	'CHARTTIME',
	'STORETIME',
	'CATEGORY',
	'DESCRIPTION',
	'CGID',
	'ISERROR',
	'TEXT'
]


def get_patient_sample() -> Tuple[ps.series.Series[int], ps.frame.DataFrame, ps.frame.DataFrame]:
    ''' Extracts the patient_ids, patient records and deceased_patients from the PATIENTS table '''
    patients = ps.read_csv(PATIENTS_PATH)
    sample_ids = patients.SUBJECT_ID
    # Moratality set
    deceased_patients = patients[patients.EXPIRE_FLAG == 1] 
    deceased_patients = deceased_patients[['SUBJECT_ID', 'DOD']]
    # first 10 characters of DOD column is date (we're ignoring time)
    deceased_patients = deceased_patients.to_spark()
    deceased_patients = deceased_patients.select('SUBJECT_ID', F.substring('DOD', 0, 10).alias('DOD'))
    deceased_patients = deceased_patients.to_pandas_on_spark()

    return sample_ids, patients, deceased_patients


def _get_data_for_sample(patient_ids: ps.series.Series[int], file_name: str, skip_sampling: bool = True) -> ps.frame.DataFrame:
    ''' Get the data only relevant for the sample. '''
    full_path = RAW_BASE_PATH.format(fname=file_name)
    raw = ps.read_csv(full_path)

    relevant_data = raw.to_spark()
    if skip_sampling == False:
        patient_ids = patient_ids.to_dataframe().to_spark()
        relevant_data = raw.join(F.broadcast(patient_ids), raw.SUBJECT_ID == patient_ids.SUBJECT_ID, 'left_semi')
    relevant_data = relevant_data.to_pandas_on_spark()

    return relevant_data


def preprocess(patient_ids: ps.series.Series[int]) -> Tuple[ps.frame.DataFrame, ps.frame.DataFrame, ps.frame.DataFrame, ps.frame.DataFrame]:
    ''' Returns preprocessed dfs containg records for @patient_ids
    '''
    #### Admissions
    admissions = _get_data_for_sample(patient_ids, ADMISSIONS_FNAME)
    # first 10 characters of DOD column is date (we're ignoring time)
    admissions_sp = admissions.to_spark()
    all_cols = [col for col in all_admissions_cols if col != 'ADMITTIME']
    admissions_sp = admissions_sp.select(*all_cols, F.substring('ADMITTIME', 0, 10).alias('ADMITTIME'))
    admissions = admissions_sp.to_pandas_on_spark()
    print('done processing admissions')

    #### Diagnoses
    diagnoses = _get_data_for_sample(patient_ids, DIAGNOSES_FNAME)
    diagnoses['ICD9_CODE'] = 'ICD9_' + diagnoses['ICD9_CODE']
    adm_cols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']
    diagnoses = diagnoses.merge(admissions[adm_cols], on=['SUBJECT_ID', 'HADM_ID'])
    dropper = ['ROW_ID', 'SEQ_NUM', 'HADM_ID']
    renamer = {'ICD9_CODE': 'FEATURE_NAME', 'ADMITTIME': 'DATE'}
    diag_preprocessed = diagnoses.drop(columns=dropper).rename(columns=renamer)
    diag_preprocessed['VALUE'] = 1
    print('done processing diags')

    #### Labs
    lab_results = _get_data_for_sample(patient_ids, LABEVENTS_FNAME)
    # first 10 characters of DOD column is date (we're ignoring time)
    lab_results_sp = lab_results.to_spark()
    # renames CHARTTIME to DATE
    all_cols2 = [col for col in all_lab_results_cols if col != 'CHARTTIME']
    lab_results_sp = lab_results_sp.select(*all_cols2, F.substring('CHARTTIME', 0, 10).alias('DATE'))
    lab_results_sp = lab_results_sp.withColumn('FEATURE_NAME', F.concat(F.lit('LAB_'), F.col('ITEMID').cast(T.StringType())))
    lab_results = lab_results_sp.to_pandas_on_spark()
    dropper = ['ROW_ID', 'HADM_ID', 'VALUE', 'VALUEUOM', 'FLAG', 'ITEMID', 'CHARTTIME']
    renamer = {'VALUENUM': 'VALUE'}
    lab_preprocessed = lab_results.drop(columns=dropper).rename(columns=renamer)
    print('done processing labs')

    #### Meds
    meds = _get_data_for_sample(patient_ids, PRESCRIPTIONS_FNAME)
    meds = meds[meds.ENDDATE.notna()]
    meds['DOSE_VAL_RX'] = meds['DOSE_VAL_RX'].fillna(0)

    meds_sp = meds.to_spark()

    # renames ENDDATE to DATE
    all_cols3 = [col for col in all_meds_cols if col != 'ENDDATE']
    meds_sp = meds_sp.select(*all_cols3, F.substring('ENDDATE', 0, 10).alias('DATE'))
    all_cols3_2 = [col for col in all_cols3 if col != 'STARTDATE'] + ['DATE']
    meds_sp = meds_sp.select(*all_cols3_2, F.substring('STARTDATE', 0, 10).alias('STARTDATE'))

    # cleanse aphanums from dose
    meds_sp = meds_sp.withColumn('DOSE_VAL_RX', F.regexp_replace(F.col('DOSE_VAL_RX'), '[A-Za-z,>< ]', ''))
    meds_sp = meds_sp.select(*all_cols3_2 + ['STARTDATE'], F.split(F.col('DOSE_VAL_RX'), '-').alias('dose_arr_temp'))
    meds_sp = meds_sp.withColumn('dose_arr_temp', F.col('dose_arr_temp').cast('array<float>'))

    query = '''aggregate(
        `{col}`,
        CAST(0.0 AS double),
        (acc, x) -> acc + x,
        acc -> acc / size(`{col}`)
    ) AS  `{new_col}`'''.format(col='dose_arr_temp', new_col='VALUE')
    meds_sp = meds_sp.selectExpr('*', query).drop('dose_arr_temp')

    meds_sp = meds_sp.withColumn('FEATURE_NAME', F.concat(F.lit('MED_'), F.col('GSN').cast(T.StringType())))
    meds = meds_sp.to_pandas_on_spark()

    dropper = [col for col in meds.columns if col not in {'SUBJECT_ID', 'DATE', 'FEATURE_NAME', 'VALUE'}]
    meds_preprocessed = meds.drop(columns=dropper).rename(columns=renamer)
    print('done processing meds')

    #### Notes
    notes_preprocessed = _get_data_for_sample(patient_ids, NOTES_FNAME, skip_sampling=True)

    notes_preprocessed_sp = notes_preprocessed.to_spark()
    all_cols4 = [col for col in all_notes_cols if col != 'CHARTDATE']
    notes_preprocessed_sp = notes_preprocessed_sp.select(*all_cols4, F.substring('CHARTDATE', 0, 10).alias('CHARTDATE'))

    notes_preprocessed_sp = notes_preprocessed_sp.withColumn('CLEAN_TEXT', F.regexp_replace(F.col('TEXT'), '[^\w]', ' '))
    notes_preprocessed_sp = notes_preprocessed_sp.withColumn('CLEAN_TEXT', F.regexp_replace(F.col('CLEAN_TEXT'), '_', ' '))
    notes_preprocessed_sp = notes_preprocessed_sp.withColumn('CLEAN_TEXT', F.regexp_replace(F.col('CLEAN_TEXT'), '\\s+', ' '))
    notes_preprocessed_sp = notes_preprocessed_sp.select(*all_notes_cols, F.lower(F.col('CLEAN_TEXT')).alias('CLEAN_TEXT'))
    # rename to DATE for consistency
    notes_preprocessed_sp = notes_preprocessed_sp.withColumnRenamed('CHARTDATE', 'DATE')
    notes_preprocessed = notes_preprocessed_sp.to_pandas_on_spark()
    print('done processing notes')

    return diag_preprocessed, lab_preprocessed, meds_preprocessed, notes_preprocessed


## Feature engr. helpers
def define_train_period(deceased_to_date: ps.frame.DataFrame, *feature_sets: List[ps.frame.DataFrame],
                        obs_w: int = OBSERVATION_WINDOW,
                        pred_w: int = PREDICTION_WINDOW) -> ps.frame.DataFrame:
	''' 
    Create SUBJECT_ID -> earliest_date and SUBJECT_ID -> last date dfs.
    Returns DF (SUBJECT_ID, EARLIEST_DATE, LAST_DATE_OR_DOD)
    '''
	cols = ['SUBJECT_ID', 'DATE']
	# union of feature sets on 'SUBJECT_ID' and 'DATE'
	all_feats = ps.concat([feats[cols] for feats in feature_sets])

	last_date_base = all_feats.groupby('SUBJECT_ID').DATE.max()
	last_date_base_sp = last_date_base.to_frame().reset_index().to_spark()
	deceased_to_date_sp = deceased_to_date.to_spark()

	deceased_to_date_sp = deceased_to_date_sp.withColumn('DOD_MINUS_PREDW_TMP', F.date_sub(F.col('DOD'), pred_w)).drop('DOD')
	data_sp = last_date_base_sp.join(F.broadcast(deceased_to_date_sp), last_date_base_sp.SUBJECT_ID == deceased_to_date_sp.SUBJECT_ID, 'left_outer')
	data_sp = data_sp.withColumn('LAST_DATE_OR_(DOD_MINUS_PREDW)', F.coalesce(F.col('DOD_MINUS_PREDW_TMP'), F.col('DATE')))
	data_sp = data_sp.withColumn('EARLIEST_DATE',  F.date_sub(F.col('DATE'), obs_w))
	data_sp = data_sp.drop(deceased_to_date_sp.SUBJECT_ID).drop('DATE').drop('DOD_MINUS_PREDW_TMP')

	data = data_sp.to_pandas_on_spark()
	return data


def _clean_up_feature_sets(*feature_sets: List[ps.frame.DataFrame], date: ps.frame.DataFrame, is_notes: bool = False) -> List[ps.frame.DataFrame]:
    '''Leave only features from inside the observation window.
    Returned DF schema: (SUBJECT_ID, DATE, FEATURE_NAME, VALUE)
    if is_notes: (SUBJECT_ID, DATE, existing ntoes feats.....)
    '''
    results = []
    for feat_set in feature_sets:
        # each record should be in >= earliest_date and < last_date 
        # join feat_set with date
        data_sp = feat_set.merge(date, on='SUBJECT_ID', how='inner')
        data_sp = data_sp[data_sp['DATE'] >= data_sp['EARLIEST_DATE']]
        data_sp = data_sp[data_sp['DATE'] < data_sp['LAST_DATE_OR_(DOD_MINUS_PREDW)']]
        if not is_notes:
            data_sp = data_sp[['SUBJECT_ID', 'DATE', 'FEATURE_NAME', 'VALUE']]
        results.append(data_sp)
    return results


def get_last_note(patient_ids: ps.series.Series[int], notes_preprocessed: ps.frame.DataFrame, date: ps.frame.DataFrame, as_tokenized=False) -> ps.frame.DataFrame:
    '''
    Returns dataframe containing latest note for each patient. If @as_tokenized is true, performs extra cleansing in to prepare the text blob for tokenizer,
    otherwise, returns the latest note without cleansing
    '''
    if as_tokenized:
        last_note = _clean_up_feature_sets(notes_preprocessed, date=date, is_notes=True)[0]
        select_cols = ['SUBJECT_ID', 'DATE', 'TEXT']
        last_note = last_note.sort_values(by=select_cols, ascending=False).drop_duplicates('SUBJECT_ID')[select_cols]
        ###
        last_note_sp = last_note.to_spark()
        patient_ids_sp = patient_ids.to_dataframe().to_spark()
        last_note_sp = last_note_sp.join(F.broadcast(patient_ids_sp), last_note_sp.SUBJECT_ID == patient_ids_sp.SUBJECT_ID, 'left_semi') 

        ## Prepare text for tokenizer
        last_note_sp = last_note.to_spark()
        # replace punctuation
        last_note_sp = last_note_sp.withColumn('TO_TOK', F.regexp_replace(F.col('TEXT'), '[\[\]\*\_#:?!]+', ' ')).drop('TEXT')
        # remove spaces
        last_note_sp = last_note_sp.withColumn('TO_TOK', F.regexp_replace(F.col('TO_TOK'), '\\s+', ' '))
        # remove dots
        last_note_sp = last_note_sp.withColumn('TO_TOK', F.regexp_replace(F.col('TO_TOK'), '\. \.| \.', '.'))
        # remove duplicated dots
        last_note_sp = last_note_sp.withColumn('TO_TOK', F.regexp_replace(F.col('TO_TOK'), '\.+', '.'))
        last_note = last_note_sp.to_pandas_on_spark()
        ###
        last_note = last_note.reset_index(drop=True)
    else:
        last_note = _clean_up_feature_sets(notes_preprocessed, date=date, is_notes=True)[0]
        select_cols = ['SUBJECT_ID', 'DATE', 'CLEAN_TEXT']
        last_note = last_note.sort_values(by=select_cols, ascending=False).drop_duplicates('SUBJECT_ID')[select_cols]
        ##
        last_note_sp = last_note.to_spark()
        patient_ids_sp = patient_ids.to_dataframe().to_spark()
        last_note_sp = last_note_sp.join(F.broadcast(patient_ids_sp), last_note_sp.SUBJECT_ID == patient_ids_sp.SUBJECT_ID, 'left_semi') 
        last_note = last_note_sp.to_pandas_on_spark()
        ##
    return last_note


def build_feats(df: ps.frame.DataFrame, aggs: list, train_ids: ps.frame.DataFrame = None, low_thresh: int = None, is_diag: bool = False) -> ps.frame.DataFrame:
    '''Build feature aggregations for patient.

    Args:
        agg: list of aggregations to use
        train_ids: if not empty, only features that exist in the train set 
            will be used
        low_thresh: if not empty, only features that more than low_thresh
            patients have will be used
        
        Returns schema: SUBJECT_ID FEATURE_NAME  ...agg_cols...
    '''
    cols_to_use = ['SUBJECT_ID', 'FEATURE_NAME']
    df_sp = df.to_spark()
    if train_ids is not None:
        train_ids_sp = train_ids.to_spark()
        train_df_sp = df_sp.join(F.broadcast(train_ids_sp), df_sp.SUBJECT_ID == train_ids_sp.SUBJECT_ID, 'left_semi')
        df_sp = df_sp.join(F.broadcast(train_df_sp), df_sp.FEATURE_NAME == train_df_sp.FEATURE_NAME, 'left_semi')
        df = df_sp.to_pandas_on_spark()

    if low_thresh is not None:
        deduplicated = df.drop_duplicates(cols_to_use)
        # Count freq. of FEATURE_NAME (categorical variable)
        count = deduplicated.FEATURE_NAME.value_counts()
        count = count.to_frame().reset_index()
        count = count.rename(columns={'FEATURE_NAME': 'COUNT_TMP', 'index': 'FEATURE_NAME'})
        features_to_leave = count[count['COUNT_TMP'] > low_thresh]
        features_to_leave_sp = features_to_leave.to_spark().drop('COUNT_TMP')
        df_sp = df_sp.join(F.broadcast(features_to_leave_sp), df_sp.FEATURE_NAME == features_to_leave_sp.FEATURE_NAME, 'left_semi')
        df = df_sp.to_pandas_on_spark()

    # drop extraneous col
    df = df.drop('DATE')
    if is_diag == True: # TODO fix VALUE should be True or false
        # Diag requires custom pyspark aggregation
        df_sp = df.to_spark()
        grouped_sp = df_sp.groupBy(*cols_to_use).agg(
            F.sum('VALUE').alias('TOTAL_VALUE_TMP'),
        )
        grouped = grouped_sp.to_pandas_on_spark()
        grouped = grouped[grouped['TOTAL_VALUE_TMP'] > 0].drop('TOTAL_VALUE_TMP')
    else:
        grouped = df.groupby(cols_to_use).agg(aggs)

    return grouped


def _write_spark_dfs_to_disk(deceased_to_date: ps.frame.DataFrame, train_ids: ps.frame.DataFrame, test_ids: ps.frame.DataFrame, last_note_tokenized: ps.frame.DataFrame, diag_built: ps.frame.DataFrame, labs_built: ps.frame.DataFrame, meds_built: ps.frame.DataFrame, last_note: ps.frame.DataFrame):
    '''
    Writes all input dataframes to disk
    '''
    deceased_to_date_sp = deceased_to_date.to_spark()
    train_ids_sp = train_ids.to_spark()
    test_ids_sp = test_ids.to_spark()
    diag_built_sp = diag_built.to_spark()
    meds_build_sp = meds_built.to_spark()
    labs_built_sp = labs_built.to_spark()
    last_note_sp = last_note.to_spark()
    last_note_tokenized_sp = last_note_tokenized.to_spark()

    deceased_to_date_sp.write.mode('overwrite').json(os.path.join(PATH_PROCESSED, 'spark-etl', 'deceased_to_date.json'))
    print('done writing deceased_to_date')
    train_ids_sp.write.mode('overwrite').json(os.path.join(PATH_PROCESSED, 'spark-etl', 'train_ids.json'))
    print('done writing train_ids')
    test_ids_sp.write.mode('overwrite').json(os.path.join(PATH_PROCESSED, 'spark-etl', 'test_ids.json'))
    print('done writing test_ids')

    diag_built_sp.write.mode('overwrite').csv(os.path.join(PATH_PROCESSED, 'spark-processed-features', 'diag_built.csv'))
    print('done writing diag_built')
    meds_build_sp.write.mode('overwrite').csv(os.path.join(PATH_PROCESSED, 'spark-processed-features', 'meds_built.csv'))
    print('done writing meds_built')

    ###
    labs_built_sp.write.mode('overwrite').csv(os.path.join(PATH_PROCESSED, 'spark-processed-features', 'labs_built.csv'))
    print('done writing labs_built')
    last_note_tokenized_sp.write.mode('overwrite').csv(os.path.join(PATH_PROCESSED, 'spark-processed-features', 'last_note_tokenized.csv'))
    print('done writing last_note_tokenized')
    last_note_sp.write.mode('overwrite').csv(os.path.join(PATH_PROCESSED, 'spark-processed-features', 'last_note.csv'))
    print('done writing last_note')


def main():
    '''
    Main preprocessing logic
    '''
    # Get patient sample
    patient_ids, patients_sample, deceased_to_date = get_patient_sample()
    # get relevant MIMIC data for sample
    diag_preprocessed, lab_preprocessed, meds_preprocessed, notes_preprocessed = preprocess(patient_ids)
    feature_sets = [diag_preprocessed, lab_preprocessed, meds_preprocessed]

    date = define_train_period(deceased_to_date, *feature_sets)

    # Choose last note for each patient
    last_note = get_last_note(patient_ids, notes_preprocessed, date, as_tokenized=False)

    ### Add transformer embeddings used in pretrained model
    last_note_tokenized = get_last_note(patient_ids, notes_preprocessed, date, as_tokenized=True)

    # All features in feature_prepocessed form are features with columns ['SUBJECT_ID', 'FEATURE_NAME', 'DATE', 'VALUE], which can be later used for any of the aggregations we'd like.

    ### Feature construction
    # We are going to do a train test split based on patients to validate our model. We will only use those features that appear in the train set. Also, we will only use features that are shared between many patients (we will define 'many' manually for each of the feature sets).  
    # This way we will lose some patients who don't have 'popular' features, but that's fine since our goal is to compare similar patients, not to train the best model.
    ##
    patient_ids_sp = patient_ids.to_dataframe().to_spark()
    train_ids_sp, test_ids_sp = patient_ids_sp.randomSplit([TRAIN_SIZE, 1.0 - TRAIN_SIZE], seed=RANDOM_SEED)
    train_ids, test_ids = train_ids_sp.to_pandas_on_spark(), test_ids_sp.to_pandas_on_spark()

    #### Feat calculations
    diag, lab, med = _clean_up_feature_sets(*feature_sets, date=date)
    meds_built = build_feats(med, aggs=['mean', 'count'], train_ids=train_ids, low_thresh=50)
    # Diag requires custom handling
    diag_built = build_feats(diag, aggs=None, train_ids=train_ids, low_thresh=30, is_diag=True)
    labs_built = build_feats(lab, aggs=['mean', 'max', 'min'], train_ids=train_ids, low_thresh=50)

    _write_spark_dfs_to_disk(deceased_to_date, train_ids, test_ids, last_note_tokenized, diag_built, labs_built, meds_built, last_note)


if __name__ == '__main__':
    main()
