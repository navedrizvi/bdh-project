import os
from typing import Dict, List, NamedTuple, Set, Tuple, Union
import re
import json
import datetime as dt
from collections import Counter

# from tqdm.auto import tqdm
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
from pyspark.pandas import read_csv


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

from pyspark.sql.functions import to_timestamp, to_date, substring


def get_patient_sample() -> Tuple[pyspark.pandas.series.Series[int], pyspark.pandas.frame.DataFrame, pyspark.pandas.frame.DataFrame]:
    patients = read_csv(PATIENTS_PATH)
    sample_ids = patients.SUBJECT_ID
    # Moratality set
    deceased_patients = patients[patients.EXPIRE_FLAG == 1] 
    deceased_patients = deceased_patients[['SUBJECT_ID', 'DOD']]
    # first 10 characters of DOD column is date (we're ignoring time)
    deceased_patients = deceased_patients.to_spark()
    deceased_patients = deceased_patients.select('SUBJECT_ID', substring('DOD', 0, 10).alias('DOD'))
    deceased_patients = deceased_patients.to_pandas_on_spark()

    return sample_ids, patients, deceased_patients


def _get_data_for_sample(patient_ids: pyspark.pandas.series.Series[int]) -> pyspark.pandas.frame.DataFrame:
    '''Get the data only relevant for the sample.'''
    full_path = RAW_BASE_PATH.format(fname=file_name)
    iterator = read_csv(path=full_path)
    return pd.concat([chunk[chunk.SUBJECT_ID.isin(patient_ids)]])


def _find_mean_dose(dose: str) -> Union[float, None]:
    if pd.isnull(dose):
        return 0
    try:
        cleaned = re.sub(r'[A-Za-z,>< ]', '', dose)
        parts = cleaned.split('-')
        return np.array(parts).astype(float).mean()
    except:
        # print(dose) # TODO fix to address the following conditions in src data:
        # ['50/500', '250/50', '500//50', '800/160', '-0.5-2', '0.3%', 'About-CM1000', 'one', '500/50', '12-', '-15-30', '1%', 'Hold Dose', '1.25/3', '1%', ': 5-10', '0.63/3', '0.63/3', '20-', '1.26mg/6', '1.26mg/6', '0.63 mg/3', '1.2/1']
        return None
        

def _clean_text(note: str) -> str:
    cleaned = re.sub(r'[^\w]', ' ', note).replace('_', ' ')
    removed_spaces = re.sub(' +', ' ', cleaned)
    lower = removed_spaces.lower()
    return lower


def preprocess(patient_ids: pyspark.pandas.series.Series[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ''' Returns preprocessed dfs containg records for @patient_ids
    '''
    admissions = _get_data_for_sample(patient_ids, ADMISSIONS_FNAME)
    lab_results = _get_data_for_sample(patient_ids, LABEVENTS_FNAME)
    diagnoses = _get_data_for_sample(patient_ids, DIAGNOSES_FNAME)
    meds = _get_data_for_sample(patient_ids, PRESCRIPTIONS_FNAME)
    notes_preprocessed = _get_data_for_sample(patient_ids, NOTES_FNAME)

    admissions['ADMITTIME'] = pd.to_datetime(admissions.ADMITTIME).dt.date
    print('done processing admissions')
    #### Diagnoses

    diagnoses['ICD9_CODE'] = 'ICD9_' + diagnoses['ICD9_CODE']
    adm_cols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']
    diagnoses = diagnoses.merge(admissions[adm_cols], on=['SUBJECT_ID', 'HADM_ID'])
    dropper = ['ROW_ID', 'SEQ_NUM', 'HADM_ID']
    renamer = {'ICD9_CODE': 'FEATURE_NAME', 'ADMITTIME': 'DATE'}
    diag_preprocessed = diagnoses.drop(columns=dropper).rename(columns=renamer)
    diag_preprocessed['VALUE'] = 1
    print('done processing diags')

    #### Labs
    lab_results['DATE'] = pd.to_datetime(lab_results['CHARTTIME']).dt.date
    lab_results['FEATURE_NAME'] = 'LAB_' + lab_results['ITEMID'].astype(str)
    dropper = ['ROW_ID', 'HADM_ID', 'VALUE', 'VALUEUOM', 'FLAG', 'ITEMID', 'CHARTTIME']
    renamer = {'VALUENUM': 'VALUE'}
    lab_preprocessed = lab_results.drop(columns=dropper).rename(columns=renamer)
    print('done processing labs')

    #### Meds
    meds = meds[meds.ENDDATE.notna()]
    meds['DATE'] = pd.to_datetime(meds['ENDDATE']).dt.date
    meds['VALUE'] = meds['DOSE_VAL_RX'].map(_find_mean_dose)
    meds['FEATURE_NAME'] = 'MED_' + meds['GSN'].astype(str)
    dropper = [col for col in meds.columns if col not in {'SUBJECT_ID', 'DATE', 'FEATURE_NAME', 'VALUE'}]
    meds_preprocessed = meds.drop(columns=dropper).rename(columns=renamer)
    print('done processing meds')

    # Here we can preprocess notes. Later the same things can be done using Spark # TODO 2
    #### Notes
    notes_preprocessed['DATE'] = pd.to_datetime(notes_preprocessed['CHARTDATE']).dt.date
    notes_preprocessed['CLEAN_TEXT'] = notes_preprocessed['TEXT'].map(_clean_text)
    print('done processing notes')

    return diag_preprocessed, lab_preprocessed, meds_preprocessed, notes_preprocessed
##########################

####### QA
#ensuring every patient is unique
# print(f'{patients.SUBJECT_ID.nunique()} unique patients in {len(patients)} rows')
# TODO (add more form nb) calculate summary stats for data to ensure quality (asserts, can be approx)
# print(f'{patients.SUBJECT_ID.nunique()} unique patients in {len(patients)} rows')
###########


## Feature engr. helpers
def define_train_period(deceased_to_date: pd.Series, *feature_sets: List[pd.DataFrame], 
                        obs_w: int = OBSERVATION_WINDOW, 
                        pred_w: int = PREDICTION_WINDOW) -> Tuple[Dict, Dict]:
    '''Create SUBJECT_ID -> earliest_date and SUBJECT_ID -> last date dicts.'''
    cols = ['SUBJECT_ID', 'DATE']
    all_feats = pd.concat([feats[cols] for feats in feature_sets])
    last_date_base = all_feats.groupby('SUBJECT_ID').DATE.max().to_dict()
    last_date = {subj_id: date
                 for subj_id, date in last_date_base.items()
                 if subj_id not in deceased_to_date}
    subtracted_pred_w = {subj_id: date - dt.timedelta(days=pred_w)
                         for subj_id, date in deceased_to_date.items()}
    last_date.update(subtracted_pred_w)
    earliest_date = {subj_id: date - dt.timedelta(days=obs_w)
                     for subj_id, date in last_date.items()}
    return earliest_date, last_date


def _clean_up_feature_sets(*feature_sets: List[pd.DataFrame], earliest_date: dict, last_date: dict) -> List[pd.DataFrame]:
    '''Leave only features from inside the observation window.'''
    results = []
    for feats in feature_sets:
        results.append(feats[(feats.DATE < feats.SUBJECT_ID.map(last_date))
                             & (feats.DATE >= feats.SUBJECT_ID.map(earliest_date))])
    return results


# TODO 2 use spark pandas and assert content correctness
def _prepare_text_for_tokenizer(text: str) -> str:
    cleaned = ('. ').join(text.splitlines())
    removed_symbols = re.sub('[\[\]\*\_#:?!]+', ' ', cleaned)
    removed_spaces = re.sub(' +', ' ', removed_symbols)
    removed_dots = re.sub('\. \.| \.', '.', removed_spaces)
    removed_duplicated_dots = re.sub('\.+', '.', removed_dots)
    return removed_duplicated_dots


def get_last_note(patient_ids: set, notes_preprocessed: pd.DataFrame, earliest_date: Dict, last_date: Dict, as_tokenized=False) -> pd.Series:
    if as_tokenized:
        last_note = _clean_up_feature_sets(notes_preprocessed, earliest_date=earliest_date, last_date=last_date)[0]
        select_cols = ['SUBJECT_ID', 'DATE', 'TEXT']
        last_note = last_note.sort_values(by=select_cols, ascending=False).drop_duplicates('SUBJECT_ID')[select_cols]
        last_note = last_note[last_note.SUBJECT_ID.isin(patient_ids)]
        last_note['TO_TOK'] = last_note.TEXT.map(_prepare_text_for_tokenizer)
        last_note = last_note.reset_index(drop=True)
    else:
        last_note = _clean_up_feature_sets(notes_preprocessed, earliest_date=earliest_date, last_date=last_date)[0]
        select_cols = ['SUBJECT_ID', 'DATE', 'CLEAN_TEXT']
        last_note = last_note.sort_values(by=select_cols, ascending=False).drop_duplicates('SUBJECT_ID')[select_cols]
        last_note = last_note[last_note.SUBJECT_ID.isin(patient_ids)]
    return last_note


def build_feats(df: pd.DataFrame, agg: list, train_ids: list = None, low_thresh: int = None) -> pd.DataFrame:
    '''Build feature aggregations for patient.
    
    Args:
        agg: list of aggregations to use
        train_ids: if not empty, only features that exist in the train set 
            will be used
        low_thresh: if not empty, only features that more than low_thresh
            patients have will be used
    '''
    cols_to_use = ['SUBJECT_ID', 'FEATURE_NAME']
    print(f'Total feats: {df.FEATURE_NAME.nunique()}')
    if train_ids is not None:
        train_df = df[df.SUBJECT_ID.isin(train_ids)]
        train_feats = set(train_df.FEATURE_NAME) #py
        df = df[df.FEATURE_NAME.isin(train_feats)]
        print(f'Feats after leaving only train: {len(train_feats)}') #py
        
    if low_thresh is not None:
        deduplicated = df.drop_duplicates(cols_to_use)
        count: Dict[str, int] = Counter(deduplicated.FEATURE_NAME) #py
        features_to_leave = set(feat for feat, cnt in count.items() if cnt > low_thresh) #py
        df = df[df.FEATURE_NAME.isin(features_to_leave)]
        print(f'Feats after removing rare: {len(features_to_leave)}') #py
    
    grouped = df.groupby(cols_to_use).agg(agg)
    return grouped


# TODO 3 use spark pandas and assert content correctness
def pivot_aggregation(df: pd.DataFrame, fill_value: int = None, use_sparse: bool = True) -> pd.DataFrame:
    '''Make sparse pivoted table with SUBJECT_ID as index.'''
    pivoted = df.unstack()
    if fill_value is not None:
        pivoted = pivoted.fillna(fill_value)
    
    if use_sparse:
        pivoted = pivoted.astype(pd.SparseDtype('float', fill_value))
    
    pivoted.columns = [f'{col[-1]}_{col[1]}' for col in pivoted.columns]
    return pivoted


# TODO 4 cleanup for TF-IDF lemmatise, remove stopwords


# TODO 4 need to do in spark somehow
def get_tf_idf_feats(last_note: pd.DataFrame) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(max_features=200)
    tf_idf = vectorizer.fit_transform(last_note.CLEAN_TEXT)
    select_cols = [f'TFIDF_{feat}' for feat in vectorizer.get_feature_names()]

    tf_idf_feats = pd.DataFrame.sparse.from_spmatrix(tf_idf, columns=select_cols, index=last_note.SUBJECT_ID)
    return tf_idf_feats

def write_to_disk(deceased_to_date: Dict[int, dt.date], train_ids: Set[int], test_ids: Set[int], feats_to_train_on: List[pd.DataFrame], tf_idf_notes_feats: pd.DataFrame, last_note_tokenized: pd.DataFrame):
    deceased_to_date = {k: v.isoformat() for k, v in deceased_to_date.items()}
    with open(os.path.join(PATH_PROCESSED, 'etl', "deceased_to_date.json"), 'w') as f:
        json.dump(deceased_to_date, f)

    with open(os.path.join(PATH_PROCESSED, 'etl', "train_ids.json"), 'w') as f:
        json.dump({'train_ids': list(train_ids)}, f)

    with open(os.path.join(PATH_PROCESSED, 'etl', "test_ids.json"), 'w') as f:
        json.dump({'test_ids': list(test_ids)}, f)

    for i, feat in enumerate(feats_to_train_on):
        feat.to_csv(os.path.join(PATH_PROCESSED, f'training_feat{i}.csv'))
    
    tf_idf_notes_feats.to_csv(os.path.join(PATH_PROCESSED, 'tf_idf_notes_feats.csv'))
    last_note_tokenized.to_csv(os.path.join(PATH_PROCESSED, 'last_note_tokenized.csv'))

def main():
    # Get patient sample
    sample_ids, patients_sample, deceased_to_date = get_patient_sample()
    # get relevant MIMIC data for sample
    diag_preprocessed, lab_preprocessed, meds_preprocessed, notes_preprocessed = preprocess(sample_ids)
    use_feature_sets = [diag_preprocessed, lab_preprocessed, meds_preprocessed]

    earliest_date, last_date = define_train_period(deceased_to_date, *use_feature_sets)
    # Choose last note for each patient
    last_note = get_last_note(sample_ids, notes_preprocessed, earliest_date, last_date, as_tokenized=False)

    ### Add transformer embeddings used in pretrained model
    last_note_tokenized = get_last_note(sample_ids, notes_preprocessed, earliest_date, last_date, as_tokenized=True)

    # All features in feature_prepocessed form are features with columns ['SUBJECT_ID', 'FEATURE_NAME', 'DATE', 'VALUE], which can be later used for any of the aggregations we'd like.

    ### Feature construction
    # We are going to do a train test split based on patients to validate our model. We will only use those features that appear in the train set. Also, we will only use features that are shared between many patients (we will define 'many' manually for each of the feature sets).  
    # This way we will lose some patients who don't have 'popular' features, but that's fine since our goal is to compare similar patients, not to train the best model.
    train_ids, test_ids = train_test_split(list(sample_ids), train_size=TRAIN_SIZE, random_state=RANDOM_SEED)
    diag, lab, med = _clean_up_feature_sets(*use_feature_sets, earliest_date=earliest_date, last_date=last_date)

    #### Feat calculations
    diag_built = build_feats(diag, agg=[lambda x: x.sum() > 0], train_ids=train_ids, low_thresh=30)
    labs_built = build_feats(lab, agg=['mean', 'max', 'min'], train_ids=train_ids, low_thresh=50)
    meds_built = build_feats(med, agg=['mean', 'count'], train_ids=train_ids, low_thresh=50)

    # make sparse pivoted tables
    diag_final = pivot_aggregation(diag_built, fill_value=0)
    labs_final = pivot_aggregation(labs_built, fill_value=0)
    meds_final = pivot_aggregation(meds_built, fill_value=0)

    feats_to_train_on = [diag_final, meds_final, labs_final]
    tf_idf_notes_feats = get_tf_idf_feats(last_note)

    write_to_disk(deceased_to_date, train_ids, test_ids, feats_to_train_on, tf_idf_notes_feats, last_note_tokenized)
    return deceased_to_date, train_ids, feats_to_train_on, tf_idf_notes_feats, last_note_tokenized
