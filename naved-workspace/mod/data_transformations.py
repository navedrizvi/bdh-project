import json
from typing import Dict, List, Tuple
import re
import datetime as dt
from collections import Counter
import os

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import ml

RAW_BASE_PATH = "../../data/raw/"
ADMISSIONS_FNAME =  "ADMISSIONS.csv.gz"
DIAGNOSES_FNAME = "DIAGNOSES_ICD.csv.gz"
LABEVENTS_FNAME = "LABEVENTS.csv.gz"
PRESCRIPTIONS_FNAME = "PRESCRIPTIONS.csv.gz"
PATIENTS_FNAME = "PATIENTS.csv.gz"
PATIENTS_PATH = os.path.join(RAW_BASE_PATH, PATIENTS_FNAME)

PATH_PROCESSED = "../../data/processed/"
NOTES_PATH = os.path.join(PATH_PROCESSED, 'SAMPLE_NOTES.csv')

PATIENT_SAMPLE_SIZE = 10000
TRAIN_SIZE = 0.8
# We need to take into account only the events that happened during the observation window. The end of observation window is N days before death for deceased patients and date of last event for alive patients. We can have several sets of events (e.g. labs, diags, meds), so we need to choose the latest date out of those.
OBSERVATION_WINDOW = 2000
PREDICTION_WINDOW = 50

RANDOM_SEED = 1


# TODO 1 use spark pandas and assert content correctness
def get_patient_sample() -> Tuple[set, pd.Series, pd.Series]:
    patients = pd.read_csv(PATIENTS_PATH)
    #sampling random patients
    patients_sample = patients.sample(n=PATIENT_SAMPLE_SIZE, random_state=RANDOM_SEED)
    sample_ids = set(patients_sample.SUBJECT_ID)  #py
    with open(os.path.join(PATH_PROCESSED, "SAMPLE_IDS.json"), 'w') as f:
        json.dump({'ids': list(sample_ids)}, f)
    with open(os.path.join(PATH_PROCESSED, "SAMPLE_IDS.json"), 'r') as f:
        sample_ids = set(json.load(f)['ids'])
    patients_sample = patients[patients.SUBJECT_ID.isin(sample_ids)]
    # Moratality set
    deceased_to_date = patients_sample[patients_sample.EXPIRE_FLAG == 1] \
        .set_index('SUBJECT_ID').DOD.map(lambda x: pd.to_datetime(x).date()).to_dict()
    return sample_ids, patients_sample, deceased_to_date


##########################
# TODO 5 use spark pandas and assert content correctness
def _get_data_for_sample(sample_ids: set,
                        file_name: str,
                        chunksize: int = 10_000) -> pd.DataFrame:
    """Get the data only relevant for the sample."""
    full_path = os.path.join(RAW_BASE_PATH, file_name)
    iterator = pd.read_csv(full_path, iterator=True, chunksize=chunksize)
    return pd.concat([chunk[chunk.SUBJECT_ID.isin(sample_ids)] for chunk in tqdm(iterator)])


def _find_mean_dose(dose: str) -> float:
    if pd.isnull(dose):
        return 0
    try:
        cleaned = re.sub(r'[A-Za-z,>< ]', '', dose)
        parts = cleaned.split('-')
        return np.array(parts).astype(float).mean()
    except:
        print(dose)


def _clean_text(note: str) -> str:
    cleaned = re.sub(r'[^\w]', ' ', note).replace("_", " ")
    removed_spaces = re.sub(' +', ' ', cleaned)
    lower = removed_spaces.lower()
    return lower


def preprocess(sample_ids: set) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ''' Returns preprocessed dfs containg records for @sample_ids '''
    admissions = _get_data_for_sample(sample_ids, ADMISSIONS_FNAME)
    diagnoses = _get_data_for_sample(sample_ids, DIAGNOSES_FNAME)
    lab_results = _get_data_for_sample(sample_ids, LABEVENTS_FNAME, chunksize=100_000)
    meds = _get_data_for_sample(sample_ids, PRESCRIPTIONS_FNAME)

    admissions['ADMITTIME'] = pd.to_datetime(admissions.ADMITTIME).dt.date

    #### Diagnoses
    diagnoses['ICD9_CODE'] = "ICD9_" + diagnoses['ICD9_CODE']
    adm_cols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']
    diagnoses = diagnoses.merge(admissions[adm_cols], on=['SUBJECT_ID', 'HADM_ID'])
    dropper = ['ROW_ID', 'SEQ_NUM', 'HADM_ID']
    renamer = {'ICD9_CODE': 'FEATURE_NAME', 'ADMITTIME': 'DATE'}
    diag_preprocessed = diagnoses.drop(columns=dropper).rename(columns=renamer)
    diag_preprocessed['VALUE'] = 1

    #### Labs
    lab_results['DATE'] = pd.to_datetime(lab_results['CHARTTIME']).dt.date
    lab_results['FEATURE_NAME'] = "LAB_" + lab_results['ITEMID'].astype(str)
    dropper = ['ROW_ID', 'HADM_ID', 'VALUE', 'VALUEUOM', 'FLAG', 'ITEMID', 'CHARTTIME']
    renamer = {'VALUENUM': 'VALUE'}
    lab_preprocessed = lab_results.drop(columns=dropper).rename(columns=renamer)
    lab_preprocessed = lab_results.drop(columns=dropper)

    #### Meds
    meds = meds[meds.ENDDATE.notna()]
    meds['DATE'] = pd.to_datetime(meds['ENDDATE']).dt.date
    meds['VALUE'] = meds['DOSE_VAL_RX'].map(_find_mean_dose)
    meds['FEATURE_NAME'] = "MED_" + meds['GSN'].astype(str)
    dropper = [col for col in meds.columns if col not in {'SUBJECT_ID', 'DATE', 'FEATURE_NAME', 'VALUE'}]
    meds_preprocessed = meds.drop(columns=dropper).rename(columns=renamer)

    # Here we can preprocess notes. Later the same things can be done using Spark # TODO 2
    #### Notes
    notes_preprocessed = pd.read_csv(NOTES_PATH)
    notes_preprocessed['DATE'] = pd.to_datetime(notes_preprocessed['CHARTDATE']).dt.date
    notes_preprocessed['CLEAN_TEXT'] = notes_preprocessed['TEXT'].map(_clean_text)

    return diag_preprocessed, lab_preprocessed, meds_preprocessed, notes_preprocessed
##########################

####### QA
#ensuring every patient is unique
# print(f"{patients.SUBJECT_ID.nunique()} unique patients in {len(patients)} rows")
# TODO (add more form nb) calculate summary stats for data to ensure quality (asserts, can be approx)
# print(f"{patients.SUBJECT_ID.nunique()} unique patients in {len(patients)} rows")
###########


## Feature engr. helpers
def define_train_period(deceased_to_date: pd.Series, *feature_sets: List[pd.DataFrame], 
                        obs_w: int = OBSERVATION_WINDOW, 
                        pred_w: int = PREDICTION_WINDOW) -> Tuple[Dict, Dict]:
    """Create SUBJECT_ID -> earliest_date and SUBJECT_ID -> last date dicts."""
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


# TODO 2 use spark pandas and assert content correctness
def _clean_up_feature_sets(*feature_sets: List[pd.DataFrame], earliest_date: dict, last_date: dict) -> List[pd.DataFrame]:
    """Leave only features from inside the observation window."""
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


# TODO 2 use spark pandas and assert content correctness
def get_last_note(sample_ids: set, notes_preprocessed: pd.DataFrame, earliest_date: Dict, last_date: Dict, as_tokenized=False) -> pd.Series:
    if as_tokenized:
        last_note = _clean_up_feature_sets(notes_preprocessed, earliest_date=earliest_date, last_date=last_date)[0]
        select_cols = ['SUBJECT_ID', 'DATE', 'TEXT']
        last_note = last_note.sort_values(by=select_cols, ascending=False).drop_duplicates('SUBJECT_ID')[select_cols]
        last_note = last_note[last_note.SUBJECT_ID.isin(sample_ids)]
        last_note['TO_TOK'] = last_note.TEXT.map(_prepare_text_for_tokenizer)
        last_note = last_note.reset_index(drop=True)
    else:
        last_note = _clean_up_feature_sets(notes_preprocessed, earliest_date=earliest_date, last_date=last_date)[0]
        select_cols = ['SUBJECT_ID', 'DATE', 'CLEAN_TEXT']
        last_note = last_note.sort_values(by=select_cols, ascending=False).drop_duplicates('SUBJECT_ID')[select_cols]
        last_note = last_note[last_note.SUBJECT_ID.isin(sample_ids)]
    return last_note

# TODO 3 use spark pandas and assert content correctness
def build_feats(df: pd.DataFrame, agg: list, train_ids: list = None, low_thresh: int = None) -> pd.DataFrame:
    """Build feature aggregations for patient.
    
    Args:
        agg: list of aggregations to use
        train_ids: if not empty, only features that exist in the train set 
            will be used
        low_thresh: if not empty, only features that more than low_thresh
            patients have will be used
    """
    cols_to_use = ['SUBJECT_ID', 'FEATURE_NAME']
    print(f"Total feats: {df.FEATURE_NAME.nunique()}")
    if train_ids is not None:
        train_df = df[df.SUBJECT_ID.isin(train_ids)]
        train_feats = set(train_df.FEATURE_NAME) #py
        df = df[df.FEATURE_NAME.isin(train_feats)]
        print(f"Feats after leaving only train: {len(train_feats)}") #py
        
    if low_thresh is not None:
        deduplicated = df.drop_duplicates(cols_to_use)
        count = Counter(deduplicated.FEATURE_NAME)
        features_to_leave = set(feat for feat, cnt in count.items() if cnt > low_thresh) #py
        df = df[df.FEATURE_NAME.isin(features_to_leave)]
        print(f"Feats after removing rare: {len(features_to_leave)}") #py
    
    grouped = df.groupby(cols_to_use).agg(agg)
    return grouped

# TODO 3 use spark pandas and assert content correctness
def pivot_aggregation(df: pd.DataFrame, fill_value: int = None, use_sparse: bool = True) -> pd.DataFrame:
    """Make sparse pivoted table with SUBJECT_ID as index."""
    pivoted = df.unstack()
    if fill_value is not None:
        pivoted = pivoted.fillna(fill_value)
    
    if use_sparse:
        pivoted = pivoted.astype(pd.SparseDtype("float", fill_value))
    
    pivoted.columns = [f"{col[-1]}_{col[1]}" for col in pivoted.columns]
    return pivoted


# TODO 4 cleanup for TF-IDF lemmatise, remove stopwords

# TODO 4 need to do in spark somehow
def get_tf_idf_feats(last_note: pd.DataFrame) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(max_features=200)
    tf_idf = vectorizer.fit_transform(last_note.CLEAN_TEXT)
    select_cols = [f'TFIDF_{feat}' for feat in vectorizer.get_feature_names()]

    tf_idf_feats = pd.DataFrame.sparse.from_spmatrix(tf_idf, columns=select_cols, index=last_note.SUBJECT_ID)
    return tf_idf_feats




def main():
    # Get patient sample
    patient_sample_ids, patients_sample, deceased_to_date = get_patient_sample()
    # get relevant MIMIC data for sample
    diag_preprocessed, lab_preprocessed, meds_preprocessed, notes_preprocessed = preprocess(patient_sample_ids)
    use_feature_sets = [diag_preprocessed, lab_preprocessed, meds_preprocessed]

    earliest_date, last_date = define_train_period(deceased_to_date, *use_feature_sets)
    ### Add note TF-IDF
    # Choose last note for each patient
    last_note = get_last_note(patient_sample_ids, notes_preprocessed, earliest_date, last_date, as_tokenized=False)

    ### Add transformer embeddings used in pretrained model
    last_note_tokenized = get_last_note(patient_sample_ids, notes_preprocessed, earliest_date, last_date, as_tokenized=True)

    # All features in feature_prepocessed form are features with columns ['SUBJECT_ID', 'FEATURE_NAME', 'DATE', 'VALUE], which can be later used for any of the aggregations we'd like.

    ### Feature construction
    # We are going to do a train test split based on patients to validate our model. We will only use those features that appear in the train set. Also, we will only use features that are shared between many patients (we will define "many" manually for each of the feature sets).  
    # This way we will lose some patients who don't have "popular" features, but that's fine since our goal is to compare similar patients, not to train the best model.
    train_ids, test_ids = train_test_split(list(patient_sample_ids), train_size=TRAIN_SIZE, random_state=RANDOM_SEED)
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

    ml.main(deceased_to_date, feats_to_train_on, train_ids, tf_idf_notes_feats, last_note_tokenized)




