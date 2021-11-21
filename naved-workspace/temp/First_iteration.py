# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

print(_IMPORTS)

import re

# +
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

from sklearn.model_selection import train_test_split
# -

# ### Sampling on patients

PATH = "../data/raw/"
PATH_PROCESSED = "../data/processed/"
PATH_DATASETS = "../data/train/"

SAMPLE_SIZE = 10000
RANDOM_SEED = 1

patients = pd.read_csv(os.path.join(PATH, "PATIENTS.csv.gz"))
patients.EXPIRE_FLAG.value_counts()

#ensuring every patient is unique
print(f"{patients.SUBJECT_ID.nunique()} unique patients in {len(patients)} rows")
#sampling random patients
patients_sample = patients.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

# +
SAMPLE_IDS = set(patients_sample.SUBJECT_ID)

with open(os.path.join(PATH_PROCESSED, "SAMPLE_IDS.json"), 'w') as f:
    json.dump({'ids': list(SAMPLE_IDS)}, f)
    
with open(os.path.join(PATH_PROCESSED, "SAMPLE_IDS.json"), 'r') as f:
    SAMPLE_IDS = set(json.load(f)['ids'])
# -

patients_sample = patients[patients.SUBJECT_ID.isin(SAMPLE_IDS)]

patients_sample.head()

DECEASED_TO_DATE = patients_sample[patients_sample.EXPIRE_FLAG == 1]\
    .set_index('SUBJECT_ID').DOD.map(lambda x: pd.to_datetime(x).date()).to_dict()


def get_data_for_sample(sample_ids: set, 
                        file_name: str, 
                        chunksize: int = 10_000) -> pd.DataFrame:
    """Get the data only relevant for the sample."""
    full_path = os.path.join(PATH, file_name)
    iterator = pd.read_csv(full_path, iterator=True, chunksize=chunksize)
    return pd.concat([chunk[chunk.SUBJECT_ID.isin(SAMPLE_IDS)] for chunk in tqdm(iterator)])


# +
### Incomplete: Using AWS glue, for now I will do stuff locally
# import boto3
# Creating the low level functional client
# client = boto3.client(
#     'glue',
#     aws_access_key_id = '',
#     aws_secret_access_key = '',
#     region_name = 'us-east-1'
# )
# clientResponse = client.get_table(DatabaseName="mimiciii",Name="admissions")
# -

# ### Build features

# All features in feature_prepocessed form are features with columns ['SUBJECT_ID', 'FEATURE_NAME', 'DATE', 'VALUE], which can be later used for any of the aggregations we'd like.

admissions = get_data_for_sample(SAMPLE_IDS, "ADMISSIONS.csv.gz")

admissions['ADMITTIME'] = pd.to_datetime(admissions.ADMITTIME).dt.date

diagnoses = get_data_for_sample(SAMPLE_IDS, "DIAGNOSES_ICD.csv.gz")

lab_results = get_data_for_sample(SAMPLE_IDS, "LABEVENTS.csv.gz", chunksize=100_000)

meds = get_data_for_sample(SAMPLE_IDS, "PRESCRIPTIONS.csv.gz")

# #### Diagnoses

diagnoses['ICD9_CODE'] = "ICD9_" + diagnoses['ICD9_CODE']

adm_cols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']
diagnoses = diagnoses.merge(admissions[adm_cols], on=['SUBJECT_ID', 'HADM_ID'])

dropper = ['ROW_ID', 'SEQ_NUM', 'HADM_ID']
renamer = {'ICD9_CODE': 'FEATURE_NAME', 'ADMITTIME': 'DATE'}
diag_preprocessed = diagnoses.drop(columns=dropper).rename(columns=renamer)
diag_preprocessed['VALUE'] = 1

diag_preprocessed.head()

# #### Labs

lab_results['DATE'] = pd.to_datetime(lab_results['CHARTTIME']).dt.date

lab_results['FEATURE_NAME'] = "LAB_" + lab_results['ITEMID'].astype(str)

dropper = ['ROW_ID', 'HADM_ID', 'VALUE', 'VALUEUOM', 'FLAG', 'ITEMID', 'CHARTTIME']
renamer = {'VALUENUM': 'VALUE'}
lab_preprocessed = lab_results.drop(columns=dropper).rename(columns=renamer)

lab_preprocessed.head()

# #### Meds

meds = meds[meds.ENDDATE.notna()]

meds['DATE'] = pd.to_datetime(meds['ENDDATE']).dt.date


def find_mean_dose(dose: str) -> float:
    if pd.isnull(dose):
        return 0
    try:
        cleaned = re.sub(r'[A-Za-z,>< ]', '', dose)
        parts = cleaned.split('-')
        return np.array(parts).astype(float).mean()
    except:
        print(dose)


meds['VALUE'] = meds['DOSE_VAL_RX'].map(find_mean_dose)

meds['FEATURE_NAME'] = "MED_" + meds['GSN'].astype(str)

dropper = [col for col in meds.columns if col not in {'SUBJECT_ID', 'DATE', 'FEATURE_NAME', 'VALUE'}]
meds_preprocessed = meds.drop(columns=dropper).rename(columns=renamer)

meds_preprocessed.head()


# #### Notes

# Here we can preprocess notes. Later the same things can be done using Spark.

def get_notes(sample_ids: set = None,
              note_path: str = "NOTEEVENTS.csv.gz",
              chunksize: int = 10_000) -> pd.DataFrame:
    """Get all notes or only those relevant for the sample."""
    if sample_ids is None:
        return pd.read_csv(os.path.join(PATH, note_path))
    return get_data_for_sample(sample_ids, note_path, chunksize)


# +
notes_sample = get_notes(SAMPLE_IDS)
# notes_sample.to_csv(os.path.join(PATH_PROCESSED, 'SAMPLE_NOTES.csv'), index=False)

notes = pd.read_csv(os.path.join(PATH_PROCESSED, 'SAMPLE_NOTES.csv'))
# -

notes.head()

notes['DATE'] = pd.to_datetime(notes['CHARTDATE']).dt.date


def clean_text(note: str):
    cleaned = re.sub(r'[^\w]', ' ', note).replace("_", " ")
    removed_spaces = re.sub(' +', ' ', cleaned)
    lower = removed_spaces.lower()
    return lower


notes['CLEAN_TEXT'] = notes['TEXT'].map(clean_text)

notes.head()

# ### Feature construction

# We need to take into account only the events that happened during the observation window. The end of observation window is N days before death for deceased patients and date of last event for alive patients. We can have several sets of events (e.g. labs, diags, meds), so we need to choose the latest date out of those.

OBSERVATION_WINDOW = 2000
PREDICTION_WINDOW = 50


def define_train_period(*feature_sets, 
                        obs_w: int = OBSERVATION_WINDOW, 
                        pred_w: int = PREDICTION_WINDOW) -> (dict, dict):
    """Create SUBJECT_ID -> earliest_date and SUBJECT_ID -> last date dicts."""
    cols = ['SUBJECT_ID', 'DATE']
    all_feats = pd.concat([feats[cols] for feats in feature_sets])
    last_date_base = all_feats.groupby('SUBJECT_ID').DATE.max().to_dict()
    last_date = {subj_id: date
                 for subj_id, date in last_date_base.items()
                 if subj_id not in DECEASED_TO_DATE}
    subtracted_pred_w = {subj_id: date - dt.timedelta(days=pred_w)
                         for subj_id, date in DECEASED_TO_DATE.items()}
    last_date.update(subtracted_pred_w)
    earliest_date = {subj_id: date - dt.timedelta(days=obs_w)
                     for subj_id, date in last_date.items()}
    return earliest_date, last_date


def clean_up_feature_sets(*feature_sets, earliest_date: dict, last_date: dict) -> list:
    """Leave only features from inside the observation window."""
    results = []
    for feats in feature_sets:
        results.append(feats[(feats.DATE < feats.SUBJECT_ID.map(last_date))
                             & (feats.DATE >= feats.SUBJECT_ID.map(earliest_date))])
    return results


use_feature_sets = [diag_preprocessed, lab_preprocessed, meds_preprocessed]

earliest_date, last_date = define_train_period(*use_feature_sets)

diag, lab, med = clean_up_feature_sets(*use_feature_sets, earliest_date=earliest_date, last_date=last_date)

# #### Feat calculations

# We are going to do a train test split based on patients to validate our model. We will only use those features that appear in the train set. Also, we will only use features that are shared between many patients (we will define "many" manually for each of the feature sets).  
#
# This way we will lose some patients who don't have "popular" features, but that's fine since our goal is to compare similar patients, not to train the best model.

TRAIN_SIZE = 0.8

train_ids, test_ids = train_test_split(list(SAMPLE_IDS), train_size=TRAIN_SIZE, random_state=RANDOM_SEED)


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
        train_feats = set(train_df.FEATURE_NAME)
        df = df[df.FEATURE_NAME.isin(train_feats)]
        print(f"Feats after leaving only train: {len(train_feats)}")
        
    if low_thresh is not None:
        deduplicated = df.drop_duplicates(cols_to_use)
        count = Counter(deduplicated.FEATURE_NAME)
        features_to_leave = set(feat for feat, cnt in count.items() if cnt > low_thresh)
        df = df[df.FEATURE_NAME.isin(features_to_leave)]
        print(f"Feats after removing rare: {len(features_to_leave)}")
    
    grouped = df.groupby(cols_to_use).VALUE.agg(agg)
    return grouped


def pivot_aggregation(df: pd.DataFrame, fill_value = None, use_sparse: bool = True) -> pd.DataFrame:
    """Make sparse pivoted table with SUBJECT_ID as index."""
    pivoted = df.unstack()
    if fill_value is not None:
        pivoted = pivoted.fillna(fill_value)
    
    if use_sparse:
        pivoted = pivoted.astype(pd.SparseDtype("float", fill_value))
    
    pivoted.columns = [f"{col[-1]}_{col[1]}" for col in pivoted.columns]
    return pivoted


diag_built = build_feats(diag, agg=[lambda x: x.sum() > 0], train_ids=train_ids, low_thresh=30)

diag_final = pivot_aggregation(diag_built, fill_value=0)

diag_final.head()

labs_built = build_feats(lab, agg=['mean', 'max', 'min'], train_ids=train_ids, low_thresh=50)

labs_final = pivot_aggregation(labs_built, fill_value=0)

labs_final.head()

meds_built = build_feats(med, agg=['mean', 'count'], train_ids=train_ids, low_thresh=50)

meds_final = pivot_aggregation(meds_built, fill_value=0)

meds_final.head()

# ### Model Training

# We will use random forest to automatically incorporate feature interrelations into our model.

from sklearn.ensemble import RandomForestClassifier

feats_to_train_on = [diag_final, meds_final, labs_final]

df_final = pd.concat(feats_to_train_on, axis=1).fillna(0)

target = pd.Series(df_final.index.isin(DECEASED_TO_DATE), index=df_final.index, name='target')

target.value_counts()

# +
# pd.concat([df_final, target], axis=1).to_csv(os.path.join(PATH_DATASETS, 'diag_med_lab.csv'))
# -

train_loc = df_final.index.isin(train_ids)


def train_and_predict(df: pd.DataFrame, target: pd.Series, train_loc: pd.Series, classifier) -> np.array:
    try:
        classifier.fit(df_final[train_loc], target[train_loc])
        pred = classifier.predict_proba(df_final[~train_loc])[:, 1]
    except AttributeError:
        classifier.fit(df_final[train_loc].sparse.to_dense(), target[train_loc])
        pred = classifier.predict_proba(df_final[~train_loc].sparse.to_dense())[:, 1]
    return pred


cl = RandomForestClassifier(random_state=RANDOM_SEED)

pred = train_and_predict(df_final, target, train_loc, cl)

roc_auc_score(target[~train_loc], pred)

pd.Series(cl.feature_importances_, index=df_final.columns).sort_values(ascending=False).iloc[:10]

# ### Add note TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

last_note = clean_up_feature_sets(notes, earliest_date=earliest_date, last_date=last_date)[0]

cols = ['SUBJECT_ID', 'DATE', 'CLEAN_TEXT']
last_note = last_note.sort_values(by=cols, ascending=False).drop_duplicates('SUBJECT_ID')[cols]

last_note = last_note[last_note.SUBJECT_ID.isin(SAMPLE_IDS)]

last_note

vectorizer = TfidfVectorizer(max_features=200)
tf_idf = vectorizer.fit_transform(last_note.CLEAN_TEXT)


cols = [f"TFIDF_{feat}" for feat in vectorizer.get_feature_names()]
tf_idf_feats = pd.DataFrame.sparse.from_spmatrix(tf_idf, columns=cols, index=last_note.SUBJECT_ID)

# #### Training

# making sure no new rows are added
feats_to_train_on = [diag_final, meds_final, labs_final]
df_final = pd.concat(feats_to_train_on, axis=1).fillna(0)
tf_idf_feats = tf_idf_feats[tf_idf_feats.index.isin(df_final.index)]

feats_to_train_on = [diag_final, meds_final, labs_final, tf_idf_feats]

df_final = pd.concat(feats_to_train_on, axis=1).fillna(0)

target = pd.Series(df_final.index.isin(DECEASED_TO_DATE), index=df_final.index, name='target')

target.value_counts()

# +
# pd.concat([df_final, target], axis=1).to_csv(os.path.join(PATH_DATASETS, 'diag_med_lab_tf_idf.csv'))
# -

cl = RandomForestClassifier(random_state=RANDOM_SEED)

train_loc = df_final.index.isin(train_ids)

pred = train_and_predict(df_final, target, train_loc, cl)

roc_auc_score(target[~train_loc], pred)

pd.Series(cl.feature_importances_, index=df_final.columns).sort_values(ascending=False).iloc[:10]

# Better results, mostly from getting patient discharge information from notes.

# ### Add transformer embeddings

last_note

last_note = clean_up_feature_sets(notes, earliest_date=earliest_date, last_date=last_date)[0]

cols = ['SUBJECT_ID', 'DATE', 'TEXT']
last_note = last_note.sort_values(by=cols, ascending=False).drop_duplicates('SUBJECT_ID')[cols]

last_note = last_note[last_note.SUBJECT_ID.isin(SAMPLE_IDS)]


def prepare_text_for_tokenizer(text: str):
    cleaned = ('. ').join(text.splitlines())
    removed_symbols = re.sub('[\[\]\*\_#:?!]+', ' ', cleaned)
    removed_spaces = re.sub(' +', ' ', removed_symbols)
    removed_dots = re.sub('\. \.| \.', '.', removed_spaces)
    removed_duplicated_dots = re.sub('\.+', '.', removed_dots)
    return removed_duplicated_dots


prepare_text_for_tokenizer(last_note['TEXT'].iloc[0])

last_note['TO_TOK'] = last_note.TEXT.map(prepare_text_for_tokenizer)

last_note = last_note.reset_index(drop=True)

last_note.head()

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

import torch

tokenizer = AutoTokenizer.from_pretrained("deepset/covid_bert_base")

config = AutoConfig.from_pretrained('deepset/covid_bert_base', output_hidden_states=True, output_attentions=True)

# + tags=[]
model = AutoModelForMaskedLM.from_pretrained("deepset/covid_bert_base", config=config)
model.eval()


# -

def get_vector_for_text(text):
    """This is ugly and slow."""
    encoding = tokenizer(text, 
                         add_special_tokens=True, 
                         truncation=True, 
                         padding="max_length", 
                         return_attention_mask=True, 
                         return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
        hs = outputs.hidden_states
        token_embeddings = torch.stack(hs, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)
        token_vecs = hs[-2][0]
        text_embedding = torch.mean(token_vecs, dim=0)
        return text_embedding


# %%time
sentence = get_vector_for_text(last_note.TO_TOK.iloc[0])

torch.save(sentence, f"../data/embeddings/99283.pt")

for row_num, row in tqdm(last_note.iloc[0:].iterrows()):
    text = row['TO_TOK']
    subj_id = row['SUBJECT_ID']
    embedding = get_vector_for_text(text)
    torch.save(embedding, f"../data/embeddings/{subj_id}.pt")

# + tags=[]
from glob import glob
# -

subj_ids = []
embeds = []
for file in tqdm(glob('../data/embeddings/*')):
    name = file.split('/')[-1]
    subj_id = int(name.split('.')[0])
    embedding = torch.load(file)
    subj_ids.append(subj_id)
    embeds.append(np.array(embedding))

embed_df = pd.DataFrame(embeds, index=subj_ids)

embed_df.columns = [f"EMBED_{i}" for i in embed_df.columns]

embed_df.head()

# #### Training with embeds

# making sure no new rows are added
feats_to_train_on = [diag_final, meds_final, labs_final]
df_final = pd.concat(feats_to_train_on, axis=1).fillna(0)
embed_df = embed_df[embed_df.index.isin(df_final.index)]

feats_to_train_on = [diag_final, meds_final, labs_final, embed_df]

df_final = pd.concat(feats_to_train_on, axis=1).fillna(0)

target = pd.Series(df_final.index.isin(DECEASED_TO_DATE), index=df_final.index, name='target')

target.value_counts()

# +
# pd.concat([df_final, target], axis=1).to_csv(os.path.join(PATH_DATASETS, 'diag_med_lab_embed.csv'))
# -

cl = RandomForestClassifier(random_state=RANDOM_SEED)

train_loc = df_final.index.isin(train_ids)

pred = train_and_predict(df_final, target, train_loc, cl)

roc_auc_score(target[~train_loc], pred)

pd.Series(cl.feature_importances_, index=df_final.columns).sort_values(ascending=False).iloc[:10]


