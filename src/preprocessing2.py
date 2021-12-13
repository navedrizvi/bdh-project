'''
Downstream of preprocessing.py (runs on single machine)
'''
import os
from typing import List, Tuple

import pandas as pd
import pyspark.pandas as ps
from sklearn.feature_extraction.text import TfidfVectorizer

PATH_PROCESSED = '../data/processed/'


def pivot_aggregation(df_local: pd.DataFrame, fill_value: int = None, use_sparse: bool = True) -> pd.DataFrame:
    ''' Make sparse pivoted table with SUBJECT_ID as index. '''
    pivoted_local = df_local.unstack()
    if fill_value is not None:
        pivoted_local = pivoted_local.fillna(fill_value)
    
    if use_sparse:
        pivoted_local = pivoted_local.astype(pd.SparseDtype('float', fill_value))
    
    pivoted_local.columns = [f'{col[-1]}_{col[1]}' for col in pivoted_local.columns]
    return pivoted_local


def get_tf_idf_feats(latest_note: pd.DataFrame) -> pd.DataFrame:
    '''
    Returns TF-IDF features generated on @latest_note
    '''
    vectorizer = TfidfVectorizer(max_features=200)
    tf_idf = vectorizer.fit_transform(latest_note.CLEAN_TEXT)
    select_cols = [f'TFIDF_{feat}' for feat in vectorizer.get_feature_names()]
    tf_idf_feats = pd.DataFrame.sparse.from_spmatrix(tf_idf, columns=select_cols, index=latest_note.SUBJECT_ID)
    return tf_idf_feats


def _read_spark_dfs_from_disk() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Reads output of preprocessing.py
    '''
    diag_built = ps.read_csv(os.path.join(PATH_PROCESSED, 'spark-processed-features', 'diag_built.csv')).to_pandas()
    print('done reading diag_built')
    meds_built = ps.read_csv(os.path.join(PATH_PROCESSED, 'spark-processed-features', 'meds_built.csv')).to_pandas()
    print('done reading meds_built')
    labs_built = ps.read_csv(os.path.join(PATH_PROCESSED, 'spark-processed-features', 'labs_built.csv')).to_pandas()
    print('done reading labs_built')
    last_note = pd.read_csv(os.path.join(PATH_PROCESSED, 'spark-processed-features', 'last_note.csv')).to_pandas()
    print('done reading last_note')
    return diag_built, meds_built, labs_built, last_note


def _write_local_dfs_to_disk(feats_to_train_on: List[pd.DataFrame], tf_idf_notes_feats: pd.DataFrame):
    '''
    Writes pandas dfs in local memory to disk
    '''
    for i, feat in enumerate(feats_to_train_on):
        feat.to_csv(os.path.join(PATH_PROCESSED, 'spark-processed-features', f'training_feat{i}.csv'))
    tf_idf_notes_feats.to_csv(os.path.join(PATH_PROCESSED, 'spark-processed-features', 'tf_idf_notes_feats.csv'))


def main():
    # make sparse pivoted tables
    # use local node for computing sparse matrix and TF-IDF
    diag_built, labs_built, meds_built, last_note = _read_spark_dfs_from_disk()
    diag_final = pivot_aggregation(diag_built, fill_value=0)
    labs_final = pivot_aggregation(labs_built, fill_value=0)
    meds_final = pivot_aggregation(meds_built, fill_value=0)
    tf_idf_notes_feats = get_tf_idf_feats(last_note)

    feats_to_train_on = [diag_final, meds_final, labs_final]

    # writes feats_to_train_on and tf_idf_notes_feats which are fed into ml.py
    _write_local_dfs_to_disk(feats_to_train_on=feats_to_train_on, tf_idf_notes_feats=tf_idf_notes_feats)


if __name__ == '__main__':
    main()