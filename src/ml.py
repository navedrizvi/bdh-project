import enum
import os
from glob import glob
from typing import List, Tuple, Set

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pyspark.pandas as ps
from joblib import dump

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig


EMBEDDINGS_BASE_PATH = '../data/embeddings/'
MODELS_BASE_PATH = '../data/models/'
EMBEDDING_TEMPLATE = '../data/embeddings/{subj_id}.pt'

PRETRAINED_MODEL_NAME = 'deepset/covid_bert_base'

RANDOM_SEED = 1
PATH_PROCESSED = '../data/processed/'

class ModelType(enum.Enum):
    Baseline = 'Baseline'
    TF_IDF = 'TF_IDF'
    Embeddings = 'Embeddings'


def get_training_and_target(model_type: ModelType, deceased_to_date: pd.Series, *feats_to_train_on: List[pd.DataFrame], improved_df = None) -> Tuple[pd.DataFrame, pd.Series]:
    ''' Returns training and target dfs '''
    feats_to_train_on_1 = [*feats_to_train_on]
    df_final = pd.concat(feats_to_train_on_1, axis=1).fillna(0)
    if model_type == ModelType.Baseline:
        target = pd.Series(df_final.index.isin(deceased_to_date), index=df_final.index, name='target')
        return df_final, target
    elif model_type == ModelType.TF_IDF or model_type == ModelType.Embeddings:
        if improved_df is None:
            raise ValueError(f'should specify @improved_df if @model_type is {model_type.value}')
        improved_df = improved_df[improved_df.index.isin(df_final.index)]
        feats_to_train_on_2 = [*feats_to_train_on_1, improved_df]
        df_final = pd.concat(feats_to_train_on_2, axis=1).fillna(0)
        target = pd.Series(df_final.index.isin(deceased_to_date), index=df_final.index, name='target')
        return df_final, target
    else:
        raise NotImplementedError


def _train_and_predict(df: pd.DataFrame, target: pd.Series, train_loc: pd.Series, classifier) -> np.array:
    classifier.fit(df[train_loc], target[train_loc])
    pred = classifier.predict_proba(df[~train_loc])[:, 1]
    return pred


def train_cl_model(model_type: ModelType, df: pd.DataFrame, train_ids: Set[int], target: pd.Series) -> None:
    ''' Trains random classifier model '''
    train_loc = df.index.isin(train_ids)
    cl = RandomForestClassifier(random_state=RANDOM_SEED)
    pred = _train_and_predict(df, target, train_loc, cl)
    dump(cl, os.path.join(MODELS_BASE_PATH, f'{model_type.value}.joblib')) 
    print(f'Roc score RandomForestClassifier {model_type.value}: {roc_auc_score(target[~train_loc], pred)}')
    feature_importances = pd.Series(cl.feature_importances_, index=df.columns).sort_values(ascending=False).iloc[:10]
    print(f'Feature importances  {model_type.value}: {feature_importances}\n')


# embedding ML
# TODO refactor this to work on batches of notes
    # '''This is slow.'''
def _get_vector_for_text(text: str, tokenizer: AutoTokenizer, model: AutoModelForMaskedLM) -> torch.Tensor:
    encoding = tokenizer(text, 
                        add_special_tokens=True,
                        truncation=True, 
                        padding='max_length', 
                        return_attention_mask=True, 
                        return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoding)
        hs = outputs.hidden_states
        token_embeddings = torch.stack(hs, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)
        token_vecs = hs[-2][0]
        text_embedding = torch.mean(token_vecs, dim=0)
        return text_embedding


def _generate_and_save_embedding(last_note: pd.DataFrame, tokenizer: AutoTokenizer, model: AutoModelForMaskedLM) -> None:
    ''' Generate and save the embeddings to disk b/c they're expensive to compute '''
    for row_num, row in tqdm(last_note.iloc[0:].iterrows()):
        text = row['TO_TOK']
        subj_id = row['SUBJECT_ID']
        embedding = _get_vector_for_text(text, tokenizer, model)
        torch.save(embedding, EMBEDDING_TEMPLATE.format(subj_id=subj_id))


def _get_saved_embeddings() -> Tuple[List[int], List[np.array]]:
    subj_ids = []
    embeddings = []
    for file in tqdm(glob(os.path.join(EMBEDDINGS_BASE_PATH, '*'))):
        name = file.split('/')[-1]
        subj_id = int(name.split('.')[0])
        embedding = torch.load(file)
        subj_ids.append(subj_id)
        embeddings.append(np.array(embedding))
    return subj_ids, embeddings


def generate_and_get_embeddings_df(last_note_tokenized: pd.Series) -> pd.DataFrame:
    ''' Returns dataframe containing embeddings for pretrained model '''
    config = AutoConfig.from_pretrained(PRETRAINED_MODEL_NAME, output_hidden_states=True, output_attentions=True)
    model = AutoModelForMaskedLM.from_pretrained(PRETRAINED_MODEL_NAME, config=config)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    _generate_and_save_embedding(last_note_tokenized, tokenizer, model)

    subj_ids, embeds = _get_saved_embeddings()
    embed_df = pd.DataFrame(embeds, index=subj_ids)
    embed_df.columns = [f'EMBED_{i}' for i in embed_df.columns]
    return embed_df


def _read_spark_dfs_from_disk() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Reads output of preprocessing.py and preprocessing2.py
    '''
    deceased_to_date = ps.read_csv(os.path.join(PATH_PROCESSED, 'spark-etl', 'deceased_to_date.json')).to_pandas()
    train_ids = ps.read_csv(os.path.join(PATH_PROCESSED, 'spark-etl', 'train_ids.json')).to_pandas()

    tf_idf_notes_feats = ps.read_csv(os.path.join(PATH_PROCESSED, 'spark-processed-features', 'tf_idf_notes_feats.csv')).to_pandas()
    feats_to_train_on = []
    for i in range(3):
        feats_to_train_on.append(ps.read_csv(os.path.join(PATH_PROCESSED, 'spark-processed-features', f'training_feat{i}.csv')).to_pandas())
    
    last_note_tokenized = ps.read_csv(os.path.join(PATH_PROCESSED, 'spark-processed-features', f'last_note_tokenized.csv')).to_pandas()
    
    return deceased_to_date, train_ids, feats_to_train_on, tf_idf_notes_feats, last_note_tokenized


def main():
    '''
    Bare-bones logic for training and evaluating the ML models
    '''
    # get input
    deceased_to_date, train_ids, feats_to_train_on, tf_idf_notes_feats, last_note_tokenized = _read_spark_dfs_from_disk()
    
    ### Train Baseline model
    # We will use random forest to automatically incorporate feature interrelations into our model.
    df_final_baseline, target_baseline = get_training_and_target(ModelType.Baseline, deceased_to_date, *feats_to_train_on)
    train_cl_model(ModelType.Baseline, df_final_baseline, train_ids, target_baseline)

    ### Train model with note TF-IDF
    df_final_tfidf, target_tfidf = get_training_and_target(ModelType.TF_IDF, deceased_to_date, *feats_to_train_on, improved_df=tf_idf_notes_feats)
    train_cl_model(ModelType.TF_IDF, df_final_tfidf, train_ids, target_tfidf)
    # Better results, mostly from getting patient discharge information from notes.

    ### Train model with transformer embeddings
    embed_df = generate_and_get_embeddings_df(last_note_tokenized)
    df_final_embeddings, target_embeddings = get_training_and_target(ModelType.Embeddings, deceased_to_date, *feats_to_train_on, improved_df=embed_df)
    train_cl_model(ModelType.Embeddings, df_final_embeddings, train_ids, target_embeddings)
