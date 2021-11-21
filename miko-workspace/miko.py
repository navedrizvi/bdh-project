import datetime as dt
from collections import defaultdict, Counter, namedtuple
from itertools import product, combinations, permutations
import json
import os
import pickle
import warnings

#from IPython.display import HTML, display, set_matplotlib_formats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core import series
from pyspark.sql.types import ArrayType, DoubleType
import seaborn as sns

from sqlalchemy import create_engine
from tqdm.auto import tqdm


import re
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

from sklearn.model_selection import train_test_split

#add nltk, seaborn in yml

import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
# $example off$
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
 


PATH = "../data/raw/"
PATH_PROCESSED = "../data/processed/"
PATH_DATASETS = "../data/train/"

SAMPLE_SIZE = 10000
RANDOM_SEED = 1



def get_notes(sample_ids: set = None,
              note_path: str = "NOTEEVENTS.csv.gz",
              chunksize: int = 10_000) -> pd.DataFrame:
    """Get all notes or only those relevant for the sample."""
    if sample_ids is None:
        return pd.read_csv(os.path.join(PATH, note_path))
    return get_data_for_sample(sample_ids, note_path, chunksize)

def clean_text(note: str):

    # remove \w and set string to lower
    cleaned = re.sub(r'[^\w]', ' ', note).replace("_", " ").lower()
    # remove numbers
    cleaned = re.sub(r'\d+', '', cleaned)
    # remove single characters
    cleaned = re.sub(r"\b[a-zA-Z]\b", "", cleaned)
    # remove spaces
    cleaned = re.sub(' +', ' ', cleaned)
    
    tokenized = cleaned.split(" ")

    lemmatizer = WordNetLemmatizer()
    filtered_words = [lemmatizer.lemmatize(word) for word in tokenized if word not in stopwords.words('english')]

    # remove duplicates
    res = []
    [res.append(x) for x in filtered_words if x not in res]

    # medical stop words


    return res

def subnotes (note: str):    
    # remove \w and set string to lower
    cleaned = re.sub(r'[^\w]', ' ', note).replace("_", " ").lower()
    
    #merge with Ivan's note filtering process
    filter = "discharge diagnosis"
    
    try:
        res = cleaned.split(filter.lower(), 1)[1]
    except:
        res = "none"
    return res

def drop_blank (note: series):
    return list(filter(None, note))


#for initial processing
#notes_sample = get_notes(SAMPLE_IDS)
#notes_sample.to_csv(os.path.join(PATH_PROCESSED, 'SAMPLE_NOTES.csv'), index=False)

#for full processing, remove nrows argument
#notes = pd.read_csv(os.path.join(PATH_PROCESSED, 'SAMPLE_NOTES.csv'))
notes = pd.read_csv(os.path.join(PATH_PROCESSED, 'SAMPLE_NOTES.csv'), nrows=100)

patients_cols = ["SUBJECT_ID", "EXPIRE_FLAG"]
patients = pd.read_csv(os.path.join(PATH, 'PATIENTS.csv'), usecols=patients_cols)

#notes.head()
#print (notes.head())
#print (patients.head())

notes['DATE'] = pd.to_datetime(notes['CHARTDATE']).dt.date

#get subnotes first
notes['SUBNOTES'] = notes['TEXT'].map(subnotes)

#tester
"""
print (type(notes['SUBNOTES']))
print ("Records with values")
print (len(notes[notes['SUBNOTES'] == 'none']))
print ("Total records")
print (len(notes['SUBNOTES']))
print (notes['SUBNOTES'])
"""

#clean text - stopwords, lemmatize, tokenize
notes['CLEAN_TEXT'] = notes['SUBNOTES'].map(clean_text)
#remove blanks in the list
notes['CLEAN_TEXT'] = notes['CLEAN_TEXT'].map(drop_blank)

#merge note data with patient mortality
df = pd.merge (notes, patients, on="SUBJECT_ID")
df.rename(columns={"EXPIRE_FLAG":"label", "CLEAN_TEXT":"words"}, inplace=True)
df = df[["label", "words"]]

#print (df.head())

spark = SparkSession\
        .builder\
        .appName("TfIdfExample")\
        .getOrCreate()

wordsData = spark.createDataFrame(df)
  
# increase numFeatures to reduce hashing collision
# numFeatures = 100, based on the len of list of features
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=100)
featurizedData = hashingTF.transform(wordsData)

# alternatively, CountVectorizer can also be used to get term frequency vectors
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=3)
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

rescaledData.select("label", "words", "features").show()
print (rescaledData)

spark.stop()


