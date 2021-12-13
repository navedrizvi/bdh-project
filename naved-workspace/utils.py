
"""
Demo script for reading a CSV file from S3 into a pandas data frame using the boto3 library
"""

import os
from typing import Tuple

import boto3
import pandas as pd
from urlparse import urlparse


AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

def split_s3_path(path: str) -> Tuple[str, str]:
	'''
	>>> o = urlparse('s3://bucket_name/folder1/folder2/file1.json', allow_fragments=False)
	>>> o
	ParseResult(scheme='s3', netloc='bucket_name', path='/folder1/folder2/file1.json', params='', query='', fragment='')
	>>> o.netloc
	'bucket_name'
	>>> o.path
	'/folder1/folder2/file1.json'
	'''
	o = urlparse(path, allow_fragmets=False)
	return o.netloc, o.path[1:]

def get_gz_s3_file_as_df(gz_s3_path: str) -> pd.DataFrame:
	s3_client = boto3.client(
		"s3",
		aws_access_key_id=AWS_ACCESS_KEY_ID,
		aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
		aws_session_token=AWS_SESSION_TOKEN,
	)

	b, k = split_s3_path(gz_s3_path)
	response = s3_client.get_object(Bucket=b, Key=k)

	status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

	if status == 200:
		print(f"Successful S3 get_object response. Status - {status}")
		books_df = pd.read_csv(response.get("Body"), compression='gzip')
		return books_df
	else:
		print(f"Unsuccessful S3 get_object response. Status - {status}")