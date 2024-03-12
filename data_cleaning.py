# Imports
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MultiLabelBinarizer

def clean_data(json_path):

    # Load the JSON data
    with open(json_path, 'r') as file:
        data = [ details | { 'name': name.replace('_', ' ') } for item in json.load(file) for name, details in item.items() ]

    # Drop na rows and the screenplay column due to missing data
    df = pd.DataFrame(data).dropna(subset=['budget', 'box_office', 'language', 'starring', 'producers']).drop(columns=['screenplay'])

    # Move name columns to the start
    df = df[['name'] + [col for col in df.columns if col != 'name']]

    # Convert release_date to datetime format
    df['release_date'] = pd.to_datetime(df['release_date'])

    # Split release_date into year, month, day
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df['release_day'] = df['release_date'].dt.day

    # Drop release_date
    df = df.drop(columns=['release_date'])

    # Define list of features and target
    categorical_features = [ 'director', 'producers', 'starring', 'country', 'language', 'release_month','release_day' ]
    numerical_features = [ 'budget', 'box_office', 'vote_average', 'vote_count', 'runtime', 'release_year' ]
    target = 'genres'

    # Separate the subsets and targets
    df_categorical = df[ categorical_features ]
    df_numerical = df[ numerical_features ]

    # One-hot encode the targets
    mlb = MultiLabelBinarizer()
    y = pd.DataFrame(mlb.fit_transform(df[target]), columns=mlb.classes_)

    return df_numerical, y
