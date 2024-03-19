# Imports
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string

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
    df = df.drop(columns=['release_date'])

    # Separate the nlp df and the numerical df
    df_numerical = df[ [ 'budget', 'box_office', 'vote_average', 'vote_count', 'runtime', 'release_year' ] ]
    df_categorical = df[['director', 'producers', 'starring', 'country', 'language', 'release_month', 'release_day']]
    df_nlp = preprocess_nlp_df(df[ [ 'name', 'overview', 'plot' ] ])

    # Zscore the numerical df
    df_numerical = (df_numerical - df_numerical.mean()) / df_numerical.std()

    # Reset indexes before concatenation
    df_numerical = df_numerical.reset_index(drop=True)
    df_nlp = df_nlp.reset_index(drop=True)

    # Concatenate different category dfs
    final_df = pd.concat([ df_numerical, df_nlp ], axis=1)

    # One-hot encode the targets
    mlb = MultiLabelBinarizer()
    y = pd.DataFrame(mlb.fit_transform(df['genres']), columns=mlb.classes_)

    return final_df, y




def preprocess_nlp_df( nlp_df ):
    """
	Function to preprocess each text entry in the DataFrame.
	"""

    # Create set of stopwords and the lemmatizer object
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def text_preprocessing( text ):
        """
		Function to preprocess a single text entry.
		"""
        # Tokenize the text, remove punctuation and lowercase
        tokens = [ word.lower() for word in word_tokenize(text) if word.lower() not in string.punctuation ]

        # Remove stopwords
        tokens = [ word for word in tokens if word not in stop_words ]

        # Lemmatize the tokens and return the processed string
        return ' '.join([ lemmatizer.lemmatize(token) for token in tokens ])

    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # List to store TF-IDF matrices for each column
    tfidf_matrices = [ ]

    # Iterate over each column in the DataFrame, preprocess and vectorize the text
    for column in nlp_df.columns:
        preprocessed_texts = nlp_df[ column ].apply(text_preprocessing)
        tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_texts)
        tfidf_matrices.append(tfidf_matrix)

    # Combine the TF-IDF matrices. Note: This requires the matrices to have the same number of rows.
    combined_tfidf_matrix = np.hstack([ tfidf_matrix.toarray() for tfidf_matrix in tfidf_matrices ])

    # Reduce using SVD down to n-features
    n_components = 300
    svd = TruncatedSVD(n_components = n_components)
    reduced_tfidf_matrix = svd.fit_transform(combined_tfidf_matrix)

    # Apply L2 normalization to the rows of the reduced matrix
    normalized_tfidf_matrix = normalize(reduced_tfidf_matrix, norm = 'l2', axis = 1)

    # Convert back to a DataFrame
    column_names = [ 'tfidf_svd_' + str(i) for i in range(n_components) ]
    reduced_df = pd.DataFrame(normalized_tfidf_matrix, columns = column_names)

    return reduced_df