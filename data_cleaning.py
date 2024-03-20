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


def clean_data( json_path ):

	# Load the JSON data
	with open(json_path, 'r') as file:
		data = [ details | { 'name': name.replace('_', ' ') } for item in json.load(file) for name, details in item.items() ]

	# Drop na rows and the screenplay column due to missing data
	df = pd.DataFrame(data).dropna(subset = [ 'budget', 'box_office', 'language', 'starring', 'producers' ]).drop(columns = [ 'screenplay' ])

	# Move name columns to the start
	df = df[ [ 'name' ] + [ col for col in df.columns if col != 'name' ] ]

	# Convert release_date to datetime format
	df[ 'release_date' ] = pd.to_datetime(df[ 'release_date' ])

	# Split release_date into year, month, day
	df[ 'release_year' ] = df[ 'release_date' ].dt.year
	df = df.drop(columns = [ 'release_date' ])

	# Separate the nlp df and the numerical df
	df_numerical = df[ [ 'budget', 'box_office', 'vote_average', 'vote_count', 'runtime', 'release_year' ] ]
	df_nlp = preprocess_nlp_df(df[ [ 'name', 'overview', 'plot', 'language' ] ])

	# Zscore the numerical df
	df_numerical = (df_numerical - df_numerical.mean()) / df_numerical.std()

	# Reset indexes before concatenation
	df_numerical = df_numerical.reset_index(drop = True)
	df_nlp = df_nlp.reset_index(drop = True)

	# Concatenate different category dfs
	final_df = pd.concat([ df_numerical, df_nlp ], axis = 1)

	# One-hot encode the targets
	mlb = MultiLabelBinarizer()
	y = pd.DataFrame(mlb.fit_transform(df[ 'genres' ]), columns = mlb.classes_)

	return final_df, y


def preprocess_nlp_df( nlp_df ):
	"""
	Function to preprocess each text entry in the DataFrame.
	"""

	# Create set of stopwords and the lemmatizer object
	stop_words = set(stopwords.words('english'))
	lemmatizer = WordNetLemmatizer()

	# Function to preprocess a single text entry
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

	# Calculate all tfidf matrices
	tfidf_matrices = [ TfidfVectorizer().fit_transform(nlp_df[ column ].apply(text_preprocessing)) for column in nlp_df.columns ]

	# Combine the TF-IDF matrices
	combined_tfidf_matrix = np.hstack([ tfidf_matrix.toarray() for tfidf_matrix in tfidf_matrices ])

	# L2 normalized reduced SVD TF-IDF matrix
	normalized_tfidf_matrix = normalize(TruncatedSVD(1000).fit_transform(combined_tfidf_matrix), norm = 'l2', axis = 1)

	return pd.DataFrame(normalized_tfidf_matrix, columns = [ 'nlp_' + str(i) for i in range(1000) ])
