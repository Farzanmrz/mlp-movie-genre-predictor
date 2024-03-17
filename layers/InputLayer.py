from layers.Layer import Layer
import numpy as np
import json
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string

class InputLayer(Layer):

	#Input: dataIn, an NxD matrix #Output : None
	def __init__ (self):
		super().__init__()




	#Input: dataIn, an NxD matrix #Output : An NxD matrix
	def forward(self ,dataIn):

		self.setPrevIn(dataIn)

		# Separate dataframes
		df_numerical = dataIn[[ 'budget', 'box_office', 'vote_average', 'vote_count', 'runtime', 'release_year' ]]
		df_categorical = dataIn[ [ 'director', 'producers', 'starring', 'country', 'language', 'release_month', 'release_day' ] ]
		df_nlp = dataIn[[ 'name', 'overview', 'plot' ]]

		# Zscore numerical features
		df_numerical = (df_numerical - np.mean(df_numerical)) / np.std(df_numerical)

		# Process NLP features
		df_nlp = self.preprocess_nlp_df(df_nlp)

		# Concatenate different category dfs
		y = pd.concat([df_numerical, df_nlp], axis=1)

		self.setPrevOut(y)

		return y


	def testGetFeatures(self):
		return self.df_numerical, self.df_categorical, self.preprocess_nlp_df(self.df_nlp)

	def preprocess_nlp_df(self, nlp_df ):
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

		# Convert back to a DataFrame
		column_names = [ 'tfidf_svd_' + str(i) for i in range(n_components) ]
		reduced_df = pd.DataFrame(reduced_tfidf_matrix, columns = column_names)

		return reduced_df

	def gradient(self): pass
	def backward(self,gradIn): pass



