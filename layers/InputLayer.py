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
	def __init__ (self, dataIn):
		super().__init__()

	#Input: dataIn, an NxD matrix #Output : An NxD matrix
	def forward(self ,dataIn):

		self.setPrevIn(dataIn)

		# Columns to normalize and rest
		fts_numerical = [ 'budget', 'box_office', 'vote_average', 'vote_count', 'runtime', 'release_year' ]
		fts_rest = [ col for col in dataIn.columns if col not in fts_numerical ]

		# Separate the dataframe into columns to normalize and columns to leave as is
		df_numerical = dataIn[ fts_numerical ]
		df_rest = dataIn[ fts_rest ]

		# Apply Z-score normalization to numerical features
		df_numerical = (df_numerical - df_numerical.mean()) / df_numerical.std()

		# Reset the index before concatenation
		df_numerical = df_numerical.reset_index(drop=True)
		df_rest = df_rest.reset_index(drop=True)

		# Concatenate the normalized and non-normalized dataframes
		final_df = pd.concat([ df_numerical, df_rest ], axis = 1)

		# You might want to adjust the following part according to what you actually
		# want to do with `final_df`, as the `y = ` line is incomplete in your provided code.
		# For now, I'll assume you want to set and return the final_df as the output.
		self.setPrevOut(final_df)

		return final_df

	def gradient(self): pass
	def backward(self,gradIn): pass



