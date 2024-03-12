from layers.Layer import Layer
import numpy as np
import pandas as pd
import scipy

class SoftmaxLayer(Layer):

	#Input: None #Output : None
	def __init__ (self):
		super().__init__()

	#Input: dataIn, an NxD matrix #Output : An NxD matrix
	def forward(self ,dataIn):

		# Convert to numpy array
		if isinstance(dataIn, pd.DataFrame):
			dataIn = dataIn.values

		self.setPrevIn(dataIn)
		numer = np.exp(dataIn - np.max(dataIn, axis=1, keepdims=True))
		denom = np.sum(numer, axis = 1, keepdims = True)
		y = numer/denom
		self.setPrevOut(y)
		return y

	#We’ll worry about these later...
	def gradient(self):
		return np.array([ np.diag(y) - np.outer(y, y) for y in self.getPrevOut() ])

	def backward(self , gradIn ):
		selfGrad = self.gradient()
		if scipy.sparse.issparse(gradIn):
			gradIn = gradIn.toarray()
		if scipy.sparse.issparse(selfGrad):
			selfGrad = selfGrad.toarray()
		return np.einsum('...i,...ij', gradIn, selfGrad)


