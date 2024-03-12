import numpy as np
import scipy

class CrossEntropy():

	def eval(self, Y, Yhat, epsilon = 1e-7):
		if scipy.sparse.issparse(Y):
			Y = Y.toarray()
		if scipy.sparse.issparse(Yhat):
			Yhat = Yhat.toarray()
		return -np.mean(np.sum(Y * np.log(Yhat + epsilon), axis=1))

	def gradient(self, Y, Yhat, epsilon = 1e-7):
		return -((Y)/(Yhat+epsilon))