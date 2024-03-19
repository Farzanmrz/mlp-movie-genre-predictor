import numpy as np

class LogLoss():

	def eval(self, Y, Yhat, epsilon = 1e-7):
		return -np.mean(Y * np.log(Yhat + epsilon) + (1 - Y) * np.log(1 - Yhat + epsilon))

	def gradient(self, Y, Yhat, epsilon = 1e-7):
		return -((Y - Yhat) / (Yhat * (1 - Yhat) + epsilon))