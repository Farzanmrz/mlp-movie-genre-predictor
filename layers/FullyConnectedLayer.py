from layers.Layer import Layer
import numpy as np
import pandas as pd

class FullyConnectedLayer(Layer):
	def __init__(self, sizeIn, sizeOut ):
		self.sizeIn = sizeIn
		self.sizeOut = sizeOut

		# Compute xavier weight for weights and biases
		xav_weight = (6/(sizeIn + sizeOut))**(1/2)
		self.weights = np.random.uniform(-xav_weight, xav_weight, (sizeIn, sizeOut))
		self.bias = np.random.uniform(-xav_weight, xav_weight, (1, sizeOut))

		# Set variables for adam learning
		self.s = 0
		self.r = 0
		self.p1 = 0.9
		self.p2 = 0.999
		self.delta = 1e-8

	# Input : None
	# Output : The sizeIn x sizeOut weight matrix
	def getWeights( self ):
		return self.weights

	# Input : The sizeIn x sizeOut weight matrix
	# Output : None
	def setWeights( self, weights ):
		self.weights = weights

	# Input : None
	# Output : The 1 x sizeOut bias vector
	def getBiases( self ):
		return self.bias

	# Input : The 1 x sizeOut bias vector
	# Output : None
	def setBiases( self, biases ):
		self.bias = biases

	#Input: dataIn, an NxD matrix
	#Output : An NxK data matrix
	def forward(self ,dataIn):

		if isinstance(dataIn, pd.DataFrame):
			dataIn = dataIn.values

		self.setPrevIn(dataIn)
		y = np.dot(dataIn,self.getWeights()) + self.getBiases()
		self.setPrevOut(y)
		return y

	#We’ll worry about these later...
	def gradient(self):
		return self.weights.T

	def backward(self , gradIn ):
		return  gradIn @ self.gradient()

	def updateWeights( self, gradIn,t, eta = 0.0001 ):

		# Ensure numpy array for consistency
		if isinstance(gradIn, (pd.Series, pd.DataFrame)):
			gradIn = gradIn.values

		# Compute gradient
		dJdb = np.sum(gradIn, axis = 0) / gradIn.shape[ 0 ]
		dJdW = (np.array(self.getPrevIn()).T @ gradIn) / gradIn.shape[ 0 ]

		# First moment update
		self.s = (self.p1 * self.s) + ((1 - self.p1) * dJdW)

		# Second moment update
		self.r = (self.p2 * self.r) + ((1 - self.p2) * (dJdW * dJdW))

		# Gradient descent numerator
		numer = self.s / (1 - (self.p1 ** (t+1)))

		# Gradient descent denominator
		denom = (((self.r) / (1 - (self.p2 ** (t+1)))) ** (1 / 2)) + self.delta

		# Final update term
		update_term = numer / denom

		# Update weights
		self.setWeights(self.getWeights() - (eta * update_term))
		self.setBiases(self.getBiases() - (eta * dJdb))




