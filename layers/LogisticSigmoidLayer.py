from layers.Layer import Layer
import numpy as np

class LogisticSigmoidLayer(Layer):

	#Input: None #Output : None
	def __init__ (self):
		super().__init__()

	#Input: dataIn, an NxD matrix #Output : An NxD matrix
	def forward(self ,dataIn):
		self.setPrevIn(dataIn)
		y = 1/(1+np.exp(-dataIn))
		self.setPrevOut(y)
		return y

	#We’ll worry about these later...
	def gradient(self):
		y_hat = self.getPrevOut()
		return y_hat*(1-y_hat)

	def backward( self , gradIn ):
		return self.gradient() * gradIn
