from layers.Layer import Layer
import numpy as np
class InputLayer(Layer):

	#Input: dataIn, an NxD matrix #Output : None
	def __init__ (self ,dataIn):
		self.meanX = dataIn.mean()
		self.stdX = dataIn.std()
		self.stdX = np.where(self.stdX == 0, 1, self.stdX)

	#Input: dataIn, an NxD matrix #Output : An NxD matrix
	def forward(self ,dataIn):
		self.setPrevIn(dataIn)
		y = (dataIn - self.meanX)/self.stdX
		self.setPrevOut(y)
		return y

	#We’ll worry about these later...
	def gradient(self): pass
	def backward( self , gradIn ): pass





