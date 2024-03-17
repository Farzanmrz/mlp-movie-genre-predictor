from layers.Layer import Layer
import numpy as np

class ReLULayer(Layer):

	#Input: None #Output : None
	def __init__ (self):
		super().__init__()

	#Input: dataIn, an NxD matrix #Output : An NxD matrix
	def forward(self ,dataIn):
		self.setPrevIn(dataIn)
		y = np.maximum(0,dataIn)
		self.setPrevOut(y)
		return y

	#We’ll worry about these later...
	def gradient(self):
		return np.where(self.getPrevIn() > 0, 1, 0)

	def backward( self , gradIn ):
		return gradIn * self.gradient()
