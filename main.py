# Imports
from layers import InputLayer, FullyConnectedLayer, SoftmaxLayer, CrossEntropy, ReLULayer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_cleaning import clean_data
from sklearn.model_selection import train_test_split

# Retrieve the x and y data
x, y = clean_data('data/raw_data.json')

# # Pass through InputLayer
# inputlayer = InputLayer.InputLayer()
#
# # check the dfs
# nlp = inputlayer.forward(x)
#
# print(nlp.shape)
# print(nlp[:5])
# print(np.count_nonzero(nlp))




# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Define the layers being used
L1 = InputLayer.InputLayer(x_train)
L2 = FullyConnectedLayer.FullyConnectedLayer(x_train.shape[1],x_train.shape[1])
L3 = ReLULayer.ReLULayer()
L4 = FullyConnectedLayer.FullyConnectedLayer(x_train.shape[1],y_train.shape[1])
L5 = SoftmaxLayer.SoftmaxLayer()
L6 = CrossEntropy.CrossEntropy()
layers = [L1, L2, L3, L4, L5, L6]

# Forward Propogation function
def forward_propagation( layers, x, y ):
	"""
	Performs forward propagation through the network layers and returns the output of the second last layer
	and the loss value from the last layer.

	:param layers: List of network layers, including the loss layer as the last layer.
	:param x: Input data for the network.
	:param y: Target labels for computing the loss.
	:return: Output of the second last layer and loss value from the last layer.
	"""
	h = x
	# Iterate through all layers except the last one to perform forward steps.
	for layer in layers[ :-1 ]:
		h = layer.forward(h)

	# The current_input now holds the output of the second-last layer.
	second_last_output = h

	# Get loss value from the last layer.
	loss = layers[ -1 ].eval(y, second_last_output)

	return second_last_output, loss

# Backward Propogation function
def backward_propagation( layers, y_hat, y, t,  eta = 0.001 ):
	"""
	Performs backward propagation given the output from forward propagation.

	:param layers: List of network layers, including the loss layer as the last layer.
	:param y_hat: The output from the forward pass (hidden layer 2 output after softmax).
	:param y: Target labels for computing the gradients.
	:param eta: Learning rate for weight updates.
	"""
	# Compute the gradients for hidden layer 2
	ce_back = layers[ 5 ].gradient(y, y_hat)
	sm_back = layers[ 4 ].backward(ce_back)
	fc2_back = layers[ 3 ].backward(sm_back)

	# Weight update FC2 layer
	layers[ 3 ].updateWeights(sm_back, t, eta)

	# Compute the gradients for hidden layer 1
	relu_back = layers[ 2 ].backward(fc2_back)

	# Weight update FC1 layer
	layers[ 1 ].updateWeights(relu_back, t, eta)


# Function to run the MLP network
def run_mlp( layers, x, y, epochs, eta = 0.001 ):
	"""
	Runs the MLP neural network for a specified number of epochs.

	:param layers: List of network layers, including the loss layer as the last layer.
	:param x: Input data for the network.
	:param y: Target labels for the network.
	:param epochs: Number of epochs to train the network.
	:param eta: Learning rate for weight updates.
	:return: The final output from the last forward_propagation and a list of losses every 10 epochs.
	"""
	losses = [ ]  # To store the loss every 10 epochs

	for epoch in range(epochs):
		# Forward propagation
		output, loss = forward_propagation(layers, x, y)

		# Backward propagation
		backward_propagation(layers, output, y, epoch, eta)

		# Print and save the loss every 10 epochs
		if epoch % 100 == 0:
			print(f"Epoch {epoch}, Cross Entropy Loss: {loss}")
			losses.append(loss)

	return output, losses


# Example of usage
epochs = 1000  # Set the number of epochs
learning_rate = 0.001  # Set the learning rate
final_output, epoch_losses = run_mlp(layers, x_train, y_train, epochs, learning_rate)

# Print final output and loss for verification
print(f"Final Cross Entropy Loss: {epoch_losses[-1]}")



# # Define variables to store J for train and test
# jtrain = []
# jtest = []
#
# # Define variable to store previous cross entropy
# prevtestce = 0
# prevtraince = 0
#
# final_sm = []


# # Forwards backwards loop
# for epoch in range(3000):
# 	if epoch%100==0:
# 		print(epoch)
#
# 	# Forward propogation and ce loss for training set
# 	h = layers[ 0 ].forward(x_train)
# 	fc_forward = layers[ 1 ].forward(h)
# 	sm_forward = layers[2].forward(fc_forward)
# 	ce_forward = layers[3].eval(y_train, sm_forward)
# 	jtrain.append(ce_forward)
#
# 	# Backpropogation and update weights
# 	ce_back = L4.gradient(y_train, sm_forward)
# 	sm_back = L3.backward(ce_back)
# 	L2.updateWeights(sm_back,epoch,0.001)
#
#
# 	# Forward propogation and ce loss for test set
# 	htest = layers[ 0 ].forward(x_test)
# 	fctest_forward = layers[ 1 ].forward(htest)
# 	smtest_forward = layers[2].forward(fctest_forward)
# 	cetest_forward = layers[3].eval(y_test, smtest_forward)
# 	jtest.append(cetest_forward)
#
# 	# Terminating conditions
# 	if epoch > 0 and cetest_forward > prevtestce and np.abs(prevtraince - ce_forward)<1e-5:
#
# 		# Calculate train accuracies
# 		train_pred = np.argmax(sm_forward,1)
# 		train_act = np.argmax(y_train,1)
# 		train_act = train_act.flatten()
# 		train_acc = np.mean(train_pred == train_act )
#
# 		# Calculate test accuracies
# 		test_pred = np.argmax(smtest_forward,1)
# 		test_act = np.argmax(y_test,1)
# 		test_act = test_act.flatten()
# 		test_acc = np.mean(test_pred == test_act )
#
# 		# Print results
# 		print(f"Convergence at Epoch: {epoch + 1}")
# 		print(f"Final Training Accuracy: {train_acc * 100:.4f}%")
# 		print(f"Final Testing Accuracy: {test_acc * 100:.4f}%")
#
#
# 		# Break loop
# 		break
#
# 	# Set previous ce to current ce
# 	prevtestce = cetest_forward
# 	prevtraince = ce_forward
#
#
# # Plot the figure
# epochs = list(range(1, len(jtrain) + 1))
# plt.figure(figsize=(10, 5))
# plt.plot(epochs, jtrain, label='Training J')
# plt.plot(epochs, jtest, label='Test J')
# plt.xlabel('Epochs')
# plt.ylabel('J (Cross Entropy Loss)')
# plt.title('Training and Validation Cross-Entropy Loss vs Epoch')
# plt.legend()
# plt.grid(True)
# plt.show()


