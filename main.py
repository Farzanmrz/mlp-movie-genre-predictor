# Imports
from layers import InputLayer, FullyConnectedLayer, SoftmaxLayer, CrossEntropy, ReLULayer, LogisticSigmoidLayer, LogLoss
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_cleaning import clean_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Define the layers being used
L1 = FullyConnectedLayer.FullyConnectedLayer(x_train.shape[1],x_train.shape[1])
L2 = ReLULayer.ReLULayer()
L3 = FullyConnectedLayer.FullyConnectedLayer(x_train.shape[1],y_train.shape[1])
L4 = LogisticSigmoidLayer.LogisticSigmoidLayer()
L5 = LogLoss.LogLoss()
layers = [L1, L2, L3, L4, L5]

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

	# Hidden layer 2 gradient and weight update
	ll_back = layers[ 4 ].gradient(y, y_hat)
	ls_back = layers[ 3 ].backward(ll_back)
	fc2_back = layers[2 ].backward(ls_back)
	layers[ 2 ].updateWeights(ls_back, t, eta)

	# Hidden Layer 1 gradient and weight update
	relu_back = layers[ 1 ].backward(fc2_back)
	layers[ 0 ].updateWeights(relu_back, t, eta)

# Function to run the MLP network
def run_mlp( layers, x_train, y_train, x_test, y_test, epochs, eta = 0.001, threshold = 0.5 ):
	"""
	Runs the MLP neural network for a specified number of epochs.

	:param layers: List of network layers, including the loss layer as the last layer.
	:param x: Input data for the network.
	:param y: Target labels for the network.
	:param epochs: Number of epochs to train the network.
	:param eta: Learning rate for weight updates.
	:return: The final output from the last forward_propagation and a list of losses every 10 epochs.
	"""

	# Arrays to store training and testing loss values
	jtrain = []
	jtest = []
	prev_jtrain = None
	prev_jtest = None

	# Arrays to store final train and test predictions
	yhat_train = None
	yhat_test = None

	# Loop through all epochs
	for epoch in range(epochs):

		# Training Set Forward Backward Propagation
		train_yhat, train_j = forward_propagation(layers, x_train, y_train)
		backward_propagation(layers, train_yhat, y_train, epoch, eta)
		jtrain.append(train_j)

		# Testing Set Forward Propagation
		test_yhat, test_j = forward_propagation(layers, x_test, y_test)
		jtest.append(test_j)

		# Show Loss every 100 epochs
		if epoch % 100 == 0:
			print(f"Epoch {epoch}, Training Log Loss: {train_j}, Testing Log Loss: {test_j}")

		# Early termination condition
		if (epoch == 9999) or (prev_jtrain is not None and abs(prev_jtrain - train_j) < 1e-5) or (prev_jtest is not None and prev_jtest < test_j) :
			print(f"Early termination at epoch {epoch} due to minimal loss improvement.")
			yhat_train = (train_yhat >= threshold).astype(int)
			yhat_test = (test_yhat >= threshold).astype(int)
			break

		# Set previous values for next epoch
		prev_jtrain = train_j
		prev_jtest = test_j


	return yhat_train, yhat_test, jtrain, jtest,

# Function to calculate the accuracy of the network

def evaluate_performance( y, yhat ):
	"""
	Evaluate and print the performance of the model on multiple metrics.

	:param y_true: True binary labels in binary indicator format.
	:param y_pred: Predicted probabilities.
	"""

	# Calculating metrics
	accuracy = accuracy_score(y, yhat)
	precision = precision_score(y, yhat, average = 'micro')
	recall = recall_score(y, yhat, average = 'micro')
	f1 = f1_score(y, yhat, average = 'micro')
	jaccard = jaccard_score(y, yhat, average = 'micro')

	# Printing the metrics
	print(f"Accuracy: {accuracy:.4f}")
	print(f"Precision: {precision:.4f}")
	print(f"Recall: {recall:.4f}")
	print(f"F1 Score: {f1:.4f}")
	print(f"Jaccard Index: {jaccard:.4f}")

# set parameters
epochs = 10000  # Set the number of epochs
learning_rate = 0.001  # Set the learning rate
threshold = 0.4  # Set the threshold for binary classification
train_pred, test_pred, jtrain, jtest = run_mlp(layers, x_train, y_train,x_test, y_test, epochs, learning_rate, threshold)

# Print final output and loss for verification
print(f"\nFinal: Training Log Loss: {jtrain[-1]}, Testing Log Loss: {jtest[-1]}")

print("\nTraining Set Performance:")
evaluate_performance(y_train, train_pred)

print("\nTesting Set Performance:")
evaluate_performance(y_test, test_pred)

# Plot
# Generating a list of epoch numbers to match the length of jtrain/jtest lists
loss_epochs = list(range(1, len(jtrain)  + 1))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(loss_epochs, jtrain, label='Training Loss')
plt.plot(loss_epochs, jtest, label='Testing Loss')
plt.title('Training and Testing Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)
plt.show()