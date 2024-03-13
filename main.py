# Imports
from layers import InputLayer, FullyConnectedLayer, SoftmaxLayer, CrossEntropy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_cleaning import clean_data
from sklearn.model_selection import train_test_split

# Retrieve the x and y data
x, y = clean_data('data/raw_data.json')

# I need to zscore x
x = (x - np.mean(x)) / np.std(x)


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Define the layers being used
L1 = InputLayer.InputLayer(x_train)
L2 = FullyConnectedLayer.FullyConnectedLayer(x_train.shape[1],y_train.shape[1])
L3 = SoftmaxLayer.SoftmaxLayer()
L4 = CrossEntropy.CrossEntropy()
layers = [L1,L2,L3, L4]

# Define variables to store J for train and test
jtrain = []
jtest = []

# Define variable to store previous cross entropy
prevtestce = 0
prevtraince = 0

final_sm = []

# Forwards backwards loop
for epoch in range(3000):
	if epoch%100==0:
		print(epoch)

	# Forward propogation and ce loss for training set
	h = layers[ 0 ].forward(x_train)
	fc_forward = layers[ 1 ].forward(h)
	sm_forward = layers[2].forward(fc_forward)
	ce_forward = layers[3].eval(y_train, sm_forward)
	jtrain.append(ce_forward)

	# Backpropogation and update weights
	ce_back = L4.gradient(y_train, sm_forward)
	sm_back = L3.backward(ce_back)
	L2.updateWeights(sm_back,epoch,0.001)


	# Forward propogation and ce loss for test set
	htest = layers[ 0 ].forward(x_test)
	fctest_forward = layers[ 1 ].forward(htest)
	smtest_forward = layers[2].forward(fctest_forward)
	cetest_forward = layers[3].eval(y_test, smtest_forward)
	jtest.append(cetest_forward)

	# Terminating conditions
	if epoch > 0 and cetest_forward > prevtestce and np.abs(prevtraince - ce_forward)<1e-5:

		# Calculate train accuracies
		train_pred = np.argmax(sm_forward,1)
		train_act = np.argmax(y_train,1)
		train_act = train_act.flatten()
		train_acc = np.mean(train_pred == train_act )

		# Calculate test accuracies
		test_pred = np.argmax(smtest_forward,1)
		test_act = np.argmax(y_test,1)
		test_act = test_act.flatten()
		test_acc = np.mean(test_pred == test_act )

		# Print results
		print(f"Convergence at Epoch: {epoch + 1}")
		print(f"Final Training Accuracy: {train_acc * 100:.4f}%")
		print(f"Final Testing Accuracy: {test_acc * 100:.4f}%")


		# Break loop
		break

	# Set previous ce to current ce
	prevtestce = cetest_forward
	prevtraince = ce_forward


# Plot the figure
epochs = list(range(1, len(jtrain) + 1))
plt.figure(figsize=(10, 5))
plt.plot(epochs, jtrain, label='Training J')
plt.plot(epochs, jtest, label='Test J')
plt.xlabel('Epochs')
plt.ylabel('J (Cross Entropy Loss)')
plt.title('Training and Validation Cross-Entropy Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.show()


