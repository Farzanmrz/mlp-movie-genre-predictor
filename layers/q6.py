# Imports
import InputLayer, FullyConnectedLayer, SoftmaxLayer, CrossEntropy
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


# Read the training and testing files
train_df = pd.read_csv("mnist_train_100.csv")
test_df = pd.read_csv("mnist_valid_10.csv")

# Set random seed to 0 and shuffle both datasets
np.random.seed(0)
train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

# Separate features and target
ytrain = train_df.iloc[:,0].values.reshape(-1,1)
ytest = test_df.iloc[:,0].values.reshape(-1,1)
xtrain = train_df.iloc[:,1:].values
xtest = test_df.iloc[:,1:].values


# One-hot encoding ytrain and ytest
encoder = OneHotEncoder()
ytrain = encoder.fit_transform(ytrain)
ytest = encoder.transform(ytest)

# Define the layers being used
L1 = InputLayer.InputLayer(xtrain)
L2 = FullyConnectedLayer.FullyConnectedLayer(xtrain.shape[1],ytrain.shape[1])
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
for epoch in range(10000):
	if epoch%100==0:
		print(epoch)

	# Forward propogation and ce loss for training set
	h = layers[ 0 ].forward(xtrain)
	fc_forward = layers[ 1 ].forward(h)
	sm_forward = layers[2].forward(fc_forward)
	ce_forward = layers[3].eval(ytrain, sm_forward)
	jtrain.append(ce_forward)

	# Backpropogation and update weights
	ce_back = L4.gradient(ytrain, sm_forward)
	sm_back = L3.backward(ce_back)
	L2.updateWeights(sm_back,epoch,0.001)


	# Forward propogation and ce loss for test set
	htest = layers[ 0 ].forward(xtest)
	fctest_forward = layers[ 1 ].forward(htest)
	smtest_forward = layers[2].forward(fctest_forward)
	cetest_forward = layers[3].eval(ytest, smtest_forward)
	jtest.append(cetest_forward)

	# Terminating conditions
	if epoch > 0 and cetest_forward > prevtestce and np.abs(prevtraince - ce_forward)<1e-5:

		# Calculate train accuracies
		train_pred = np.argmax(sm_forward,1)
		train_act = np.argmax(ytrain,1)
		train_act = train_act.flatten()
		train_acc = np.mean(train_pred == train_act )

		# Calculate test accuracies
		test_pred = np.argmax(smtest_forward,1)
		test_act = np.argmax(ytest,1)
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


