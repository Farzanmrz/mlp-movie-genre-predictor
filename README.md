# Deep Learning from Scratch: Neural Network Implementations

This project contains a native implementation of deep learning algorithms and examples of shallow networks being built and analyzed for various purposes. The goal is to provide a hands-on experience with the fundamentals of neural network layers and operations without the use of high-level libraries such as TensorFlow or PyTorch.

## Structure
- `/layers`: Contains the implementation of input, fully connected, activation, and objective layers.
- `/tests`: Contains unit tests for each layer which can be run individually to validate functionality.
- `/datasets`: Includes datasets for various analyses.
- `/examples`: Contains example scripts demonstrating the application of neural networks using the layers implemented in this project.

## Layers
The layers are built on an abstract class `Layer.py` that defines a general architecture with `forward`, `backward`, and `gradient` methods. Each type of layer is specialized for its role within a neural network.

### Input Layer
The `InputLayer` is responsible for preprocessing the input data by normalizing it. This involves adjusting the data to have a mean of zero and a standard deviation of one, commonly known as z-scoring. This process helps with the convergence of the neural network during training.

### Fully Connected Layer
The `FullyConnectedLayer` is a fundamental component of neural networks where each input neuron is connected to every output neuron. It performs a linear transformation on the input data through learned weights and biases. In this implementation, weights and biases are initially set to random values within the range of \(-10^{-4}\) to \(10^{-4}\), to start the learning process from a neutral point. The default learning rate, \(\eta\), is set to 0.0001, as this rate has demonstrated consistent performance across different scenarios. This layer uses gradient descent to iteratively update its weights and biases, aiming to minimize the loss function during training.

### FullyConnectedLayer2
The `FullyConnectedLayer2` extends the capabilities of the standard fully connected layer by incorporating Xavier weight initialization and ADAM learning for optimization. Xavier Initialization helps in setting the weights to values that are suitable for the activation function, promoting efficient learning from the outset. ADAM, an adaptive learning rate optimization algorithm, is used to update the weights and biases. It calculates adaptive learning rates for each parameter from estimates of first and second moments of the gradients, providing a more sophisticated approach to converging towards the minima of the loss function.

### Activation Function Layers
Implemented activation functions include:
- **Linear**: For linear transformations.
- **ReLU**: Rectified Linear Unit for non-linear activations.
- **LogisticSigmoid**: Sigmoid activation function for binary classification tasks.
- **Softmax**: For multi-class classification problems.
- **Tanh**: Hyperbolic tangent function for scaled activations.

### Objective Function Layers
Objective functions implemented for network evaluation include:
- **LogLoss**: For binary classification.
- **CrossEntropy**: For multi-class classification.
- **SquaredError**: For regression tasks.

## Future Work
- Implement more complex network architectures.
- Include more examples and tutorials on how to use the network for classification, recommendation, and other tasks.
- Enhance the library with additional features and optimizations.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
* Farzan Mirza: [farzan.mirza@drexel.edu](mailto:farzan.mirza@drexel.edu) | [LinkedIn](https://www.linkedin.com/in/farzan-mirza13/)
