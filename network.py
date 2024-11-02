import numpy as np

class Network:
    def __init__(self, sizes: list):
        """
        sizes: list of integers representing the number of neurons in each layer
        e.g. [2, 3, 1] is a network with 2 neurons in the input layer, 3 in the hidden layer and 1 in the output layer
        The biases and weights for the network are initialized randomly,
        using a Gaussian distribution with mean 0, and variance 1.
        Note that the first layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers.
        """
        self.numOfLayers = len(sizes)
        self.sizes = sizes
        # list of (y x 1) matrices
        # matrix b is the bias of the neurons in the layer n
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # list of (y x X) matrices
        # element Wjk is the weight of the connection between the kth neuron in the nth layer and the jth neuron in the n+1 layer
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def train(self, training_data: list, epochs: int, mini_batch_size: int, lr: float):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        training_data: list of tuples (x, y) representing the training inputs and the desired outputs.
        epochs: number of epochs to train for
        mini_batch_size: size of the mini-batches to use when sampling
        lr: learning rate
        test_data: if provided then the network will be evaluated against the test data after each epoch, and partial progress printed out.
        """
        n = len(training_data)
        for i in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)
            print(f"Epoch {i}: done")
    
    def feedforward(self, input: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        input: matrix representing the input layer
        Return a tuple (activations, zs) representing the activations (a) and weighted input (z) vectors for all layers
        """
        activation = input
        activations = [input] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        return (activations, zs)  

    def update_mini_batch(self, mini_batch: list, lr: float):
        """
        Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
        mini_batch: list of tuples (x, y)
        lr: learning rate
        """
        sum_del_b = [np.zeros(b.shape) for b in self.biases]
        sum_del_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # forward pass
            activations, zs = self.feedforward(x)
            # backward pass
            del_b, del_w = self.backprop(activations, zs, y)
            sum_del_b = [nb + dnb for nb, dnb in zip(sum_del_b, del_b)]
            sum_del_w = [nw + dnw for nw, dnw in zip(sum_del_w, del_w)]
        self.weights = [w - (lr / len(mini_batch)) * nw for w, nw in zip(self.weights, sum_del_w)]
        self.biases = [b - (lr / len(mini_batch)) * nb for b, nb in zip(self.biases, sum_del_b)]
    
    def backprop(self, activations: list[np.ndarray], zs: list[np.ndarray], ground_truth: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Return a tuple ``(del_b, del_w)`` representing the
        gradient vector for the cost function C_x. ``del_b`` and
        ``del_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        """
        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]
        
        # backward pass
        delta = self.cost_derivative(activations[-1], ground_truth) * \
            self.sigmoid_prime(zs[-1])
        del_b[-1] = delta
        del_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable i in the loop below is used a little
        # differently to the notation in Chapter 2 of the book. Here,
        # i = 1 means the last layer of neurons, i = 2 is the
        # second-last layer, and so on. It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for i in range(2, self.num_layers):
            z = zs[-i]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sp
            del_b[-i] = delta
            del_w[-i] = np.dot(delta, activations[-i-1].transpose())
        return (del_b, del_w)
    
    def cost_derivative(self, output_activations: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        """
        Cost function: C = 1/2 * ||y - a||^2
        Derivative = partial C / partial a = a - y
        Return the vector of partial derivatives partial C_x / partial a for the output activations.
        """
        return (output_activations - ground_truth)

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        z: weighted input vector = w.a + b
        """
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z: np.ndarray) -> np.ndarray:
        """
        Derivative of the sigmoid function
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))