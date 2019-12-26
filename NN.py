import numpy as np
import activation as ac


class NeuralNetwork:
    inputv = np.zeros(1)
    depth = 1
    network_matrices = []  # Last matrix is weight matrix of the output, otherwise, each hidden layer has a weight matrix
    outputv = np.zeros(1)
    desiredv = np.zeros(1)

    # A list of numpy arrays to save the outputs of hidden layers at a specific state of the network
    hidden_layers_outputs = []

    # Activation of all the neurons
    activation_function = ac.Sigmoid()
    activation: 'a class'
    activation = None

    def __init__(self, input_nodes, depth, hidden_layers_nodes, output_nodes):
        self.inputv = np.zeros(input_nodes)
        columns = input_nodes
        self.depth = depth
        # Add hidden layers matrices
        for k in range(depth - 1):
            rows = hidden_layers_nodes[k]
            matrix = np.ones([rows, columns])
            self.network_matrices.append(matrix)
            # Update columns
            columns = rows

        # Add output matrix
        columns = hidden_layers_nodes[-1]
        rows = output_nodes
        matrix = np.ones([rows, columns])
        self.network_matrices.append(matrix)

        # Initialize the output vector
        self.outputv = np.zeros(output_nodes)

        # Initialize the activation function
        self.activation = ac.Activation(self.activation_function)

    def cost(self, desired: 'numpy array'):
        return np.sum(np.exp2(self.outputv - desired))

    def forward_pass(self, inputv_):
        self.inputv = inputv_
        temp_vector = self.inputv
        self.hidden_layers_outputs.clear()
        for k in range(self.depth):
            z = (self.network_matrices[k]).dot(temp_vector)
            layer_output = self.activation.output(z)
            temp_vector = layer_output
            # Add this layer output to the list of hidden layers
            if k < self.depth - 1:
                self.hidden_layers_outputs.append(temp_vector)
        self.outputv = temp_vector

    def backpropagate(self, x: np.array(0), y: np.array(0)):
        # Start by making a forward pass
        self.forward_pass(x)
        # Computer the error (mean square error)
        error = self.cost(y)

        # Start by adjusting the weight matrix of the output layer
        output_matrix = self.network_matrices[-1]
        before_last_layer = self.hidden_layers_outputs[-1]

        rows = output_matrix.shape[0]
        cols = output_matrix.shape[1]
        for j in range(rows):
            a_j = self.outputv[j]
            derivative_sigma = a_j * (1 - a_j)  # Assuming activation is sigmoid
            y_j = y[j]
            for k in range(cols):
                # WARNING!! This should be averaged for all input examples, update later
                del_c = 2 * before_last_layer[k] * (a_j - y_j) * derivative_sigma
                # So, this update is invalid
                output_matrix[j][k] -= del_c
        self.network_matrices[-1] = output_matrix
        # Loop through the hidden layers to adjust the weight matrices

    def backpropagation(self, epochs: int):
        for i in range(epochs):
            self.backpropagate(self.inputv, self.desiredv)
