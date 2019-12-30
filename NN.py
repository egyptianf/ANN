import numpy as np
import activation as ac

class NeuralNetwork:
    learning_rate = 0.01
    depth = 1
    num_of_hidden_layers = 0
    inputv = np.zeros(1)
    network_matrices = []  # Last matrix is weight matrix of the output, otherwise, each hidden layer has a weight matrix
    outputv = np.zeros(1)
    desiredv = np.zeros(1)

    # A list of numpy arrays to save the outputs of hidden layers at a specific state of the network
    hidden_layers_outputs = []

    hidden_layers_errors = []  # size = d-1

    # Activation of all the neurons
    activation_function = ac.Sigmoid()
    activation: 'a class'
    activation = None

    def __init__(self, input_nodes, depth, hidden_layers_nodes, output_nodes, learning_rate):
        self.inputv = np.zeros(input_nodes)
        columns = input_nodes
        self.depth = depth
        self.num_of_hidden_layers = self.depth - 1
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

        # Setting learning rate
        self.learning_rate = learning_rate

    def cost(self, desired: 'numpy array'):
        return np.sum(np.exp2(self.outputv - desired))

    def output_layer_error(self):
        return 2 * (self.outputv - self.desiredv)

    def unit_error(self, l: 'layer index', k: 'unit index'):
        unit_error = 0
        # We will loop through next layer l+1
        if (l+1) <= self.depth:
            layer_matrix = self.network_matrices[l+1]
            weights = layer_matrix[:, k]
            # This is the last hidden layer
            if l+1 == self.depth:
                layer_error = self.output_layer_error()
                derivative_sigma = self.outputv * (1 - self.outputv)
                unit_error = np.dot(np.multiply(weights, derivative_sigma), layer_error)
            else:
                layer_error = self.hidden_layers_errors[(self.num_of_hidden_layers - 1) - 2]
                derivative_sigma = self.hidden_layers_outputs[l-1]
                unit_error = np.dot(np.multiply(weights, derivative_sigma), layer_error)

            return unit_error
        else:
            print('Invalid layer number')


    def forward_pass(self, inputv_):
        self.inputv = inputv_
        temp_vector = self.inputv
        self.hidden_layers_outputs.clear()
        for k in range(self.depth):
            z = (self.network_matrices[k]).dot(temp_vector)
            layer_output = self.activation.output(z)
            temp_vector = layer_output
            # Add this layer output to the list of hidden layers outputs
            if k < self.depth - 1:
                self.hidden_layers_outputs.append(layer_output)
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
            derivative_sigma = a_j * (1 - a_j)  # sigmoid
            y_j = y[j]
            for k in range(cols):
                a_k = before_last_layer[k]
                # Assuming stochastic gradient descent
                del_cost = 2 * self.learning_rate * a_k * (a_j - y_j) * derivative_sigma
                output_matrix[j][k] -= del_cost

        self.network_matrices[-1] = output_matrix

        """#################-------------------------##########################"""


        # Loop through the hidden layers to adjust the weight matrices
        hidden = -2  # Hidden matrix starting from the end of the list
        while hidden >= 0:
            hidden_matrix = self.network_matrices[hidden]
            previous_layer = self.hidden_layers_outputs[hidden]
            hidden_layer_errors = np.empty([])
            rows = hidden_matrix.shape[0]
            cols = hidden_matrix.shape[1]
            for k in range(rows):
                a_k = self.hidden_layers_outputs[hidden + 1]
                derivative_sigma = a_k * (1 - a_k)
                error_of_unit_k = self.unit_error(hidden, k)
                for i in range(cols):
                    a_i = previous_layer[i]
                    del_cost = a_i * derivative_sigma * error_of_unit_k
                    hidden_matrix[k][i] -= del_cost
                np.append(hidden_layer_errors, error_of_unit_k)
            self.network_matrices[hidden] = hidden_matrix
            self.hidden_layers_errors.append(hidden_layer_errors)




    def backpropagation(self, epochs: int):
        for i in range(epochs):
            self.forward_pass(self.inputv)
            self.backpropagate(self.inputv, self.desiredv)
