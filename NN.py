import numpy as np
import activation as ac


class NeuralNetwork:
    learning_rate = 0.01
    depth = 1
    num_of_hidden_layers = 0
    inputv = np.zeros(1)
    network_matrices = []  # Last matrix is weight matrix of the output, otherwise, each hidden layer has a weight matrix
    outputv = np.zeros(1)
    outputv_derivative = np.zeros(1)
    desiredv = np.zeros(1)

    # A list of numpy arrays to save the outputs of layers at a specific state of the network
    nets = []
    nets_derivatives = []

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
        self.outputv_derivative = np.zeros(output_nodes)

        # Initialize the activation function
        self.activation = ac.Activation(self.activation_function)

        # Setting learning rate
        self.learning_rate = learning_rate

    def cost(self):
        return np.sum(np.power(self.outputv - self.desiredv, 2))

    def output_layer_error(self):
        return 2 * (self.outputv - self.desiredv)

    def unit_error(self, l: 'layer index', k: 'unit index'):
        # We will loop through next layer l+1
        if (l + 1) <= self.depth:
            layer_matrix = self.network_matrices[l]
            weights = layer_matrix[:, k]
            # This is the last hidden layer (layer l=L-1)
            if l + 1 == self.depth:
                layer_error = self.output_layer_error()
                derivative_sigma = self.outputv_derivative
                unit_error = np.dot(np.multiply(weights, derivative_sigma), layer_error)
            else:  # Any other hidden layer
                layer_error = self.hidden_layers_errors[(self.num_of_hidden_layers - 1) - 2]
                derivative_sigma = self.nets_derivatives[l - 1]
                unit_error = np.dot(np.multiply(weights, derivative_sigma), layer_error)

            return unit_error
        else:
            print('Invalid layer number')

    def forward_pass(self, inputv_):
        self.inputv = inputv_
        temp_vector = self.inputv
        self.nets.clear()
        self.nets_derivatives.clear()
        self.nets.append(inputv_)
        layer_output_derivative = None
        for k in range(self.depth):
            z = (self.network_matrices[k]).dot(temp_vector)
            layer_output = self.activation.output(z)
            layer_output_derivative = self.activation.derivative(z)
            temp_vector = layer_output
            # Add this layer output to the list of hidden layers outputs
            if k < self.depth - 1:
                self.nets.append(layer_output)
                self.nets_derivatives.append(layer_output_derivative)
        self.outputv = temp_vector
        # put output vector derivative
        self.outputv_derivative = layer_output_derivative

    def backpropagate(self, x: np.array(0), y: np.array(0)):

        # Start by adjusting the weight matrix of the output layer
        output_matrix = self.network_matrices[-1]
        before_last_layer = self.nets[-1]

        rows = output_matrix.shape[0]
        cols = output_matrix.shape[1]
        for j in range(rows):
            a_j = self.outputv[j]
            derivative_sigma = self.outputv_derivative[j]
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
        while hidden >= -self.depth:
            hidden_matrix = self.network_matrices[hidden]
            previous_layer_output = self.nets[hidden + 2]
            rows = hidden_matrix.shape[0]
            hidden_layer_errors = np.zeros(rows)
            cols = hidden_matrix.shape[1]
            for k in range(rows):
                layer_output_derivatives = self.nets_derivatives[hidden + 1]
                derivative_sigma = layer_output_derivatives[k]
                error_of_unit_k = self.unit_error(-(hidden+1), k)
                for i in range(cols):
                    a_i = previous_layer_output[i]
                    del_cost = self.learning_rate * a_i * derivative_sigma * error_of_unit_k
                    hidden_matrix[k][i] -= del_cost
                hidden_layer_errors[k] = error_of_unit_k
            self.network_matrices[hidden] = hidden_matrix
            self.hidden_layers_errors.append(hidden_layer_errors)

            hidden -= 1


        # End by making a forward pass
        self.forward_pass(x)

    def backpropagation(self, epochs: int):
        self.forward_pass(self.inputv)
        for i in range(epochs):
            self.backpropagate(self.inputv, self.desiredv)
            # print("weights after epoch", i)
            # print(self.network_matrices)
