import numpy as np


class NeuralNetwork:
    inputv = np.zeros(1)
    depth = 1
    hidden_layers_matrices = []
    outputv = np.zeros(1)

    def __init__(self, input_nodes, depth, hidden_layers_nodes, output_nodes):
        self.inputv = np.zeros(input_nodes)
        columns = input_nodes
        self.depth = depth
        # Add hidden layers matrices
        for k in range(depth-1):
            rows = hidden_layers_nodes[k]
            matrix = np.ones([rows, columns])
            self.hidden_layers_matrices.append(matrix)
            # Update columns
            columns = rows

        # Add output matrix
        columns = hidden_layers_nodes[-1]
        rows = output_nodes
        matrix = np.ones([rows, columns])
        self.hidden_layers_matrices.append(matrix)

        # Initialize the output vector
        self.outputv = np.zeros(output_nodes)


    def forward_pass(self, inputv_):
        self.inputv = inputv_
        temp_vector = self.inputv
        for k in range(self.depth):
            layer_output = (self.hidden_layers_matrices[k]).dot(temp_vector)
            temp_vector = layer_output
        self.outputv = temp_vector