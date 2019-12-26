from NN import NeuralNetwork
import numpy as np
if __name__ == '__main__':
    # A network with depth=d has d-1 hidden layers
    nn = NeuralNetwork(input_nodes=2, depth=2, hidden_layers_nodes=(2,), output_nodes=2)
    inputv = np.array([0.9, -0.1])
    print(nn.network_matrices)
    nn.network_matrices[0] = np.array([[2.1, -0.9], [-0.8, 0.4]])
    nn.network_matrices[1] = np.array([[1.2, -0.7], [-0.2, -1.1]])
    print(nn.network_matrices)
    nn.forward_pass(inputv)
    print('output_vector', nn.outputv)
    print('cost', nn.cost(np.array([1, 1])))
    print('Hidden layers outputs', nn.hidden_layers_outputs)
    nn.desiredv = np.array([1, 1]) # Just for testing
    nn.backpropagation(1)
    # Check the output matrix after performing one iteration of backpropagation
    print(nn.network_matrices)

