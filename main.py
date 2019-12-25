from NN import NeuralNetwork
import numpy as np
if __name__ == '__main__':
    # A network with depth=d has d-1 hidden layers
    nn = NeuralNetwork(input_nodes=2, depth=2, hidden_layers_nodes=(2,), output_nodes=1)
    inputv = np.array([0.3, 0.3])
    print(nn.hidden_layers_matrices)
    nn.forward_pass(inputv)
    print('output_vector', nn.outputv)
