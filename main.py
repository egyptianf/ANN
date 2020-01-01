from NN import NeuralNetwork
import numpy as np
if __name__ == '__main__':
    # A network with depth=d has d-1 hidden layers
    nn = NeuralNetwork(input_nodes=2, depth=2, hidden_layers_nodes=(2,), output_nodes=2, learning_rate=1)

    nn.network_matrices[0] = np.array([[500, 70], [30, 100]])
    nn.network_matrices[1] = np.array([[700, 1000], [966, 875]])
    print("Matrices before bp", nn.network_matrices)
    nn.desiredv = np.array([1000, 1000])  # Just for testing
    nn.inputv = np.array([0.9, -0.1])
    before = nn.cost()
    print("Cost before backpropagation: ", nn.cost())
    epochs = 10
    nn.backpropagation(epochs)
    # Check the output matrix after performing one iteration of backpropagation

    print("Network matrices after", epochs, "epochs: ", nn.network_matrices)
    after = nn.cost()
    print("Cost after ", epochs, "epochs: ", after)
    print("Difference between errors are: ", after - before)
