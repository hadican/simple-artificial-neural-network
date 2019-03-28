import decimal
import numpy as np


class ArtificialNeuralNetwork:

    def __init__(self):
        # assign random weights to a 4 by 1 matrix with values from -1 to 1
        self.synaptic_weights = 2 * np.random.random((4, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def think(self, inputs):
        # e.g. [1,0,0,1]
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

    def train(self, training_inputs, training_outputs, number_of_iterations):
        for iteration in range(number_of_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments


if __name__ == "__main__":
    ann = ArtificialNeuralNetwork()

    print("Weights in the Beginning:")
    print(ann.synaptic_weights)

    training_inputs = np.array([[1, 0, 0, 1],
                                [0, 1, 1, 0],
                                [1, 0, 1, 0],
                                [0, 0, 0, 1]])

    training_outputs = np.array([[1],
                                 [0],
                                 [1],
                                 [0]])

    ann.train(training_inputs, training_outputs, 10000)

    print("Weights After Training:")
    print(ann.synaptic_weights)

    input = np.array([1, 0, 1, 0])
    print("Think about the answer:", input)

    output = ann.think(input)  # e.g. 0.9989834
    result = decimal.Decimal(output[0]).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP)
    print("The answer for {} is: {}".format(input, result))
