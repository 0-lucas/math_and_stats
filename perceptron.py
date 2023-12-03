import numpy as np

np.random.seed(0)


class Perceptron:
    def __init__(self):
        self.ru_target = 4584538
        self.weights = np.random.rand(7)
        self.bias = -1
        self.learning_rate = 0.005
        self.training_data = self.get_samples()

    def get_samples(self):
        samples = np.random.normal(self.ru_target, 25, 50)
        samples = [int(number) for number in samples]
        x = []
        for z in samples:
            x.append([int(i) for i in str(z)])
        return x

    def predict(self, input):
        return int(np.dot(input, self.weights) + self.bias)

    def fit(self, epochs=5):
        for epoch in range(epochs):
            error_per_epoch = []
            for inputs in self.training_data:
                prediction = self.predict(inputs)
                error = self.ru_target - prediction

                for weight in range(len(inputs)):
                    delta = self.learning_rate * error * inputs[weight]
                    self.weights[weight] += delta

                self.bias += self.learning_rate * error
                error_per_epoch.append(error)

            print(f"Erro médio por época: {np.mean(error_per_epoch)}")


x = Perceptron()
# print(x.training_data)
x.fit()
print(x.weights)