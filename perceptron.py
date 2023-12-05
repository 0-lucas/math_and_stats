import numpy as np

np.random.seed(0)


class Perceptron:
    def __init__(self, ru=4584538, learning_rate=0.1, bias=-1):
        self.target = ru
        self.bias = bias
        self.learning_rate = learning_rate
        self.weights = [1 for i in range(len(str(self.target)))]
        self.training_data = None

    @staticmethod
    def get_as_list(integer):
        return [int(digit) for digit in str(integer)]

    @staticmethod
    def step_function(output):
        if output >= 0:
            return 1
        else:
            return -1

    def get_target_class(self, sample):
        if sample >= self.target:
            return 1
        else:
            return -1

    @staticmethod
    def get_lowest_and_highest_numbers(number):
        number_digits = len(str(number))

        if number_digits <= 0:
            raise ValueError("Number of digits must be positive")

        lowest_number = 10 ** (number_digits - 1)
        highest_number = (10 ** number_digits) - 1

        return lowest_number, highest_number

    def get_samples(self, sample_size=100):
        # Half the training data is uniform, and half is drawn from a normal distribution built around the target.
        value_range = self.get_lowest_and_highest_numbers(self.target)

        uniform_samples = np.random.uniform(value_range[0], value_range[1], int(sample_size / 2))
        uniform_samples = [int(number) for number in uniform_samples]

        gaussian_samples = np.random.normal(self.target, 100, int(sample_size / 2))
        gaussian_samples = [int(number) for number in gaussian_samples]

        shuffled_data = uniform_samples + gaussian_samples
        np.random.shuffle(shuffled_data)
        return shuffled_data

    def sum(self, inputs):
        if isinstance(inputs, (int, float)):
            inputs = self.get_as_list(inputs)

        output = (np.dot(inputs, self.weights)) + self.bias
        return output

    def improve_weights(self, output, error):
        if not error:
            return

        output = self.get_as_list(output)
        for index, value in enumerate(self.weights):
            delta = error * output[index] * self.learning_rate

            self.weights[index] += delta

    def fit(self, epochs=15, samples=100, early_callback=None):
        self.training_data = self.get_samples(samples)

        for epoch in range(epochs):
            error_per_epoch = []

            for sample in self.training_data:
                target_class = self.get_target_class(sample)

                intermediary_output = self.sum(sample)
                predicted_class = self.step_function(intermediary_output)

                error = target_class - predicted_class

                self.improve_weights(sample, error)

                error_per_epoch.append(error)

            global_error = np.mean([i ** 2 for i in error_per_epoch])
            print(f"Erro global por época: {global_error}")

            if early_callback:
                if global_error <= early_callback:
                    print(f"Última epóca treinada: {epoch}")
                    break

    def test(self, test_data=None, detailed=False):
        if not test_data:
            test_data = self.get_samples(100)

        accuracy = 0
        for sample in test_data:
            output = self.sum(sample)

            target_class = self.get_target_class(sample)
            prediction = self.step_function(output)

            if prediction == target_class:
                accuracy += 1

            if detailed:
                print(f"RU {sample}. Previsto: {prediction}. Real: {target_class}")

        print(f"Acurácia: {accuracy} acertos de {len(test_data)}.")
        return accuracy

    @staticmethod
    def get_optimal_parameters():
        best_accuracy = 0

        for learning_rate in [0.1, 0.01, 0.005, 0.001]:
            for epochs in [10, 15, 20, 50, 100]:
                for samples in [50, 100, 200, 500, 1000]:
                    perceptron = Perceptron(learning_rate=learning_rate)
                    perceptron.fit(epochs=epochs, samples=samples)
                    accuracy = perceptron.test()

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        optimal_parameters = {
                            'learning_rate': learning_rate,
                            'epochs': epochs,
                            'samples': samples,
                        }

        print(f"Melhores parâmetros: {optimal_parameters}, Acurácia: {best_accuracy}")