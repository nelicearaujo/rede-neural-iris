"""Rede neural simples para classificar o conjunto de dados Iris.

As operações da rede (forward pass, perda, backpropagation e atualização dos
parâmetros) são implementadas apenas com NumPy. O Scikit-Learn é usado somente
para carregar e dividir os dados.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def relu(x):
    """Função de ativação ReLU."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivada da ReLU."""
    return (x > 0).astype(float)


def softmax(x):
    """Softmax numericamente estável, aplicada linha a linha."""
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exponentials = np.exp(x_shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def one_hot_encode(labels, number_of_classes):
    """Converte rótulos inteiros em vetores One-Hot."""
    return np.eye(number_of_classes)[labels]


def standardize(train_data, test_data):
    """Padroniza os dados usando apenas estatísticas do conjunto de treino."""
    mean = np.mean(train_data, axis=0)
    standard_deviation = np.std(train_data, axis=0)

    # Evita divisão por zero caso uma característica seja constante.
    standard_deviation = np.where(standard_deviation == 0, 1, standard_deviation)

    train_standardized = (train_data - mean) / standard_deviation
    test_standardized = (test_data - mean) / standard_deviation
    return train_standardized, test_standardized


class NeuralNetwork:
    """Rede 4-8-3 treinada por gradiente descendente em lote."""

    def __init__(self, input_size=4, hidden_size=8, output_size=3, seed=42):
        rng = np.random.default_rng(seed)

        # Inicialização de He, apropriada para a camada com ReLU.
        self.weights_1 = rng.normal(
            0, np.sqrt(2 / input_size), size=(input_size, hidden_size)
        )
        self.bias_1 = np.zeros((1, hidden_size))
        self.weights_2 = rng.normal(
            0, np.sqrt(2 / hidden_size), size=(hidden_size, output_size)
        )
        self.bias_2 = np.zeros((1, output_size))

    def forward(self, inputs):
        """Calcula as ativações das camadas escondida e de saída."""
        hidden_input = np.dot(inputs, self.weights_1) + self.bias_1
        hidden_activation = relu(hidden_input)
        output_input = np.dot(hidden_activation, self.weights_2) + self.bias_2
        output_activation = softmax(output_input)

        cache = (inputs, hidden_input, hidden_activation, output_activation)
        return output_activation, cache

    @staticmethod
    def cross_entropy(predictions, targets):
        """Calcula a entropia cruzada média."""
        epsilon = 1e-12
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.mean(np.sum(targets * np.log(predictions), axis=1))

    def train(self, inputs, targets, epochs=1000, learning_rate=0.05):
        """Treina a rede usando backpropagation e gradiente descendente."""
        number_of_samples = inputs.shape[0]

        for epoch in range(1, epochs + 1):
            predictions, cache = self.forward(inputs)
            _, hidden_input, hidden_activation, _ = cache

            # Derivadas da entropia cruzada combinada com Softmax.
            output_error = (predictions - targets) / number_of_samples
            gradient_weights_2 = np.dot(hidden_activation.T, output_error)
            gradient_bias_2 = np.sum(output_error, axis=0, keepdims=True)

            hidden_error = np.dot(output_error, self.weights_2.T) * relu_derivative(
                hidden_input
            )
            gradient_weights_1 = np.dot(inputs.T, hidden_error)
            gradient_bias_1 = np.sum(hidden_error, axis=0, keepdims=True)

            self.weights_2 -= learning_rate * gradient_weights_2
            self.bias_2 -= learning_rate * gradient_bias_2
            self.weights_1 -= learning_rate * gradient_weights_1
            self.bias_1 -= learning_rate * gradient_bias_1

            if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
                loss = self.cross_entropy(predictions, targets)
                print(f"Época {epoch:4d}/{epochs} - Loss: {loss:.4f}")

    def predict(self, inputs):
        """Retorna a classe mais provável para cada amostra."""
        probabilities, _ = self.forward(inputs)
        return np.argmax(probabilities, axis=1)


def main():
    # 1. Carregamento e preparação do conjunto Iris.
    iris = load_iris()
    features = iris.data.astype(float)
    labels = iris.target
    number_of_classes = len(np.unique(labels))

    features_train, features_test, labels_train, labels_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    features_train, features_test = standardize(features_train, features_test)
    targets_train = one_hot_encode(labels_train, number_of_classes)

    # 2 e 3. Criação e treinamento da rede 4 -> 8 -> 3.
    network = NeuralNetwork(
        input_size=features_train.shape[1],
        hidden_size=8,
        output_size=number_of_classes,
        seed=42,
    )
    network.train(features_train, targets_train, epochs=1000, learning_rate=0.05)

    # 4. Avaliação no conjunto de teste.
    predicted_labels = network.predict(features_test)
    accuracy = np.mean(predicted_labels == labels_test)
    print(f"\nAcurácia final no conjunto de teste: {accuracy:.2%}")


if __name__ == "__main__":
    main()
