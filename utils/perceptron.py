import numpy as np
from . import activation

# input: x in Rn (n,1)
# weights : w in Rn (n,1)
# bias : b in R (1,1)
# output : sigma( w x + b) (1,1)


class Perceptron():

    def __init__(self, eta=0.01, n_iter=1000, input_dim=None, activation="sigmoid", bias=0):
        self.activation = activation.lower()
        self.eta = eta
        self.n_iter = n_iter
        self.W = np.random.random(input_dim)
        self.b = bias

    def weighted_sum(self, X):
        return np.dot(X, self.W) + self.b

    def predict(self, X):
        return np.where(self.weighted_sum(X) >= 0, 1, -1)

    def update(self, y, yhat, X):
        self.W += self.eta * (y - yhat) * X

    def sigma(self, X):
        z = np.dot(self.W, X) + self.b
        if self.activation == "sigmoid":
            return activation.sigmoid(z)
        if self.activation == "relu":
            return activation.relu(z)
        if self.activation == "step":
            return activation.step(z)

    def fit(self, X, Y):
        for _ in range(self.n_iter):  # epoch loop for statistical learning
            for x, y in zip(X, Y):  # loop through each data point
                yhat = self.predict(x)
                self.update(y, yhat, x)
