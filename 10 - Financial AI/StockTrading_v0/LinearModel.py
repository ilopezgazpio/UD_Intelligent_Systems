import numpy as np

class LinearModel:
    '''
    A linear regression model
    '''

    def __init__(self, input_dim, n_action):

        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)

        # momentum terms
        self.vW = 0
        self.vb = 0

        self.losses = []


    def predict(self, X):

        assert(len(X.shape) == 2)
        # X -> Tuples x Observation
        # W -> Observation x actions
        # out -> tuples x actions
        return X.dot(self.W) + self.b


    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):

        assert(len(X.shape) == 2)

        # The loss values are 2-D, normally we would divide by N only, but now we divide by N x K
        num_values = np.prod(Y.shape)

        # gradient descent derivatives of Loss
        Yhat = self.predict(X)
        gW = 2 * X.T.dot(Yhat-Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values

        # update momentum terms
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        # update params
        self.W += self.vW
        self.b += self.vb

        mse = np.mean((Yhat - Y)**2)
        self.losses.append(mse)


    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']


    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)