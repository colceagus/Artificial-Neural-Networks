# Tudor Berariu, 2016

import numpy as np

def sigmoid(arr):
    return 1.0 / (1.0 + np.exp(-arr))

class SigmoidClassifier:

    def __init__(self):
        self.params = None

    def output(self, X):
        assert self.params is not None, "No parameters"

        ## TODO: Replace this code with a correct implementation
        (N, _) = X.shape
        (_, K) = self.params.shape

        Y = np.zeros((N, K))
        ## ----

        return Y

    def update_params(self, X, T, lr):
        (N, D) = X.shape
        (_N, K) = T.shape
        assert _N == N, "X and T should have the same number of rows"

        if self.params is None:
            self.params = np.random.randn(D + 1, K) / 100

        ## TODO: Compute the gradient and update the parameters
