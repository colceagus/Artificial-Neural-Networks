# Tudor Berariu, 2016

import numpy as np

def sigmoid(arr):
    return 1.0 / (1.0 + np.exp(-arr))

sigmoid = np.vectorize(sigmoid)

class SigmoidClassifier:

    def __init__(self):
        self.params = None


    def output(self, X):
        assert self.params is not None, "No parameters"

        ## TODO: Replace this code with a correct implementation
        (N, D) = X.shape
        (pD, K) = self.params.shape
        
        assert pD == D + 1, "Parameters are not augmented with biases"

        Y = np.zeros((N, K))
 
        # We add the bias component to the inputs
        XBiasComponents = np.ones((N, 1))
        XwithBias = np.hstack((X, XBiasComponents))

        (aN, aD) = XwithBias.shape
        assert aN == N, "Bias Augmented Inputs' Matrix should have the same number of rows" 
        assert aD == (D + 1), "Bias Augmented Inputs should have added another dimension to the columns"

        # We presume that the Parameters have the shape of (D+1, K)
        Y = sigmoid(np.dot(np.transpose(self.params), np.transpose(XwithBias)))
                
        return Y

    def update_params(self, X, T, lr):
        (N, D) = X.shape
        (_N, K) = T.shape
        assert _N == N, "X and T should have the same number of rows"

        if self.params is None:
            self.params = np.random.randn(D + 1, K) / 100

        ## TODO: Compute the gradient and update the parameters
        XwithBias = np.hstack((X, np.ones((N, 1))))
        Y = self.output(X)
        # Gradient wrt Parameters = ((Y - T) * Y * (1 - Y))' * X
        error = Y - np.transpose(T)
        dotErrorY = np.dot(error, np.transpose(Y))
        onesMinusY = np.ones(Y.shape) - Y
        product = np.dot(dotErrorY, onesMinusY)
        gradientWRTParams = np.dot(product, XwithBias) 
        self.params = self.params - np.transpose(lr * gradientWRTParams)
