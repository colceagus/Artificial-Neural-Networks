# Tudor Berariu, 2016

import numpy as np

def sigmoid(arr):
    return 1.0 / (1.0 + np.exp(-arr))

sigmoid = np.vectorize(sigmoid, otypes=[np.float])

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
        Y = np.transpose(Y)
        assert Y.shape == (N, K)
                
        return Y


    def update_params(self, X, T, lr):
        (N, D) = X.shape
        (_N, K) = T.shape
        assert _N == N, "X and T should have the same number of rows"

        if self.params is None:
            print "Generating random parameters"
            self.params = np.random.randn(D + 1, K) / 100

        # TODO: Compute the gradient and update the parameters
        Y = self.output(X)
        #print "Output: ", Y
        
        XwithBias = np.hstack((X, np.ones((N, 1))))
        # Gradient wrt Parameters = ((Y - T) * Y * (1 - Y))' * X
        error = Y - T

        assert error.shape == (N, K)
        #print "Error: ", error 

        dotErrorY = error * Y
        assert dotErrorY.shape == (N, K)

        onesMinusY = np.ones(Y.shape) - Y
        assert onesMinusY.shape == (N, K)

        product = dotErrorY * onesMinusY
        assert product.shape == (N, K)

        gradientWRTParams = np.dot(np.transpose(product), XwithBias) / N
        assert gradientWRTParams.shape == (K, D+1)
        #print "Gradient Mean: ", np.mean(np.abs(gradientWRTParams))

        self.params = self.params - np.transpose(lr * gradientWRTParams)
        assert self.params.shape == (D+1, K)
