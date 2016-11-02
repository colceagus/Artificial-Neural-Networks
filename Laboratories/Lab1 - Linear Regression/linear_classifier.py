# Tudor Berariu, 2016

import numpy as np
from time import time

class LinearClassifier:

    def __init__(self):
        self.params = None

    def closed_form(self, X, T):
        (N, D) = X.shape
        print "X shape: %s %s" % (N, D)
        (_N, K) = T.shape
        assert _N == N, "X and T should have the same number of rows"

        ## Compute the values of the parameters
        self.params = np.zeros((D+1, K))

        # Compute (D, K) params
        # Augment X with the biases' coefficients (ones)
        # X is of (N, D) dimension
        # We add one to the D
        augForX = np.ones((N, 1))
        augX = np.hstack((X, augForX))
        (aN, aD) = augX.shape
        assert aN == N, "Bias Augmented Inputs' Matrix should have the same number of rows" 
        assert aD == (D + 1), "Bias Augmented Inputs should have added another dimension to the columns"

        print "Computing Closed-Form PseudoInverse. Please wait..."
        tstart = time()
        Xpseudoinv = np.linalg.pinv(augX)
        tstop = time()
        print "Done. Time elapsed: %s" % (tstop - tstart)

        print "Computing parameters. Please wait..."
        tstart = time()
        W = np.dot(Xpseudoinv, T)
        tstop = time()
        print "Done. Time elapsed: %s" % (tstop - tstart)

        self.params = W
        print "%s %s" % self.params.shape, (D+1, K)
        assert self.params.shape == (D+1, K)

    def output(self, X):
        assert self.params is not None, "No parameters"

        (N, _) = X.shape
        (_, K) = self.params.shape

        Y = np.zeros((N, K))
        transposedW = np.transpose(self.params)
        #print "Transposed W shape: (%s, %s)" % transposedW.shape

        bias_components = np.ones((N, 1))
        #print "Bias Components for X shape: (%s, %s)" % bias_components.shape

        XwithBiasComponents = np.hstack((X, bias_components))
        #print "X with Bias Components shape: (%s, %s)" % XwithBiasComponents.shape

        transposedXwithBiasComponents = np.transpose(XwithBiasComponents)
        #print "Transposed X with Bias Components shape: (%s, %s)" % transposedXwithBiasComponents.shape

        Y = np.dot(transposedW, transposedXwithBiasComponents)
        Y = np.transpose(Y)
        #print "Y shape: %s %s" % Y.shape

        assert Y.shape == (N, K)

        return Y

    def update_params(self, X, T, lr):
        (N, D) = X.shape
        (_N, K) = T.shape
        assert _N == N, "X and T should have the same number of rows"

        if self.params is None:
            self.params = np.random.randn(D + 1, K) / 100

        ## Compute gradient and update parameters
        XwithBias = np.hstack((X, np.ones((N, 1))))
        #print "X with Bias Components shape: (%s, %s)" % XwithBias.shape
        
        # We have N examples, of dimension D+bias (D+1)
        # We generate N outputs with K classes = dimension (N, K)
        #print self.params
        XW = np.dot(XwithBias, self.params)
        #print "XW dot product shape: (%s, %s)" % XW.shape
        
        # We compute the error (difference between computed output and target 
        # output classes) of dimension (N,K))
        targetDifference = XW - T
        #print "XW - T shape: (%s, %s)" % targetDifference.shape
        
        # The gradient is the difference above transposed (of dimension
        # (K, N)) multiplied by X (of dimension (N, D+1)) resulting
        # a matrix of dimension (K, D+1)
        targetDifferenceTransposed = np.transpose(targetDifference)
        #print "XW - T transposed shape: (%s, %s)" % targetDifferenceTransposed.shape
        
        # Gradient with respect to the parameters = multiplication of the error
        # transposed (target difference transposed, of dimension (K, N)) with
        # the examples of dimension (N, D+1) resulting in a matrix of dimension
        # (K, D+1)
        gradientWithRespectToWeights = np.dot(targetDifferenceTransposed, XwithBias)
        #print "Gradient with respect to the Parameters shape: (%s, %s)" % gradientWithRespectToWeights.shape
        
        # Gradients sum up because we process a batch of inputs
        # a set of inputs consisting of equally distributed samples of digits from 0 to 9
        # We have to divide the summed up gradient values by the number of examples in the batch
        gradientWithRespectToWeights = gradientWithRespectToWeights / N
        #print np.mean(np.abs(gradientWithRespectToWeights))
        
        # We adjust the gradient values with the learning rate
        # resulting a matrix of dimension (K, D+1)
        # gradient = np.dot(np.transpose(np.dot(np.append(X, np.ones((X.shape[0], 1)), axis=1), self.params) - T), X)
        adjustedGradient = lr * gradientWithRespectToWeights
        #print "Learning rate adjusted gradient shape: (%s, %s)" % adjustedGradient.shape
        
        # And subtract the adjusted gradient values transposed (of dimension (D+1, K)
        # from the parameters (of dimension (D+1, K), resulting of a matrix of
        # of dimension (K, D+1)
        updatedParams = self.params - np.transpose(adjustedGradient)
        #print "Adjusted params shape: (%s, %s)" % updatedParams.shape
        self.params = updatedParams
