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
	print "Computing Closed-Form PseudoInverse. Please wait..."
	tstart = time()
	Xpseudoinv = np.linalg.pinv(X)
	tstop = time()
	print "Done. Time elapsed: %s" % (tstop - tstart)

	print "Computing parameters. Please wait..."
	tstart = time()
	W = np.dot(Xpseudoinv, T)
	tstop = time()
	print "Done. Time elapsed: %s" % (tstop - tstart)

	# Add the bias (D+1, K)
	print "Params w/o bias shape: (%s, %s)" % (W.shape)
	biases = np.ones((1, W.shape[1]))
	print "Biases shape: (%s, %s)" % biases.shape
	self.params = np.append(W, biases, axis=0)

	print "%s %s" % self.params.shape, (D+1, K)
        assert self.params.shape == (D+1, K)

    def output(self, X):
        assert self.params is not None, "No parameters"

        (N, _) = X.shape
        (_, K) = self.params.shape

        Y = np.zeros((N, K))
	transposedW = np.transpose(self.params)
	print "Transposed W shape: %s %s" % transposedW.shape

	bias_components = np.ones((X.shape[0], 1))
	print "Bias Components for X shape: %s %s" % bias_components.shape

	XwithBiasComponents = np.append(X, bias_components, axis=1)
	print "X with Bias Components shape: %s %s" % XwithBiasComponents.shape

	transposedXwithBiasComponents = np.transpose(XwithBiasComponents)
	print "Transposed X with Bias Components shape: %s %s" % transposedXwithBiasComponents.shape

	Y = np.dot(transposedW, transposedXwithBiasComponents)
	Y = np.transpose(Y)
	print "Y shape: %s %s" % Y.shape

	assert Y.shape == (N, K)

        return Y

    def update_params(self, X, T, lr):
        (N, D) = X.shape
        (_N, K) = T.shape
        assert _N == N, "X and T should have the same number of rows"

        if self.params is None:
            self.params = np.random.randn(D + 1, K) / 100

        ## Compute gradient and update parameters
	XwithBias = np.append(X, np.ones((X.shape[0], 1)), axis=1)
	XW = np.dot(XwithBias, self.params)
	targetDifference = XW - T
	targetDifferenceTransposed = np.transpose(targetDifference)
	gradientWithRespectToWeights = np.dot(targetDifferenceTransposed, X)
	# gradient = np.dot(np.transpose(np.dot(np.append(X, np.ones((X.shape[0], 1)), axis=1), self.params) - T), X)
	updatedParams = self.params - lr * gradientWithRespectToWeights
	self.params = updatedParams

