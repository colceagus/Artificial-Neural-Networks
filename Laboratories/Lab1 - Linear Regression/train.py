# Tudor Berariu, 2016

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from time import time as timeit

from mnist_loader import load_mnist

from linear_classifier import LinearClassifier
from sigmoid_classifier import SigmoidClassifier

####### BENCHMARK #######
t_program_start = timeit()
#########################

EPOCHS_NO = 400
LEARNING_RATE = 0.004
REPORT_EVERY = 40

def evaluate(model, X, L):
    Y = model.output(X)
    C = Y.argmax(1)
    (N, K) = Y.shape

    accuracy = (np.sum(C == L) * 1.0) / N

    conf_matrix = np.zeros((K, K), dtype="int")
    for i in range(N):
        conf_matrix[L[i],C[i]] += 1

    return accuracy, conf_matrix

def plot_confusion_matrix(conf_matrix, figure_id, title):
    plt.figure(figure_id)
    (N,_) = conf_matrix.shape
    plt.imshow(conf_matrix, interpolation='nearest')
    plt.xticks(np.arange(0,N), map(str,range(N)))
    plt.yticks(np.arange(0,N), map(str,range(N)))
    plt.title(title)
tstart = timeit()
data = load_mnist()
tstop = timeit()
print "Load MNIST: %s" % (tstop - tstart) # prints in seconds

tstart = timeit()
N_train = data["train_no"]
#N_train = 20000
X_train = data["train_imgs"].squeeze()[:N_train,:]
L_train = data["train_labels"][:N_train]
T_train = np.zeros((N_train, L_train.max() + 1))
T_train[np.arange(N_train), L_train] = 1
tstop = timeit()
print "MNIST Train Dataset: %s" % (tstop - tstart) # in seconds

tstart = timeit()
N_test = data["test_no"]
X_test = data["test_imgs"].squeeze()
L_test = data["test_labels"]
T_test = np.zeros((N_test, L_test.max() + 1))
T_test[np.arange(N_test), L_test] = 1
tstop = timeit()
print "MNIST Test Dataset: %s" % (tstop - tstart) # ditto

# ------------------------------------------------------------------------------
# ------ Closed form solution

cf_model = LinearClassifier()
tstart = timeit()
cf_model.closed_form(X_train, T_train)
tstop = timeit()

print "Closed-Form Train Time: %s" % (tstop - tstart)

acc, conf = evaluate(cf_model, X_test, L_test)

print("[Closed Form] Accuracy on test set: %f" % acc)
print(conf)
plot_confusion_matrix(conf, 1, "Closed form")

acc1 = np.ones(EPOCHS_NO) * acc

tstop = timeit()
print "Closed-Form Execution Time: %s" % (tstop - tstart)

print("-------------------")

# ------------------------------------------------------------------------------
# ------ Gradient optimization of linear model
tstart = timeit()
grad_model = LinearClassifier()

acc2 = np.zeros(EPOCHS_NO)

ep = 1
while ep <= EPOCHS_NO:
    grad_model.update_params(X_train, T_train, LEARNING_RATE)
    acc, conf = evaluate(grad_model, X_test, L_test)

    acc2[ep-1] = acc

    if ep % REPORT_EVERY == 0:
        print("[Linear-grad] Epoch %4d; Accuracy on test set: %f" % (ep, acc))

    ep = ep + 1

print(conf)
plot_confusion_matrix(conf, 2, "Linear model - gradient")

tstop = timeit()
print "Gradient Optimization on Linear Model Execution Time: %s" \
        % (tstop - tstart)

print("-------------------")

# ------------------------------------------------------------------------------
# ------ Non-linear model
tstart = timeit()
sig_model = SigmoidClassifier()

ep = 1
acc3 = np.zeros(EPOCHS_NO)

while ep <= EPOCHS_NO:
    sig_model.update_params(X_train, T_train, LEARNING_RATE)
    acc, conf = evaluate(sig_model, X_test, L_test)

    acc3[ep-1] = acc

    if ep % REPORT_EVERY == 0:
        print("[Linear-grad] Epoch %4d; Accuracy on test set: %f" % (ep, acc))

    ep = ep + 1

print(conf)
plot_confusion_matrix(conf, 3, "Sigmoid model")

tstop = timeit()
print "Gradient Optimization on Non-Linear Model Execution Time: %s" \
        % (tstop - tstart)
print("-------------------")

####### BENCHMARK END #######
tstop = timeit()
print "Gradient Optimization on Non-Linear Model Execution Time: %s" \
        % (tstop - t_program_start)
#############################
plt.figure(4)

plt.plot(np.arange(1, EPOCHS_NO+1), acc1, label="Closed form")
plt.plot(np.arange(1, EPOCHS_NO+1), acc2, label="Linear model")
plt.plot(np.arange(1, EPOCHS_NO+1), acc3, label="Non-linear model")
plt.legend(loc="lower right")
plt.show()


