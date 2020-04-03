import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
from sklearn.metrics import accuracy_score
from scipy import io
import os
import math
import warnings

numpy.random.seed(42)

cwd = os.chdir('/Users/ernestng/Desktop/NUS/Y2S2/DSA3102/HMWK2')
data = io.loadmat('emaildata.mat')
ytrain = data['ytrain']  # shape = (3065,1)
ytest = data['ytest']
xtrain = data['Xtrain']  # shape = (3065,57)
xtest = data['Xtest']
# w = np.zeros((57,1)) #shape = (57,1)
m = xtrain.shape[0]  # m  = 3065
p = xtrain.shape[1]  # p = 57
w = np.random.randn(p)*np.sqrt(2/m)  # shape = (57,1)


def sigmoid(x):
    a = 1/(1+np.exp(-x))
    return a


def prediction(w, Data):
    pred = []
    z = np.dot(w.T, Data.T)
    a = sigmoid(z)
    for i in range(0, len(a[0])):
        if (a[0][i] > 0.5):
            pred.append(1)
        elif (a[0][i] <= 0.5):
            pred.append(-1)
    return pred


def obj(y, w, X):  # scalar
    z = np.dot(X, w)
    val = -np.multiply(y, z)
    fx = np.sum(np.log(1+np.exp(val)))
    return fx


def grad(y, w, X):  # shape = (1,57)
    z = np.dot(X, w)
    val = np.multiply(y, z)
    denom = -y/(1 + np.exp(val))
    gradient = np.dot(denom.T, X)
    return gradient


fOld = []  # list of all function values
grads = []  # list of all gradient values
ctrlist = []
acc = []
tol = 0.1
ctr = 1  # counter
batch = 50  # change batch size
maxit = 10000  # change max iterations
#maxit = maxit/(int(m/batch))
w = np.random.randn(p, 1)  # shape = (57,1)
sgrad = grad(ytrain, w, xtrain)
while ctr < maxit:
    for i in range(1, int(m/batch)):
        # Generating random batch indexes
        weight = np.random.randint(low=1, high=m, size=batch)
        Xup = xtrain[weight]  # xtrain batch
        yup = ytrain[weight]  # ytrain batch
        sgrad = grad(yup, w, Xup)
        # set stepsize
        step = 1/ctr**2
        # update w
        w = w - step*sgrad.T
    fx = obj(ytrain, w, xtrain)
    # print(len(fOld))
    # print(fx)
    if fx != math.inf:
        fOld.append((sum(fOld)+fx)/ctr)
    gradw = np.linalg.norm(sgrad)
    grads.append(gradw)
    ctrlist.append(ctr)

    ypred = prediction(w, xtest)
    score = accuracy_score(ytest, ypred)*100
    acc.append(score)

    print("Epoch", ctr, "accuracy", score)

    ctr += 1
    if gradw < tol:
        break
