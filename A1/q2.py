### Author: Carlos G. Oliver
### Kernelized logistic regression

import numpy as np
import sys
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import binarize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import linear_model
from sklearn.model_selection import KFold

def w_alpha(lam, alpha, y, X):
    """
        compute w vector in terms of alphas
    """

    m, n = X.shape
    w_a = np.array([np.sum([alpha[i] * y[i] * X[i][j] for i in range(m)])\
            for j in range(n)])

    return 1/lam * np.array(w_a)

def g_i(alpha, a_i, x_i, y_i, X, lam):
    """
        compute first derivative at a given a_i 
    """
    w_a = w_alpha(lam, alpha, y, X)
    return y_i * np.dot(w_a, x_i) + np.log(a_i / (1-a_i))

def new_alpha(a_i, g, lam, x_i):
    """
        compute new value of alpha_i
    """
    
    return a_i - (g / ((1/lam) * np.dot(x_i.T, x_i) * (1  / (a_i * (1 - a_i)))))

def update_alphas(alpha, X, y, lam):
    """
        update each element of alpha to compute new alpha vector
    """
    for i, a_i in enumerate(alpha):
        g = g_i(alpha, a_i, X[i], y[i], X, lam)

        a_new = new_alpha(a_i, g, lam, X[i])
        if a_new >= 1 or a_new <= 0:
            a_new = a_i

        alpha[i] = a_new

    return alpha

def logistic_fit(X, y, lam=1, iterations=25):
    alpha = np.random.rand(X.shape[0])
    for i in range(iterations):
        print("updating alphas iteration: %s of %s" % (i, iterations))
        alpha = update_alphas(alpha, X, y, lam)

    return alpha

def poly_kernel(x, z, d):
    """
        Evaluates polynomial kernel on vectors x and z
    """
    return (np.dot(x, z) + 1) ** d

def logistic(x):
    """
        simple logistic function
    """
    return (1 / (1 + np.exp(-x)))

def kernel_logistic(alpha, X, x, d=1):
   return logistic(np.sum([alpha[i] * poly_kernel(X[i], x, d) for i in \
        range(X.shape[0])]))

def predict(probs):
    return binarize(probs, threshold=0.5).flatten()
if __name__ == "__main__":
    x_file = "x.dat"
    y_file = "y.dat"

    x = np.loadtxt(x_file)
    y = np.loadtxt(y_file)

    X = np.insert(x, 0, 1, axis=1)

    K = int((len(X) / (len(X) * 0.2)))

    kf = KFold(K)
    #run kernelized logistic regression on d=1,2,3

    degree_acc_train = {}
    degree_acc_test = {}

    for d in [1, 2, 3]:
        print("evaluating kernel degree %s" % d)
        for train, test in kf.split(X):
            X_train, X_test, y_train, y_test = X[train], X[test],\
                y[train], y[test]

            print(X_train.shape)
            alpha = logistic_fit(X_train, y_train)

            train_yhat = np.array([kernel_logistic(alpha, X_train, X_train[i],\
                d=d) for i in range(X_train.shape[0])])

            test_yhat = np.array([kernel_logistic(alpha, X_test, X_test[i], d=d) for i\
                in range(X_test.shape[0])])

            binary_predicted_y_train = predict(train_yhat)
            binary_predicted_y_test = predict(test_yhat)

            degree_acc_train.setdefault(d, []).append(accuracy_score(y_train,\
                binary_predicted_y_train))
            degree_acc_test.setdefault(d, []).append(accuracy_score(y_test,\
                binary_predicted_y_test))

    log_acc_train_df = pd.DataFrame.from_dict(degree_acc_train)
    log_acc_test_df = pd.DataFrame.from_dict(degree_acc_test)

    log_acc_train_df.to_csv("log_kernel_train.csv")
    log_acc_test_df.to_csv("log_kernel_test.csv")

    #run regular logistic regression
    
    logistic_acc = {}

    print("training regular logistic regression")
    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test],\
            y[train], y[test]
        clf = linear_model.LogisticRegression()
        clf.fit(X_train, y_train)
        
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(Y_test, y_test)

        logistic_acc.setdefault('train', []).append(train_score)
        logistic_acc.setdefault('test', []).append(test_score)


    log_df = pd.DataFrame.from_dict(logistic_acc)
    log_df.to_csv("logistic_acc.csv")
    pass
