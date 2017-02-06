import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import linear_model
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.stats import norm

x_file = "x.dat"
y_file = "y.dat"

x = np.loadtxt(x_file)
y = np.loadtxt(y_file)

x_data = np.insert(x, 0, 1, axis=1)


data_split = train_test_split(x, y, test_size=0.2)

lambdas = [1e-50, 0.1, 1, 10, 100, 1000, 10000]
sigmas_g = [0.1, 0.5, 1, 5, 10]
mu = np.arange(-10, 11, 5)

def basis(X, sigma=[0.1]):
    phi = np.array(X[:,0])

    for j, col in enumerate(X.T):
        if j == 0:
            continue
        else:
            for s in sigma:
                for m in mu:
                    gauss_vec = norm.pdf(col, m, s)
                    phi = np.vstack((phi, gauss_vec))
    return phi.T

def reg_test(X, y, param=[1e-50, 0.1, 1, 10, 100, 10000], ylabel=None,\
    metric='loss', title=None, save=None, xlabel=None, log=False, reg=False,\
        basis_inp=False, sigmas=[0.1]):

    X, y = shuffle(X, y)

    train_vals = {}
    test_vals = {}

    weights_matrix = {}
    weights = {}
    K = int((len(X) / (len(X) * 0.2)))

    print("Doing {0} splits".format(K))
    kf = KFold(n_splits=K)

    current_split = 1
    for train, test in kf.split(X):
        print("running split {0}".format(current_split))

        for i, l in enumerate(param):
            print("param {0}".format(i))
            if basis_inp:
                print("computing basis")
                X_basis = basis(X, sigma=sigmas)

                print("computed basis")
                X_train, X_test, y_train, y_test = X_basis[train], X_basis[test], y[train], y[test]

            else:
                X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            if reg:
                c = 1/l
                print(c, l)
                clf = linear_model.LogisticRegression(penalty='l2', C=c,\
                fit_intercept=False)
            else:
                print("training")
                clf = linear_model.LogisticRegression(penalty='l2', C=10e50,\
                    fit_intercept=False)
            clf.fit(X_train, y_train)
            print("trained")

            if l in weights_matrix.keys():
                weights_matrix[l] = np.vstack((weights_matrix[l],\
                    clf.coef_.flatten()))
            else:
                weights_matrix[l] = clf.coef_

            weights.setdefault(l, []).append(clf.coef_.flatten())


            if metric == 'accuracy':
                train_vals.setdefault(l, []).append(clf.score(X_train, y_train))
                test_vals.setdefault(l, []).append(clf.score(X_test, y_test))
    
            if metric == 'loss':
                train_vals.setdefault(l, []).append(log_loss(y_train,\
                    clf.predict(X_train)))
                test_vals.setdefault(l, []).append(log_loss(y_test,\
                    clf.predict(X_test)))

        current_split += 1

    stat = lambda d, fun: [fun(d[k]) for k in d]

    as_matrix = lambda d: np.array([d[k] for k in d])

    # split each weight vector by sigma and take norm. [norm(s1), norm(s2), ..]
    norms_split = lambda w: [np.linalg.norm(x) for x in 
        np.split(w[1:], (x_data.shape[1] -1) / len(sigmas))]

    test_mean = stat(test_vals, np.mean)
    test_std = stat(test_vals, np.std)
    train_mean = stat(train_vals, np.mean)
    train_std = stat(train_vals, np.std)

    norms = {l:[np.linalg.norm(w) for w in weights[l]] for l in param}
    # get ditionary of norms for each lambda {l1: [[norm(s1), norm(s2)], [...]]}
    # norms_sigma = {l:[norms_split(w) for w in weights[l]] \
        # for l in param}

    # get mean of norm for each lambda {l1: [s1, s2, s3], ...}
    sigma_norms_mean = {}
    sigma_norms_std  = {}

    for l in param:
        weights_array = np.asarray(weights[l])
        print(x_data.shape)
        print("weights array")
        print(weights_array.shape)
        print(len(sigmas))
        norms_array = np.asarray([[np.linalg.norm(s) for s in np.split(w[1:],\
            len(sigmas))] for w in weights_array])
        sigma_norms_mean[l] = np.mean(norms_array, axis=0)
        print(sigma_norms_mean[l])
        sigma_norms_std[l] = np.std(norms_array, axis=0)


    norm_means = stat(norms, np.mean)
    norm_std = stat(norms, np.mean)

    weight_val_means = {lam:[np.mean(weights_matrix[lam][:,j]) for j in
        range(weights_matrix[lam].shape[1])] for lam in weights_matrix}

    weight_val_std= {lam:[np.std(weights_matrix[lam][:,j]) for j in
        range(weights_matrix[lam].shape[1])] for lam in weights_matrix}
    
    fig, ax = plt.subplots()

    if metric == "accuracy" or metric == "loss":
        ax.errorbar(param, train_mean, yerr=train_std, label='train')
        ax.errorbar(param, test_mean, yerr=test_std, label='test')

    elif metric == "norm":
        ax.errorbar(param, norm_means, yerr=norm_std)

    elif metric == "sigma_norm":
        print(sigma_norms_mean)
        for i in range(len(sigmas)):
            line_mean = [sigma_norms_mean[l][i] for l in param]
            line_std = [sigma_norms_std[l][i] for l in param]
            ax.errorbar(param, line_std, yerr=line_std,\
                label=sigmas[i])
        
    elif metric == "values":
        means = as_matrix(weight_val_means).T
        stds = as_matrix(weight_val_std).T
        l = np.array(param)
        lam_bcast = np.broadcast_to(l, (means.shape[0],l.shape[0]))
        for i, w in enumerate(means):
            plt.plot(l, w, 'o')

    else:
        print("invalid metric")
    if log:
        ax.set_xscale("log")

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.set_ylim(ymin=0)
    plt.legend(loc="upper left")

    if save:
        plt.savefig(save, format="pdf")

    else:
        plt.show()

if __name__ == "__main__":
    X_gauss = basis(x_data)
    reg_test(x_data, y, metric="sigma_norm", ylabel="L2 Norm",\
         title="", xlabel=r"$\log\lambda$", param=lambdas, basis_inp=True, \
            reg=True,log=True, sigmas=sigmas_g,\
                save="Figures/all_sigma_norms.pdf")
