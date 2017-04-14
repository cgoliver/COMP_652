import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

data_path = "../Data/hw3pca.txt"

def pca_fits(data):
    kf = KFold(n_splits=5)

    total_train_err = []
    total_test_err = []

    total_var = []

    for train_index, test_index in kf.split(data): 
        X_train, X_test = data[train_index], data[test_index]

        train_errors = []
        test_errors = []

        features = X_train.shape[1]
        examples = X_train.shape[0]

        for c in range(1, features):
            pca = PCA(n_components=c)
            pca.fit(X_train)

            X_train_pca = pca.transform(X_train)
            X_test_pca = pca.transform(X_test)

            train_proj = pca.inverse_transform(X_train_pca)
            test_proj = pca.inverse_transform(X_test_pca)

            train_loss = ((X_train - train_proj) ** 2 ).mean()
            test_loss = ((X_test - test_proj) ** 2 ).mean()

            train_errors.append(train_loss)
            test_errors.append(test_loss)


        total_test_err.append(test_errors)
        total_train_err.append(train_errors)

    np_tot_train = np.asarray(total_train_err)
    np_tot_test = np.asarray(total_test_err)

    c = 20

    train_means = np_tot_train.mean(axis=0)[:c]
    train_stds = np_tot_train.std(axis=0)[:c]

    test_means= np_tot_test.mean(axis=0)[:c]
    test_stds= np_tot_test.std(axis=0)[:c]


    plt.errorbar(np.arange(1, len(train_means)+1), train_means,\
        yerr=train_stds, label="train error")
    plt.errorbar(np.arange(1, len(test_means)+1), test_means,\
        yerr=test_stds, label="test error")

    plt.xticks(np.arange(1, len(train_means)+1, 4))
    plt.xlabel("Number of components")
    plt.ylabel("Mean reconstruction error")
    plt.legend()
    plt.savefig("../Tex/Figures/reconstruction_cut.pdg", format="pdf")
    plt.show()

    pca_full = PCA()
    pca_full.fit(data)
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
    plt.xticks(np.arange(1,len(pca_full.explained_variance_ratio_) + 1, 24))
    plt.ylabel("Fraction explained variance")
    plt.xlabel("Number of components")
    plt.savefig("../Tex/Figures/var.pdf", format="pdf")
    plt.show()

    pass
if __name__ == "__main__":
    data = np.loadtxt(data_path)
    pca_fits(data)
    pass

