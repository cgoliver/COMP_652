import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

data_path = "../Data/hw3pca.txt"

def pca_fits(data):
    #split
    X_train, X_test = train_test_split(data, test_size=0.2)
    #shuffle data

    train_errors = []
    test_errors = []

    features = X_train.shape[1]

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


    plt.plot(range(1, features), train_errors, label="train error")
    plt.plot(range(1, features), test_errors, label="test error")
    plt.xlabel("Number of components")
    plt.ylabel("Mean reconstruction error")
    plt.legend()
    plt.show()

    pass
if __name__ == "__main__":
    data = np.loadtxt(data_path)
    pca_fits(data)
    pass

