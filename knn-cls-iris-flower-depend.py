import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# load dataset
dataset = load_iris()

# dataset keys info:
print(dataset.keys())

# training set, test sets splitting
X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], random_state=0)

# shape of training data
print(X_train.shape)
print(y_train.shape)

# creating a pandas dataframe from feature names in dataset
dataframe = pd.DataFrame(X_train, columns=dataset.feature_names)

# visualizing dataframe
scatter_matrix(dataframe, c=y_train, figsize=(15,15), marker=".", 
hist_kwds={'bins':20}, s=60, alpha=0.8, cmap=cm.get_cmap('Spectral'))
plt.show()

# we'll predict new data with K-Nearest Neighbor algorithm with max neighbor 1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# test the accuracy with the test set
y_pred = knn.predict(X_test)
print(dataset['target_names'][y_pred])
print("accuracy = " + str(knn.score(X_test, y_test)))
