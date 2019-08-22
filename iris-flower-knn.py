import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

#load dataset
dataframe = pd.read_csv("test-datasets/classification/iris.csv",sep=",")

#dataframe shuffling
dataframe = dataframe.reindex(np.random.permutation(dataframe.index))

#training set, test set splitting
train_set = dataframe.sample(frac=0.8, random_state=200)
test_set = dataframe.drop(train_set.index)

#training set features and target
X_train = train_set[["sepal width (cm)", "sepal length (cm)", "petal width (cm)", "petal length (cm)"]]
y_train = train_set[["species"]]

#test set features and target
X_test = test_set[["sepal width (cm)", "sepal length (cm)", "petal width (cm)", "petal length (cm)"]]
y_test = test_set[["species"]]

#inspect dataframe
print(dataframe.describe())

#train the model with K-Nearest Neighbor algorithm with max neighbor 'n'
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

#make predictions on the test set 
prediction = knn.predict(X_test)

#determine the accuracy
print("Training set accuracy : ", knn.score(X_train, y_train))
print("Test set accuracy : ", knn.score(X_test, y_test))
