import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

#load dataset
dataframe = pd.read_csv("../test-datasets/classification/breast_cancer.csv",sep=",")

#dataframe shuffling
dataframe = dataframe.reindex(np.random.permutation(dataframe.index))

#training set, test set splitting
train_set = dataframe.sample(frac=0.8, random_state=200)
test_set = dataframe.drop(train_set.index)

#training set features and target
X_train = train_set[["mean radius","mean texture","mean perimeter","mean area","mean smoothness",
"mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
"radius error","texture error","perimeter error","area error","smoothness error","compactness error",
"concavity error","concave points error","symmetry error","fractal dimension error","worst radius",
"worst texture","worst perimeter","worst area","worst smoothness","worst compactness","worst concavity",
"worst concave points","worst symmetry","worst fractal dimension"]]
y_train = train_set[["cancerous"]]

#test set features and target
X_test = test_set[["mean radius","mean texture","mean perimeter","mean area","mean smoothness",
"mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
"radius error","texture error","perimeter error","area error","smoothness error","compactness error",
"concavity error","concave points error","symmetry error","fractal dimension error","worst radius",
"worst texture","worst perimeter","worst area","worst smoothness","worst compactness","worst concavity",
"worst concave points","worst symmetry","worst fractal dimension"]]
y_test = test_set[["cancerous"]]

mlp = MLPClassifier(solver='lbfgs',random_state=10,hidden_layer_sizes=[6,3], activation="relu",alpha=5)
mlp.fit(X_train,y_train)
training_accuracy = mlp.score(X_train,y_train)
test_accuracy = mlp.score(X_test,y_test)

#determine the accuracy
print("Training set accuracy : ", training_accuracy)
print("Test set accuracy : ", test_accuracy)