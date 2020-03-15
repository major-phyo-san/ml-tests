import numpy as np
import pandas as pd
from sklearn.svm import SVC

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

#train the model with SVC-RBFKernel
svm = SVC(kernel='rbf',C=10,gamma=0.1)
svm.fit(X_train,y_train)
training_accuracy = svm.score(X_train,y_train)
test_accuracy = svm.score(X_test,y_test)

#determine the accuracy
print("Training set accuracy with raw data : ", training_accuracy)
print("Test set accuracy with raw data : ", test_accuracy)

#pre-processing the data
#compute the minium value per feature on the training set
min_on_training = X_train.min(axis=0)
#compute the range of each feature (max - min) on the training set
range_on_training = (X_train - min_on_training).max(axis=0)
#subtract the min, and divide by range, afterward min = 0 and max = 1 for each feature
X_train_scaled = (X_train - min_on_training)/range_on_training

#do the same pre-processing on the test set
min_on_test = X_test.min(axis=0)
range_on_test = (X_test - min_on_test).max(axis=0)
X_test_scaled = (X_test - min_on_test)/range_on_test

svm.fit(X_train_scaled,y_train)
training_accuracy = svm.score(X_train_scaled,y_train)
test_accuracy = svm.score(X_test_scaled,y_test)

#determine the accuracy
print("Training set accuracy : ", training_accuracy)
print("Test set accuracy : ", test_accuracy)