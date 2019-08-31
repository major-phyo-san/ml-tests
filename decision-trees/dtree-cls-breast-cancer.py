import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt

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

#train the model with Decision Tree
depth_settings = [1,2,4,8]
training_accuracy = []
test_accuracy = []
for depth_setting in depth_settings:
	d_tree = DecisionTreeClassifier(max_depth=depth_setting,random_state=0)
	d_tree.fit(X_train,y_train)
	training_accuracy.append(d_tree.score(X_train,y_train))
	test_accuracy.append(d_tree.score(X_test,y_test))

#determine the accuracy
print("Training set accuracy : ", training_accuracy)
print("Test set accuracy : ", test_accuracy)
plt.plot(depth_settings, training_accuracy, color="blue", label="training accuracy")
plt.plot(depth_settings, test_accuracy, color="red", label="test accuracy")
plt.legend()
plt.xlabel("Max Tree Depth")
plt.ylabel("Accuracy")
plt.show()