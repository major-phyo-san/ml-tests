import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

#train the model with Random Forest
tree_counts = [1,2,3,4,5,6,7,8,9,10]
training_accuracy = []
test_accuracy = []
for tree_count in tree_counts:
	d_tree_forest = RandomForestClassifier(n_estimators=tree_count,random_state=0)
	d_tree_forest.fit(X_train,y_train)
	training_accuracy.append(d_tree_forest.score(X_train,y_train))
	test_accuracy.append(d_tree_forest.score(X_test,y_test))

#determine the accuracy
for i in training_accuracy.count():
	print("Training set accuracy : ", training_accuracy)
	print("Test set accuracy : ", test_accuracy)

plt.plot(tree_counts, training_accuracy, color="blue", label="training accuracy")
plt.plot(tree_counts, test_accuracy, color="red", label="test accuracy")
plt.legend()
plt.xlabel("No: of trees")
plt.ylabel("Accuracy")
plt.show()