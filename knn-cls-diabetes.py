import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

#load dataset
dataframe = pd.read_csv("test-datasets/classification/pima-indians-diabetes.data.csv",sep=",")

#dataframe shuffling
dataframe = dataframe.reindex(np.random.permutation(dataframe.index))

#training set, test set splitting
train_set = dataframe.sample(frac=0.8, random_state=200)
test_set = dataframe.drop(train_set.index)

#training set features and target
X_train = train_set[["no: of pregnancy","plasma glucose concentration","diastolic blood pressure(mmHg)",
"triceps skin fold thickness(mm)","2hr serum insulin (mu U/ml)","BMI","diabetes pedigree","age"]]
y_train = train_set[["diabetic"]]

#test set features and target
X_test = test_set[["no: of pregnancy","plasma glucose concentration","diastolic blood pressure(mmHg)",
"triceps skin fold thickness(mm)","2hr serum insulin (mu U/ml)","BMI","diabetes pedigree","age"]]
y_test = test_set[["diabetic"]]

#inspect dataframe
print(dataframe.describe())

#train the model with K-Nearest Neighbor algorithm with max neighbor 'n_neighbor'
training_accuracy = []
test_accuracy = []
neighbors_settings = range(1,5)
for n_neighbor in neighbors_settings:
	knn = KNeighborsClassifier(n_neighbors=n_neighbor)
	knn.fit(X_train, y_train)
	training_accuracy.append(knn.score(X_train, y_train))
	test_accuracy.append(knn.score(X_test, y_test))

#make predictions on the test set 
prediction = knn.predict(X_test)

#determine the accuracy
print("Training set accuracy : ", training_accuracy)
print("Test set accuracy : ", test_accuracy)
plt.plot(neighbors_settings, training_accuracy, color="blue", label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, color="red", label="test accuracy")
plt.xlabel("No: of neighbors")
plt.ylabel("Accuracy")
plt.legend()
plt.show()