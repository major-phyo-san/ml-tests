import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt

#load dataset
dataframe = pd.read_csv("../test-datasets/regression/california_housing_train.csv",sep=",")

#dataframe shuffling
dataframe = dataframe.reindex(np.random.permutation(dataframe.index))

#training set, test set splitting
train_set = dataframe.sample(frac=0.8, random_state=200)
test_set = dataframe.drop(train_set.index)

#training set features and target
X_train = train_set[["longitude","latitude","housing_median_age","total_rooms",
"total_bedrooms","population","households","median_income"]]
y_train = train_set[["median_house_value"]]

#test set features and target
X_test = test_set[["longitude","latitude","housing_median_age","total_rooms",
"total_bedrooms","population","households","median_income"]]
y_test = test_set[["median_house_value"]]

#inspect dataframe
print(dataframe.describe())

#train the model with Linear Regressor with various Alpha values
training_accuracy = []
test_accuracy = []
intercept_b = []
coefficients_w = []
alpha_settings = [0.0,0.1,0.25,0.5,0.75,1]
for alpha_setting in alpha_settings:
	lin = Ridge(alpha=alpha_setting)
	lin.fit(X_train,y_train)
	training_accuracy.append(lin.score(X_train, y_train))
	test_accuracy.append(lin.score(X_test, y_test))
	intercept_b.append(lin.intercept_)
	coefficients_w.append(lin.coef_)

#determine the accuracy
print("Training set accuracy : ", training_accuracy)
print("Test set accuracy : ", test_accuracy)
print("Coefficient Values : ", coefficients_w)
print("Intercept Values : ", intercept_b)
plt.plot(alpha_settings, training_accuracy, color="blue", label="training accuracy")
plt.plot(alpha_settings, test_accuracy, color="red", label="test accuracy")
#plt.plot(alpha_settings, intercept_b, color="yellow", label="intercept values")
plt.xlabel("Alpha values")
plt.ylabel("Accuracy")
plt.show()