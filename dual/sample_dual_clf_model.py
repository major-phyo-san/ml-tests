import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

def dataframe_read_shuffle(path):
    dataframe = pd.read_csv(path,sep=",")
    np.random.seed(83)
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
    return dataframe

def dataframe_splitter(dataframe):
    train_set = dataframe.sample(frac=0.7, random_state=200)
    remainframe = dataframe.drop(train_set.index)
    validate_set = remainframe.sample(frac=0.5, random_state=200)
    test_set = remainframe.drop(validate_set.index)
    return train_set,validate_set,test_set

def features_target_splitter(dataset, features, target):
    X = dataset[features]
    y = dataset[target]
    return X,y

def train_classifier_model(model, train_features, train_target, validate_features, validate_target):
    model.fit(train_features, train_target)
    r_squared = model.score(validate_features, validate_target)
    
    predictions = model.predict(validate_features)
    mse = mean_squared_error(validate_target, predictions)
    rmse = np.sqrt(mse)
    
    return r_squared, rmse

def final_test(model, test_features, test_target):
    r_squared = model.score(test_features,test_target)
    predictions = model.predict(test_features)
    mse = mean_squared_error(test_target,predictions)
    rmse = np.sqrt(mse)
    return r_squared, rmse

dataframe = dataframe_read_shuffle("../test-datasets/classification/breast_cancer.csv")
train_set, validate_set, test_set = dataframe_splitter(dataframe)
X_train, y_train = features_target_splitter(train_set, ["mean radius","mean texture","mean perimeter","worst radius","worst texture","worst perimeter"],["cancerous"])
X_validate, y_validate = features_target_splitter(validate_set, ["mean radius","mean texture","mean perimeter","worst radius","worst texture","worst perimeter"],["cancerous"])
X_test, y_test = features_target_splitter(test_set, ["mean radius","mean texture","mean perimeter","worst radius","worst texture","worst perimeter"],["cancerous"])

knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_r_squared, knn_rmse = train_classifier_model(knn_clf, X_train, y_train, X_validate, y_validate)
print("KNN R Squared: {:.2f}".format(knn_r_squared))
print("KNN RMSE: {:.3f}".format(knn_rmse))

log_reg = LogisticRegression(C=1)
logreg_r_squared, logreg_rmse = train_classifier_model(log_reg, X_train, y_train, X_validate, y_validate)
print("LogReg R Squared: {:.2f}".format(logreg_r_squared))
print("LogReg RMSE: {:.3f}".format(logreg_rmse))

knn_r_squared_fin, knn_rmse_fin = final_test(knn_clf, X_test, y_test)
print("KNN R Squared (on test set): {:.2f}".format(knn_r_squared_fin))
print("KNN RMSE (on test set): {:.3f}".format(knn_rmse_fin))

logreg_r_squared_fin, logreg_rmse_fin = final_test(log_reg, X_test, y_test)
print("LogReg R Squared (on test set): {:.2f}".format(logreg_r_squared_fin))
print("LogReg RMSE (on test set): {:.3f}".format(logreg_rmse_fin))
