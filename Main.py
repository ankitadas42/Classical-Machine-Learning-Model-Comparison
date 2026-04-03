import pandas as pd
import numpy as ny
# from sklearn library of ml we import datasets,functions and 3 ai models 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


data=load_iris()
X=data.data #output data
Y=data.target  #output data


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
#LogisticRegression for learning patterns of both variable
lr=LogisticRegression()
lr.fit(X_train,y_train)


#KNeighborsClassifier for learning patterns of both variable
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)


#Decision tree
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)

# func to predict input data  from dataset use for testing
lr_pred=lr.predict(X_test)
knn_pred=knn.predict(X_test)
dt_pred=knn.predict(X_test)


# func to compare predicted/input data and answers
print("LogisticRegression",accuracy_score(y_test,lr_pred),"%")
print("Knn",accuracy_score(y_test,knn_pred),"%")
print("DecisionTree",accuracy_score(y_test,dt_pred),"%")

