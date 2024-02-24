import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
# read dataframe
df = pd.read_csv('csvfile/diabetes_binary_health_indicators_BRFSS2015.csv')
# print first five row


print(df.head())
print(df.shape)

# check null operation for dataframe

print(df.isnull().sum())

# get the summary statisics


print(df.describe().round())
print(df.info())


# diabetes binary
# 0 -> non_diabetic
# 1 -> diabetic
# the best feature and traget
x = df.iloc[:, 1:]
y = df['Diabetes_binary']

feature=SelectKBest(score_func=chi2)
feature.fit(x,y)
x=feature.transform(x)
print(x.shape)
print(feature.get_support())

print("---------shapes for feature and target----------- ")
print(x.shape)
print(y.shape)
print("*"*40)



# split data

print("*"*40)
print("------------train test split && shapes-----------------------/n")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print("---------shapes for feature and target after splits and scaling----------\n")
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print("*"*40)

# svc classfier
print("-----------svc classfier--------------")
classfier = svm.SVC(max_iter=2000, kernel="linear", tol=0.0001,random_state=0, C=0.1)
classfier.fit(X=x_train, y=y_train)

print("\n train score is : {:.3f}".format(classfier.score(X=x_train, y=y_train)))
print("test score is : {:.3f}".format(classfier.score(X=x_test, y=y_test)))
print("*"*40)

# prediction x_test
print("---------prediction for x_test -------")
prediction = classfier.predict(X=x_test)
print(prediction)
print("*"*40)

# confusion matrix
print("--------confusion matrix---------")
cm = confusion_matrix(y_true=y_test, y_pred=prediction)
print(cm)
print("*"*40)

# drow confusion matrix svc
print("*"*40)
print(sns.heatmap(cm, center=True).set_title(" drow confusion matrix svc "))
plt.show()
print("*"*40)
#
print("--------------classfication_rebort------------------")
all_model = classification_report(y_true=y_test, y_pred=prediction)
print(all_model)
print("*"*40)

print(sns.countplot(x=prediction, data=df).set_title(" prediction for one person by svm "))
plt.show()
