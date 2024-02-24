import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
from sklearn import svm

import seaborn as sns
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
#load dateframe
df=pd.read_csv('csvfile/diabetes_binary_health_indicators_BRFSS2015.csv')
#check null operation for dataframe
print(df.head())
print(df.isnull().sum())
print(df.info())
#get the summary statisics
print(df.describe().round())

#feature && target
x=df.iloc[:,1:]
y=df['Diabetes_binary']
feature=SelectKBest(score_func=chi2)
feature.fit(x,y)
x=feature.transform(x)
print(x.shape)
print(feature.get_support())

print(x.shape)
print(y.shape)
#show data 
print(sns.countplot(x='Diabetes_binary',data=df).set_title("0 -> non_diabetic   1-> diabetic"))
plt.show()

#corelation data
coor=df.corr()
print(sns.heatmap(coor,annot=True).set_title("corelation data"))
plt.show()
#split data
print("*"*40)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
# first operation of machine lerning
lr=LogisticRegression(max_iter=2000,C=1.0,tol=0.0001,solver="lbfgs",penalty="l2")
# train model
lr.fit(X=x_train,y=y_train)
print(lr.fit(X=x_train,y=y_train))

#prediction model
print("*"*40)
prediction=lr.predict(x_test)
print("prediction for x_test: ",prediction)
# train and test score
print("\n train score is :{:.3f} ".format(lr.score(X=x_train,y=y_train)))
print("test score is :{:.3f} ".format(lr.score(X=x_test,y=y_test)))

#probabilty for each  predicted
pre_prob=lr.predict_proba(x_test)
print(pre_prob)
print("*"*40)

# confusion metrics first model
cm=confusion_matrix(y_true=y_test,y_pred=prediction)
print(cm)

#clasifiction report to confused all model 
all_model=classification_report(y_true=y_test,y_pred=prediction)
print(all_model)
print("*"*40)
# ##"""
# drow confusion matrices for all prediction features  
sns.heatmap(cm,center=True).set_title("confusion matrices for all prediction features")
plt.show()
print("*"*100)


# predic for one person by testing a disease 
prediction=lr.predict([[1.0,1.0,40.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0]])
print(prediction)
print("*"*40)

# drow prediction for one person by testing a disease
print(sns.countplot(x=prediction,data=df).set_title(" prediction for one person logisic "))
plt.show()


print("*"*100)

#end of logistic regression

# svc classfier
print("-----------svc classfier--------------")
classfier = svm.SVC(max_iter=2000,kernel="rbf", tol=0.0001,random_state=0, C=0.1)
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



