import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("csvfile/diabetes_binary_health_indicators_BRFSS2015.csv")
# explore data
print(df.head())
print( df.dropna())
print(df.drop_duplicates())
print(df.describe().round())
print("summution of null",df.isnull().sum())

#coorelation data
coor=df.corr()
print(sns.heatmap(coor,annot=True).set_title("corelation data"))
plt.show()

# show dataframe 
print(sns.countplot(x='Diabetes_binary',data=df).set_title("0 -> non_diabetic   1-> diabetic"))
plt.show()


#feature selection
x=df.iloc[:,1:22]
y=df['Diabetes_binary']

#feature selection
feature=SelectKBest(score_func=chi2)
feature.fit(x,y)
x1=feature.transform(x)
print(x1.shape)
print(y.shape)
print(feature.get_support())

#data scaling 
scaling=StandardScaler(copy=True,with_mean=True,with_std=True)
scaling.fit(X=x)
x=scaling.transform(X=x)
print (" -----------data scaling for features decision tree -------")
print(x)
print(x.shape)

#show data 
print(sns.countplot(x='Diabetes_binary',data=df).set_title("0 -> non_diabetic   1-> diabetic"))
plt.show()

#split data
X_train, X_test, y_train, y_test = train_test_split(x1, y, test_size = 0.20,random_state=0)

ml = LogisticRegression(solver='liblinear', C=1.0, random_state=0)
ml.fit(X_train, y_train)



#prediction data for x
y_pred = ml.predict(X_test)
print(y_pred)
print(X_train.shape)

# train and test score
print("\n train score is :{:.3f} ".format(ml.score(X=X_train,y=y_train)))
print("test score is :{:.3f} ".format(ml.score(X=X_test,y=y_test)))
print("*"*40)

#probabilty for each  predicted
pre_prob=ml.predict_proba(X_test)
print(pre_prob)
print("*"*40)

#confusion matrix
cm=confusion_matrix(y_test,y_pred)
print("confusion_matrix :",cm)
print("*"*40)

# drow confusion matrices for all prediction features  
sns.heatmap(cm,center=True).set_title("confusion matrices for all prediction features logistic")
plt.show()
print("*"*100)


#classfication rebort
print("--------classfication rebort for all data  logistic---------")
all_model=classification_report(y_test, y_pred) 
print(all_model)
print("*"*40)


 #-------------------------   SVM classfier   -----------------------------------------------------------------------

# svc classfier
print("-----------svc classfier--------------")
classfier = svm.SVC(kernel="linear",max_iter=2000 ,tol=0.000001,random_state=0, C=0.1)
classfier.fit(X=X_train, y=y_train)

print("\n train score is : {:.3f}".format(classfier.score(X=X_train, y=y_train)))
print("test score is : {:.3f}".format(classfier.score(X=X_test, y=y_test)))
print("*"*40)

# prediction x_test
print("---------prediction for x_test -------")
prediction = classfier.predict(X=X_test)
print(prediction)
print("*"*40)

# confusion matrix
print("--------confusion matrix---------")
cm = confusion_matrix(y_true=y_test, y_pred=prediction)
print(cm)
print("*"*40)

# drow confusion matrix svc
print("*"*40)
print(sns.heatmap(cm, center=True).set_title("  confusion matrix svc "))
plt.show()
print("*"*40)



#("--------classfication rebort for all data  svm---------")
print("--------classfication rebort for all data  svm ---------")
all_model = classification_report(y_true=y_test, y_pred=prediction)
print(all_model)
print("*"*40)

print(sns.countplot(x=prediction, data=df).set_title(" prediction for one person by svm "))
plt.show()


#----------------------------- Decision Tree ----------------------------------------------


x=df.iloc[:,1:]
y=df['Diabetes_binary']

#data scaling 
scaling=StandardScaler(copy=True,with_mean=True,with_std=True)
scaling.fit(X=x)
standardizd=scaling.transform(X=x)
print (" -----------data scaling for features decision tree -------")
print(standardizd)
print(standardizd.shape)


#after scaling featura and target
X_decision=standardizd
y_decision=y

#split data for decision tree
print("*"*40)
print("------------train test split && shapes decision tree  -----------------------/n")
x_train,x_test,y_train,y_test=train_test_split(X_decision,y_decision,test_size=0.2,random_state=0)
print("---------shapes for feature and target after splits and scaling----------\n")
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
print("*"*40)

model_tree=DecisionTreeRegressor(max_depth=6,random_state=0)
model_tree.fit(X=x_train,y=y_train)


# importance feature
importance =model_tree.feature_importances_
print(importance)
def plot_feature_importance(model3):
    plt.figure(figsize=(8,6))
    n_feature=21
    plt.barh(range(n_feature),model3.feature_importances_,align='center')
    plt.yticks(np.arange(n_feature),x)
    plt.xlabel("feature importance")
    plt.xlabel("feature")
    plt.ylim(-1,n_feature)
plot_feature_importance(model_tree)
plt.savefig('feature')
plt.show()


feature=SelectKBest(score_func=chi2)
feature.fit(x,y)
x=feature.transform(x)
print(x.shape)
print(feature.get_support())
x_train,x_test,y_train,y_test=train_test_split(x,y_decision,test_size=0.2,random_state=0)



#Decision Tree classfier
clf = DecisionTreeClassifier( random_state=0)
clf.fit(x_train, y_train)
print("\n train score is :{:.3f} ".format(clf.score(X=x_train,y=y_train)))
print("test score is :{:.3f} ".format(clf.score(X=x_test,y=y_test)))


y_pred = clf.predict(x_test)
print("--------classfication rebort for all data  decision tree---------")

print(classification_report(y_test, y_pred))
cm=confusion_matrix(y_test,y_pred)

print("confusion_matrix :",cm)

# drow confusion matrix svc
print("*"*40)
print(sns.heatmap(cm, center=True).set_title(" drow confusion matrix decision tree "))
plt.show()
print("*"*40)
#