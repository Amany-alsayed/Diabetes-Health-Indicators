import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
# read dataframe
#start decision tree
df=pd.read_csv('csvfile/diabetes_binary_health_indicators_BRFSS2015.csv')

#print first five row 
def five_row():
    print(df.head())
    print(df.shape)

#check null operation for dataframe
def check_null():
    print(df.isnull().sum())

#get the summary statisics
def descript_data():
    print(df.describe().round())
    print(df.info())

# diabetes binary
#0 -> non_diabetic
#1 -> diabetic
print("*"*40)
print(df.iloc[:,[0]].value_counts())

# group by target
df.groupby(['Diabetes_binary']).mean()
print("*"*40)

# the best feature and traget
x=df.iloc[:,1:]
y=df['Diabetes_binary']


def best_feature():
    print("---------shapes for feature and target----------- ")
    print(x.shape)
    print(y.shape)
    print("*"*40)


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

#split data
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

print(classification_report(y_test, y_pred))
cm=confusion_matrix(y_test,y_pred)

print("confusion_matrix :",cm)