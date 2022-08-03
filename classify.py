import numpy
import scipy
import pandas
from sklearn.model_selection import train_test_split    ##split dataset
from sklearn.model_selection import cross_val_score     ##to do cross validation for best algorithm
from sklearn.model_selection import StratifiedKFold
###algorithms to predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
##to measure accuracy and performance of a model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

da = pandas.read_csv("IRIS.csv") ###creating dataframe
##creating test and train sets(validation)
arr =da.values
x= arr[:,0:4]
y = arr[:,4]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.15,random_state=1)

"""model = [] ###creating list of models to find best option
model.append(("LR",LogisticRegression(solver ="liblinear",multi_class ="ovr" )))
model.append(("DTC",DecisionTreeClassifier()))
model.append(("KNC",KNeighborsClassifier()))
model.append(("LDA",LinearDiscriminantAnalysis()))
model.append(("snm",SVC(gamma="auto")))
model.append(("NB",GaussianNB()))"""

"""names=[]
result = []
for name,models in model:
  k = StratifiedKFold(n_splits=10,random_state=1,shuffle = True)
  model_result = cross_val_score(model,x,y,cv =k,scoring = "accuracy")
  result.append(model_result)
  names.append(name)
  print("%s %f %f"%(name,result.mean(),result.std()))"""

model = SVC(gamma = "auto")
model.fit(x_train,y_train)
y1 = model.predict(x_test) ##training and predicting
##testing and evluating
print(classification_report(y_test,y1))
print(accuracy_score(y_test,y1))
print(confusion_matrix(y_test,y1))

print(model.predict(input())
