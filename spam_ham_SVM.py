import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import joblib
#Loading the Dataset
dataframe = pd.read_csv("G:\Python\Project_ML_Sem6\spam.csv")
print(dataframe.describe())

#Spliting into Training and Test Data 80:20

x = dataframe["EmailText"]
y = dataframe["Label"]

x_train,y_train = x[0:4457],y[0:4457]
x_test,y_test = x[4457:],y[4457:]

#Extracting Features using CountVectorizer to extract the frequency of a word occuring in a specific label 
cv = CountVectorizer()  
features = cv.fit_transform(x_train)

#Building the model using SVM and SVC as it is a binary classifier

#tuned_parameters = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}

#using GridSearchCV to find the best parameter using trial and error.
#model = GridSearchCV(svm.SVC(), tuned_parameters)
model = svm.SVC(C = 100, gamma = 0.001, kernel = 'rbf')

model.fit(features,y_train)

#print(model.best_params_)

#Testing the Accuracy
print(model.score(cv.transform(x_test),y_test))

#joblib.dump(model,"G:\Python\Project_ML_Sem6\SVM_Model.pkl")