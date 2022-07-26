import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import accuracy_score

#Loading the Dataframe
dataframe = pd.read_csv("G:\Python\Project_ML_Sem6\spam.csv",sep=',',engine='python')
x = dataframe["EmailText"]
y = dataframe["Label"]
print(dataframe.describe())

#Performing text vectorization to go from strings to lists of numbers
x_train,y_train = x[0:4457],y[0:4457]
x_test,y_test = x[4457:],y[4457:]
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
x_train_transformed = vectorizer.fit_transform(x_train)
x_test_transformed  = vectorizer.transform(x_test)

# Defining the classifier

#using a big depth, as the data set is small.
clf = RandomForestClassifier(max_depth=100, random_state=0)
#Fitting the training data.
clf.fit(x_train_transformed, y_train)

#Testing the data.
pred = clf.predict(x_test_transformed)

#Accuracy Score.
print("Accuracy of Decision Trees Algorithm: ", accuracy_score(pred,y_test))