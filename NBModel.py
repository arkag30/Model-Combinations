from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import joblib
dataframe = pd.read_csv("G:\Python\Project_ML_Sem6\spam.csv")
x = dataframe["EmailText"]
y = dataframe["Label"]

x_train,y_train = x[0:4457],y[0:4457]
x_test,y_test = x[4457:],y[4457:]

clf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
joblib.dump(clf,"G:\Python\Project_ML_Sem6\spam_model.pkl")