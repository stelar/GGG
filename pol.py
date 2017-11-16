import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Hinge
from sklearn.linear_model import Huber
import numpy
file='train.csv'
data=pd.read_csv(file)
number = LabelEncoder()
data['type']=number.fit_transform(data['type'].astype('str'))
data['color']=number.fit_transform(data['color'].astype('str'))
print(data.describe())
data=data.replace("",numpy.NaN)
print(data.isnull().sum())
data.dropna(inplace=True)
y = data['type']
del data['type']

print data.head(20)
from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(data.values, y.values,
                                                                    test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=51,)
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
svm = SVC()

classifiers = [rf, gnb, mnb, bnb, svm]
classifier_names = ["Random Forest", "Gaussian NB", "Multinomial NB", "Bernoulli NB", "SVC(rbf)"]
feature_weights = []

for classifier, classifier_name in zip(classifiers, classifier_names):

    classifier.fit(train_data, train_labels)
    predicted_labels = classifier.predict(test_data)
    if classifier_name == "Random Forest":
        feature_weights = classifier.feature_importances_
    print ("--------------------------------------\n")
    print "Accuracy for Classifier ", classifier_name, " : ", metrics.accuracy_score(test_labels, predicted_labels)
    print "Confusion Matrix for ", classifier_name, " :\n ", metrics.confusion_matrix(test_labels, predicted_labels)
    print "Classification Report for ", classifier_name, " :\n ", metrics.classification_report(test_labels,
                                                                                                predicted_labels)
    print "----------------------------------------\n"

print "Weights of the selected features : \n", feature_weights
