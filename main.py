import pandas as pd
import sys
sys. path. append(".")
from LinearClassifier import LinearClassifier
from TreesClassifier import TreesClassifier
from data_preprocess import DataPreprocess
from sklearn.model_selection import train_test_split
from Csv import Csv
import matplotlib.pyplot as plt

train_ds = pd.read_csv('spotify_dataset_train.csv')


dp = DataPreprocess(train_ds)
dp.statisticalProperties()

# numpy array of data from the dataset
(X_data, y_data) = dp.preprocessDs(True)

# train the data set, predict Genre and show metrics to analyze prediction
def train():
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=42)

    lc = LinearClassifier(X_data, y_data)

    lc_prediction = lc.prediction(X_test)

    lc.ScoreMetrics(list(lc_prediction), list(y_test))

    print('prediction with linear classifier : ' + str(lc_prediction))
    print()

    tc = TreesClassifier(X_data, y_data)
    tc_pred = tc.decision_tree_prediction(X_test)

    tc.ScoreMetrics(list(tc_pred), list(y_test))

    #clf score
    clf = tc.getClf()
    score = clf.score(X_test, y_test)
    print('clf score = %f' %score)

    # Random Forest

    rf_pred = tc.random_forest_prediction(X_test)

    tc.ScoreMetrics(list(rf_pred), list(y_test))

# Challenge ! Predict the genre of the dataset 'spotify_dataset_test'

print('-----')
print('Challenge !')
print('-----')

test_ds = pd.read_csv('spotify_dataset_test.csv')

# preprocess datas
dp = DataPreprocess(train_ds)
X_test = dp.preprocessDs(False)

# prediction with tree classifier
tc = TreesClassifier(X_data, y_data)

# get labels from RandomForest
rf_pred = tc.random_forest_prediction(X_test)
dp.printLabes(rf_pred)

# get labels from DecisionTree
dt_pred = tc.decision_tree_prediction(X_test)
genresLabels = dp.printLabes(dt_pred)

# write genres into a csv file
CSV = Csv(list(genresLabels))
fileName = 'genres.csv'
CSV.clearCsv(fileName)
CSV.writeToCsv(fileName)

# see if both agrees
print('-----')
print((dt_pred != rf_pred).sum())
print('-----')

train()




