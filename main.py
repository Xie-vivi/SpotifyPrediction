import pandas as pd
import sys
sys. path. append(".")
from data_preprocess import DataPreprocess
from sklearn.model_selection import train_test_split
from Classifiers import Classifiers
from sklearn.preprocessing import StandardScaler

train_ds = pd.read_csv('spotify_dataset_train.csv')


dp = DataPreprocess(train_ds)
dp.statisticalProperties()

# numpy array of data from the dataset
(X_data, y_data) = dp.preprocessDs(True)



"""

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

"""

#Classification

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=0, test_size=0.25)

standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train)
X_test = standardScaler.transform(X_test)

Clf = Classifiers(X_train, y_train)

# => 150 0.3
def bestParamettersGradientBoostPrediction():
    for n_est in [100, 150, 200, 250, 300, 350]:
        for lr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            print(str(n_est) + ' ' + str(lr))
            gc_pred = Clf.gradientBoostPrediction(X_test, n_est, lr)

            Clf.ScoreMetrics(gc_pred, y_test)

def bestParamettersSgdPred():
    sgd_pred = Clf.sgdPrediction(X_test)

    Clf.ScoreMetrics(sgd_pred, y_test)


rf = Clf.random_forest_prediction(X_test)
Clf.ScoreMetrics(rf, y_test)

bgc = Clf.Bagging(X_test)
Clf.ScoreMetrics(bgc, y_test)


sc = Clf.stacking(X_test)
Clf.ScoreMetrics(sc, y_test)

