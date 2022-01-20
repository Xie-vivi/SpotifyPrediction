import pandas as pd
from data_preprocess import DataPreprocess
from pandas.plotting import scatter_matrix
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
from TreesClassifier import TreesClassifier
from sklearn.model_selection import train_test_split
from LinearClassifier import LinearClassifier
from sklearn.metrics import mean_squared_error, r2_score

ds = pd.read_csv('spotify_dataset_subset.csv')

dp = DataPreprocess(ds)

dp.dsInfos()

preprocessedDs = dp.preprocessSubset()


def correlations():
    corr = preprocessedDs.corr()
    plt.figure()
    sns.heatmap(corr, annot=True)
    plt.show()

y_data = preprocessedDs['popularity']
preprocessedDs.drop(['popularity'], axis=1, inplace=True)

X_data = preprocessedDs.to_numpy()

def ErrorMetrics(y_pred, y_test):
    MSE = mean_squared_error(y_test, y_pred)

    R2_score = r2_score(y_test, y_pred)

    print('MSE = %f' % MSE)
    print()
    print('R2 = %f' % R2_score)
    print()

def plot_tree(clf):
    fn = preprocessedDs.columns
    cn = list(map(str, y_data))
    tree.plot_tree(clf,
                   feature_names=fn,
                   class_names=cn,
                   filled=True);
    plt.show()

def popularityPrediction():
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=42)

    lc = LinearClassifier(X_data, y_data)

    lc_prediction = lc.prediction(X_test)

    ErrorMetrics(lc_prediction, y_test)

    print('prediction with linear classifier : ' + str(lc_prediction))
    print()

    tc = TreesClassifier(X_data, y_data)
    tc_pred = tc.decision_tree_prediction(X_test)

    clf = tc.getClf()
    plot_tree(clf)

    print('prediction with decision tree : ' + str(tc_pred))
    print()

    ErrorMetrics(tc_pred, y_test)

    # Random Forest

    rf_pred = tc.random_forest_prediction(X_test)

    print('prediction with random forest : ' + str(rf_pred))
    print()

    ErrorMetrics(rf_pred, y_test)

popularityPrediction()





