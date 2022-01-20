from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from Classifiers import Classifiers


class TreesClassifier(Classifiers):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def getClf(self):
        return self.clf

    def decision_tree_prediction(self, X_test):
        print('-----')
        print('decision tree classifier')
        print('-----')
        clf = DecisionTreeClassifier()
        clf = clf.fit(self.X_data, self.y_data)
        self.clf = clf
        y_pred = clf.predict(X_test)
        return y_pred

    def random_forest_prediction(self, X_test):
        print('-----')
        print('Random forest classifier')
        print('-----')
        clf = RandomForestClassifier()
        clf = clf.fit(self.X_data, self.y_data)
        self.clf = clf
        y_pred = clf.predict(X_test)
        return y_pred