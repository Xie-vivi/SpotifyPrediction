from sklearn.metrics import mean_squared_error, r2_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import  RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

class Classifiers:

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def stacking(self, X_test):
        estimators = [('rf', RandomForestClassifier(random_state=0)) , ('sgd', make_pipeline(StandardScaler(), SGDClassifier(penalty='l1')))]
        clf = StackingClassifier(estimators = estimators, final_estimator = RandomForestClassifier(random_state=0)).fit(self.X_data, self.y_data)
        print('-----')
        print('prediction with Stacking')
        print('-----')
        y_predict = clf.predict(X_test)
        return y_predict

    def Bagging(self, X_test):
        clf = BaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators = 15).fit(self.X_data, self.y_data)
        print('-----')
        print('prediction with Bagging')
        print('-----')
        y_predict = clf.predict(X_test)
        return y_predict

    def ScoreMetrics(self, y_pred, y_test):
        MSE = mean_squared_error(y_test, y_pred)

        R2_score = r2_score(y_test, y_pred)

        print('MSE = %f' % MSE)
        print()
        print('R2 = %f' % R2_score)
        print()
        try:
            score_f1 = f1_score(y_test, y_pred, average='weighted')
            print('f1 = %f' % score_f1)
            print()
        except Exception:
            print('continuous values - Cannot process f1 score')

    def mlpPrediction(self, X_test):
        clf = MLPClassifier().fit(self.X_data, self.y_data)
        print('-----')
        print('prediction with neural network')
        print('-----')
        y_predict = clf.predict(X_test)
        return y_predict

    def sgdPrediction(self, X_test):
        clf = make_pipeline(StandardScaler(), SGDClassifier(penalty='l1'))
        clf.fit(self.X_data, self.y_data)
        print('-----')
        print('prediction with SGD')
        print('-----')
        y_predict = clf.predict(X_test)
        return y_predict

    def random_forest_prediction(self, X_test, nb_est=100):
        print('-----')
        print('Random forest classifier')
        print('-----')
        print('nb estimators = %d' % nb_est)
        clf = RandomForestClassifier(n_estimators=nb_est, random_state=0)
        clf = clf.fit(self.X_data, self.y_data)
        y_pred = clf.predict(X_test)
        return y_pred

    def gradientBoostPrediction(self, X_test, n_est=100, lr=0.1):
        clf = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr)
        clf.fit(self.X_data, self.y_data)
        print('-----')
        print('prediction with gradient Boost Classifier')
        print('-----')
        y_predict = clf.predict(X_test)
        return y_predict