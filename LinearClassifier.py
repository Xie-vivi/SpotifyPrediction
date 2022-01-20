from sklearn.linear_model import LinearRegression
from Classifiers import Classifiers

class LinearClassifier(Classifiers):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def prediction(self, X_test):
        reg = LinearRegression().fit(self.X_data, self.y_data)
        print('-----')
        print('prediction with Linear classifier')
        print('-----')
        y_predict = reg.predict(X_test)
        return y_predict
