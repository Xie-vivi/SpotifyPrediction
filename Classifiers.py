from sklearn.metrics import mean_squared_error, r2_score, f1_score

class Classifiers:

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