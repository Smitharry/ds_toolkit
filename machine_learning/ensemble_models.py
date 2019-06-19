from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier


class Pac_RF():

    def __init__(self):
        self.pac = PassiveAggressiveClassifier(tol=0.001)
        self.forest = RandomForestClassifier()
        self.classes_ = []

    def set_params(self, params):
        for param in params:
            model, param_ = param.split('.')
            if model == 'pac':
                self.pac.set_params(**{param_: params[param]})
            else:
                self.forest.set_params(**{param_: params[param]})

    def fit(self, X, y):
        self.pac.fit(X, y)
        des_matrix = self.pac.decision_function(X)
        self.forest.fit(des_matrix, y)
        self.find_class_order()

    def predict_proba(self, X):
        des_matrix = self.pac.decision_function(X)
        probs = self.forest.predict_proba(des_matrix)
        return probs

    def find_class_order(self):
        self.classes_ = self.forest.classes_
