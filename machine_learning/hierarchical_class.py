
import numpy as np


class HierarchicalClassification:
    def __init__(self, models, params=dict()):
        self.models = models
        if params:
            self.set_params(params)

    def set_params(self, params):
        for model_index in params:
            param = params[model_index]
            self.models[model_index].set_params(**param)

    def get_params(self, deep=False):
        params = {}
        for index, model in enumerate(self.models):
            params[index] = model.get_params()
        return params

    def collect_info(self, levels):
        self.levels = levels
        self.n_classes = [len(np.unique(level)) for level in self.levels.transpose()]
        self.uniques = self.find_class_labels(self.levels)
        self.path = np.unique(self.levels, axis=0)

    def find_class_labels(self, levels):
        uniques = []
        for level in levels.transpose():
            uniques.append(set(np.unique(level)))
        return uniques

    def create_mask(self, levels):
        masks = []
        for index, level in enumerate(levels.transpose()):
            slice = np.logical_not(np.isin(level, list(self.uniques[index])))
            level[slice] = 0
            mask = np.eye(self.n_classes[index])[level]
            mask[slice] = mask[slice] * 0
            masks.append(mask)
        return masks

    def calculate_likelihood(self, x, y_true):
        probs = self.predict_proba(x)
        likelihood = np.zeros(len(x))
        for index in range(len(x)):
            true_label = y_true[index][-1]
            if true_label not in probs:
                likelihood[index] = 0
            else:
                likelihood[index] = probs[true_label][index]
        likelihood = np.sum(likelihood)
        return likelihood

    def find_predicted_classes(self, y_train):
        self.predicted_classes = self.find_class_labels(y_train)

    def fit(self, X, y_train):
        self.collect_info(y_train)
        for model, y_true in zip(self.models, y_train.transpose()):
            model.fit(X, y_true)
            X = np.concatenate((X, model.predict_proba(X)), axis=1)

    def predict(self, x):
        y_pred = []
        for model in self.models:
            y_pred.append(model.predict_proba(x))
            x = np.concatenate((x, model.predict_proba(x)), axis=1)
        return y_pred

    def predict_proba(self, x):
        y_pred = self.predict(x)
        probs = self.calculate_prob(y_pred, len(x))
        return probs

    def calculate_prob(self, y_pred, length):
        probs_list = {}
        for path in self.path:
            y_true = np.array([np.ones(length, dtype=np.int32) * path[i] for i in range(len(path))]).transpose()
            masks = self.create_mask(y_true)
            level_true_probs = []
            for level, mask in zip(y_pred, masks):
                level_true_prob = np.multiply(level, mask)
                level_true_prob = level_true_prob.sum(axis=1)
                level_true_probs.append(level_true_prob)
            probs = np.ones(length)
            for level in level_true_probs:
                probs = probs * np.array(level)
            probs_list[path[-1]] = probs
        return probs_list