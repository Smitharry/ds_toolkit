from sklearn.model_selection import train_test_split
import numpy as np


class HierarchicalClassification:
    def __init__(self, levels, models):
        self.levels = levels
        self.models = models
        self.n_classes = [len(np.unique(level)) for level in self.levels.transpose()]
        self.uniques = self.find_class_labels(self.levels)
        self.predicted_classes = None
        self.path = np.unique(self.levels, axis=0)

    def find_class_labels(self, levels):
        uniques = []
        for level in levels.transpose():
            uniques.append(set(np.unique(level)))
        return uniques

    def padd_level(self, level, level_index):
        # level.shape = (number of observations, number of predicted classes)
        missed_classes = self.uniques[level_index] - self.predicted_classes[level_index]
        for class_ in missed_classes:
            level = self.pad_class(level, class_)
        return level

    def pad_class(self, level, class_):
        # level.shape = (number of observations, number of predicted classes)
        # padd absent class either to end of array or into array.

        if class_ < level.shape[1]:
            level = np.insert(level, class_, 0, axis=1)
        else:
            zeros = np.zeros(level.shape[0]).reshape(1, level.shape[0])
            level = np.concatenate((level, zeros), axis=1)
        return level

    def add_missed_classes(self, y_pred):
        # y_pred.shape = (number of hierarchy levels,
        # number of observations, number of predicted classes)
        for index, level in enumerate(y_pred):
            if level.shape[1] != self.n_classes[index]:
                y_pred[index] = self.padd_level(level, index)
        return y_pred

    def create_mask(self, levels):
        masks = []
        for index, level in enumerate(levels.transpose()):
            mask = np.eye(self.n_classes[index])[level]
            masks.append(mask)
        return masks

    def calculate_likelihood(self, y_true, y_pred):
        level_true_probs = []
        masks = self.create_mask(y_true)
        for level, mask in zip(y_pred, masks):
            level_true_prob = np.multiply(level, mask)
            level_true_prob = level_true_prob.sum(axis=1)
            level_true_probs.append(level_true_prob)
        likelihood = np.array(level_true_probs).sum()
        return likelihood

    def find_predicted_classes(self, y_train):
        self.predicted_classes = self.find_class_labels(y_train)

    def fit(self, X):
        x_train, x_val, y_train, y_val = train_test_split(X, self.levels, test_size=0.2)
        self.find_predicted_classes(y_train)
        self.val = (x_val, y_val)
        self.train = (x_train, y_val)
        for model, y_true in zip(self.models, y_train.transpose()):
            model.fit(x_train, y_true)
            x_train = np.concatenate((x_train, model.predict_proba(x_train)), axis=1)

    def predict(self, x):
        y_pred = []
        for model in self.models:
            y_pred.append(model.predict_proba(x))
            x = np.concatenate((x, model.predict_proba(x)), axis=1)
        return y_pred

    def predict_proba(self, x):
        y_pred = []
        for model in self.models:
            y_pred.append(model.predict_proba(x))
            x = np.concatenate((x, model.predict_proba(x)), axis=1)
        y_pred = self.add_missed_classes(y_pred)
        probs = self.calculate_prob(y_pred, len(x))
        return probs

    def calculate_prob(self, y_pred, length):
        probs_list = {}
        for path in self.path:
            y_true = np.array([np.ones(length, dtype=np.int32)*path[i] for i in range(len(path))]).transpose()
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
        prob_array = []
        for i in range(self.n_classes[-1]):
            prob_array.append(probs_list[i])
        prob_array = np.array(prob_array)
        return prob_array
