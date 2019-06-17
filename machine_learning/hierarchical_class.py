from sklearn.model_selection import train_test_split
import numpy as np


class HierarchicalClassification:
    def __init__(self, levels, models):
        self.levels = levels
        self.models = models
        self.n_classes = [len(np.unique(level)) for level in self.levels.transpose()]
        self.uniques = self.find_class_labels(self.levels)

    def find_class_labels(self, levels):
        uniques = []
        for level in levels.transpose():
            uniques.append(set(np.unique(level)))
        return uniques

    def add_missed_classes(self, y_pred, y_true):
        uniques = self.find_class_labels(y_true)
        for index, level in enumerate(y_pred):
            level_copy = level
            if len(level[0]) != self.n_classes[index]:
                missed_classes = self.uniques[index] - uniques[index]
                for class_ in missed_classes:
                    if class_ < len(level_copy[0]):
                        level_copy = np.insert(level_copy, class_, 0, axis=1)
                    else:
                        zeros = np.zeros(len(level_copy)).reshape(1, len(level_copy))
                        level_copy = np.concatenate((level_copy, zeros), axis=1)
            y_pred[index] = level_copy
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
            print(mask.shape, level.shape)
            level_true_prob = np.multiply(level, mask)
            level_true_prob = level_true_prob.sum(axis=1)
            level_true_probs.append(level_true_prob)
        likelihood = np.array(level_true_probs).sum()
        return likelihood

    def fit_model(self, X):
        x_train, x_val, y_train, y_val = train_test_split(X, self.levels, test_size=0.2)
        y_pred_train = []
        y_pred_val = []
        for model, y_true, y_true_val in zip(self.models, y_train.transpose(), y_val.transpose()):
            model.fit(x_train, y_true)
            y_pred_train.append(model.predict_proba(x_train))
            y_pred_val.append(model.predict_proba(x_val))
            x_train = np.concatenate((x_train, model.predict_proba(x_train)), axis=1)
            x_val = np.concatenate((x_val, model.predict_proba(x_val)), axis=1)
        y_pred_train = self.add_missed_classes(y_pred_train, y_train)
        y_pred_val = self.add_missed_classes(y_pred_val, y_val)
        return y_pred_val