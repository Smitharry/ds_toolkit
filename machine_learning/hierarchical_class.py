import numpy as np


class HierarchicalClassification:
    def __init__(self, models, params=dict()):
        self.models = models
        self.class_order = {}
        self.levels = None
        self.n_classes = None
        self.uniques = None
        self.path = None
        if params:
            self.set_params(params)

    def set_params(self, params):
        for param in params:
            model, param_ = param.split('/')
            model = int(model)
            self.models[model].set_params({param_: params[param]})

    def get_params(self):
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
            unseen_instances = np.isin(level, list(self.uniques[index]), invert=True)
            level[unseen_instances] = 0
            mask = np.eye(self.n_classes[index])[level]
            mask[unseen_instances] = mask[unseen_instances] * 0
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

    def find_class_order(self):
        for index, model in enumerate(self.models):
            keys = model.classes_
            items = [i for i in range(len(keys))]
            local_class_order = dict(zip(keys, items))
            self.class_order[index] = local_class_order

    def fit(self, X, y_train):
        self.collect_info(y_train)
        for model, y_true in zip(self.models, y_train.transpose()):
            model.fit(X, y_true)
            X = np.concatenate((X, model.predict_proba(X)), axis=1)
        self.find_class_order()

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

    def find_actual_path(self, path):
        actual_path = [self.class_order[index][class_] for index, class_ in enumerate(path)]
        return actual_path

    def calculate_prob(self, y_pred, length):
        all_probs = {}
        for path in self.path:
            actual_path = self.find_actual_path(path)
            masks = self.collect_pseudo_true(actual_path, length)
            path_probs = self.calculate_class_prob(masks, y_pred, length)
            all_probs[path[-1]] = path_probs
        return all_probs

    def collect_pseudo_true(self, actual_path, length):
        y_true = np.array([np.ones(length, dtype=np.int32) * class_ for class_ in actual_path]).transpose()
        masks = self.create_mask(y_true)
        return masks

    def calculate_class_prob(self, masks, y_pred, length):
        class_probs = []
        for level, mask in zip(y_pred, masks):
            true_prob = self.calculate_level_prob(level, mask)
            class_probs.append(true_prob)
        return self.multiply_along_path(class_probs, length)

    def multiply_along_path(self, class_probs, length):
        probs = np.ones(length)
        for level in class_probs:
            probs = probs * np.array(level)
        return probs

    def calculate_level_prob(self, level, mask):
        true_prob = np.multiply(level, mask)
        true_prob = true_prob.sum(axis=1)
        return true_prob
