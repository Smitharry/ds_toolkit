from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import hierarchical_class
import numpy as np

forest_0 = RandomForestClassifier()
forest_1 = RandomForestClassifier()
forest_2 = RandomForestClassifier()
forest_3 = RandomForestClassifier()
forest_4 = RandomForestClassifier()

dataset = pd.read_csv('dataset.csv', encoding='utf-8')
bow_df = pd.read_csv('bow_df.csv', encoding='utf-8')
X_train, X_test, y_train, y_test = train_test_split(bow_df.drop(columns='ClassCode', axis=1),
                                                    np.array(dataset[['level_0', 'level_1', 'level_2', 'level_3', 'level_4']]),
                                                    test_size=0.2, shuffle=True)
params = [{0 : {'max_depth' : 10}, 1  : {'max_depth' : 10}, 2 : {'max_depth' : 10},\
           3 : {'max_depth' : 10}, 4 : {'max_depth' : 10}}, {0 : {'max_depth' : 10},\
            1  : {'max_depth' : 20}, 2 : {'max_depth' : 30}, 3 : {'max_depth' : 40}, 4 : {'max_depth' : 50}}]

scores = {}
for index, param in enumerate(params):
    forest_0 = RandomForestClassifier()
    forest_1 = RandomForestClassifier()
    forest_2 = RandomForestClassifier()
    forest_3 = RandomForestClassifier()
    forest_4 = RandomForestClassifier()
    hier = hierarchical_class.HierarchicalClassification([forest_0, forest_1, forest_2, forest_3, forest_4])
    hier.set_params(param)
    scores[index] = []
    for i in range(5):
        x_train, x_test, y_train_, y_test_ = train_test_split(X_train, y_train,
                                                    test_size=0.2, shuffle=True)
        hier.fit(x_train, y_train_)
        likelihood = hier.calculate_likelihood(x_test, y_test_)
        scores[index].append(likelihood)

