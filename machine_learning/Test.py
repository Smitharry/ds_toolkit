from sklearn.ensemble import RandomForestClassifier
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
hier = hierarchical_class.HierarchicalClassification(np.array(dataset[['level_0', 'level_1', 'level_2', 'level_3', 'level_4']]),
                                 [forest_0, forest_1, forest_2, forest_3, forest_4])
a = hier.fit_model(np.array(bow_df.drop(columns='ClassCode', axis=1)))
