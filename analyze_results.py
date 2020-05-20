import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from itertools import product
import pandas as pd
from pandas.plotting import scatter_matrix


with open('saved/classifier_results.pkl', 'rb') as f:
    classifier_results = pkl.load(f)
losses = classifier_results['loss']
accuracies = classifier_results['accuracy']
with open('saved/mutual_informations.pkl', 'rb') as f:
    mis = pkl.load(f)
mi_x_e = mis['I(embeddings, embeddings + noise)']
mi_p_l = mis['I(predictions, labels)']
mi_e_l = mis['I(embeddings, labels)']

dicts = [accuracies, losses, mi_x_e, mi_p_l, mi_e_l]
dict_names = ['acc', 'loss', 'MI(X, E)', 'MI(P, L)', 'MI(E, L)']
# for dic in [losses, accuracies, mi_x_e, mi_p_l, mi_e_l]:
    # pprint(dic)


observations = []
# model_names = ['ResNet50']
model_names = ['ResNet50', 'ResNet50V2', 'ResNet101', 'ResNet101V2', 'InceptionV3', 'Xception']
stops = [1]
# stops = [1, 30, 200]
for stop, model_name in product(stops, model_names):
    id = f'{model_name}_{stop}'
    observations.append(np.array([dic[id] for dic in dicts]))

observations = np.array(observations)

# pprint(mis)

# pprint(np.corrcoef(observations))
dfs = {}

for stop in stops:
    pandas_dict = {name: [dic[f'{model_name}_{stop}'] for model_name in model_names]
                   for dic, name in zip(dicts, dict_names)}
    df = pd.DataFrame(pandas_dict)
    dfs[stop] = df
    print(df.corr())
    scatter_matrix(df, figsize=(7, 7))
    plt.show()

