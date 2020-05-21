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
dict_names = ['accuracy', 'loss', 'I(X; Ê)', 'I(P; Y)', 'I(E; Y)']
# for dic in [losses, accuracies, mi_x_e, mi_p_l, mi_e_l]:
    # pprint(dic)


# model_names = ['ResNet50']
model_names = ['ResNet50', 'ResNet50V2', 'ResNet101', 'ResNet101V2', 'InceptionV3', 'Xception']
# stops = [1]
stops = [1, 30, 200]


for dic in [mi_x_e, mi_e_l]:
    for name in model_names:
        avg = sum([dic[f'{name}_{stop}'] for stop in stops]) / 3
        for stop in stops:
            dic[f'{name}_{stop}'] = avg


#
# for stop, model_name in product(stops, model_names):
#     observations = []
#     id = f'{model_name}_{stop}'
#     observations.append(np.array([dic[id] for dic in dicts]))
#
#     observations = np.array(observations)
#
#     # pprint(mis)
#
#     pprint(np.corrcoef(observations))
dfs = {}
#
for stop in stops:
    pandas_dict = {name: [dic[f'{model_name}_{stop}'] for model_name in model_names]
                   for dic, name in zip(dicts, dict_names)}
    df = pd.DataFrame(pandas_dict)
    dfs[stop] = df
    print(df.corr())
#     scatter_matrix(df, figsize=(7, 7))
#     plt.show()
#

#
# fig, axes = plt.subplots(2, 3, sharex=False)
# axes = axes.reshape((6,))
# ticks = [1, 2, 3]
# labels = ['1', '30', '200']
# plt.xticks(ticks, labels)
# for i, (dic, title) in enumerate(zip(dicts, dict_names)):
#     axes[i].set_title(title)
#     for name in model_names:
#         y = [dic[f'{name}_{stop}'] for stop in stops]
#         axes[i].plot(ticks, y, label=name)
# handles, labels = axes[0].get_legend_handles_labels()
# axes[-1].axis('off')
# fig.legend(handles, labels, loc=(.74, .14))
# plt.savefig('plots/results/5.png')
# plt.show()