import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from skopt import Optimizer
from skopt.optimizer import dummy_minimize
from skopt import dummy_minimize
import scipy.io as sio
import keras
from photonai import Hyperpipe
from photonai import PipelineElement
from photonai import PhotonRegister
from photonai.validation import ResultsTreeHandler
from scipy.stats import itemfreq
from photonai.investigator.Investigator import Investigator
import matplotlib.pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"


mat_data = sio.loadmat('Dosenbach_matrices_1044.mat')
FC_matrices = mat_data['matrices']
FC_reordered = np.swapaxes(FC_matrices, 0, 2)


FC_labels = np.loadtxt('Dosenbach_labels_1044.txt')
FC_labels = FC_labels - 1


filter = np.where((FC_labels ==0) | (FC_labels == 1))
FC_matrices_simplified, FC_labels_simplified = FC_reordered[filter], FC_labels[filter]



#RandomCtrlData1 = np.random.randn(587, FC_matrices_simplified.shape[1], FC_matrices_simplified.shape[2])
#RandomCtrlLabels1 = np.zeros(587)
#RandomCtrlData2 = np.random.randn(410, FC_matrices_simplified.shape[1], FC_matrices_simplified.shape[2])
#RandomCtrlLabels2 = np.ones(410)
#RandomMatrices = np.concatenate((RandomCtrlData1, RandomCtrlData2), axis = 0)
#RandomLabels = np.concatenate((RandomCtrlLabels1, RandomCtrlLabels2))

print(FC_matrices_simplified.shape)
#print(RandomMatrices.shape)
print(FC_labels_simplified.shape)
#print(RandomLabels.shape)




PhotonRegister.save(photon_name='Biclustering2d',
                        class_str='photonai.modelwrapper.Biclustering2d.Biclustering2d', element_type="Transformer")

PhotonRegister.save(photon_name='SimpleCNN',
                        class_str='photonai.modelwrapper.SimpleCNN.SimpleCNN', element_type="Estimator")

my_pipe = Hyperpipe('BiClusteringPipe',
                        optimizer='grid_search',
                        optimizer_params= {'num_iterations': 1},
                        metrics=['accuracy'],
                        best_config_metric='accuracy',
                        inner_cv=KFold(n_splits=5, shuffle=True, random_state=42),
                        outer_cv=KFold(n_splits=5),
                        eval_final_performance=True)

my_pipe += PipelineElement('Biclustering2d')

my_pipe += PipelineElement('SimpleCNN')

my_pipe.fit(FC_matrices_simplified, FC_labels_simplified)



inner_performances = list()
for i, fold in enumerate(my_pipe.result_tree.outer_folds[0].tested_config_list):
    inner_performances.append((fold.config_dict, fold.metrics_test[0].value))
print(inner_performances)

plt.ylim(0.2, 0.8)
plt.xticks(rotation=90)
plt.margins(0.3)

for i, lelles in inner_performances:
    print(i, lelles)
    Benis = ",".join(("{}={}".format(*p) for p in i.items()))
    plt.plot(Benis, lelles, 'ro')


plt.show()

rer = ResultsTreeHandler.get_performance_outer_folds(my_pipe)

print(rer)


#fct(foldobject, hyperparameter_key):
    #results = list()
    #for i, fold in enumerate(my_pipe.result_tree.outerfolds[0].tested_config_list):
        #if fold.config_dict.get(hyperparameter_key) != None:
            #results.append(fold.config_dict)




