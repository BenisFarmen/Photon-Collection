import numpy as np
import tensorflow
from sklearn.model_selection import KFold
import scipy.io as sio
import keras
import photonai
from photonai.optimization.Hyperparameters import IntegerRange

#Example Code to debug image classification models
#Adjust the array sizes to make the random data as similar to your data as possible
#Adjust the group sizes to test for effects under class imbalance

# make "control" data
RandomCtrlData = np.random.rand(500, 114, 114)
RandomCtrlLabels = np.zeros(RandomCtrlData.shape[0])

# make "patient" data
RandomPatData = np.random.rand(500, 114, 114)
RandomPatLabels = np.ones(RandomPatData.shape[0])

# make third group data
RandomThirdData = np.random.rand(500, 114, 114)
RandomThirdlabels = np.ones(RandomPatData.shape[0]) * 2

#Make a binary classification dataset
RandomBinaryData = np.concatenate((RandomCtrlData, RandomPatData), axis=0)
RandomBinaryLabels = np.concatenate((RandomCtrlLabels, RandomPatLabels))

#Make a multi-class classification dataset
RandomMultiData = np.concatenate((RandomCtrlData, RandomPatData, RandomThirdData), axis=0)
RandomMultiLabels = np.concatenate((RandomCtrlData, RandomPatData, RandomThirdData))

photonai.PhotonRegister.save(photon_name='SimpleCNN',photon_package='PhotonCore',
                             class_str='photonai.modelwrapper.SimpleCNN.SimpleCNN', element_type="Estimator")


my_pipe = photonai.Hyperpipe('RandomPipe',
                             optimizer='grid_search',
                             metrics=['accuracy'],
                             best_config_metric='accuracy',
                             inner_cv=KFold(n_splits=5, shuffle=True, random_state=42),
                             outer_cv=KFold(n_splits=5),
                             eval_final_performance=True)

my_pipe += photonai.PipelineElement('SimpleCNN', hyperparameters= {'final_activation' : 'sigmoid', 'loss' : 'binary_crossentropy'})

my_pipe.fit(RandomBinaryData, RandomBinaryLabels)
