import numpy as np
import tensorflow
from sklearn.model_selection import KFold
import scipy.io as sio
import keras
import photonai
from photonai.optimization.Hyperparameters import IntegerRange
from keras.datasets import mnist


#MNIST Dataset with Preprocessing
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#train_filter = np.where((y_train == 0 ) | (y_train == 1))
#test_filter = np.where((y_test == 0) | (y_test == 1))
#x_train, y_train = x_train[train_filter], y_train[train_filter]
#x_test, y_test = x_test[test_filter], y_test[test_filter]


mat_data = sio.loadmat('data_control.mat')
ControlData = mat_data['data_control']
print(ControlData.shape)
ControlData = ControlData[:,:,:,0]
ControlLabels = np.zeros(ControlData.shape[0])

#ControlTopData = ControlData[1:10,:,:]
#ControlTopLabels = np.zeros(ControlTopData.shape[0])
#RandomCtrlData = np.random.rand(347, 114, 114)
#RandomCtrlLabels = np.zeros(RandomCtrlData.shape[0])

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

FC_matrices_simplified = np.reshape(FC_matrices_simplified, (FC_matrices_simplified.shape[0], -1))



photonai.PhotonRegister.save(photon_name='PhotonVectorizer',photon_package='PhotonCore',
                             class_str='photonai.modelwrapper.PhotonVectorizer.PhotonVectorizer', element_type="Transformer")

my_pipe = photonai.Hyperpipe('SVMReferencePipe',
                             optimizer='sk_opt',
                             metrics=['accuracy'],
                             best_config_metric='accuracy',
                             inner_cv=KFold(n_splits=3, shuffle=True, random_state=42),
                             outer_cv=KFold(n_splits=5),
                             eval_final_performance=True)

#my_pipe+= photonai.PipelineElement('PhotonVectorizer')

my_pipe += photonai.PipelineElement('SVC', hyperparameters={'kernel': ['linear']})

my_pipe.fit(FC_matrices_simplified, FC_labels_simplified)
