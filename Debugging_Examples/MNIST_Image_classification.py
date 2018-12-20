import numpy as np
import tensorflow
from sklearn.model_selection import KFold
import scipy.io as sio
import keras
import photonai
from photonai.optimization.Hyperparameters import IntegerRange
from keras.datasets import mnist

#MNIST Dataset with Preprocessing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#train/test filters for binary classification dataset
train_filter = np.where((y_train == 0 ) | (y_train == 1))
test_filter = np.where((y_test == 0) | (y_test == 1))

#train/test set for binary classification
x_train_binary, y_train_binary = x_train[train_filter], y_train[train_filter]
x_test_binary, y_test_binary = x_test[test_filter], y_test[test_filter]


#register your custom element (here CNN for MNIST is used)
photonai.PhotonRegister.save(photon_name='SimpleCNN',photon_package='PhotonCore',
                             class_str='photonai.modelwrapper.SimpleCNN.SimpleCNN', element_type="Estimator")

my_pipe = photonai.Hyperpipe('BiClusteringPipe',
                             optimizer='grid_search',
                             metrics=['accuracy'],
                             best_config_metric='accuracy',
                             inner_cv=KFold(n_splits=5, shuffle=True, random_state=42),
                             outer_cv=KFold(n_splits=5),
                             eval_final_performance=True)

my_pipe += photonai.PipelineElement('SimpleCNN', hyperparameters= {'final_activation' : 'sigmoid', 'loss' : 'binary_crossentropy'})

#choose the right dataset to evaluate your pipe, both binary and multiclass classification is possible
my_pipe.fit(x_train_binary, y_train_binary)
