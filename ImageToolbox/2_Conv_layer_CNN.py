import numpy as np
import tensorboard
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dropout, Dense
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ShuffleSplit
from keras.utils import to_categorical
from photonai.photonlogger.Logger import Logger
from photonai.helpers.TFUtilities import binary_to_one_hot
from photonai.modelwrapper.KerasBaseEstimator import KerasBaseEstimator

class SimpleCNN(BaseEstimator, ClassifierMixin, KerasBaseEstimator):

    def __init__(self, hidden_layer_sizes=[128, 128],
                 first_layer_dim = 8, first_layer_kernel = 2,
                 first_layer_activation='relu', first_layer_batchnorm = 1,
                 first_layer_pooling = 0, first_layer_poolsize = 2,
                 second_layer_dim = 12, second_layer_kernel = 2,
                 second_layer_activation = 'relu', second_layer_batchnorm = 1,
                 second_layer_pooling = 1, second_layer_poolsize = 2,
                 dropout_rate=0.7, target_dimension=2, act_func='relu',
                 learning_rate=0.001, batch_normalization=True, nb_epoch=5, early_stopping_flag=True,
                 eaSt_patience=20, reLe_factor = 0.4, reLe_patience=5, batch_size=32, verbosity=2,
                 loss = 'binary_crossentropy', final_activation = 'sigmoid'):
        super(KerasBaseEstimator, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes

        #first layer hyperparameters
        self.first_layer_dim = first_layer_dim
        self.first_layer_kernel = first_layer_kernel
        self.first_layer_activation = first_layer_activation
        self.first_layer_batchnorm = first_layer_batchnorm
        self.first_layer_pooling = first_layer_pooling
        self.first_layer_poolsize = first_layer_poolsize

        #second layer hyperparameters
        self.second_layer_dim = second_layer_dim
        self.second_layer_kernel = second_layer_kernel
        self.second_layer_activation = second_layer_activation
        self.second_layer_batchnorm = second_layer_batchnorm
        self.second_layer_pooling = second_layer_pooling
        self.second_layer_poolsize = second_layer_poolsize

        self.dropout_rate = dropout_rate
        self.act_func = act_func
        self.learning_rate = learning_rate
        self.target_dimension = target_dimension
        self.batch_normalization = batch_normalization
        self.nb_epoch = nb_epoch
        self.early_stopping_flag = early_stopping_flag
        self.eaSt_patience = eaSt_patience
        self.reLe_factor = reLe_factor
        self.reLe_patience = reLe_patience
        self.batch_size = batch_size
        self.loss = loss
        self.final_activation = final_activation

        #self.model = None

        # Todo: Check why Logger singleton doesn't work in this scenario
        # if Logger().verbosity_level == 2:
        #     self.verbosity = 2
        # else:
        #     self.verbosity = 0

        self.verbosity = verbosity


    def fit(self, X, y):

        # prepare target values
        # Todo: calculate number of classes?
        # Todo: smarter way to check input dimensions?

        if self.target_dimension > 1:
            y = to_categorical(y, self.target_dimension)

        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))

        # 1. make model
        self.model = self.create_model((X.shape[1], X.shape[2], X.shape[3]))

        # 2. fit model
        # start_time = time.time()

        # use callbacks only when size of training set is above 100
        if X.shape[0] > 100:
            # get pseudo validation set for keras callbacks
            splitter = ShuffleSplit(n_splits=1, test_size=0.2)
            for train_index, val_index in splitter.split(X):
                X_train = X[train_index]
                X_val = X[val_index]
                y_train = y[train_index]
                y_val = y[val_index]

            # register callbacks
            callbacks_list = []
            # use early stopping (to save time;
            # does not improve performance as checkpoint will find the best model anyway)
            if self.early_stopping_flag:
                early_stopping = EarlyStopping(monitor='val_loss',
                                               patience=self.eaSt_patience)
                callbacks_list += [early_stopping]

            # adjust learning rate when not improving for patience epochs
            reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                          factor=self.reLe_factor,
                                          patience=self.reLe_patience,
                                          min_lr=0.001, verbose=0)
            callbacks_list += [reduce_lr]

            # fit the model
            results = self.model.fit(X_train, y_train,
                                     validation_data=(X_val, y_val),
                                     batch_size=self.batch_size,
                                     epochs=self.nb_epoch,
                                     verbose=self.verbosity,
                                     callbacks=callbacks_list)
        else:
            # fit the model
            Logger().warn('Cannot use Keras Callbacks because of small sample size...')
            results = self.model.fit(X, y, batch_size=self.batch_size,
                                     epochs=self.nb_epoch,
                                     verbose=self.verbosity)

        return self

    def predict(self, X):
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
        predict_result = self.model.predict(X, batch_size=self.batch_size)
        max_index = np.argmax(predict_result, axis=1)
        return max_index

    def predict_proba(self, X):
        """
        Predict probabilities
        :param X: array-like
        :type data: float
        :return: predicted values, array
        """
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))

        return self.model.predict(X, batch_size=self.batch_size)

    def score(self, X, y_true):
        return np.zeros(1)

    def create_model(self, input_size):

        model = Sequential()


        #constructing the first convlayer
        model.add(Conv2D(self.first_layer_dim,
                         kernel_size=(self.first_layer_kernel, self.first_layer_kernel),
                         activation=self.first_layer_activation,
                         input_shape=(input_size)))
        if self.first_layer_batchnorm == 1:
            model.add(BatchNormalization())
        if self.first_layer_pooling == 1:
            model.add(MaxPooling2D(pool_size=self.first_layer_poolsize))

        #constructing the second convlayer
        model.add(Conv2D(self.second_layer_dim,
                         kernel_size=(self.second_layer_kernel, self.second_layer_kernel),
                         activation=self.second_layer_activation))
        if self.second_layer_batchnorm == 1:
            model.add(BatchNormalization())
        if self.second_layer_pooling == 1:
            model.add(MaxPooling2D(pool_size=self.second_layer_poolsize))


        model.add(Dropout(self.dropout_rate))


        model.add(Flatten())

        for i, dim in enumerate(self.hidden_layer_sizes):
            if i == 0:
                model.add(Dense(dim, input_shape=(input_size,),  kernel_initializer='random_uniform'))
            else:
                model.add(Dense(dim, activation='relu', kernel_initializer='he_uniform'))

            if self.batch_normalization == True:
                model.add(BatchNormalization())

            model.add(Dropout(self.dropout_rate))

        model.add(Dense(self.target_dimension, activation=self.final_activation))

        # Compile model
        optimizer = Adam(lr=self.learning_rate)
        model.compile(loss=self.loss,
                      optimizer=optimizer, metrics=['accuracy'])


        return model


    @staticmethod
    def dense_to_one_hot(labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
