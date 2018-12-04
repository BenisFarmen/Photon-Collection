import itertools
import numpy as np
import math
import random
from scipy.spatial.distance import pdist
import pandas
import tensorflow as tf
from sklearn.model_selection import KFold
import scipy.io as sio
import keras
from scipy.stats import itemfreq
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.datasets import boston_housing



def _stirling(n):

    # http://en.wikipedia.org/wiki/Stirling%27s_approximation

    return math.sqrt(2*math.pi*n)*(n/math.e)**n





def _nCr(n, r):

    """

    Get number of combinations of r from n items (e.g. n=10, r=2--> pairs from 0 to 10

    :param n: n items to draw from

    :param r: number of items to draw each time

    :return: approximate number of combinations

    """

    try:

        return (_stirling(n) / _stirling(r) / _stirling(n - r) if n > 20 else math.factorial(n) / math.factorial(r) / math.factorial(n - r))

    except:

        return -1





def random_pair_generator(items, rand_seed=True):

    """Return an iterator of random pairs from a list of items."""

    # Keep track of already generated pairs

    used_pairs = set()

    random.seed(rand_seed)

    i = 0

    while True:

        if rand_seed:

            random.seed(i)

            i += 1

        pair = random.sample(items, 2)

        # Avoid generating both (1, 2) and (2, 1)

        pair = tuple(sorted(pair))

        if pair not in used_pairs:

            used_pairs.add(pair)

            yield pair





def nearest_pair_generator(X, distance_metric):

    # return most similar pair (from similar to dissimilar)

    i = 0

    # compute distance matrix and get indices

    #res_order = np.argsort(pdist(X, distance_metric))



    # drop samples with missing values

    X = X[~np.isnan(X).any(axis=1)]



    from sklearn.preprocessing import StandardScaler

    s = StandardScaler()

    X = s.fit_transform(X)

    res_order = np.argsort(pdist(X, distance_metric))



    inds = np.triu_indices(X.shape[0], k=1)

    while True:

        # get index tuple for the closest draw_limit samples

        pair = (inds[0][res_order[i]], (inds[1][res_order[i]]))

        i+=1

        yield pair





def _get_combinations(X, draw_limit, rand_seed, distance_metric, generator='random_pair'):

    """

    :param X: data array

    :param draw_limit: in case the full number of combinations is > 10k, how many to draw?

    :param rand_seed: sets seed for random sampling of combinations (for reproducibility)

    :param generator: method with which to obtain sample pairs (samples until draw_limit is reached or generator stops)

                        'random_pair': sample randomly from all pairs

                        'nearest_pair': get most similar pairs

    :param distance metric: if generator is 'nearest_pair', this will set the distance metric to obtained similarity

    :return: list of tuples indicating which samples to merge

    """



    items = range(X.shape[0])

    # limit the number of new samples generated if all combinations > draw_limit

    n_combs = _nCr(n=len(items), r=2) # get number of possible pairs (combinations of 2) from this data

    if n_combs > draw_limit or n_combs == -1:



        if generator == 'random_pair':

            combs_generator = random_pair_generator(items=items, rand_seed=rand_seed)

        elif generator == 'nearest_pair':

            combs_generator = nearest_pair_generator(X=X, distance_metric=distance_metric)



        # Get draw_limit sample pairs

        combs = list()

        for i in range(draw_limit):

            combs.append(list(next(combs_generator)))

    else:

        # get all combinations of samples

        combs = list(itertools.combinations(items, 2))

    return combs





def get_samples(X, y=None, generator='random_pair', distance_metric=None, draw_limit=10000, rand_seed=True):

    """

    Generates "new samples" by computing the mean between all or n_draws pairs of existing samples and appends them to X

    The target for each new sample is computed as the mean between the constituent targets

    :param X: data

    :param y: targets (optional)

    :param draw_limit: in case the full number of combinations is > 10k, how many to draw?

    :param rand_seed: sets seed for random sampling of combinations (for reproducibility only)

    :return: X_new: X and X_augmented; (y_new: the correspoding targets)

    """



    #print('\n\n Sample Pairing: ' + str(generator) + ', draw_limit = ' + str(draw_limit))



    # generate combinations

    combs = _get_combinations(X=X,

                              generator=generator,

                              distance_metric=distance_metric,

                              draw_limit=draw_limit,

                              rand_seed=rand_seed)



    # compute mean over sample pairs

    X_aug = np.empty([len(combs), X.shape[1]])

    i = 0

    for c in combs:

        X_aug[i] = np.mean([X[c[0]], X[c[1]]], axis=0)

        i += 1

    # add augmented samples to existing data

    X_new = np.concatenate((X, X_aug), axis=0)



    # get the corrsponding targets

    if y is not None:

        y_aug = np.empty(len(combs))

        i = 0

        for c in combs:

            y_aug[i] = np.mean([y[c[0]], y[c[1]]])

            i += 1

        # add augmented samples to existing data

        y_new = np.concatenate((y, y_aug))

        return X_new, y_new

    else:

        return X_new





def get_samples_classification(X, y,

                               generator='random_pair',

                               distance_metric=None, draw_limit=10000, balance_classes=True, rand_seed=True):



    # ensure class balance in the training set if balance_classes is True

    nDiff = list()

    for t in np.unique(y):

        if balance_classes == True:

            nDiff.append(draw_limit - np.sum(y == t))

        else:

            nDiff.append(draw_limit)





    # run get_samples for each class independently

    for t, limit in zip(np.unique(y), nDiff):

        X_new_class, y_new_class = get_samples(X[y == t], y[y == t],

                                   generator=generator, distance_metric=distance_metric, draw_limit=limit, rand_seed=rand_seed)

        if 'X_new' not in locals():

            X_new = X_new_class

            y_new = y_new_class

        else:

            X_new = np.concatenate((X_new, X_new_class))

            y_new = np.concatenate((y_new, y_new_class))

    return X_new, y_new





def run_regression(X_tr, y_tr):

    from sklearn.preprocessing import StandardScaler

    from sklearn.decomposition import PCA

    from sklearn.pipeline import Pipeline

    from sklearn.svm import SVR, LinearSVR

    from sklearn.preprocessing import Imputer

    from sklearn.neural_network import MLPRegressor

    from sklearn.ensemble import RandomForestRegressor

    from sklearn.feature_selection import SelectKBest, f_regression, RFE

    estimators = []

    estimators.append(('imputer', Imputer(missing_values=np.nan, strategy='median', axis=0)))

    estimators.append(('standardize', StandardScaler()))

    estimators.append(('SVR', LinearSVR()))

    # estimators.append(('PCA', PCA()))

    #estimators.append(('fsSel', RFE(SVR(kernel="linear"), 30, step=1)))

    # #estimators.append(('fsSel', SelectKBest(f_regression, k=40)))

    #estimators.append(('SVR', SVR(kernel='rbf')))

    #estimators.append(('RF', RandomForestRegressor(criterion='mse', min_samples_split=2)))

    #estimators.append(('Reg', MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='identity')))

    #estimators.append(('Reg', MLPRegressor(hidden_layer_sizes=(5), activation='identity')))



    pipeline = Pipeline(estimators)

    pipeline.fit(X_tr, y_tr)

    return pipeline





def run_classification(X_tr, y_tr):

    from sklearn.svm import SVC

    from sklearn.preprocessing import StandardScaler

    from sklearn.pipeline import Pipeline



    estimators = []

    estimators.append(('standardize', StandardScaler()))

    estimators.append(('SVC', SVC()))

    pipeline = Pipeline(estimators)

    pipeline.fit(X_tr, y_tr)



    return pipeline





def get_ENIGMA_data():

    # Regression with the ENIGMA BrainAge dataset

    # folder = '/media/tim/Data/myGoogleDrive/work/all_skripts/py_code/checks/ENIGMA_age/final/'

    folder = 'D:/myGoogleDrive/work/all_skripts/py_code/checks/ENIGMA_age/final/'

    # Regression with the ENIGMA BrainAge dataset

    # gender = 'Female'

    gender = 'Male'

    d = ''



    from sklearn.preprocessing import Imputer

    setID = 'train'

    X = pandas.read_csv(folder + d + setID + 'Controls' + gender + 's_raw.csv', header=0, delimiter=',')

    y = X.Age.values

    X = X.drop(labels=['Age'], axis=1).values



    # get performance for test data

    setID = 'test'

    X_te = pandas.read_csv(folder + d + setID + 'Controls' + gender + 's_raw.csv', header=0, delimiter=',')

    y_te = X_te.Age.values

    X_te = X_te.drop(labels=['Age'], axis=1).values



    # get performance for BD data

    setID = 'test'

    X_BD = pandas.read_csv(folder + d + setID + 'BD' + gender + 's_raw.csv', header=0, delimiter=',')

    y_BD = X_BD.Age.values

    X_BD = X_BD.drop(labels=['Age'], axis=1).values



    # get performance for MDD data

    setID = 'test'

    X_MDD = pandas.read_csv(folder + d + setID + 'MDD' + gender + 's_raw.csv', header=0, delimiter=',')

    y_MDD = X_MDD.Age.values

    X_MDD = X_MDD.drop(labels=['Age'], axis=1).values



    return X, y, X_te, y_te, X_BD, y_BD, X_MDD, y_MDD





if __name__ == '__main__':

    # seed = 42

    # drawLimitList = [0, 2000, 5000, 10000, 20000]

    # for dl in drawLimitList:

    #     # Regression with the ENIGMA BrainAge dataset

    #     X, y, X_te, y_te, X_BD, y_BD, X_MDD, y_MDD = get_ENIGMA_data()

    #

    #     # sample pairing

    #     X, y = get_samples(X, y, generator='random_pair', draw_limit=dl, rand_seed=True)

    #     #X, y = get_samples(X, y, generator='nearest_pair', distance_metric='correlation', draw_limit=10000, rand_seed=True)

    #

    #     # evaluate model with standardized dataset

    #     np.random.seed(seed)

    #     print('Plain Data: data shape: ' + str(X.shape))

    #     my_model = run_regression(X_tr=X, y_tr=y)

    #     preds = my_model.predict(X_te)

    #     results = np.mean(np.abs(y_te - preds))

    #     print("MAE %.2f" % results)

    #

    #     # get performance for BD data

    #     preds_ind = my_model.predict(X_BD)

    #     mae = np.mean(np.abs(y_BD - preds_ind))

    #     print('BD: MAE overall = ' + str(mae))

    #

    #     # get performance for MDD data

    #     preds_ind = my_model.predict(X_MDD)

    #     mae = np.mean(np.abs(y_MDD - preds_ind))

    #     print('MDD: MAE overall = ' + str(mae))



    ####################################################

    ####################################################

    ####################################################



    # Classification with the sonar dataset

    from sklearn.model_selection import cross_val_score

    from sklearn.preprocessing import LabelEncoder

    from sklearn import datasets

    drawLimitList = [0, 2000, 5000, 10000, 20000]


    for dl in drawLimitList:

        boston = datasets.load_boston()



        #(X_tr, y_tr), (X_te, y_te) = boston_housing.load_data(test_split= 0.95, seed = 42)



        #get matrix data for functional connectivity
        #mat_data = sio.loadmat('Dosenbach_matrices_1044.mat')
        #FC_matrices = mat_data['matrices']
        #FC_reordered = np.swapaxes(FC_matrices, 0, 2)

        #FC_labels = np.loadtxt('Dosenbach_labels_1044.txt')
        #FC_labels = FC_labels - 1

        #filter = np.where((FC_labels == 0) | (FC_labels == 1))
        #X, y = FC_reordered[filter], FC_labels[filter]
        #X = np.reshape(X, (X.shape[0], -1))
        #X, y = shuffle(X, y, random_state=42)


        # get training and test set

        X_tr, X_te, y_tr, y_te = train_test_split(boston.data, boston.target, test_size=0.9, random_state=42)


        # evaluate model with standardized dataset

        seed = 42

        np.random.seed(seed)



        # evaluate model with augmented data

        X_tr, y_tr = get_samples(X_tr, y_tr, generator='random_pair', distance_metric=None,

                                                draw_limit=dl, rand_seed=True)

        # X_tr, y_tr = get_samples_classification(X_tr, y_tr, generator='nearest_pair', distance_metric='correlation',

        #                                         draw_limit=dl, rand_seed=True)



        print('Augmented Data (random pairs): draw_limit = ' + str(dl) + '; data shape: ' + str(X_tr.shape))

        my_model = run_regression(X_tr=X_tr, y_tr=y_tr)



        # predict test set

        preds = my_model.predict(X_te)

        results = my_model.score(X_te, y_te)

        print("Coefficient of determination R^2 %.2f" % results)
