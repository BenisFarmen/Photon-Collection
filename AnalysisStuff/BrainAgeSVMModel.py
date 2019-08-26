"""

Paper Manufacturing 2019 - Brain Age Prediction - Correlations of brain age gap with Psychometric Measures in FOR2107 depressive patients



- use Tim's pretrained brain age models

"""

from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings, PreprocessingPipe
from photonai.optimization.Hyperparameters import FloatRange, Categorical, IntegerRange
from photonai.neuro.AtlasMapping import AtlasMapper
from sklearn.model_selection import KFold
import time
import os
import pandas as pd
import glob
from collections import OrderedDict
import numpy as np
from numpy.matlib import repmat


#load nifti files
file_path_list = glob.glob('/spm-data/Scratch/spielwiese_vincent/PAC2019/CAT12_JC/mri/mwp1age*.nii')
X = sorted(file_path_list)

#load labels
PAClabels = pd.DataFrame(pd.read_excel(r'/spm-data/Scratch/spielwiese_ramona/PAC2019/old_files/PAC2019_data.xlsx'))
PAClabels = PAClabels[['id', 'age']]
PACIDs = [z.split("/mri/mwp1")[1].split("_")[0] for z in file_path_list]
SelectedLabels = PAClabels[PAClabels['id'].isin(PACIDs)]
y = SelectedLabels['age'].to_numpy()

# YOU CAN SAVE THE TRAINING AND TEST RESULTS AND ALL THE PERFORMANCES IN THE MONGODB
mongo_settings = OutputSettings(save_predictions='best')


#Pick the Atlases you want to use
atlases = ['AAL', 'HarvardOxford-cort-maxprob-thr25', 'HarvardOxford-sub-maxprob-thr25', ',mni_icbm152_gm_tal_nlin_sym_09a']
#loop over the different atlases
for atlas_name in atlases:
    # DESIGN YOUR PIPELINE
    my_pipe = Hyperpipe('basic_svm_pipe',  # the name of your pipeline
                        optimizer='grid_search',  # which optimizer PHOTON shall use
                        metrics=['mean_absolute_error', 'pearson_correlation'],  # the performance metrics of your interest
                        best_config_metric='mean_absolute_error',
                        inner_cv=KFold(n_splits=5),  # test each configuration ten times respectively,
                        verbosity=1,
                        output_settings=mongo_settings)  # get error, warn and info message

    preprocessing = PreprocessingPipe()
    preprocessing += PipelineElement('BrainAtlas', atlas_name=atlas_name, extract_mode='vec')
    my_pipe += preprocessing
    my_pipe += PipelineElement('SVR', hyperparameters={}, kernel='linear')

    # NOW TRAIN YOUR PIPELINE
    start_time = time.time()
    atlas_mapper = AtlasMapper()
    my_folder = os.path.join('/spm-data/Scratch/spielwiese_vincent/PAC2019/CAT12_JC/AtlasMapper_' + atlas_name)
    atlas_mapper.generate_mappings(my_pipe, my_folder)
    atlas_mapper.fit(X, y)
    elapsed_time = time.time() - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

#load the FOR Data for Predictions
FOR_labels = pd.DataFrame(pd.read_csv(r'/spm-data/Scratch/spielwiese_vincent/PAC2019/CAT12_JC/data_brain_age_disorder_comparison.csv'))
FOR_age = FOR_labels[['age']]
FOR_Data_paths = FOR_labels[['NeuroDerivatives_r1184_gray_matter']]

#reload the trained model
for atlas_name in atlases:
    #reload trained model
    atlas_mapper = AtlasMapper()
    atlas_mapper.load_from_file('/spm-data/Scratch/spielwiese_vincent/PAC2019/CAT12_JC/AtlasMapper_' + atlas_name + '/basic_svm_pipe_atlas_mapper_meta.json')
    #predict on FOR data
    age_prediction_results = atlas_mapper.predict(FOR_Data_paths)
    #turn predictions into dataframe
    age_prediction_results = pd.DataFrame(age_prediction_results)
    age_prediction_results = age_prediction_results.to_numpy()
    #calculate brain age gap via numpy
    age_targets = repmat(FOR_age, 1, age_prediction_results.shape[1])
    age_gap = age_prediction_results - age_targets
    np.savetxt(fname='/spm-data/Scratch/spielwiese_vincent/PAC2019/CAT12_JC/age_gap.csv', X=age_gap, fmt='%.2f', delimiter=',')
#use the trained model to predict on new data

debug = True


