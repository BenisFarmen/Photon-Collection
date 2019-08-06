"""

Paper Manufacturing 2019 - Brain Age Prediction - Correlations of brain age gap with Psychometric Measures in FOR2107 depressive patients



- train Model on James Cole Data and predict on FOR

"""

from photonai.base.PhotonBase import Hyperpipe, PipelineElement, OutputSettings, PreprocessingPipe
from photonai.optimization.Hyperparameters import FloatRange, Categorical, IntegerRange
from photonai.neuro.AtlasMapping import AtlasMapper
from photonai.neuro.NeuroBase import NeuroModuleBranch
from sklearn.model_selection import KFold
import time
import os
import pandas as pd
import glob
import numpy as np

from photonai.base.PhotonBase import PipelineElement, PreprocessingPipe, Hyperpipe
from photonai.base.PhotonBatchElement import PhotonBatchElement

#load nifti files
file_path_list = glob.glob('/spm-data/Scratch/spielwiese_vincent/PAC2019/TestRun_PreProcessing/mri/mwp1age*.nii')
print(file_path_list)
X = sorted(file_path_list)
print(X)

#load labels
PAClabels = pd.DataFrame(pd.read_excel(r'/spm-data/Scratch/spielwiese_ramona/PAC2019/old_files/PAC2019_data.xlsx'))
PAClabels = PAClabels[['id', 'age']]
PACIDs = [z.split("/mri/mwp1")[1].split("_")[0] for z in file_path_list]
SelectedLabels = PAClabels[PAClabels['id'].isin(PACIDs)]
y = SelectedLabels['age'].to_numpy()
print(np.isnan(y))
print(type(y))


# YOU CAN SAVE THE TRAINING AND TEST RESULTS AND ALL THE PERFORMANCES IN THE MONGODB
mongo_settings = OutputSettings(save_predictions='best')

# DESIGN YOUR PIPELINE
my_pipe = Hyperpipe('basic_svm_pipe',  # the name of your pipeline
                    optimizer='grid_search',  # which optimizer PHOTON shall use
                    metrics=['mean_absolute_error'],  # the performance metrics of your interest
                    best_config_metric='mean_absolute_error',
                    inner_cv=KFold(n_splits=2),  # test each configuration ten times respectively,
                    verbosity=2,
                    output_settings=mongo_settings)  # get error, warn and info message


preprocessing = PreprocessingPipe()
#neuro_branch = NeuroModuleBranch('neuro_branch', nr_of_processes=10, cache_folder='/spm-data/Scratch/spielwiese_vincent/PAC2019/CAT12_JC/Analysis_Cache')
preprocessing += PipelineElement('BrainAtlas', atlas_name='AAL', extract_mode='vec', rois='all')
#transformer = PipelineElement("your_trans", hyperparameters={})
#batched_transformer = PhotonBatchElement("batched_neuro", batch_size=10, base_element=neuro_branch)
#preprocessing +=  neuro_branch
my_pipe += preprocessing
my_pipe += PipelineElement('SVR', hyperparameters={}, kernel='linear')

    # NOW TRAIN YOUR PIPELINE
start_time = time.time()
atlas_mapper = AtlasMapper()
my_folder = os.path.join('/spm-data/Scratch/spielwiese_vincent/PAC2019/CAT12_JC/AtlasMapper_AAL')
atlas_mapper.generate_mappings(my_pipe, my_folder)
atlas_mapper.fit(X, y)
elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


#reload the trained model
atlas_mapper = AtlasMapper()
atlas_mapper.load_from_file('/spm-data/Scratch/spielwiese_vincent/PAC2019/CAT12_JC/AtlasMapper_AAL/basic_svm_pipe_atlas_mapper_meta.json')

#load the new data
FOR_Data = 1 #load stuff

#use the trained model to predict on new data
print(atlas_mapper.predict(X))
debug = True


