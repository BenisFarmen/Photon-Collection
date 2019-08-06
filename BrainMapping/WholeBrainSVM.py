#A Whole-Brain SVM pipeline for brain age prediction
import glob
import numpy as np
import pandas as pd
import tensorflow
from sklearn.model_selection import KFold
import scipy.io as sio
import photonai
from photonai.base.PhotonBase import PreprocessingPipe
from photonai import PipelineElement
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
print(y.shape)


# Run the pipe with a 5 x 5 nested cross-validation and with accuracy as the metric by which to choose the best configuration
my_pipe = photonai.Hyperpipe('WholeBrainSVMPipe',
                             optimizer='grid_search',
                             metrics=['mean_absolute_error'],
                             best_config_metric='mean_absolute_error',
                             inner_cv=KFold(n_splits=5, shuffle=True, random_state=42))

#we are using a SVM as a comparable standard
preprocessing = PreprocessingPipe()

atlas = PipelineElement('BrainAtlas', atlas_name='AAL', extract_mode='vec', rois='all')

preprocessing += PhotonBatchElement('BrainAtlas', batch_size=100, atlas_name='AAL', extract_mode='vec', rois='all')

my_pipe += preprocessing

my_pipe += PipelineElement('StandardScaler')

my_pipe += PipelineElement('SVR')

my_pipe.fit(X, y)

debug = True


