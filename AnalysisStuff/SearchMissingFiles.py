import glob
import subprocess


def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

#find niftis
nifti_list = glob.glob('/spm-data/Scratch/spielwiese_ramona/PAC2019/raw_data_JC/age*.nii.gz')
preprocessed_nifti_list = glob.glob('/spm-data/Scratch/spielwiese_vincent/PAC2019/CAT12_JC/mri/mwp1age*.nii')
#find all needed nifits
Havelist = [z.split("/mri/mwp1")[1].split("_")[0] for z in preprocessed_nifti_list]
Wantlist = [z.split("/raw_data_JC/")[1].split("_")[0] for z in nifti_list]
#list of needed files
Needlist = diff(Wantlist, Havelist)
#take these indices/labels and copy the needed files to the server, or to a folder for preprocessing
Missing_File_Paths = [('/spm-data/Scratch/spielwiese_ramona/PAC2019/raw_data_JC/' + z + '*' + '.nii.gz') for z in Needlist]
Missing_File_Paths.sort()
with open('/spm-data/Scratch/spielwiese_vincent/PAC2019/missing/missing_files.txt', 'w') as f:
    for item in Missing_File_Paths:
        f.write("%s\n" % item)

for missing_file in Missing_File_Paths:
    subprocess.run(['cp' + ' ' + missing_file + ' ' + '/spm-data/Scratch/spielwiese_vincent/PAC2019/missing/'])

#find missing niftis
#missing_nifti_list = []

