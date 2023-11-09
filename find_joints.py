from typing import Union
import numpy as np
import torch
from body_model.body_model import BodyModel
from torch import nn
from tools.omni_tools import copy2cpu as c2c
from os import path as osp

support_dir = 'Amass_files' #Home-folder for Amass poses files (obtain from https://amass.is.tue.mpg.de/download.php)
bm_fname =  osp.join('PATH_TO_SMPLX_model.npz') (obtain from https://smplx.is.tue.mpg.de/downloads)
amass_npz_fname = osp.join(support_dir, 'NAME_OF_AMASS_FILE.npz')

comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bdata = np.load(amass_npz_fname)

data_arr = []
motion_indices = np.arange(0, bdata['poses'].shape[0], 12) #Downsampling the data to 12 fps

for i in motion_indices:
    bm = BodyModel(bm_fname)(**{
        'pose_body': torch.tensor(bdata['poses'][i:i+1,3:66]).type(torch.float),
        'root_orient': torch.tensor(np.zeros((1,3))).type(torch.float),
        'trans':torch.tensor(np.zeros((1,3))).type(torch.float),
    })
    
    fId = 0
    joints = c2c(bm.Jtr[fId])
    
    joint_indices = [0,1,2,4,5,6,7,8,12,15,16,17,18,19,20,21] #Indices corresponding to the 16 joints -'pelvis', 'left_hip',        'right_hip', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'neck', 'head', 'left_shoulder','right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist','right_wrist',

    joints_considered = []
    for j in joint_indices:
        joints_considered.append(joints[j])
    data_arr.append(joints_considered)
    
data_arr = np.asarray(data_arr)
np.save('NAME_OF_AMASS_FILE_joints.npy', data_arr)
