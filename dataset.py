# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import os
import numpy as np
import h5py
import torch

class point_color_voxel_dataset(torch.utils.data.Dataset):
    '''
    This dataset returns:
    1. Sampled points on the surfaces of 3D shapes;
    2. The normals of the sampled points;
    3. The RGB or RGBA colors of the sampled points;
    4. Colored voxels of 3D shapes.
    '''
    def __init__(self, data_dir, point_batch_size, train):
        self.data_dir = data_dir
        self.point_batch_size = point_batch_size
        self.train = train
    
        obj_names = os.listdir(self.data_dir)
        obj_names = sorted(obj_names)

        if self.train is None:
            self.start_idx = 0
            self.obj_names = obj_names
            print("Total#", "all", len(self.obj_names))
        elif self.train:
            self.start_idx = 0
            self.obj_names = obj_names[:int(len(obj_names)*0.8)]
            print("Total#", "train", len(self.obj_names))
        else:
            self.start_idx = int(len(obj_names)*0.8)
            self.obj_names = obj_names[int(len(obj_names)*0.8):]
            print("Total#", "test", len(self.obj_names))

    def __len__(self):
        return len(self.obj_names)

    def __getitem__(self, index):
        hdf5_dir = self.data_dir+"/"+self.obj_names[index]+"/vertices_normals_colors_voxels.hdf5"
        grid_size = 64

        hdf5_file = h5py.File(hdf5_dir, 'r')
        rand_idcs = np.random.randint(len(hdf5_file["vertices"])-self.point_batch_size+1)
        vertices = hdf5_file["vertices"][rand_idcs:rand_idcs+self.point_batch_size]
        normals = hdf5_file["normals"][rand_idcs:rand_idcs+self.point_batch_size]
        colors = hdf5_file["colors"][rand_idcs:rand_idcs+self.point_batch_size]
        voxels = hdf5_file["voxel_color"][:]
        hdf5_file.close()

        colors = colors[:,:3] #RGB only, remove alpha
        voxels = np.transpose(voxels, (3,0,1,2)).astype(np.float32)

        return vertices, normals, colors, voxels