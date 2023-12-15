# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import os
import numpy as np
import cv2
import argparse
import time
import h5py
from skimage import measure

import sys
sys.path.append("../")

from utilities import mesh_utilities
from utilities import point_cloud_utilities
import point_cloud_utilities_cy


parser = argparse.ArgumentParser()
parser.add_argument("class_id", type=str, help="shapenet category id")
parser.add_argument("share_id", type=int, help="id of the share [0]")
parser.add_argument("share_total", type=int, help="total num of shares [1]")
FLAGS = parser.parse_args()

class_id = FLAGS.class_id
target_dir = "./"+class_id+"_simplified_textured/"
if not os.path.exists(target_dir):
    print("ERROR: this dir does not exist: "+target_dir)
    exit(-1)

share_id = FLAGS.share_id
share_total = FLAGS.share_total

obj_names = os.listdir(target_dir)
obj_names = sorted(obj_names)

obj_names_ = []
for i in range(len(obj_names)):
    if i%share_total==share_id:
        obj_names_.append(obj_names[i])
obj_names = obj_names_


num_of_points = 100000
voxel_size = 64
visualize_data = False #set to True to do sanity check by storing point clouds and voxels of the dataset for visual examination


for i in range(len(obj_names)):
    obj_dir = target_dir + obj_names[i] + "/model_simplified_textured.obj"
    texture_dir = target_dir + obj_names[i] + "/model_simplified_textured.png"
    output_dir = target_dir + obj_names[i] + "/vertices_normals_colors_voxels.hdf5"
    print(i,len(obj_names),obj_names[i])

    _,_,vertices,normals,colors = point_cloud_utilities.sample_points(obj_dir,texture_dir,num_of_points,exact_num=True,normalize=False)
    
    if visualize_data:
        point_cloud_utilities.write_ply_point_normal_color(output_dir+".ply",vertices,normals,colors[:,2::-1])


    #get colored voxels
    voxel_count = np.zeros([voxel_size,voxel_size,voxel_size], np.int32)
    voxel_color = np.zeros([voxel_size,voxel_size,voxel_size,4], np.int32)
    vertices_int = ((vertices+0.5)*voxel_size).astype(np.int32)
    vertices_int = np.clip(vertices_int,0,voxel_size-1)

    #numpy's += only performs assignments to the last values!
    #in numpy, a = np.array([0,0,0,0,1]), b = np.zeros([2]), then b[a]+=1, b is now [1,1] instead of [4,1]!!!!
    #voxel_count[vertices_int[:,0],vertices_int[:,1],vertices_int[:,2]] += 1  <-- wrong!!!
    #voxel_color[vertices_int[:,0],vertices_int[:,1],vertices_int[:,2]] += colors  <-- wrong!!!
    #the following customized functions do the correct thing.
    point_cloud_utilities_cy.indexed_add_constant_3d_int(voxel_count,vertices_int,1)
    point_cloud_utilities_cy.indexed_add_array_3d_color(voxel_color,vertices_int,colors)
    
    voxel_count = np.expand_dims(voxel_count,3)
    voxel_color = voxel_color.astype(np.float32)/np.maximum(voxel_count,1)
    voxel_count = (voxel_count>0).astype(np.float32)

    if visualize_data:
        verts, faces, _, _ = measure.marching_cubes(0.5-voxel_count[:,:,:,0], 0)
        verts_int1 = (verts).astype(np.int32)
        verts_int2 = (verts+0.5).astype(np.int32)
        verts = (verts+0.5)/voxel_size-0.5
        vcolors = np.maximum(voxel_color[verts_int1[:,0],verts_int1[:,1],verts_int1[:,2]],voxel_color[verts_int2[:,0],verts_int2[:,1],verts_int2[:,2]])
        vcolors = vcolors.astype(np.uint8)
        mesh_utilities.write_ply_triangle_color(output_dir+".surfacevoxel"+str(voxel_size)+".ply", verts, vcolors[:,2::-1], faces)


    hdf5_file = h5py.File(output_dir, 'w')
    hdf5_file.create_dataset("vertices", [num_of_points,3], np.float32, compression=9)
    hdf5_file.create_dataset("normals", [num_of_points,3], np.float32, compression=9)
    hdf5_file.create_dataset("colors", [num_of_points,4], np.float32, compression=9)
    hdf5_file.create_dataset("voxel_color", [voxel_size,voxel_size,voxel_size,5], np.float32, compression=9)
    hdf5_file["vertices"][:] = vertices
    hdf5_file["normals"][:] = normals
    hdf5_file["colors"][:] = colors.astype(np.float32)/256
    hdf5_file["voxel_color"][:] = np.concatenate([voxel_count,voxel_color/256],axis=3)
    hdf5_file.close()
