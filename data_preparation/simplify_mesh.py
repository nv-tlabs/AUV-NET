
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
import mcubes
import open3d as o3d

import sys
sys.path.append("../")

from utilities import mesh_utilities
from utilities import voxel_utilities

parser = argparse.ArgumentParser()
parser.add_argument("class_id", type=str, help="shapenet category id")
parser.add_argument("share_id", type=int, help="id of the share [0]")
parser.add_argument("share_total", type=int, help="total num of shares [1]")
FLAGS = parser.parse_args()

class_id = FLAGS.class_id
target_dir = "./"+class_id+"/"
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




for i in range(len(obj_names)):

    if os.path.exists(target_dir + obj_names[i] + "/rendering_poses.txt"):
        continue
    
    ### normalize mesh into unit cube
    raw_obj_name = target_dir + obj_names[i] + "/model.obj"
    normalized_obj_name = target_dir + obj_names[i] + "/model_normalized.obj"
    print(i,len(obj_names),raw_obj_name)
    mesh_utilities.normalize_obj(raw_obj_name,normalized_obj_name)

    ### obtain 1024^3 voxels using binvox

    command = "./binvox -bb -0.5 -0.5 -0.5 0.5 0.5 0.5 -d 1024 -e "+normalized_obj_name
    os.system(command)

    #rename
    old_binvox_name = target_dir + obj_names[i] + "/model_normalized.binvox"
    binvox_name = target_dir + obj_names[i] + "/model.binvox"
    command = "mv "+ old_binvox_name + " " + binvox_name
    os.system(command)
    
    ### obtain simplified 1024^3 voxels

    raw_voxels = voxel_utilities.read_voxels(binvox_name,fix_coords=False)
    voxel_utilities.depth_fusion(raw_voxels)
    state_ctr = voxel_utilities.alpha_hull(raw_voxels)
    
    simplified_binvox_name = target_dir + obj_names[i] + "/model_coarse.binvox"
    voxel_utilities.write(simplified_binvox_name, [1024,1024,1024], state_ctr)
    del state_ctr
    
    ### obtain simplified mesh and save as obj file

    simplified_voxels_padded = np.zeros([1024+2,1024+2,1024+2], np.uint8)
    simplified_voxels = voxel_utilities.read_voxels(simplified_binvox_name)
    simplified_voxels_padded[1:-1,1:-1,1:-1] = simplified_voxels
    vertices, triangles = mcubes.marching_cubes(simplified_voxels_padded, 0.5)
    vertices += np.random.uniform(-0.1, 0.1, size=vertices.shape)
    del simplified_voxels_padded

    mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices),triangles=o3d.utility.Vector3iVector(triangles))
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=20000)

    vertices = np.asarray(mesh.vertices)
    vertices = (vertices-0.5)/1024 - 0.5
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    simplified_obj_name = target_dir + obj_names[i] + "/model_simplified.obj"
    o3d.io.write_triangle_mesh(simplified_obj_name, mesh)
    
    ### obtain additional rendering camera poses, to handle occlusion in rendering

    poses_file_name = target_dir + obj_names[i] + "/rendering_poses.txt"
    voxel_utilities.get_occlusion_cut(simplified_voxels,poses_file_name)
    del simplified_voxels



