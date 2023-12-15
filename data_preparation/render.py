# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import bpy, sys, os
import argparse
from math import pi


preferences = bpy.context.preferences
cycles_preferences = preferences.addons["cycles"].preferences
cuda_devices, opencl_devices = cycles_preferences.get_devices()

for device in cuda_devices:
    device.use = True
cycles_preferences.compute_device_type = "CUDA"
bpy.context.scene.cycles.device = "GPU"


argv = sys.argv[sys.argv.index('--') + 1:]
parser = argparse.ArgumentParser()
parser.add_argument("class_id", type=str, help="shapenet category id")
parser.add_argument("share_id", type=int, help="id of the share [0]")
parser.add_argument("share_total", type=int, help="total num of shares [1]")
FLAGS = parser.parse_known_args(argv)[0]

class_id = FLAGS.class_id
target_dir = "./"+class_id+"/"
if not os.path.exists(target_dir):
    print("ERROR: this dir does not exist: "+target_dir)
    exit(-1)

share_id = FLAGS.share_id
share_total = FLAGS.share_total

root_dir = './'+class_id
obj_dir_set = os.listdir(root_dir)
obj_dir_set = sorted(obj_dir_set)

obj_dir_set_ = []
for i in range(len(obj_dir_set)):
    if i%share_total==share_id:
        obj_dir_set_.append(obj_dir_set[i])
obj_dir_set = obj_dir_set_

for obj_dir in obj_dir_set:
    
    #uncomment this line to skip shapes that are already rendered
    if os.path.exists(root_dir+'/'+obj_dir+'/15.png'): continue
    
    print(obj_dir)
    
    fout = open("current_rendering"+str(share_id)+".txt", 'w')
    fout.write(obj_dir)
    fout.close()

    #remove
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    for this_obj in bpy.data.objects:
        if this_obj.type=="MESH":
            this_obj.select_set(True)
            bpy.ops.object.delete(use_global=False, confirm=False)


    file_loc = root_dir+'/'+obj_dir+'/model_normalized.obj'
    imported_object = bpy.ops.import_scene.obj(filepath=file_loc, use_edges=False, use_smooth_groups=False)

    #load
    for this_obj in bpy.data.objects:
        if this_obj.type=="MESH":
            this_obj.select_set(True)
            bpy.context.view_layer.objects.active = this_obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.split_normals()

    bpy.ops.object.mode_set(mode='OBJECT')


    #get additional camera poses
    poses_file_name = root_dir+'/'+obj_dir+'/rendering_poses.txt'
    fin = open(poses_file_name, 'r')
    x_cut, y_cut, z_cut_1, z_cut_2 = [float(i) for i in fin.readline().split()]
    fin.close()


    # list of camera locations
    cam_location_list = [(0.0, 0.0, 1.0), (0.0, 0.0, y_cut), (0.0, 0.0, -1.0), (0.0, 0.0, y_cut), (0.0, -1.0, 0.0), 
                         (0.0, -z_cut_2, 0.0), (0.0, 0.0, 0.0), (0.0,  -z_cut_1, 0.0), (0.0, 1.0, 0.0), (0.0,  -z_cut_1, 0.0), 
                         (0.0, 0.0, 0.0), (0.0, -z_cut_2, 0.0), (1.0, 0.0, 0.0), (x_cut, 0.0, 0.0), (-1.0, 0.0, 0.0), 
                         (x_cut, 0.0, 0.0)   ]

    # list of euler angles
    euler_list = [ (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (pi, 0.0, 0.0), (pi, 0.0, 0.0), (0.5*pi, 0.0, 0.0), 
                   (0.5*pi, 0.0, 0.0), (0.5*pi, 0.0, 0.0), (0.5*pi, 0.0, 0.0), (0.5*pi, 0.0, pi), (0.5*pi, 0.0, pi), 
                   (0.5*pi, 0.0, pi), (0.5*pi, 0.0, pi), (0.5*pi, 0.0, 0.5*pi), (0.5*pi, 0.0, 0.5*pi), (0.5*pi, 0.0, -0.5*pi), 
                   (0.5*pi, 0.0, -0.5*pi)  ]

    for view_id in range(16):

        cam_loc = cam_location_list[ view_id ]
        cam_eulers = euler_list[ view_id ]

        cam = bpy.data.objects['Camera']
        cam.location.x = cam_loc[0]
        cam.location.y = cam_loc[1]
        cam.location.z = cam_loc[2]

        cam.rotation_euler[0] = cam_eulers[0]
        cam.rotation_euler[1] = cam_eulers[1]
        cam.rotation_euler[2] = cam_eulers[2]

        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.filepath = root_dir+'/'+obj_dir+'/' + str(view_id) + '.png'
        bpy.ops.render.render(write_still=True)

    

