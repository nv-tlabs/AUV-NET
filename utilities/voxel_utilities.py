

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import numpy as np
import struct
import cutils

def read_header(fp):
    line = fp.readline().strip()
    dims = [int(i) for i in fp.readline().strip().split(b' ')[1:]]
    fp.readline() #omit translate
    fp.readline() #omit scale
    fp.readline() #omit "data\n"
    return dims

def read_voxels(filename, fix_coords=True):
    '''
    Read voxels as 3D uint8 numpy array from a binvox file.
    If fix_coords then i->x, j->y, k->z; otherwise i->x, j->z, k->y.
    '''
    fp = open(filename, 'rb')
    dims = read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    fp.close()
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(bool)
    data = data.reshape(dims).astype(np.uint8)
    if fix_coords:
        data = np.ascontiguousarray(np.transpose(data, (0, 2, 1)))
    return data

def bwrite(fp,s):
    fp.write(s.encode())

def write_pair(fp,state,ctr):
    fp.write(struct.pack('B',state))
    fp.write(struct.pack('B',ctr))

def write(filename, voxel_size, encoding):
    '''
    Write voxels into a binvox file.
    It uses precomputed run-length encoding results (encoding).
    '''
    fp = open(filename, 'wb')
    bwrite(fp,'#binvox 1\n')
    bwrite(fp,'dim '+str(voxel_size[0])+' '+str(voxel_size[1])+' '+str(voxel_size[2])+'\n')
    bwrite(fp,'translate 0 0 0\nscale 1\ndata\n')
    c = 0
    while encoding[c,0]!=2:
        write_pair(fp, encoding[c,0], encoding[c,1])
        c += 1
    fp.close()




### voxel processing

def depth_fusion(raw_voxels):
    rendering = np.full([5120,5120,6], 65536, np.int32)
    cutils.depth_fusion_XZY_5views(raw_voxels, rendering)


def alpha_hull(raw_voxels):
    voxel_size = 1024
    cube_sizex = 32
    cube_sizey = 32
    cube_sizez = 80
    padding_size = 2

    tmp_voxel1 = np.zeros([voxel_size+cube_sizex*2,voxel_size+cube_sizey*2,voxel_size+cube_sizez*2], np.uint8)
    tmp_accu1 = np.zeros([voxel_size+cube_sizex*2,voxel_size+cube_sizey*2,voxel_size+cube_sizez*2], np.int32)
    tmp_voxel1[cube_sizex:-cube_sizex,cube_sizey:-cube_sizey,cube_sizez:-cube_sizez] = raw_voxels
    cutils.cube_alpha_hull(tmp_voxel1,tmp_accu1,cube_sizex,cube_sizey,cube_sizez)
    alpha_hull = np.ascontiguousarray(tmp_voxel1[cube_sizex:-cube_sizex,cube_sizey:-cube_sizey,cube_sizez:-cube_sizez])
    del tmp_voxel1
    del tmp_accu1


    tmp_voxel1 = np.zeros([voxel_size+padding_size*2,voxel_size+padding_size*2,voxel_size+padding_size*2], np.uint8)
    tmp_accu1 = np.zeros([voxel_size+padding_size*2,voxel_size+padding_size*2,voxel_size+padding_size*2], np.int32)
    tmp_voxel2 = np.zeros([voxel_size+padding_size*2,voxel_size+padding_size*2,voxel_size+padding_size*2], np.uint8)
    tmp_accu2 = np.zeros([voxel_size+padding_size*2,voxel_size+padding_size*2,voxel_size+padding_size*2], np.int32)
    encoding = np.zeros([voxel_size*voxel_size*64,2], np.int32)

    #X-
    cutils.get_transpose(tmp_voxel1,alpha_hull,padding_size,0,0)
    cutils.get_transpose(tmp_voxel2,raw_voxels,padding_size,0,0)
    cutils.boundary_cull(tmp_voxel1,tmp_accu1,tmp_voxel2,tmp_accu2,encoding)
    cutils.recover_transpose(tmp_voxel1,alpha_hull,padding_size,0,0)
    #X+
    cutils.get_transpose(tmp_voxel1,alpha_hull,padding_size,0,1)
    cutils.get_transpose(tmp_voxel2,raw_voxels,padding_size,0,1)
    cutils.boundary_cull(tmp_voxel1,tmp_accu1,tmp_voxel2,tmp_accu2,encoding)
    cutils.recover_transpose(tmp_voxel1,alpha_hull,padding_size,0,1)
    #Y-
    cutils.get_transpose(tmp_voxel1,alpha_hull,padding_size,1,0)
    cutils.get_transpose(tmp_voxel2,raw_voxels,padding_size,1,0)
    cutils.boundary_cull(tmp_voxel1,tmp_accu1,tmp_voxel2,tmp_accu2,encoding)
    cutils.recover_transpose(tmp_voxel1,alpha_hull,padding_size,1,0)
    #Y+
    cutils.get_transpose(tmp_voxel1,alpha_hull,padding_size,1,1)
    cutils.get_transpose(tmp_voxel2,raw_voxels,padding_size,1,1)
    cutils.boundary_cull(tmp_voxel1,tmp_accu1,tmp_voxel2,tmp_accu2,encoding)
    cutils.recover_transpose(tmp_voxel1,alpha_hull,padding_size,1,1)
    #Z-
    cutils.get_transpose(tmp_voxel1,alpha_hull,padding_size,2,0)
    cutils.get_transpose(tmp_voxel2,raw_voxels,padding_size,2,0)
    cutils.boundary_cull(tmp_voxel1,tmp_accu1,tmp_voxel2,tmp_accu2,encoding)
    cutils.recover_transpose(tmp_voxel1,alpha_hull,padding_size,2,0)

    del tmp_voxel1
    del tmp_accu1
    del tmp_voxel2
    del tmp_accu2
    del raw_voxels

    cutils.get_run_length_encoding(alpha_hull,encoding)
    del alpha_hull
    
    return encoding


def get_occlusion_cut(simplified_voxels,poses_file_name):
    ray_x1 = np.zeros([1024], np.int32)
    ray_y1 = np.zeros([1024], np.int32)
    ray_z1 = np.zeros([1024], np.int32)
    ray_x2 = np.zeros([1024], np.int32)
    ray_y2 = np.zeros([1024], np.int32)
    ray_z2 = np.zeros([1024], np.int32)
    visibility_flag = np.ones([1024,1024,1024], np.uint8)

    cutils.get_rays(simplified_voxels, ray_x1, ray_y1, ray_z1, ray_x2, ray_y2, ray_z2, visibility_flag)
    del visibility_flag

    #get balance point

    optimal_idx = 0
    optimal_sum = 0
    current_pos = 0
    current_neg = np.sum(ray_x1)
    for i in range(1024):
        current_pos += ray_x2[i]
        current_neg -= ray_x1[i]
        current_sum = current_pos+current_neg
        if current_sum>optimal_sum:
            optimal_sum = current_sum
            optimal_idx = i
    x_cut = optimal_idx+1
    
    optimal_idx = 0
    optimal_sum = 0
    current_pos = 0
    current_neg = np.sum(ray_y1)
    for i in range(1024):
        current_pos += ray_y2[i]
        current_neg -= ray_y1[i]
        current_sum = current_pos+current_neg
        if current_sum>optimal_sum:
            optimal_sum = current_sum
            optimal_idx = i
    y_cut = optimal_idx+1
    
    optimal_idx = 0
    optimal_sum = 0
    current_pos = 0
    current_neg = np.sum(ray_z1)
    for i in range(512):
        current_pos += ray_z2[i]
        current_neg -= ray_z1[i]
        current_sum = current_pos+current_neg
        if current_sum>optimal_sum:
            optimal_sum = current_sum
            optimal_idx = i
    z_cut_1 = optimal_idx+1
    
    optimal_idx = 0
    optimal_sum = 0
    current_pos = 0
    current_neg = np.sum(ray_z1)
    for i in range(512,1024):
        current_pos += ray_z2[i]
        current_neg -= ray_z1[i]
        current_sum = current_pos+current_neg
        if current_sum>optimal_sum:
            optimal_sum = current_sum
            optimal_idx = i
    z_cut_2 = optimal_idx+1


    
    fout = open(poses_file_name, 'w')
    fout.write(str(x_cut/1024.0-0.5)+" ")
    fout.write(str(y_cut/1024.0-0.5)+" ")
    fout.write(str(z_cut_1/1024.0-0.5)+" ")
    fout.write(str(z_cut_2/1024.0-0.5)+" ")
    fout.close()