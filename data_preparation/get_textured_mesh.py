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

import sys
sys.path.append("../")

from utilities import voxel_utilities

parser = argparse.ArgumentParser()
parser.add_argument("class_id", type=str, help="shapenet category id")
parser.add_argument("share_id", type=int, help="id of the share [0]")
parser.add_argument("share_total", type=int, help="total num of shares [1]")
FLAGS = parser.parse_args()

class_id = FLAGS.class_id
if class_id!="02958343":
    print("ERROR: this script is designed for cars (02958343)")
    exit(-1)
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


output_dir = "./"+class_id+"_simplified_textured/"

for idx in range(len(obj_names)):
    in_folder = target_dir + obj_names[idx] +"/"
    out_folder = output_dir + obj_names[idx] +"/"
    obj_file_name = "model_simplified_textured.obj"
    obj_png_name = "model_simplified_textured.png"
    obj_mtl_name = "model_simplified_textured.mtl"
    print(idx,len(obj_names),in_folder)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if os.path.exists( out_folder+obj_file_name ) and os.path.exists( out_folder+obj_png_name ) and os.path.exists( out_folder+obj_mtl_name ):
        print("model exists. skip.")
        continue

    # check the rendered images
    valid_input = True
    for i in range(16):
        if not os.path.exists( in_folder+str(i)+".png" ):
            valid_input = False
    if not valid_input:
        print("rendering incomplete. skip.")
        continue


    #additional camera poses
    poses_file_name = in_folder+'rendering_poses.txt'
    fin = open(poses_file_name, 'r')
    x_cut, y_cut, z_cut_1, z_cut_2 = [float(i) for i in fin.readline().split()]
    fin.close()

    voxel_name = in_folder+"model_coarse.binvox"
    batch_voxels = voxel_utilities.read_voxels(voxel_name)

    x_cut_int = round((x_cut+0.5)*1024)
    y_cut_int = round((y_cut+0.5)*1024)
    z_cut_1_int = round((z_cut_1+0.5)*1024)
    z_cut_2_int = round((z_cut_2+0.5)*1024)

    texture_image = np.zeros([1024*4,1024*4,4], np.uint8)

    for i in range(16):
        h_id = i//4
        w_id = i%4
        img_in = cv2.imread(in_folder+str(i)+".png", cv2.IMREAD_UNCHANGED)
        if i in [1,3,5,6,7,9,10,11,13,15]:
            if i==1:
                mask = np.minimum(batch_voxels[:,y_cut_int-1,:], batch_voxels[:,y_cut_int,:])
                mask = np.transpose(mask)
            elif i==3:
                mask = np.minimum(batch_voxels[:,y_cut_int-1,:], batch_voxels[:,y_cut_int,:])
                mask = np.transpose(mask)[::-1]
            elif i==5:
                mask = np.minimum(batch_voxels[:,:,z_cut_2_int-1], batch_voxels[:,:,z_cut_2_int])
                mask = np.transpose(mask)[::-1]
            elif i==6:
                mask = np.minimum(batch_voxels[:,:,512-1], batch_voxels[:,:,512])
                mask = np.transpose(mask)[::-1]
            elif i==7:
                mask = np.minimum(batch_voxels[:,:,z_cut_1_int-1], batch_voxels[:,:,z_cut_1_int])
                mask = np.transpose(mask)[::-1]
            elif i==9:
                mask = np.minimum(batch_voxels[:,:,z_cut_1_int-1], batch_voxels[:,:,z_cut_1_int])
                mask = np.transpose(mask)[::-1,::-1]
            elif i==10:
                mask = np.minimum(batch_voxels[:,:,512-1], batch_voxels[:,:,512])
                mask = np.transpose(mask)[::-1,::-1]
            elif i==11:
                mask = np.minimum(batch_voxels[:,:,z_cut_2_int-1], batch_voxels[:,:,z_cut_2_int])
                mask = np.transpose(mask)[::-1,::-1]
            elif i==13:
                mask = np.minimum(batch_voxels[x_cut_int-1,:,:], batch_voxels[x_cut_int,:,:])
                mask = mask[::-1,::-1]
            elif i==15:
                mask = np.minimum(batch_voxels[x_cut_int-1,:,:], batch_voxels[x_cut_int,:,:])
                mask = mask[::-1]
            img_in_color = np.ascontiguousarray(img_in[:,:,:3])
            mask = mask | (img_in[:,:,3]==0)
            img_in_color = cv2.inpaint(img_in_color,mask,inpaintRadius=2,flags=cv2.INPAINT_TELEA)
            img_in[:,:,:3] = img_in_color
        texture_image[h_id*1024:(h_id+1)*1024,w_id*1024:(w_id+1)*1024] = img_in


    mask = (texture_image[:,:,3:4]!=0)
    texture_image = texture_image*mask
    cv2.imwrite(out_folder+obj_png_name,texture_image)



    #save material
    fout = open(out_folder+obj_mtl_name, 'w')
    fout.write("newmtl m16\n")
    fout.write("Ka 0.200000 0.200000 0.200000\n")
    fout.write("Kd 1.000000 1.000000 1.000000\n")
    fout.write("Ks 1.000000 1.000000 1.000000\n")
    fout.write("Ke 0.000000 0.000000 0.000000\n")
    fout.write("Tr 1.000000\n")
    fout.write("Ns 0.000000\n")
    fout.write("illum 2\n")
    fout.write("map_Kd "+obj_png_name+"\n")
    fout.write("\n")
    fout.close()


    fin = open(in_folder+"model_simplified.obj",'r')
    lines = fin.readlines()
    fin.close()

    vertices = []
    triangles = []
    for i in range(len(lines)):
        line = lines[i].split()
        if len(line)==0:
            continue
        elif line[0] == 'v':
            vertices.append([float(line[1]),float(line[2]),float(line[3])])
        elif line[0] == 'f':
            triangles.append([int(line[1].split("/")[0]),int(line[2].split("/")[0]),int(line[3].split("/")[0])])
    vertices = np.array(vertices, np.float32)
    triangles = np.array(triangles, np.int32)-1


    epsilon = 1e-10
    voffsetepsilon = 0.5/1024
    triangle_texture_list = np.zeros([len(triangles)],np.int32)
    for i in range(len(triangles)):
        #area = |u x v|/2 = |u||v|sin(uv)/2
        a,b,c = vertices[triangles[i,1]]-vertices[triangles[i,0]]
        x,y,z = vertices[triangles[i,2]]-vertices[triangles[i,0]]
        ti,tj,tk = b*z-c*y,c*x-a*z,a*y-b*x
        area2 = (ti*ti+tj*tj+tk*tk)**0.5
        if area2<epsilon:
            triangle_texture_list[i] = 0
        else:
            #add normal preference here
            tj = tj*1.1
            if abs(ti)>abs(tj) and abs(ti)>abs(tk):
                if ti>0: #front
                    mean_pos = max(vertices[triangles[i,0],0],vertices[triangles[i,1],0],vertices[triangles[i,2],0])-voffsetepsilon
                    if mean_pos>x_cut:
                        triangle_texture_list[i] = 12
                    else:
                        triangle_texture_list[i] = 13
                else: #back
                    mean_pos = min(vertices[triangles[i,0],0],vertices[triangles[i,1],0],vertices[triangles[i,2],0])+voffsetepsilon
                    if mean_pos<x_cut:
                        triangle_texture_list[i] = 14
                    else:
                        triangle_texture_list[i] = 15
            elif abs(tj)>abs(ti) and abs(tj)>abs(tk):
                if tj>0: #up
                    mean_pos = max(vertices[triangles[i,0],1],vertices[triangles[i,1],1],vertices[triangles[i,2],1])-voffsetepsilon
                    if mean_pos>y_cut:
                        triangle_texture_list[i] = 0
                    else:
                        triangle_texture_list[i] = 1
                else: #down
                    mean_pos = min(vertices[triangles[i,0],1],vertices[triangles[i,1],1],vertices[triangles[i,2],1])+voffsetepsilon
                    if mean_pos<y_cut:
                        triangle_texture_list[i] = 2
                    else:
                        triangle_texture_list[i] = 3
            else:
                if tk>0: #left
                    mean_pos = max(vertices[triangles[i,0],2],vertices[triangles[i,1],2],vertices[triangles[i,2],2])-voffsetepsilon
                    if mean_pos>z_cut_2:
                        triangle_texture_list[i] = 4
                    elif mean_pos>0:
                        triangle_texture_list[i] = 5
                    elif mean_pos>z_cut_1:
                        triangle_texture_list[i] = 6
                    else:
                        triangle_texture_list[i] = 7
                else: #right
                    mean_pos = min(vertices[triangles[i,0],2],vertices[triangles[i,1],2],vertices[triangles[i,2],2])+voffsetepsilon
                    if mean_pos<z_cut_1:
                        triangle_texture_list[i] = 8
                    elif mean_pos<0:
                        triangle_texture_list[i] = 9
                    elif mean_pos<z_cut_2:
                        triangle_texture_list[i] = 10
                    else:
                        triangle_texture_list[i] = 11


    #save mesh v vt f
    fout = open(out_folder+obj_file_name, 'w')
    fout.write("mtllib "+obj_mtl_name+"\n")
    fout.write("usemtl m16\n")
    for j in range(len(vertices)):
        fout.write("v "+str(vertices[j,0])+" "+str(vertices[j,1])+" "+str(vertices[j,2])+"\n")
    vt_count = 0
    for j in range(len(triangles)):
        i = triangle_texture_list[j]
        h_id = 3-i//4
        w_id = i%4
        tmp_uv = []
        for k in range(3):
            if i in [0,1]: #up
                vt_x = vertices[triangles[j,k], 0] + 0.5
                vt_y = 0.5 - vertices[triangles[j,k], 2]
            elif i in [2,3]: #down
                vt_x = vertices[triangles[j,k], 0] + 0.5
                vt_y = vertices[triangles[j,k], 2] + 0.5
            elif i in [4,5,6,7]: #right
                vt_x = vertices[triangles[j,k], 0] + 0.5
                vt_y = vertices[triangles[j,k], 1] + 0.5
            elif i in [8,9,10,11]: #left
                vt_x = 0.5 - vertices[triangles[j,k], 0]
                vt_y = vertices[triangles[j,k], 1] + 0.5
            elif i in [12,13]: #front
                vt_x = 0.5 - vertices[triangles[j,k], 2]
                vt_y = vertices[triangles[j,k], 1] + 0.5
            elif i in [14,15]: #back
                vt_x = vertices[triangles[j,k], 2] + 0.5
                vt_y = vertices[triangles[j,k], 1] + 0.5
            else:
                print("ERROR")
                exit(-1)
            if vt_x>=1: vt_x = 0.9999
            if vt_x<=0: vt_x = 0.0001
            if vt_y>=1: vt_y = 0.9999
            if vt_y<=0: vt_y = 0.0001
            fout.write("vt "+str((vt_x+w_id)/4.0)+" "+str((vt_y+h_id)/4.0)+"\n")
            tmp_uv.append(vt_count)
            vt_count += 1
        fout.write("f")
        for k in range(3):
            fout.write(" "+str(triangles[j,k]+1)+"/"+str(tmp_uv[k]+1))
        fout.write("\n")
    fout.close()


