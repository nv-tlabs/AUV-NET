

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import numpy as np
import cv2
import os

from utilities import voxel_utilities
import point_cloud_utilities_cy

def read_obj_with_uv(obj_dir,normalize):

    fin = open(obj_dir,'r')
    lines = fin.readlines()
    fin.close()
    
    vertices = []
    triangles = [] #vertex_index/texture_index/normal_index
    uv_vertices = []
    uv_triangles = []

    for i in range(len(lines)):
        line = lines[i].split()
        if len(line)==0:
            continue
        elif line[0] == 'v':
            vertices.append([float(line[1]),float(line[2]),float(line[3])])
        elif line[0] == 'vt':
            uv_vertices.append([float(line[1]), float(line[2])])
        elif line[0] == 'f':
            flen = len(line)-1
            for tid in range(flen-2): #deal with non-triangle polygons
                tmp_tri = [ int(line[1].split("/")[0]) ]
                tmp_uv = [ int(line[1].split("/")[1]) ]
                for j in range(2):
                    tmp_tri.append( int(line[2+tid+j].split("/")[0]) )
                    tmp_uv.append( int(line[2+tid+j].split("/")[1]) )
                triangles.append(tmp_tri)
                uv_triangles.append(tmp_uv)

    vertices = np.array(vertices, np.float32)
    triangles = np.array(triangles, np.int32)-1
    uv_vertices = np.array(uv_vertices, np.float32)
    uv_triangles = np.array(uv_triangles, np.int32)-1

    if normalize:
        #normalize diagonal=1
        x_max,y_max,z_max = np.max(vertices,0)
        x_min,y_min,z_min = np.min(vertices,0)
        x_mid,y_mid,z_mid = (x_max+x_min)/2,(y_max+y_min)/2,(z_max+z_min)/2
        x_scale,y_scale,z_scale = x_max-x_min,y_max-y_min,z_max-z_min
        scale = (x_scale*x_scale + y_scale*y_scale + z_scale*z_scale)**0.5
        vertices[:,0] = (vertices[:,0]-x_mid)/scale
        vertices[:,1] = (vertices[:,1]-y_mid)/scale
        vertices[:,2] = (vertices[:,2]-z_mid)/scale

    return vertices,triangles,uv_vertices,uv_triangles


def sample_points(obj_dir,texture_dir,num_of_points,exact_num=False,normalize=False):
    '''
    Sample points with normals and RGBA colors.
    '''

    vertices,triangles,uv_vertices,uv_triangles = read_obj_with_uv(obj_dir,normalize)

    #read texture png
    if not os.path.exists(texture_dir):
        texture_dir = texture_dir[:-4]+".jpg"
    texture_img_ = cv2.imread(texture_dir, cv2.IMREAD_UNCHANGED)
    if texture_img_.shape[2]==4:
        texture_img = texture_img_
    elif texture_img_.shape[2]==3:
        texture_img = np.full([texture_img_.shape[0],texture_img_.shape[1],4],255,np.uint8)
        texture_img[:,:,:3] = texture_img_
    else:
        texture_img = np.full([texture_img_.shape[0],texture_img_.shape[1],4],255,np.uint8)
        texture_img[:,:,0] = texture_img_
        texture_img[:,:,1] = texture_img_
        texture_img[:,:,2] = texture_img_
    texture_img = texture_img.astype(np.float32)


    triangle_area_list = np.zeros([len(triangles)],np.float32)
    triangle_normal_list = np.zeros([len(triangles),3],np.float32)
    sample_prob_list = np.zeros([len(triangles)],np.float32)
    random_numbers = np.random.random((len(triangles)*3+num_of_points*3,)).astype(np.float32)
    point_normal_color_list = np.zeros([num_of_points*3,10],np.float32)

    count = point_cloud_utilities_cy.sample_points(vertices, triangles, uv_vertices, uv_triangles, texture_img, triangle_area_list, triangle_normal_list, sample_prob_list, random_numbers, point_normal_color_list, num_of_points)
    if count<=0:
        print("infinite loop here!")
        exit(0)

    point_normal_color_list = point_normal_color_list[:count]

    if exact_num:
        np.random.shuffle(point_normal_color_list)
        point_normal_color_list = point_normal_color_list[:num_of_points]

    mesh_vertices = vertices
    mesh_triangles = triangles
    vertices = np.ascontiguousarray(point_normal_color_list[:,0:3], np.float32)
    normals = np.ascontiguousarray(point_normal_color_list[:,3:6], np.float32)
    colors = np.ascontiguousarray(point_normal_color_list[:,6:10], np.int32)

    return mesh_vertices,mesh_triangles,vertices,normals,colors


def sample_surface_points(obj_dir,texture_dir,voxel_dir,num_of_points,exact_num=False,normalize=False):
    '''
    Sample points with normals and RGBA colors, same to sample_points, but this function uses occupancy voxels in voxel_dir to determine whether a sampled point is on the surface or inside the shape; only surface points are sampled.
    '''

    vertices,triangles,uv_vertices,uv_triangles = read_obj_with_uv(obj_dir,normalize)

    #read texture png
    if not os.path.exists(texture_dir):
        texture_dir = texture_dir[:-4]+".jpg"
    texture_img_ = cv2.imread(texture_dir, cv2.IMREAD_UNCHANGED)
    if texture_img_.shape[2]==4:
        texture_img = texture_img_
    elif texture_img_.shape[2]==3:
        texture_img = np.full([texture_img_.shape[0],texture_img_.shape[1],4],255,np.uint8)
        texture_img[:,:,:3] = texture_img_
    else:
        texture_img = np.full([texture_img_.shape[0],texture_img_.shape[1],4],255,np.uint8)
        texture_img[:,:,0] = texture_img_
        texture_img[:,:,1] = texture_img_
        texture_img[:,:,2] = texture_img_
    texture_img = texture_img.astype(np.float32)

    #read filled voxel
    voxels = voxel_utilities.read_voxels(voxel_dir,fix_coords=True)


    triangle_area_list = np.zeros([len(triangles)],np.float32)
    triangle_normal_list = np.zeros([len(triangles),3],np.float32)
    sample_prob_list = np.zeros([len(triangles)],np.float32)
    random_numbers = np.random.random((len(triangles)*3+num_of_points*3,)).astype(np.float32)
    point_normal_color_list = np.zeros([num_of_points*3,10],np.float32)

    count = point_cloud_utilities_cy.sample_surface_points(vertices, triangles, uv_vertices, uv_triangles, voxels, texture_img, triangle_area_list, triangle_normal_list, sample_prob_list, random_numbers, point_normal_color_list, num_of_points)
    if count<=0:
        print("infinite loop here!")
        exit(0)

    point_normal_color_list = point_normal_color_list[:count]

    if exact_num:
        np.random.shuffle(point_normal_color_list)
        point_normal_color_list = point_normal_color_list[:num_of_points]

    mesh_vertices = vertices
    mesh_triangles = triangles
    vertices = np.ascontiguousarray(point_normal_color_list[:,0:3], np.float32)
    normals = np.ascontiguousarray(point_normal_color_list[:,3:6], np.float32)
    colors = np.ascontiguousarray(point_normal_color_list[:,6:10], np.int32)

    return mesh_vertices,mesh_triangles,vertices,normals,colors



def read_ply_point_normal_color(shape_name):
    file = open(shape_name,'r')
    lines = file.readlines()

    start = 0
    while True:
        line = lines[start].strip()
        if line == "end_header":
            start += 1
            break
        line = line.split()
        if line[0] == "element":
            if line[1] == "vertex":
                vertex_num = int(line[2])
        start += 1

    vertices = np.zeros([vertex_num,3], np.float32)
    normals = np.zeros([vertex_num,3], np.float32)
    colors = np.zeros([vertex_num,3], np.int32)
    for i in range(vertex_num):
        line = lines[i+start].split()
        vertices[i,0] = float(line[0])
        vertices[i,1] = float(line[1])
        vertices[i,2] = float(line[2])
        normals[i,0] = float(line[3])
        normals[i,1] = float(line[4])
        normals[i,2] = float(line[5])
        colors[i,0] = int(line[6])
        colors[i,1] = int(line[7])
        colors[i,2] = int(line[8])
    return vertices,normals,colors

def write_ply_point_color(output_dir,vertices,colors):
    fout = open(output_dir, 'w')
    fout.write( "ply\n" +
                "format ascii 1.0\n" +
                "element vertex "+str(len(vertices))+"\n" +
                "property float x\n" +
                "property float y\n" +
                "property float z\n" +
                "property uchar red\n" +
                "property uchar green\n" +
                "property uchar blue\n" +
                "end_header\n")
    for i in range(len(vertices)):
        fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+str(int(colors[i,0]))+" "+str(int(colors[i,1]))+" "+str(int(colors[i,2]))+"\n")
    fout.close()

def write_ply_point_normal_color(output_dir,vertices,normals,colors):
    fout = open(output_dir, 'w')
    fout.write( "ply\n" +
                "format ascii 1.0\n" +
                "element vertex "+str(len(vertices))+"\n" +
                "property float x\n" +
                "property float y\n" +
                "property float z\n" +
                "property float nx\n" +
                "property float ny\n" +
                "property float nz\n" +
                "property uchar red\n" +
                "property uchar green\n" +
                "property uchar blue\n" +
                "end_header\n")
    for i in range(len(vertices)):
        fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+str(normals[i,0])+" "+str(normals[i,1])+" "+str(normals[i,2])+" "+str(int(colors[i,0]))+" "+str(int(colors[i,1]))+" "+str(int(colors[i,2]))+"\n")
    fout.close()
