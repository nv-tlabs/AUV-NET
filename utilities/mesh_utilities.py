# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import numpy as np

def normalize_obj(this_name, out_name):
    fin = open(this_name,'r')
    lines = fin.readlines()
    fin.close()
    
    #read shape
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
    
    #remove isolated points
    vertices_used_flag = np.full([len(vertices)], 0, np.int32)
    vertices_used_flag[np.reshape(triangles,[-1])] = 1
    vertices_ = vertices[vertices_used_flag>0]

    #normalize max=1
    x_max,y_max,z_max = np.max(vertices_,0)
    x_min,y_min,z_min = np.min(vertices_,0)
    x_mid,y_mid,z_mid = (x_max+x_min)/2,(y_max+y_min)/2,(z_max+z_min)/2
    x_scale,y_scale,z_scale = x_max-x_min,y_max-y_min,z_max-z_min
    scale = max(x_scale,y_scale,z_scale)

    #write normalized shape
    fout = open(out_name, 'w')
    for i in range(len(lines)):
        line = lines[i].split()
        if len(line)==0:
            continue
        elif line[0] == 'v':
            x = (float(line[1])-x_mid)/scale
            y = (float(line[2])-y_mid)/scale
            z = (float(line[3])-z_mid)/scale
            fout.write("v "+str(x)+" "+str(y)+" "+str(z)+"\n")
        else:
            fout.write(lines[i])
    fout.close()


def write_ply_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    fout.write( "ply\n" +
                "format ascii 1.0\n" +
                "element vertex "+str(len(vertices))+"\n" +
                "property float x\n" +
                "property float y\n" +
                "property float z\n" +
                "element face "+str(len(triangles))+"\n" +
                "property list uchar int vertex_index\n" +
                "end_header\n")
    for i in range(len(vertices)):
        fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+"\n")
    for i in range(len(triangles)):
        fout.write("3 "+str(triangles[i,0])+" "+str(triangles[i,1])+" "+str(triangles[i,2])+"\n")
    fout.close()


def write_ply_triangle_color(name, vertices, colors, triangles):
    fout = open(name, 'w')
    fout.write( "ply\n" +
                "format ascii 1.0\n" +
                "element vertex "+str(len(vertices))+"\n" +
                "property float x\n" +
                "property float y\n" +
                "property float z\n" +
                "property uchar red\n" +
                "property uchar green\n" +
                "property uchar blue\n" +
                "element face "+str(len(triangles))+"\n" +
                "property list uchar int vertex_index\n" +
                "end_header\n")
    for i in range(len(vertices)):
        fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+str(int(colors[i,0]))+" "+str(int(colors[i,1]))+" "+str(int(colors[i,2]))+"\n")
    for i in range(len(triangles)):
        fout.write("3 "+str(triangles[i,0])+" "+str(triangles[i,1])+" "+str(triangles[i,2])+"\n")
    fout.close()


def write_ply_triangle_UV(name, mesh_vertices, output_uv_mask, output_uv, mesh_triangles):
    #save material
    fout = open(name+".mtl", 'w')
    for i in range(len(output_uv_mask)):
        fout.write(	"newmtl m"+str(i)+"\n" +
                    "Ns 18.000005\n" +
                    "Ka 1.000000 1.000000 1.000000\n" +
                    "Kd 1.000000 1.000000 1.000000\n" +
                    "Ks 1.000000 1.000000 1.000000\n" +
                    "Ke 0.000000 0.000000 0.000000\n" +
                    "Ni 1.450000\n" +
                    "d 1.000000\n" +
                    "illum 2\n" +
                    "map_Kd "+str(i)+".png\n" +
                    "\n")
    fout.close()

    #save mesh v vt f
    fout = open(name+".obj", 'w')
    fout.write("mtllib model.mtl\n")
    for j in range(len(mesh_vertices)):
        fout.write("v "+str(mesh_vertices[j,0])+" "+str(mesh_vertices[j,1])+" "+str(mesh_vertices[j,2])+"\n")
    for j in range(len(mesh_vertices)):
        fout.write("vt "+str(output_uv[j,1])+" "+str(1-output_uv[j,0])+"\n")
    for i in range(len(output_uv_mask)):
        fout.write("usemtl m"+str(i)+"\n")
        for j in range(len(mesh_triangles)):
            if output_uv_mask[i][j]:
                fout.write("f")
                for k in range(3):
                    fout.write(" "+str(mesh_triangles[j,k]+1)+"/"+str(mesh_triangles[j,k]+1))
                fout.write("\n")
    fout.close()

