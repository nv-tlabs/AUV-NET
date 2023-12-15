# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="aligned_textures", help="data folder")
parser.add_argument("--out_dir", type=str, default="hybrid", help="output folder")
FLAGS = parser.parse_args()

num_UV_segments = 2
read_dir = FLAGS.data_dir+"/"
write_dir = FLAGS.out_dir+"/"
if not os.path.exists(write_dir):
    os.makedirs(write_dir)


#modify the list to view other shapes
row_idx = [0,1,2,5,7,11,12,15]
col_idx = [0,1,2,5,7,11,12,15]

offset_row = 1.2
offset_col = 1.2


#save material
fout = open(write_dir+"model.mtl", 'w')
for j in range(len(col_idx)):
    for i in range(num_UV_segments):
        fout.write( "newmtl col"+str(j)+"_"+str(i)+"\n" +
        			"Ns 18.000005\n" +
        			"Ka 1.000000 1.000000 1.000000\n" +
        			"Kd 1.000000 1.000000 1.000000\n" +
        			"Ks 1.000000 1.000000 1.000000\n" +
        			"Ke 0.000000 0.000000 0.000000\n" +
        			"Ni 1.450000\n" +
        			"d 1.000000\n" +
        			"illum 2\n" +
        			"map_Kd col"+str(i)+"_"+str(j)+".png\n" +
        			"\n")
        os.system("cp "+read_dir+str(col_idx[j])+"/"+str(i)+".png "+write_dir+"col"+str(i)+"_"+str(j)+".png")
fout.close()



#save mesh v vt f
fout = open(write_dir+"model.obj", 'w')
fout.write("mtllib model.mtl\n")

total_v_counter = 0
total_vt_counter = 0
for this_row in range(len(row_idx)):
    for this_col in range(len(col_idx)):
        if this_row==-1:
            shape_idx = col_idx[this_col]
        else:
            shape_idx = row_idx[this_row]

        fin = open(read_dir+str(shape_idx)+"/model.obj", 'r')
        shape_v_counter = 0
        shape_vt_counter = 0

        for line in fin.readlines():
            line_split = line.split()
            if len(line_split)==0: continue
            if line_split[0] == "mtllib": continue
            
            if line_split[0] == "v":
                x = float(line_split[1]) - offset_row*this_row
                y = float(line_split[2])
                z = float(line_split[3]) - offset_col*this_col
                fout.write("v "+str(x)+" "+str(y)+" "+str(z)+"\n")
                shape_v_counter += 1

            if line_split[0] == "vt":
                fout.write(line)
                shape_vt_counter += 1

            if line_split[0] == "f":
                fout.write("f")
                for k in range(1,len(line_split)):
                    line_k = line_split[k]
                    tri_id = int(line_k.split("/")[0])
                    tex_id = int(line_k.split("/")[1])
                    fout.write(" "+str(tri_id+total_v_counter)+"/"+str(tex_id+total_vt_counter))
                fout.write("\n")

            if line_split[0] == "usemtl":
                mid = int(line_split[1][1:])
                fout.write("usemtl col"+str(this_col)+"_"+str(mid)+"\n")
            
        fin.close()
        total_v_counter += shape_v_counter
        total_vt_counter += shape_vt_counter

fout.close()


