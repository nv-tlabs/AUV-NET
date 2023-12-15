# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import cv2
import numpy as np
import os
import time
from multiprocessing import Process
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="aligned_textures", help="data folder")
FLAGS = parser.parse_args()

num_UV_segments = 2
data_dir = FLAGS.data_dir


def get_inpaint(useless,img_names):
	for i in range(len(img_names)):
		pid = img_names[i][0]
		idx = img_names[i][1]
		tname = img_names[i][2]

		for i in range(num_UV_segments):
			img_name = data_dir+"/"+tname+"/"+str(i)+".png"
			mask_name = data_dir+"/"+tname+"/"+str(i)+"_m.png"
			tmp_img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
			tmp_mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
			tmp_mask = (tmp_mask==0).astype(np.uint8)
			tmp_img_color = (tmp_img[:,:,0:3]).astype(np.uint8)
			tmp_img_alpha = (tmp_img[:,:,3:4]).astype(np.uint8)
			tmp_img_color = cv2.inpaint(tmp_img_color,tmp_mask,inpaintRadius=2,flags=cv2.INPAINT_TELEA)
			tmp_img_alpha = cv2.inpaint(tmp_img_alpha,tmp_mask,inpaintRadius=2,flags=cv2.INPAINT_TELEA)
			tmp_img_alpha_binary = (tmp_img_alpha>10).astype(np.uint8)*255
			tmp_img_alpha = tmp_img_alpha*(1-tmp_mask) + tmp_img_alpha_binary*tmp_mask
			tmp_img = np.concatenate([tmp_img_color,np.expand_dims(tmp_img_alpha,2)],2)
			cv2.imwrite(img_name, tmp_img)


if __name__ == '__main__':

	img_names = os.listdir(data_dir)
	img_names = [name for name in img_names if name[0]!="."]
	print(len(img_names))
	img_names = sorted(img_names)

	#prepare list of names
	num_of_process = 16
	list_of_list_of_names = []
	for i in range(num_of_process):
		list_of_list_of_names.append([])
	for idx in range(len(img_names)):
		process_id = idx%num_of_process
		list_of_list_of_names[process_id].append([process_id, idx, img_names[idx]])


	#map processes
	workers = []
	for i in range(num_of_process):
		list_of_names = list_of_list_of_names[i]
		workers.append(Process(target=get_inpaint, args = (i,list_of_names)))

	for p in workers:
		p.start()

	for p in workers:
		p.join()

	print("finished")