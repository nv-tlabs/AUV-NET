# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--phase", action="store", dest="phase", default=2, type=int, help="Training stage: 0-rigid 1-correspondence 2-smoothness")
parser.add_argument("--epoch", action="store", dest="epoch", default=10000, type=int, help="Epoch to train")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.00002, type=float, help="Learning rate for adam")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints")

#for auv-net
parser.add_argument("--data_dir", action="store", dest="data_dir", default="/local-scratch2/Zhiqin/02958343_simplified_textured/", help="Root directory of dataset")
parser.add_argument("--use_all_data", action="store_true", dest="use_all_data", default=False, help="True for using all shapes for training AUV-Net, otherwise the first 80% of the shapes are used for training")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="samples", help="Directory name to save the samples")
parser.add_argument("--point_batch_size", action="store", dest="point_batch_size", default=16384, type=int, help="point batch size [16384]")
parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training AUV-Net")
parser.add_argument("--test", action="store_true", dest="test", default=False, help="True for testing; will output all training shapes with high-quality aligned textures")

parser.add_argument("--gpu", action="store", dest="gpu", default="0", help="which GPU to use")
FLAGS = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu


import dataset
from model import AUV_NET

import torch

if not os.path.exists(FLAGS.sample_dir):
	os.makedirs(FLAGS.sample_dir)

if FLAGS.train:
	auv_net = AUV_NET(FLAGS)
	auv_dataset = dataset.point_color_voxel_dataset(FLAGS.data_dir, FLAGS.point_batch_size, train=(None if FLAGS.use_all_data else True))
	dataloader = torch.utils.data.DataLoader(auv_dataset, batch_size=1, shuffle=True, num_workers=16)
	auv_net.train(FLAGS,dataloader)

elif FLAGS.test:
	auv_net = AUV_NET(FLAGS)
	auv_net.test(FLAGS)

