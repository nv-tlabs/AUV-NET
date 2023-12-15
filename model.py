# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import time
import numpy as np
import h5py
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities import point_cloud_utilities
from utilities import mesh_utilities
import point_cloud_utilities_cy


class generator(nn.Module):
    def __init__(self, gf_dim, channel_num, image_channel):
        super(generator, self).__init__()
        self.point_dim = 2
        self.gf_dim = gf_dim
        self.channel_num = channel_num
        self.image_channel = image_channel

        self.linear_1 = nn.Linear(self.point_dim, self.gf_dim, bias=True)
        self.linear_2 = nn.Linear(self.gf_dim+self.point_dim, self.gf_dim, bias=True)
        self.linear_3 = nn.Linear(self.gf_dim+self.point_dim, self.gf_dim, bias=True)
        self.linear_4 = nn.Linear(self.gf_dim+self.point_dim, self.gf_dim, bias=True)
        self.linear_5 = nn.Linear(self.gf_dim, self.gf_dim, bias=True)
        self.linear_6 = nn.Linear(self.gf_dim, self.gf_dim, bias=True)
        self.linear_7 = nn.Linear(self.gf_dim, self.channel_num, bias=True)

    def forward(self, points, z):
        out = points

        out = self.linear_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([points,out],2)

        out = self.linear_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([points,out],2)

        out = self.linear_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = torch.cat([points,out],2)

        out = self.linear_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.linear_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.linear_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.linear_7(out)

        c_ = z.view(1,self.channel_num,self.image_channel)
        out = torch.matmul(out,c_)

        out = torch.sigmoid(out)

        return out

#masker for shapes with 2 texture generators
class masker(nn.Module):
    def __init__(self, z_dim):
        super(masker, self).__init__()
        self.z_dim = z_dim
        self.point_dim = 3
        self.gf_dim = 512

        self.linear_1 = nn.Linear(self.z_dim+self.point_dim+self.point_dim, self.gf_dim, bias=True)
        self.linear_2 = nn.Linear(self.gf_dim, self.gf_dim, bias=True)
        self.linear_3 = nn.Linear(self.gf_dim, self.gf_dim, bias=True)
        self.linear_4 = nn.Linear(self.gf_dim, 1, bias=True)
        self.linear_4.bias.data[:] = 0

    def forward(self, points, z, normals):
        zs = z.view(1,1,self.z_dim).repeat(1,points.size()[1],1)
        pointz = torch.cat([points,normals,zs],2)

        out = pointz

        out = self.linear_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.linear_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.linear_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.linear_4(out)
        out = torch.sigmoid(out)

        return out

class UV_mapper(nn.Module):
    def __init__(self, z_dim):
        super(UV_mapper, self).__init__()
        self.z_dim = z_dim
        self.point_dim = 3
        self.gf_dim = 1024

        self.linear_1 = nn.Linear(self.z_dim+self.point_dim, self.gf_dim, bias=True)
        self.linear_2 = nn.Linear(self.gf_dim, self.gf_dim, bias=True)
        self.linear_3 = nn.Linear(self.gf_dim, self.gf_dim, bias=True)
        self.linear_4 = nn.Linear(self.gf_dim, 2, bias=True)

    def forward(self, points, z):
        zs = z.view(1,1,self.z_dim).repeat(1,points.size()[1],1)
        pointz = torch.cat([points,zs],2)

        out = pointz

        out = self.linear_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.linear_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.linear_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.linear_4(out)

        return out


class encoder(nn.Module):
    def __init__(self, z_dim, coefficients_dim):
        super(encoder, self).__init__()
        self.ef_dim = 32
        self.z_dim = z_dim
        self.coefficients_dim = coefficients_dim

        self.conv_1 = nn.Conv3d(5, self.ef_dim, 4, stride=2, padding=1, bias=False)
        self.norm_1 = nn.InstanceNorm3d(self.ef_dim)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 4, stride=2, padding=1, bias=False)
        self.norm_2 = nn.InstanceNorm3d(self.ef_dim*2)
        self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=False)
        self.norm_3 = nn.InstanceNorm3d(self.ef_dim*4)
        self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=False)
        self.norm_4 = nn.InstanceNorm3d(self.ef_dim*8)
        self.conv_5 = nn.Conv3d(self.ef_dim*8, self.ef_dim*16, 4, stride=1, padding=0, bias=True)

        self.conv_out1 = nn.Conv3d(self.ef_dim*16, self.coefficients_dim, 1, stride=1, padding=0, bias=True)
        self.conv_out2 = nn.Conv3d(self.ef_dim*16, self.z_dim, 1, stride=1, padding=0, bias=True)

    def forward(self, inputs):
        out = inputs
        out = F.leaky_relu(self.norm_1(self.conv_1(out)), negative_slope=0.02, inplace=True)
        out = F.leaky_relu(self.norm_2(self.conv_2(out)), negative_slope=0.02, inplace=True)
        out = F.leaky_relu(self.norm_3(self.conv_3(out)), negative_slope=0.02, inplace=True)
        out = F.leaky_relu(self.norm_4(self.conv_4(out)), negative_slope=0.02, inplace=True)
        out = F.leaky_relu(self.conv_5(out), negative_slope=0.02, inplace=True)

        out1 = self.conv_out1(out)
        out1 = out1.view(1, self.coefficients_dim)

        out2 = self.conv_out2(out)
        out2 = out2.view(1, self.z_dim)
        out2 = torch.sigmoid(out2)

        return out1, out2


class auv_network(nn.Module):
    def __init__(self, num_UV_segments, image_channel, z_dim, generator_gf_dim, generator_channel_num, channel_num_indices):
        super(auv_network, self).__init__()
        self.num_UV_segments = num_UV_segments
        self.image_channel = image_channel
        self.z_dim = z_dim
        self.generator_gf_dim = generator_gf_dim
        self.generator_channel_num = generator_channel_num
        self.channel_num_indices = channel_num_indices
        self.coefficients_dim = channel_num_indices[-1]

        self.encoder = encoder(self.z_dim,self.coefficients_dim)
        self.masker = masker(self.z_dim)
        self.UV_mapper = UV_mapper(self.z_dim)

        self.generator = nn.ModuleList()
        for i in range(self.num_UV_segments):
            self.generator.append(generator(self.generator_gf_dim[i],self.generator_channel_num[i],self.image_channel))

    def forward(self, inputs, point_coord, point_normal, texture_coords):

        t_vector, d_vector = self.encoder(inputs)

        texture_out = None
        if texture_coords is not None:
            texture_out = []
            for i in range(self.num_UV_segments):
                texture_out.append( self.generator[i](texture_coords, t_vector[:,self.channel_num_indices[i]:self.channel_num_indices[i+1]]) )

        mask_out = self.masker(point_coord, d_vector, point_normal)
        UV_coord = self.UV_mapper(point_coord, d_vector)

        net_out = []
        for i in range(self.num_UV_segments):
            net_out.append( self.generator[i](UV_coord, t_vector[:,self.channel_num_indices[i]:self.channel_num_indices[i+1]]) )

        return t_vector, d_vector, UV_coord, texture_out, net_out, mask_out


class AUV_NET(object):
    def __init__(self, config):
        self.shape_batch_size = 1
        self.point_batch_size = config.point_batch_size
        self.num_UV_segments = 2 #do not change! The code is hard-coded to handle shapes with 2 UV segments.
        self.image_channel = 9 #color (3), normal (3), 3D coord (3)
        self.z_dim = 256 #shape latent code size

        #networks depend on number of texture images
        self.generator_gf_dim = [1024,128]
        self.generator_channel_num = [64,16]


        self.channel_num_indices = [0] #sizes and splits of the predicted coefficients
        for i in range(self.num_UV_segments):
            self.channel_num_indices.append(self.channel_num_indices[-1]+self.generator_channel_num[i]*self.image_channel)


        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            print("ERROR: GPU not available!!!")
            exit(-1)

        #build model
        self.auv_network = auv_network(self.num_UV_segments,self.image_channel,self.z_dim,self.generator_gf_dim,self.generator_channel_num,self.channel_num_indices)
        self.auv_network.to(self.device)

        #pytorch does not have a checkpoint manager
        #have to define it myself to manage max num of checkpoints to keep
        self.model_dir = "ae"
        self.max_to_keep = 8
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        self.checkpoint_name='AUV_NET.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0

        #get coordinates for visualizing learned textures during training
        self.texture_coords_scale = 2 #rescale texture images
        self.texture_image_size = 256 #resolution of sample texture images
        dima = self.texture_image_size
        self.texture_coords = np.zeros([dima,dima,2],np.float32)
        for i in range(dima):
            for j in range(dima):
                self.texture_coords[i,j,0] = i
                self.texture_coords[i,j,1] = j
        self.texture_coords = (self.texture_coords+0.5)/dima-0.5
        self.texture_coords = self.texture_coords*self.texture_coords_scale
        self.texture_coords = np.reshape(self.texture_coords,[1,dima*dima,2])
        self.texture_coords = torch.from_numpy(self.texture_coords)
        self.texture_coords = self.texture_coords.to(self.device)


    def train(self, config, dataloader_train):
        #load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            with open(checkpoint_txt) as fin:
                model_dir = fin.readline().strip()
            self.auv_network.load_state_dict(torch.load(model_dir))
            start_epoch = int(model_dir.split('-')[-1].split('.')[0])+1
            print(" [*] Load SUCCESS", start_epoch)
        else:
            print(" [!] No checkpoint detected. Training from scratch...")
            start_epoch = 0

        self.optimizer = torch.optim.Adam(self.auv_network.parameters(), lr=config.learning_rate)

        #colors for visualizing segmentations
        seg_colors = [ [255,0,0],[0,255,0],[0,0,255], [255,255,0],[255,0,255],[0,255,255] ]
        seg_colors = np.array(seg_colors, np.uint8)


        start_time = time.time()
        for epoch in range(start_epoch, config.epoch):
            self.auv_network.train()
            avg_color_loss = 0
            avg_normal_loss = 0
            avg_coordinate_cycle_loss = 0
            avg_smoothness_loss = 0
            avg_human_prior_loss = 0
            avg_counter = 0

            print("epoch: ", epoch)
            
            for idx, data in enumerate(dataloader_train, 0):

                vertices_, normals_, colors_, voxels_ = data
                voxels = voxels_.to(self.device)
                vertices = vertices_.to(self.device)
                normals = normals_.to(self.device)
                colors = colors_.to(self.device)


                self.auv_network.zero_grad()

                texture_coefficients, UV_map_latent_code, UV_coord, _, network_output, UV_segments_mask = self.auv_network(voxels, vertices, normals, None)

                UV_segments_mask = [1-UV_segments_mask, UV_segments_mask]

                #supervise mask
                if epoch<5:
                    #x front, y up
                    UV_segments_mask_human_prior = ( normals[:,:,1:2]<-0.5 ) & ( vertices[:,:,1:2]<0 )
                    UV_segments_mask_human_prior = UV_segments_mask_human_prior.float()

                    #prior loss on how to segment the shape surface into charts for UV mapping
                    human_prior_loss = torch.mean( (UV_segments_mask[1]-UV_segments_mask_human_prior)**2 )
                    UV_segments_mask = [1-UV_segments_mask_human_prior, UV_segments_mask_human_prior]

                    UV_segments_mask_human_prior = None

                    #prior loss on UV mapping
                    if epoch<1:
                        human_prior_loss += torch.mean( (UV_coord[:,:,0]-vertices[:,:,0])**2 ) + torch.mean( (UV_coord[:,:,1]-vertices[:,:,2])**2 )

                else:
                    human_prior_loss = torch.zeros([1]).to(self.device)



                #color and normal and cycle loss
                color_loss = 0
                normal_loss = 0
                coordinate_cycle_loss = 0
                for i in range(self.num_UV_segments):
                    color_loss += torch.mean( (network_output[i][:,:,0:3]-colors)**2 *UV_segments_mask[i] )
                    normal_loss += torch.mean( (network_output[i][:,:,3:6]-(normals+1)*0.5)**2 *UV_segments_mask[i] )
                    coordinate_cycle_loss += torch.mean( (network_output[i][:,:,6:9]-(vertices+0.5))**2 *UV_segments_mask[i] )



                #texture mapping smoothness loss
                dist3d_threshold = 0.02**2

                #vertices [1,N,3]
                #UV_coord [1,N,2]
                #dist [M,N]
                #note: batch size must be one (shape)!

                smoothness_loss = 0
                for i in range(self.num_UV_segments):
                    selected_idx = torch.nonzero(UV_segments_mask[i][0,:,0], as_tuple=True)[0]
                    sample_num = selected_idx.size()[0]
                    if sample_num>0:
                        if sample_num>1024:
                            selected_idx = selected_idx[:1024]
                            sample_num = 1024
                        dist_3d = torch.sum( ( vertices[:,selected_idx].view(sample_num,1,3) - vertices.repeat(sample_num,1,1) )**2, dim=2)
                        dist_2d = torch.sum( ( UV_coord[:,selected_idx].view(sample_num,1,2) - UV_coord.repeat(sample_num,1,1) )**2, dim=2)
                        dist_mask = (dist_3d<dist3d_threshold).float()
                        smoothness_loss += torch.mean( torch.abs( (dist_3d+1e-10)**0.5 - (dist_2d+1e-10)**0.5 )*dist_mask )



                color_loss = color_loss*1

                if config.phase == 0:
                    normal_loss = normal_loss*0.1
                    coordinate_cycle_loss = coordinate_cycle_loss*10
                    smoothness_loss = smoothness_loss*10
                    human_prior_loss = human_prior_loss*1

                elif config.phase == 1:
                    normal_loss = normal_loss*0.1
                    coordinate_cycle_loss = coordinate_cycle_loss*1
                    smoothness_loss = smoothness_loss*10

                elif config.phase == 2:
                    gradual_weight = min(float(epoch-start_epoch)/(start_epoch//2),1)
                    normal_loss = normal_loss*1
                    coordinate_cycle_loss = coordinate_cycle_loss*(gradual_weight*99+1)
                    smoothness_loss = smoothness_loss*(gradual_weight*90+10)



                loss = color_loss + normal_loss + coordinate_cycle_loss + smoothness_loss + human_prior_loss
                loss.backward()
                self.optimizer.step()

                avg_color_loss += color_loss.item()
                avg_normal_loss += normal_loss.item()
                avg_coordinate_cycle_loss += coordinate_cycle_loss.item()
                avg_smoothness_loss += smoothness_loss.item()
                avg_human_prior_loss += human_prior_loss.item()
                avg_counter += 1


            if epoch%1==0:
                print("Epoch: [%2d/%2d] time: %4.4f, loss: %.6f %.6f %.6f %.6f %.6f" % (epoch, config.epoch, time.time() - start_time, avg_color_loss/avg_counter, avg_normal_loss/avg_counter, avg_coordinate_cycle_loss/avg_counter, avg_smoothness_loss/avg_counter, avg_human_prior_loss/avg_counter))

                #save samples

                self.auv_network.eval()

                with torch.no_grad():
                    texture_coefficients, UV_map_latent_code, UV_coord, texture_output, network_output, UV_segments_mask = self.auv_network(voxels, vertices, normals, self.texture_coords)
                    
                    large_image = np.zeros([self.texture_image_size*4,self.texture_image_size*self.num_UV_segments,3], np.uint8)

                    point_masks = UV_segments_mask.detach().cpu().numpy()
                    point_masks = [point_masks<=0.5,point_masks>0.5]

                    point_uvs = UV_coord.detach().cpu().numpy()
                    point_uvs = (point_uvs/self.texture_coords_scale+0.5)*self.texture_image_size
                    #print(np.min(point_uvs),np.max(point_uvs))
                    point_uvs = np.clip(point_uvs.astype(np.int32),0,self.texture_image_size-1)
                    for i in range(self.num_UV_segments):
                        mapped = np.zeros([self.texture_image_size,self.texture_image_size], np.uint8)
                        mapped[point_uvs[0,point_masks[i][0,:,0],0],point_uvs[0,point_masks[i][0,:,0],1]] = 255
                        large_image[:self.texture_image_size,self.texture_image_size*i:self.texture_image_size*(i+1),:] = np.expand_dims(mapped,2)

                        img = np.reshape(texture_output[i].detach().cpu().numpy(), [self.texture_image_size,self.texture_image_size,9])
                        img = np.clip(img*255,0,255).astype(np.uint8)
                        large_image[self.texture_image_size:self.texture_image_size*2,self.texture_image_size*i:self.texture_image_size*(i+1),:] = img[:,:,0:3]
                        large_image[self.texture_image_size*2:self.texture_image_size*3,self.texture_image_size*i:self.texture_image_size*(i+1),:] = img[:,:,3:6]
                        large_image[self.texture_image_size*3:self.texture_image_size*4,self.texture_image_size*i:self.texture_image_size*(i+1),:] = img[:,:,6:9]

                    cv2.imwrite(config.sample_dir+"/"+str(epoch)+".png", large_image)


                    #get the actual output and texture segmentations
                    point_colors = network_output[0].detach().cpu().numpy()*point_masks[0] + network_output[1].detach().cpu().numpy()*point_masks[1]
                    point_mask_colors = np.reshape(seg_colors[0],[1,1,3])*np.tile(point_masks[0],[1,1,3]) + np.reshape(seg_colors[1],[1,1,3])*np.tile(point_masks[1],[1,1,3])

                    point_colors = np.clip(point_colors*255,0,255).astype(np.uint8)
                    point_colors = point_colors[0]
                    point_mask_colors = np.clip(point_mask_colors,0,255).astype(np.uint8)
                    point_mask_colors = point_mask_colors[0]
                    point_coords = vertices_.numpy()
                    point_coords = point_coords[0]

                    point_cloud_utilities.write_ply_point_color(config.sample_dir+"/"+str(epoch)+".ply", point_coords,point_colors)
                    point_cloud_utilities.write_ply_point_color(config.sample_dir+"/"+str(epoch)+"_seg.ply", point_coords,point_mask_colors)



            if epoch%5==4:
                if not os.path.exists(self.checkpoint_path):
                    os.makedirs(self.checkpoint_path)
                #save checkpoint
                save_dir = os.path.join(self.checkpoint_path,self.checkpoint_name+"-"+str(epoch)+".pth")
                torch.save(self.auv_network.state_dict(), save_dir)
                #delete checkpoint
                self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer+1)%self.max_to_keep
                if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
                    if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
                        os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
                #update checkpoint manager
                self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
                #write file
                checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
                with open(checkpoint_txt, 'w') as fout:
                    for i in range(self.max_to_keep):
                        pointer = (self.checkpoint_manager_pointer+self.max_to_keep-i)%self.max_to_keep
                        if self.checkpoint_manager_list[pointer] is not None:
                            fout.write(self.checkpoint_manager_list[pointer]+"\n")



    #obtain high-quality aligned textures and mesh+uv
    def test(self, config):
        #load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            with open(checkpoint_txt) as fin:
                model_dir = fin.readline().strip()
            self.auv_network.load_state_dict(torch.load(model_dir))
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed")
            exit(-1)

        self.auv_network.eval()

        #output_uv_image_size = 1024
        #num_of_points = 50000000
        #above is too slow, use smaller resolution to speed up
        output_uv_image_size = 512
        num_of_points = 20000000


        #transform the output texture image here
        #these parameters need to be tuned manually
        #below are the default params for car
        #beta: rotate image
        #scale: rescale image
        #offset: translate image
        beta = 0
        if self.num_UV_segments==2:
            scale_u = 0.8
            scale_v = 1.5
            offset_u = 0.5
            offset_v = 0.5
        elif self.num_UV_segments==4:
            scale_u = 0.6
            scale_v = 0.6
            offset_u = 0.5
            offset_v = 0.5


        #shape names
        data_dir = config.data_dir
        obj_names = os.listdir(data_dir)
        obj_names = sorted(obj_names)
        if not config.use_all_data:
            obj_names = obj_names[:int(len(obj_names)*0.8)]

        vertices = None
        normals = None
        colors = None
        vertices_uv = None
        vertices_mask = None
        gpu_id = int(config.gpu)

        for idx in range(len(obj_names)):
            print(idx,len(obj_names))

            save_dir = config.sample_dir+"/"+str(idx)+"/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            #load input voxels
            hdf5_dir = data_dir+"/"+obj_names[idx]+"/vertices_normals_colors_voxels.hdf5"
            hdf5_file = h5py.File(hdf5_dir, 'r')
            voxels = hdf5_file["voxel_color"][:]
            hdf5_file.close()

            voxels = np.transpose(voxels, (3,0,1,2)).astype(np.float32)

            #load mesh vertices and triangles
            #sample points with normals and colors
            obj_dir = data_dir+"/"+obj_names[idx]+"/model_simplified_textured.obj"
            texture_dir = data_dir+"/"+obj_names[idx]+"/model_simplified_textured.png"
            if vertices is not None: del vertices
            if normals is not None: del normals
            if colors is not None: del colors
            if vertices_uv is not None: del vertices_uv
            if vertices_mask is not None: del vertices_mask
            mesh_vertices,mesh_triangles,vertices,normals,colors = point_cloud_utilities.sample_points(obj_dir,texture_dir,num_of_points,exact_num=False,normalize=False)
            vertices_y_min = np.min(vertices[:,1])
            vertices_uv = np.zeros([len(vertices),2],np.float32)
            vertices_mask = np.full([self.num_UV_segments,len(vertices)], False, bool)

            #uncomment to do sanity check - write sampled points
            #point_cloud_utilities.write_ply_point_color(save_dir+"/pc.ply",vertices,colors)

            #uncomment to subdivide the mesh to have smoother seams (caused by separate texture images)
            #import trimesh
            #mesh_vertices, mesh_triangles = trimesh.remesh.subdivide(mesh_vertices, mesh_triangles)

            #get triangle center points and normals
            #use them to decide which texture image this triangle belongs
            epsilon = 1e-10
            mesh_triangle_center_list = np.zeros([len(mesh_triangles),3],np.float32)
            mesh_triangle_normal_list = np.zeros([len(mesh_triangles),3],np.float32)
            for i in range(len(mesh_triangles)):
                mesh_triangle_center_list[i] = (mesh_vertices[mesh_triangles[i,0]] + mesh_vertices[mesh_triangles[i,1]] + mesh_vertices[mesh_triangles[i,2]])/3
                #area = |u x v|/2 = |u||v|sin(uv)/2
                a,b,c = mesh_vertices[mesh_triangles[i,1]]-mesh_vertices[mesh_triangles[i,0]]
                x,y,z = mesh_vertices[mesh_triangles[i,2]]-mesh_vertices[mesh_triangles[i,0]]
                ti = b*z-c*y
                tj = c*x-a*z
                tk = a*y-b*x
                area2 = (ti*ti+tj*tj+tk*tk)**0.5
                if area2<epsilon:
                    mesh_triangle_normal_list[i,0] = 0
                    mesh_triangle_normal_list[i,1] = 0
                    mesh_triangle_normal_list[i,2] = 0
                else:
                    mesh_triangle_normal_list[i,0] = ti/area2
                    mesh_triangle_normal_list[i,1] = tj/area2
                    mesh_triangle_normal_list[i,2] = tk/area2


            output_uv_image = np.zeros([self.num_UV_segments,output_uv_image_size,output_uv_image_size,4], np.int32)
            output_uv_image_mask = np.zeros([self.num_UV_segments,output_uv_image_size,output_uv_image_size,1], np.int32)
            output_uv_image_depth = np.zeros([self.num_UV_segments,output_uv_image_size,output_uv_image_size,1], np.float32)

            output_uv_image_depth[0] = -1
            output_uv_image_depth[1] = 1

            batch_num = (len(vertices)-1)//self.point_batch_size + 1

            with torch.no_grad():
                voxels_tensor = torch.from_numpy(np.expand_dims(voxels,0))
                voxels_tensor = voxels_tensor.to(self.device)

                t_vector, d_vector = self.auv_network.encoder(voxels_tensor)

                vertices_tensor = torch.from_numpy(np.expand_dims(mesh_vertices,0)).to(self.device)
                centers_tensor = torch.from_numpy(np.expand_dims(mesh_triangle_center_list,0)).to(self.device)
                normals_tensor = torch.from_numpy(np.expand_dims(mesh_triangle_normal_list,0)).to(self.device)

                mask_out = self.auv_network.masker(centers_tensor, d_vector, normals_tensor)

                mask_out = mask_out.detach().cpu().numpy()
                mask_out = np.reshape(mask_out,[-1])
                output_uv_mask = [mask_out<=0.5, mask_out>0.5]

                #compute mesh uv
                UV_coord = self.auv_network.UV_mapper(vertices_tensor, d_vector)
                UV_coord = UV_coord.detach().cpu().numpy()

                output_u = UV_coord[0,:,0]
                output_v = UV_coord[0,:,1]
                output_uv = np.zeros([len(output_u),2], np.float32)
                output_uv[:,0] = (output_u*np.cos(beta) + output_v*np.sin(beta))*scale_u + offset_u
                output_uv[:,1] = (- output_u*np.sin(beta) + output_v*np.cos(beta))*scale_v + offset_v
                output_uv = np.clip(output_uv,0.001,0.999)

                mesh_utilities.write_ply_triangle_UV(save_dir+"model", mesh_vertices, output_uv_mask, output_uv, mesh_triangles)

                #get uv for sampled points
                for bid in range(batch_num):
                    tmp_vertices = vertices[self.point_batch_size*bid:self.point_batch_size*(bid+1)]
                    tmp_normals = normals[self.point_batch_size*bid:self.point_batch_size*(bid+1)]
                    vertices_tensor = torch.from_numpy(np.expand_dims(tmp_vertices,0)).to(self.device)
                    normals_tensor = torch.from_numpy(np.expand_dims(tmp_normals,0)).to(self.device)

                    mask_out = self.auv_network.masker(vertices_tensor, d_vector, normals_tensor)

                    mask_out = mask_out.detach().cpu().numpy()
                    vertices_mask[0,self.point_batch_size*bid:self.point_batch_size*(bid+1)] = (mask_out[0,:,0]<=0.5)
                    vertices_mask[1,self.point_batch_size*bid:self.point_batch_size*(bid+1)] = (mask_out[0,:,0]>0.5)

                    UV_coord = self.auv_network.UV_mapper(vertices_tensor, d_vector)
                    vertices_uv[self.point_batch_size*bid:self.point_batch_size*(bid+1)] = UV_coord.detach().cpu().numpy()[0]

            #first pass: determine depth
            for bid in range(batch_num):
                tmp_vertices = vertices[self.point_batch_size*bid:self.point_batch_size*(bid+1)]

                output_u = vertices_uv[self.point_batch_size*bid:self.point_batch_size*(bid+1),0]
                output_v = vertices_uv[self.point_batch_size*bid:self.point_batch_size*(bid+1),1]
                UV_coord_int_x = (output_u*np.cos(beta) + output_v*np.sin(beta))*scale_u + offset_u
                UV_coord_int_y = (- output_u*np.sin(beta) + output_v*np.cos(beta))*scale_v + offset_v
                UV_coord_int_x = UV_coord_int_x * output_uv_image_size
                UV_coord_int_y = UV_coord_int_y * output_uv_image_size
                UV_coord_int_x = np.clip(UV_coord_int_x.astype(np.int32),0,output_uv_image_size-1)
                UV_coord_int_y = np.clip(UV_coord_int_y.astype(np.int32),0,output_uv_image_size-1)

                vertices_mask_front = np.reshape(vertices_mask[0,self.point_batch_size*bid:self.point_batch_size*(bid+1)], [-1,1]).astype(np.float32)
                vertices_mask_back = np.reshape(vertices_mask[1,self.point_batch_size*bid:self.point_batch_size*(bid+1)], [-1,1]).astype(np.float32)
                tmp_metric = np.sqrt( np.square(tmp_vertices[:,0:1]) + np.square(tmp_vertices[:,1:2]-vertices_y_min) + np.square(tmp_vertices[:,2:3]) )
                tmp_metric_front = tmp_metric - (1-vertices_mask_front)*10
                tmp_metric_back = tmp_metric + (1-vertices_mask_back)*10
                point_cloud_utilities_cy.indexed_max_array_2d_float_separate(output_uv_image_depth[0],UV_coord_int_x,UV_coord_int_y,tmp_metric_front)
                point_cloud_utilities_cy.indexed_min_array_2d_float_separate(output_uv_image_depth[1],UV_coord_int_x,UV_coord_int_y,tmp_metric_back)

            #second pass: determine color
            for bid in range(batch_num):
                tmp_vertices = vertices[self.point_batch_size*bid:self.point_batch_size*(bid+1)]
                tmp_colors = colors[self.point_batch_size*bid:self.point_batch_size*(bid+1)]
            
                output_u = vertices_uv[self.point_batch_size*bid:self.point_batch_size*(bid+1),0]
                output_v = vertices_uv[self.point_batch_size*bid:self.point_batch_size*(bid+1),1]
                UV_coord_int_x = (output_u*np.cos(beta) + output_v*np.sin(beta))*scale_u + offset_u
                UV_coord_int_y = (- output_u*np.sin(beta) + output_v*np.cos(beta))*scale_v + offset_v
                UV_coord_int_x = UV_coord_int_x * output_uv_image_size
                UV_coord_int_y = UV_coord_int_y * output_uv_image_size
                UV_coord_int_x = np.clip(UV_coord_int_x.astype(np.int32),0,output_uv_image_size-1)
                UV_coord_int_y = np.clip(UV_coord_int_y.astype(np.int32),0,output_uv_image_size-1)

                vertices_mask_front = np.reshape(vertices_mask[0,self.point_batch_size*bid:self.point_batch_size*(bid+1)], [-1,1])
                vertices_mask_back = np.reshape(vertices_mask[1,self.point_batch_size*bid:self.point_batch_size*(bid+1)], [-1,1])
                tmp_metric = np.sqrt( np.square(tmp_vertices[:,0:1]) + np.square(tmp_vertices[:,1:2]-vertices_y_min) + np.square(tmp_vertices[:,2:3]) )
                vertices_mask_front = ((tmp_metric>output_uv_image_depth[0,UV_coord_int_x,UV_coord_int_y]-0.01) & vertices_mask_front).astype(np.int32)
                vertices_mask_back = ((tmp_metric<output_uv_image_depth[1,UV_coord_int_x,UV_coord_int_y]+0.01) & vertices_mask_back).astype(np.int32)
                point_cloud_utilities_cy.indexed_add_array_2d_color_separate(output_uv_image[0],UV_coord_int_x,UV_coord_int_y,tmp_colors*vertices_mask_front)
                point_cloud_utilities_cy.indexed_add_array_2d_color_separate(output_uv_image_mask[0],UV_coord_int_x,UV_coord_int_y,vertices_mask_front)
                point_cloud_utilities_cy.indexed_add_array_2d_color_separate(output_uv_image[1],UV_coord_int_x,UV_coord_int_y,tmp_colors*vertices_mask_back)
                point_cloud_utilities_cy.indexed_add_array_2d_color_separate(output_uv_image_mask[1],UV_coord_int_x,UV_coord_int_y,vertices_mask_back)



            #save texture
            output_uv_image = output_uv_image/np.maximum(output_uv_image_mask,1)
            for i in range(self.num_UV_segments):
                tmp_img = (output_uv_image[i,:,:,:]).astype(np.uint8)
                tmp_mask = (output_uv_image_mask[i,:,:,:]!=0).astype(np.uint8)*255
                cv2.imwrite(save_dir+str(i)+".png", tmp_img)
                cv2.imwrite(save_dir+str(i)+"_m.png", tmp_mask)

