#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False


# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


def sample_points(float[:, ::1] vertices, int[:, ::1] triangles, float[:, ::1] uv_vertices, int[:, ::1] uv_triangles, float[:, :, ::1] texture_img, float[::1] triangle_area_list, float[:, ::1] triangle_normal_list, float[::1] sample_prob_list, float[::1] random_numbers, float[:, ::1] point_normal_color_list, int num_of_points):
    cdef float epsilon = 1e-10
    cdef int triangles_len,texture_img_h,texture_img_w
    cdef float texture_img_h_f,texture_img_w_f

    cdef float a,b,c,x,y,z,ti,tj,tk,area_2,triangle_area_sum
    cdef int t0,t1,t2,uv_t0,uv_t1,uv_t2,i,j,count,watchdog
    cdef float prob,prob_f,rand_x,rand_y
    cdef int prob_i,rand_p,rand_len
    cdef float n_x,n_y,n_z,u_x,u_y,u_z,v_x,v_y,v_z,base_x,base_y,base_z, tu_x,tu_y,tv_x,tv_y,tbase_x,tbase_y
    cdef float p_x,p_y,p_z,pn_x,pn_y,pn_z
    cdef int pni_x,pni_y,pni_z

    triangles_len = triangles.shape[0]
    rand_len = random_numbers.shape[0]
    texture_img_h = texture_img.shape[0]
    texture_img_w = texture_img.shape[1]
    texture_img_h_f = <float>texture_img_h
    texture_img_w_f = <float>texture_img_w

    for i in range(triangles_len):
        #area = |u x v|/2 = |u||v|sin(uv)/2
        t0 = triangles[i,0]
        t1 = triangles[i,1]
        t2 = triangles[i,2]
        a = vertices[t1,0]-vertices[t0,0]
        b = vertices[t1,1]-vertices[t0,1]
        c = vertices[t1,2]-vertices[t0,2]
        x = vertices[t2,0]-vertices[t0,0]
        y = vertices[t2,1]-vertices[t0,1]
        z = vertices[t2,2]-vertices[t0,2]
        ti = b*z-c*y
        tj = c*x-a*z
        tk = a*y-b*x
        area_2 = (ti*ti+tj*tj+tk*tk)**0.5
        if area_2<epsilon:
            triangle_area_list[i] = 0.0
            triangle_normal_list[i,0] = 0.0
            triangle_normal_list[i,1] = 0.0
            triangle_normal_list[i,2] = 0.0
        else:
            triangle_area_list[i] = area_2
            triangle_normal_list[i,0] = ti/area_2
            triangle_normal_list[i,1] = tj/area_2
            triangle_normal_list[i,2] = tk/area_2
    
    triangle_area_sum = 0
    for i in range(triangles_len):
        triangle_area_sum += triangle_area_list[i]
    area_2 = num_of_points/triangle_area_sum
    for i in range(triangles_len):
        sample_prob_list[i] = triangle_area_list[i]*area_2

    count = 0
    watchdog = 0
    rand_p = 0

    while(count<num_of_points):
        watchdog += 1
        if watchdog>100:
            return -1
        for i in range(triangles_len):
            prob = sample_prob_list[i]
            prob_i = <int>prob
            prob_f = prob-prob_i

            rand_x = random_numbers[rand_p]
            rand_p += 1
            if rand_p==rand_len:
                rand_p = 0

            if rand_x<prob_f:
                prob_i += 1

            n_x = triangle_normal_list[i,0]
            n_y = triangle_normal_list[i,1]
            n_z = triangle_normal_list[i,2]

            t0 = triangles[i,0]
            t1 = triangles[i,1]
            t2 = triangles[i,2]
            uv_t0 = uv_triangles[i,0]
            uv_t1 = uv_triangles[i,1]
            uv_t2 = uv_triangles[i,2]

            base_x = vertices[t0,0]
            base_y = vertices[t0,1]
            base_z = vertices[t0,2]
            u_x = vertices[t1,0]-base_x
            u_y = vertices[t1,1]-base_y
            u_z = vertices[t1,2]-base_z
            v_x = vertices[t2,0]-base_x
            v_y = vertices[t2,1]-base_y
            v_z = vertices[t2,2]-base_z

            tbase_x = uv_vertices[uv_t0,0]
            tbase_y = uv_vertices[uv_t0,1]
            tu_x = uv_vertices[uv_t1,0]-tbase_x
            tu_y = uv_vertices[uv_t1,1]-tbase_y
            tv_x = uv_vertices[uv_t2,0]-tbase_x
            tv_y = uv_vertices[uv_t2,1]-tbase_y

            for j in range(prob_i):
                #sample a point here:
                rand_x = random_numbers[rand_p]
                rand_p += 1
                if rand_p==rand_len:
                    rand_p = 0
                rand_y = random_numbers[rand_p]
                rand_p += 1
                if rand_p==rand_len:
                    rand_p = 0

                if rand_x+rand_y>1:
                    rand_x = 1-rand_x
                    rand_y = 1-rand_y

                p_x = u_x*rand_x+v_x*rand_y+base_x
                p_y = u_y*rand_x+v_y*rand_y+base_y
                p_z = u_z*rand_x+v_z*rand_y+base_z

                point_normal_color_list[count,0] = p_x
                point_normal_color_list[count,1] = p_y
                point_normal_color_list[count,2] = p_z
                point_normal_color_list[count,3] = n_x
                point_normal_color_list[count,4] = n_y
                point_normal_color_list[count,5] = n_z

                pn_x = tu_x*rand_x+tv_x*rand_y+tbase_x
                pn_y = tu_y*rand_x+tv_y*rand_y+tbase_y

                pni_x = <int>(pn_x*texture_img_w_f)
                pni_x = pni_x - pni_x//texture_img_w*texture_img_w
                if pni_x<0: pni_x += texture_img_w
                pni_y = <int>((1-pn_y)*texture_img_h_f)
                pni_y = pni_y - pni_y//texture_img_h*texture_img_h
                if pni_y<0: pni_y += texture_img_h
                point_normal_color_list[count,6] = texture_img[pni_y,pni_x,0]
                point_normal_color_list[count,7] = texture_img[pni_y,pni_x,1]
                point_normal_color_list[count,8] = texture_img[pni_y,pni_x,2]
                point_normal_color_list[count,9] = texture_img[pni_y,pni_x,3]

                count += 1
    
    return count


def sample_surface_points(float[:, ::1] vertices, int[:, ::1] triangles, float[:, ::1] uv_vertices, int[:, ::1] uv_triangles, char[:, :, ::1] voxels, float[:, :, ::1] texture_img, float[::1] triangle_area_list, float[:, ::1] triangle_normal_list, float[::1] sample_prob_list, float[::1] random_numbers, float[:, ::1] point_normal_color_list, int num_of_points):
    cdef float epsilon = 1e-10
    cdef int voxels_size,triangles_len,texture_img_h,texture_img_w
    cdef float voxels_size_f,small_step,texture_img_h_f,texture_img_w_f

    cdef float a,b,c,x,y,z,ti,tj,tk,area_2,triangle_area_sum
    cdef int t0,t1,t2,uv_t0,uv_t1,uv_t2,i,j,count,watchdog
    cdef float prob,prob_f,rand_x,rand_y
    cdef int prob_i,rand_p,rand_len
    cdef float n_x,n_y,n_z,u_x,u_y,u_z,v_x,v_y,v_z,base_x,base_y,base_z, tu_x,tu_y,tv_x,tv_y,tbase_x,tbase_y
    cdef float p_x,p_y,p_z,pn_x,pn_y,pn_z
    cdef int pni_x,pni_y,pni_z

    voxels_size = voxels.shape[0]
    voxels_size_f = <float>voxels_size
    small_step = 1.5/voxels_size_f
    triangles_len = triangles.shape[0]
    rand_len = random_numbers.shape[0]
    texture_img_h = texture_img.shape[0]
    texture_img_w = texture_img.shape[1]
    texture_img_h_f = <float>texture_img_h
    texture_img_w_f = <float>texture_img_w

    for i in range(triangles_len):
        #area = |u x v|/2 = |u||v|sin(uv)/2
        t0 = triangles[i,0]
        t1 = triangles[i,1]
        t2 = triangles[i,2]
        a = vertices[t1,0]-vertices[t0,0]
        b = vertices[t1,1]-vertices[t0,1]
        c = vertices[t1,2]-vertices[t0,2]
        x = vertices[t2,0]-vertices[t0,0]
        y = vertices[t2,1]-vertices[t0,1]
        z = vertices[t2,2]-vertices[t0,2]
        ti = b*z-c*y
        tj = c*x-a*z
        tk = a*y-b*x
        area_2 = (ti*ti+tj*tj+tk*tk)**0.5
        if area_2<epsilon:
            triangle_area_list[i] = 0.0
            triangle_normal_list[i,0] = 0.0
            triangle_normal_list[i,1] = 0.0
            triangle_normal_list[i,2] = 0.0
        else:
            triangle_area_list[i] = area_2
            triangle_normal_list[i,0] = ti/area_2
            triangle_normal_list[i,1] = tj/area_2
            triangle_normal_list[i,2] = tk/area_2
    
    triangle_area_sum = 0
    for i in range(triangles_len):
        triangle_area_sum += triangle_area_list[i]
    area_2 = num_of_points/triangle_area_sum
    for i in range(triangles_len):
        sample_prob_list[i] = triangle_area_list[i]*area_2

    count = 0
    watchdog = 0
    rand_p = 0

    while(count<num_of_points):
        watchdog += 1
        if watchdog>100:
            return -1
        for i in range(triangles_len):
            prob = sample_prob_list[i]
            prob_i = <int>prob
            prob_f = prob-prob_i

            rand_x = random_numbers[rand_p]
            rand_p += 1
            if rand_p==rand_len:
                rand_p = 0

            if rand_x<prob_f:
                prob_i += 1

            n_x = triangle_normal_list[i,0]
            n_y = triangle_normal_list[i,1]
            n_z = triangle_normal_list[i,2]

            t0 = triangles[i,0]
            t1 = triangles[i,1]
            t2 = triangles[i,2]
            uv_t0 = uv_triangles[i,0]
            uv_t1 = uv_triangles[i,1]
            uv_t2 = uv_triangles[i,2]

            base_x = vertices[t0,0]
            base_y = vertices[t0,1]
            base_z = vertices[t0,2]
            u_x = vertices[t1,0]-base_x
            u_y = vertices[t1,1]-base_y
            u_z = vertices[t1,2]-base_z
            v_x = vertices[t2,0]-base_x
            v_y = vertices[t2,1]-base_y
            v_z = vertices[t2,2]-base_z

            tbase_x = uv_vertices[uv_t0,0]
            tbase_y = uv_vertices[uv_t0,1]
            tu_x = uv_vertices[uv_t1,0]-tbase_x
            tu_y = uv_vertices[uv_t1,1]-tbase_y
            tv_x = uv_vertices[uv_t2,0]-tbase_x
            tv_y = uv_vertices[uv_t2,1]-tbase_y

            for j in range(prob_i):
                #sample a point here:
                rand_x = random_numbers[rand_p]
                rand_p += 1
                if rand_p==rand_len:
                    rand_p = 0
                rand_y = random_numbers[rand_p]
                rand_p += 1
                if rand_p==rand_len:
                    rand_p = 0

                if rand_x+rand_y>1:
                    rand_x = 1-rand_x
                    rand_y = 1-rand_y

                p_x = u_x*rand_x+v_x*rand_y+base_x
                p_y = u_y*rand_x+v_y*rand_y+base_y
                p_z = u_z*rand_x+v_z*rand_y+base_z

                #verify normal
                pn_x = (p_x+n_x*small_step+0.5)*voxels_size_f
                pn_y = (p_y+n_y*small_step+0.5)*voxels_size_f
                pn_z = (p_z+n_z*small_step+0.5)*voxels_size_f

                pni_x = <int>pn_x
                pni_y = <int>pn_y
                pni_z = <int>pn_z

                if pni_x<0 or pni_x>=voxels_size or pni_y<0 or pni_y>=voxels_size or pni_z<0 or pni_z>=voxels_size or voxels[pni_x,pni_y,pni_z]==0:
                    point_normal_color_list[count,0] = p_x
                    point_normal_color_list[count,1] = p_y
                    point_normal_color_list[count,2] = p_z
                    point_normal_color_list[count,3] = n_x
                    point_normal_color_list[count,4] = n_y
                    point_normal_color_list[count,5] = n_z

                    pn_x = tu_x*rand_x+tv_x*rand_y+tbase_x
                    pn_y = tu_y*rand_x+tv_y*rand_y+tbase_y

                    pni_x = <int>(pn_x*texture_img_w_f)
                    pni_x = pni_x - pni_x//texture_img_w*texture_img_w
                    if pni_x<0: pni_x += texture_img_w
                    pni_y = <int>((1-pn_y)*texture_img_h_f)
                    pni_y = pni_y - pni_y//texture_img_h*texture_img_h
                    if pni_y<0: pni_y += texture_img_h
                    point_normal_color_list[count,6] = texture_img[pni_y,pni_x,0]
                    point_normal_color_list[count,7] = texture_img[pni_y,pni_x,1]
                    point_normal_color_list[count,8] = texture_img[pni_y,pni_x,2]
                    point_normal_color_list[count,9] = texture_img[pni_y,pni_x,3]

                    count += 1
    
    return count




def indexed_add_constant_3d_int(int[:,:,::1] target, int[:,::1] indices, int value):
    cdef int i,indices_len
    indices_len = indices.shape[0]
    for i in range(indices_len):
        target[indices[i,0],indices[i,1],indices[i,2]] += value

def indexed_add_array_3d_color(int[:,:,:,::1] target, int[:,::1] indices, int[:,::1] values):
    cdef int i,j,indices_len,color_len
    indices_len = indices.shape[0]
    color_len = values.shape[1]
    for i in range(indices_len):
        for j in range(color_len):
            target[indices[i,0],indices[i,1],indices[i,2],j] += values[i,j]

def indexed_add_array_2d_color_separate(int[:,:,::1] target, int[::1] x, int[::1] y, int[:,::1] values):
    cdef int i,j,indices_len,color_len
    indices_len = x.shape[0]
    color_len = values.shape[1]
    for i in range(indices_len):
        for j in range(color_len):
            target[x[i],y[i],j] += values[i,j]

def indexed_max_array_2d_float_separate(float[:,:,::1] target, int[::1] x, int[::1] y, float[:,::1] values):
    cdef int i,indices_len
    indices_len = x.shape[0]
    for i in range(indices_len):
        if values[i,0]>target[x[i],y[i],0]:
            target[x[i],y[i],0] = values[i,0]

def indexed_min_array_2d_float_separate(float[:,:,::1] target, int[::1] x, int[::1] y, float[:,::1] values):
    cdef int i,indices_len
    indices_len = x.shape[0]
    for i in range(indices_len):
        if values[i,0]<target[x[i],y[i],0]:
            target[x[i],y[i],0] = values[i,0]

