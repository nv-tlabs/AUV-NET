#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cdef int depth_buffer_size = 5120

def depth_fusion_XZY_5views(char[:, :, ::1] voxels, int[:, :, ::1] rendering):
    cdef int dimx,dimy,dimz
    
    cdef int hdis = depth_buffer_size//2 
    
    cdef int c = 0
    cdef int u = 0
    cdef int v = 0
    cdef int d = 0
    
    cdef int outside_flag = 0
    
    cdef int x = 0
    cdef int y = 0
    cdef int z = 0
    
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    
    dimx = voxels.shape[0]
    dimz = voxels.shape[1]
    dimy = voxels.shape[2]
    
    #get rendering
    for x in range(dimx):
        for y in range(dimy):
            for z in range(dimz):
                if voxels[x,z,y]>0:

                    #z-buffering
                    
                    c = 0
                    u = x + hdis
                    v = z + hdis
                    d = -y #y must always be negative in d to render from top
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 1
                    u = y + hdis
                    v = z + hdis
                    d = x
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 2
                    u = y + hdis
                    v = z + hdis
                    d = -x
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 3
                    u = x + hdis
                    v = y + hdis
                    d = z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 4
                    u = x + hdis
                    v = y + hdis
                    d = -z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d    
    
    #depth fusion
    for x in range(dimx):
        for y in range(dimy):
            for z in range(dimz):
                outside_flag = 0
                
                c = 0
                u = x + hdis
                v = z + hdis
                d = -y
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 1
                u = y + hdis
                v = z + hdis
                d = x
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 2
                u = y + hdis
                v = z + hdis
                d = -x
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 3
                u = x + hdis
                v = y + hdis
                d = z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 4
                u = x + hdis
                v = y + hdis
                d = -z
                if rendering[u,v,c]>d:
                    outside_flag += 1

                if outside_flag==0:
                    voxels[x,z,y] = 1


def get_run_length_encoding(char[:, :, ::1] voxels, int[:, ::1] encoding):
    cdef int state = 0
    cdef int ctr = 0
    cdef int p = 0
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    cdef int dimx,dimy,dimz

    dimx = voxels.shape[0]
    dimy = voxels.shape[1]
    dimz = voxels.shape[2]

    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                if voxels[i,j,k]>0:
                    voxels[i,j,k] = 1
                if voxels[i,j,k]==state:
                    ctr += 1
                    if ctr==255:
                        encoding[p,0] = state
                        encoding[p,1] = ctr
                        p += 1
                        ctr = 0
                else:
                    if ctr>0:
                        encoding[p,0] = state
                        encoding[p,1] = ctr
                        p += 1
                    state = voxels[i,j,k]
                    ctr = 1

    if ctr > 0:
        encoding[p,0] = state
        encoding[p,1] = ctr
        p += 1

    encoding[p,0] = 2



def cube_alpha_hull(char[:, :, ::1] voxels, int[:, :, ::1] accu, int cubesize_x, int cubesize_y, int cubesize_z):
    cdef int i,j,k
    cdef int dimx,dimy,dimz
    cdef int p, a000,a001,a010,a011,a100,a101,a110,a111
    cdef int cube_minx, cube_miny, cube_minz, cube_maxx, cube_maxy, cube_maxz

    dimx = voxels.shape[0]
    dimy = voxels.shape[1]
    dimz = voxels.shape[2]

    #first pass

    #dynamic programming to get integral map
    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                if voxels[i,j,k]==0:
                    a111 = 0
                else:
                    a111 = 1
                if i==0:
                    a011 = 0
                else:
                    a011 = accu[i-1,j,k]
                if j==0:
                    a101 = 0
                else:
                    a101 = accu[i,j-1,k]
                if k==0:
                    a110 = 0
                else:
                    a110 = accu[i,j,k-1]
                if j==0 or k==0:
                    a100 = 0
                else:
                    a100 = accu[i,j-1,k-1]
                if i==0 or k==0:
                    a010 = 0
                else:
                    a010 = accu[i-1,j,k-1]
                if i==0 or j==0:
                    a001 = 0
                else:
                    a001 = accu[i-1,j-1,k]
                if i==0 or j==0 or k==0:
                    a000 = 0
                else:
                    a000 = accu[i-1,j-1,k-1]
                accu[i,j,k] = a111 + a011 + a101 + a110 + a000 - a100 - a010 - a001
    
    #one
    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                voxels[i,j,k] = 1
    
    #cube alpha hull
    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                cube_minx = i-cubesize_x
                cube_miny = j-cubesize_y
                cube_minz = k-cubesize_z
                cube_maxx = i+cubesize_x-1
                cube_maxy = j+cubesize_y-1
                cube_maxz = k+cubesize_z-1
                if cube_maxx>=dimx: cube_maxx = dimx-1
                if cube_maxy>=dimy: cube_maxy = dimy-1
                if cube_maxz>=dimz: cube_maxz = dimz-1

                a111 = accu[cube_maxx,cube_maxy,cube_maxz]
                if cube_minx<0:
                    a011 = 0
                else:
                    a011 = accu[cube_minx,cube_maxy,cube_maxz]
                if cube_miny<0:
                    a101 = 0
                else:
                    a101 = accu[cube_maxx,cube_miny,cube_maxz]
                if cube_minz<0:
                    a110 = 0
                else:
                    a110 = accu[cube_maxx,cube_maxy,cube_minz]
                if cube_miny<0 or cube_minz<0:
                    a100 = 0
                else:
                    a100 = accu[cube_maxx,cube_miny,cube_minz]
                if cube_minx<0 or cube_minz<0:
                    a010 = 0
                else:
                    a010 = accu[cube_minx,cube_maxy,cube_minz]
                if cube_minx<0 or cube_miny<0:
                    a001 = 0
                else:
                    a001 = accu[cube_minx,cube_miny,cube_maxz]
                if cube_minx<0 or cube_miny<0 or cube_minz<0:
                    a000 = 0
                else:
                    a000 = accu[cube_minx,cube_miny,cube_minz]
                p = a111 + a100 + a010 + a001 - a011 - a101 - a110 - a000
                if p==0:
                    voxels[i,j,k] = 0

    #second pass

    #dynamic programming to get integral map
    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                if voxels[i,j,k]==0:
                    a111 = 0
                else:
                    a111 = 1
                if i==0:
                    a011 = 0
                else:
                    a011 = accu[i-1,j,k]
                if j==0:
                    a101 = 0
                else:
                    a101 = accu[i,j-1,k]
                if k==0:
                    a110 = 0
                else:
                    a110 = accu[i,j,k-1]
                if j==0 or k==0:
                    a100 = 0
                else:
                    a100 = accu[i,j-1,k-1]
                if i==0 or k==0:
                    a010 = 0
                else:
                    a010 = accu[i-1,j,k-1]
                if i==0 or j==0:
                    a001 = 0
                else:
                    a001 = accu[i-1,j-1,k]
                if i==0 or j==0 or k==0:
                    a000 = 0
                else:
                    a000 = accu[i-1,j-1,k-1]
                accu[i,j,k] = a111 + a011 + a101 + a110 + a000 - a100 - a010 - a001

    #cube alpha hull
    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                if voxels[i,j,k] != 0:
                    cube_minx = i-cubesize_x
                    cube_miny = j-cubesize_y
                    cube_minz = k-cubesize_z
                    cube_maxx = i+cubesize_x-1
                    cube_maxy = j+cubesize_y-1
                    cube_maxz = k+cubesize_z-1
                    if cube_maxx>=dimx: cube_maxx = dimx-1
                    if cube_maxy>=dimy: cube_maxy = dimy-1
                    if cube_maxz>=dimz: cube_maxz = dimz-1

                    a111 = accu[cube_maxx,cube_maxy,cube_maxz]
                    if cube_minx<0:
                        a011 = 0
                    else:
                        a011 = accu[cube_minx,cube_maxy,cube_maxz]
                    if cube_miny<0:
                        a101 = 0
                    else:
                        a101 = accu[cube_maxx,cube_miny,cube_maxz]
                    if cube_minz<0:
                        a110 = 0
                    else:
                        a110 = accu[cube_maxx,cube_maxy,cube_minz]
                    if cube_miny<0 or cube_minz<0:
                        a100 = 0
                    else:
                        a100 = accu[cube_maxx,cube_miny,cube_minz]
                    if cube_minx<0 or cube_minz<0:
                        a010 = 0
                    else:
                        a010 = accu[cube_minx,cube_maxy,cube_minz]
                    if cube_minx<0 or cube_miny<0:
                        a001 = 0
                    else:
                        a001 = accu[cube_minx,cube_miny,cube_maxz]
                    if cube_minx<0 or cube_miny<0 or cube_minz<0:
                        a000 = 0
                    else:
                        a000 = accu[cube_minx,cube_miny,cube_minz]
                    p = a111 + a100 + a010 + a001 - a011 - a101 - a110 - a000

                    if cube_minx<-1: cube_minx = -1
                    if cube_miny<-1: cube_miny = -1
                    if cube_minz<-1: cube_minz = -1
                    if p != (cube_maxx-cube_minx)*(cube_maxy-cube_miny)*(cube_maxz-cube_minz):
                        voxels[i,j,k] = 0



def get_transpose(char[:, :, ::1] tmp_voxel, char[:, :, ::1] batch_voxels, int padding, int target_axis, int flip):
    #numpy's transpose-assign is too slow

    cdef int dim,dim1, x,y,z

    dim = batch_voxels.shape[0]
    dim1 = dim-1

    if target_axis==0 and flip==0:
        for x in range(dim):
            for y in range(dim):
                for z in range(dim):
                    tmp_voxel[x+padding,y+padding,z+padding] = batch_voxels[x,y,z]

    if target_axis==0 and flip==1:
        for x in range(dim):
            for y in range(dim):
                for z in range(dim):
                    tmp_voxel[x+padding,y+padding,z+padding] = batch_voxels[dim1-x,y,z]

    if target_axis==1 and flip==0:
        for y in range(dim):
            for x in range(dim):
                for z in range(dim):
                    tmp_voxel[x+padding,y+padding,z+padding] = batch_voxels[y,x,z]

    if target_axis==1 and flip==1:
        for y in range(dim):
            for x in range(dim):
                for z in range(dim):
                    tmp_voxel[x+padding,y+padding,z+padding] = batch_voxels[y,dim1-x,z]

    if target_axis==2 and flip==0:
        for z in range(dim):
            for y in range(dim):
                for x in range(dim):
                    tmp_voxel[x+padding,y+padding,z+padding] = batch_voxels[z,y,x]

    if target_axis==2 and flip==1:
        for z in range(dim):
            for y in range(dim):
                for x in range(dim):
                    tmp_voxel[x+padding,y+padding,z+padding] = batch_voxels[z,y,dim1-x]


def recover_transpose(char[:, :, ::1] tmp_voxel, char[:, :, ::1] batch_voxels, int padding, int target_axis, int flip):
    #numpy's transpose-assign is too slow

    cdef int dim,dim1, x,y,z

    dim = batch_voxels.shape[0]
    dim1 = dim-1

    if target_axis==0 and flip==0:
        for x in range(dim):
            for y in range(dim):
                for z in range(dim):
                    batch_voxels[x,y,z] = tmp_voxel[x+padding,y+padding,z+padding]

    if target_axis==0 and flip==1:
        for x in range(dim):
            for y in range(dim):
                for z in range(dim):
                    batch_voxels[dim1-x,y,z] = tmp_voxel[x+padding,y+padding,z+padding]

    if target_axis==1 and flip==0:
        for y in range(dim):
            for x in range(dim):
                for z in range(dim):
                    batch_voxels[y,x,z] = tmp_voxel[x+padding,y+padding,z+padding]

    if target_axis==1 and flip==1:
        for y in range(dim):
            for x in range(dim):
                for z in range(dim):
                    batch_voxels[y,dim1-x,z] = tmp_voxel[x+padding,y+padding,z+padding]

    if target_axis==2 and flip==0:
        for z in range(dim):
            for y in range(dim):
                for x in range(dim):
                    batch_voxels[z,y,x] = tmp_voxel[x+padding,y+padding,z+padding]

    if target_axis==2 and flip==1:
        for z in range(dim):
            for y in range(dim):
                for x in range(dim):
                    batch_voxels[z,y,dim1-x] = tmp_voxel[x+padding,y+padding,z+padding]


def boundary_cull(char[:, :, ::1] voxels, int[:, :, ::1] accu, char[:, :, ::1] refvoxels, int[:, :, ::1] refaccu, int[:, ::1] queue):
    #assume target direction is X-

    cdef int i,j,k,x,y,z,q_start,q_end,this_depth,tmp_depth
    cdef int dimx,dimy,dimz,queue_len,depth_channel

    dimx = voxels.shape[0]
    dimy = voxels.shape[1]
    dimz = voxels.shape[2]
    depth_channel = dimx-1
    queue_len = queue.shape[0]

    #get accu
    for y in range(dimy):
        for z in range(dimz):
            accu[0,y,z] = 0
            accu[depth_channel,y,z] = 0
    for x in range(1,depth_channel):
        for y in range(dimy):
            for z in range(dimz):
                if voxels[x,y,z]>0:
                    accu[x,y,z] = accu[x-1,y,z] + 1
                    accu[depth_channel,y,z] = x
                else:
                    accu[x,y,z] = accu[x-1,y,z]

    #get refaccu
    for y in range(dimy):
        for z in range(dimz):
            refaccu[0,y,z] = 0
    for x in range(1,depth_channel):
        for y in range(dimy):
            for z in range(dimz):
                if refvoxels[x,y,z]>0:
                    refaccu[x,y,z] = refaccu[x-1,y,z] + 1
                else:
                    refaccu[x,y,z] = refaccu[x-1,y,z]
    
    #find boundary voxels and put into queue
    q_start = 0
    q_end = 0
    for y in range(1,dimy-1):
        for z in range(1,dimz-1):
            queue[q_end,0] = y
            queue[q_end,1] = z
            q_end += 1
            if q_end==queue_len: q_end = 0
    
    while q_start!=q_end:
        y = queue[q_start,0]
        z = queue[q_start,1]
        q_start += 1
        if q_start==queue_len: q_start = 0


        this_depth = accu[depth_channel,y,z]
        if refvoxels[this_depth,y,z]==0:
            tmp_depth = accu[depth_channel,y-1,z]
            if this_depth>tmp_depth:
                if this_depth-tmp_depth != accu[this_depth,y,z] - accu[tmp_depth,y,z]:
                    x = this_depth
                    while voxels[x,y,z]>0:
                        x -= 1
                    if refaccu[this_depth,y,z] - refaccu[x,y,z] == 0:

                        #remove voxels
                        x = this_depth
                        while voxels[x,y,z]>0:
                            voxels[x,y,z] = 0
                            x -= 1

                        #update accu
                        accu[depth_channel,y,z] = 0
                        for x in range(1,depth_channel):
                            if voxels[x,y,z]>0:
                                accu[x,y,z] = accu[x-1,y,z] + 1
                                accu[depth_channel,y,z] = x
                            else:
                                accu[x,y,z] = accu[x-1,y,z]

                        #put neighbors into queue
                        queue[q_end,0] = y
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y-1
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y+1
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y
                        queue[q_end,1] = z-1
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y
                        queue[q_end,1] = z+1
                        q_end += 1
                        if q_end==queue_len: q_end = 0


        this_depth = accu[depth_channel,y,z]
        if refvoxels[this_depth,y,z]==0:
            tmp_depth = accu[depth_channel,y+1,z]
            if this_depth>tmp_depth:
                if this_depth-tmp_depth != accu[this_depth,y,z] - accu[tmp_depth,y,z]:
                    x = this_depth
                    while voxels[x,y,z]>0:
                        x -= 1
                    if refaccu[this_depth,y,z] - refaccu[x,y,z] == 0:

                        #remove voxels
                        x = this_depth
                        while voxels[x,y,z]>0:
                            voxels[x,y,z] = 0
                            x -= 1

                        #update accu
                        accu[depth_channel,y,z] = 0
                        for x in range(1,depth_channel):
                            if voxels[x,y,z]>0:
                                accu[x,y,z] = accu[x-1,y,z] + 1
                                accu[depth_channel,y,z] = x
                            else:
                                accu[x,y,z] = accu[x-1,y,z]

                        #put neighbors into queue
                        queue[q_end,0] = y
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y-1
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y+1
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y
                        queue[q_end,1] = z-1
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y
                        queue[q_end,1] = z+1
                        q_end += 1
                        if q_end==queue_len: q_end = 0

        this_depth = accu[depth_channel,y,z]
        if refvoxels[this_depth,y,z]==0:
            tmp_depth = accu[depth_channel,y,z-1]
            if this_depth>tmp_depth:
                if this_depth-tmp_depth != accu[this_depth,y,z] - accu[tmp_depth,y,z]:
                    x = this_depth
                    while voxels[x,y,z]>0:
                        x -= 1
                    if refaccu[this_depth,y,z] - refaccu[x,y,z] == 0:

                        #remove voxels
                        x = this_depth
                        while voxels[x,y,z]>0:
                            voxels[x,y,z] = 0
                            x -= 1

                        #update accu
                        accu[depth_channel,y,z] = 0
                        for x in range(1,depth_channel):
                            if voxels[x,y,z]>0:
                                accu[x,y,z] = accu[x-1,y,z] + 1
                                accu[depth_channel,y,z] = x
                            else:
                                accu[x,y,z] = accu[x-1,y,z]

                        #put neighbors into queue
                        queue[q_end,0] = y
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y-1
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y+1
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y
                        queue[q_end,1] = z-1
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y
                        queue[q_end,1] = z+1
                        q_end += 1
                        if q_end==queue_len: q_end = 0

        this_depth = accu[depth_channel,y,z]
        if refvoxels[this_depth,y,z]==0:
            tmp_depth = accu[depth_channel,y,z+1]
            if this_depth>tmp_depth:
                if this_depth-tmp_depth != accu[this_depth,y,z] - accu[tmp_depth,y,z]:
                    x = this_depth
                    while voxels[x,y,z]>0:
                        x -= 1
                    if refaccu[this_depth,y,z] - refaccu[x,y,z] == 0:

                        #remove voxels
                        x = this_depth
                        while voxels[x,y,z]>0:
                            voxels[x,y,z] = 0
                            x -= 1

                        #update accu
                        accu[depth_channel,y,z] = 0
                        for x in range(1,depth_channel):
                            if voxels[x,y,z]>0:
                                accu[x,y,z] = accu[x-1,y,z] + 1
                                accu[depth_channel,y,z] = x
                            else:
                                accu[x,y,z] = accu[x-1,y,z]

                        #put neighbors into queue
                        queue[q_end,0] = y
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y-1
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y+1
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y
                        queue[q_end,1] = z-1
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y
                        queue[q_end,1] = z+1
                        q_end += 1
                        if q_end==queue_len: q_end = 0

        #lower corners
        this_depth = accu[depth_channel,y,z]
        if refvoxels[this_depth,y,z]==0:
            k = 0
            if k==0:
                i = accu[depth_channel,y-1,z]
                j = accu[depth_channel,y,z-1]
                if i<this_depth and j<this_depth:
                    if i>j: tmp_depth = i
                    else: tmp_depth = j
                    k = 1
            if k==0:
                i = accu[depth_channel,y+1,z]
                j = accu[depth_channel,y,z-1]
                if i<this_depth and j<this_depth:
                    if i>j: tmp_depth = i
                    else: tmp_depth = j
                    k = 1
            if k==0:
                i = accu[depth_channel,y-1,z]
                j = accu[depth_channel,y,z+1]
                if i<this_depth and j<this_depth:
                    if i>j: tmp_depth = i
                    else: tmp_depth = j
                    k = 1
            if k==0:
                i = accu[depth_channel,y+1,z]
                j = accu[depth_channel,y,z+1]
                if i<this_depth and j<this_depth:
                    if i>j: tmp_depth = i
                    else: tmp_depth = j
                    k = 1

            if k>0:
                x = this_depth
                while voxels[x,y,z]>0 and refvoxels[x,y,z]==0 and x>tmp_depth:
                    voxels[x,y,z] = 0
                    x -= 1

                #update accu
                accu[depth_channel,y,z] = 0
                for x in range(1,depth_channel):
                    if voxels[x,y,z]>0:
                        accu[x,y,z] = accu[x-1,y,z] + 1
                        accu[depth_channel,y,z] = x
                    else:
                        accu[x,y,z] = accu[x-1,y,z]

                #put neighbors into queue
                queue[q_end,0] = y
                queue[q_end,1] = z
                q_end += 1
                if q_end==queue_len: q_end = 0
                queue[q_end,0] = y-1
                queue[q_end,1] = z
                q_end += 1
                if q_end==queue_len: q_end = 0
                queue[q_end,0] = y+1
                queue[q_end,1] = z
                q_end += 1
                if q_end==queue_len: q_end = 0
                queue[q_end,0] = y
                queue[q_end,1] = z-1
                q_end += 1
                if q_end==queue_len: q_end = 0
                queue[q_end,0] = y
                queue[q_end,1] = z+1
                q_end += 1
                if q_end==queue_len: q_end = 0



def get_rays(char[:, :, ::1] voxels, int[::1] ray_x1, int[::1] ray_y1, int[::1] ray_z1, int[::1] ray_x2, int[::1] ray_y2, int[::1] ray_z2, char[:, :, ::1] visibility_flag):
    #record occluded voxel count

    cdef int dimx,dimy,dimz,dimz2
    cdef int i,j,k,p

    dimx = voxels.shape[0]
    dimy = voxels.shape[1]
    dimz = voxels.shape[2]
    dimz2 = dimz//2

    
    #get visibility_flag
    for i in range(dimx):
        for k in range(dimz):
            j = dimy-1
            if voxels[i,j,k]==0:
                visibility_flag[i,j,k] = 0
            for j in range(dimy-2,-1,-1):
                if voxels[i,j,k]==0:
                    if i==0 or i==dimx-1 or k==0 or k==dimz-1:
                        visibility_flag[i,j,k] = 0
                    elif visibility_flag[i-1,j+1,k]==0:
                        visibility_flag[i,j,k] = 0
                    elif visibility_flag[i+1,j+1,k]==0:
                        visibility_flag[i,j,k] = 0
                    elif visibility_flag[i,j+1,k-1]==0:
                        visibility_flag[i,j,k] = 0
                    elif visibility_flag[i,j+1,k+1]==0:
                        visibility_flag[i,j,k] = 0
                    elif visibility_flag[i-1,j+1,k-1]==0:
                        visibility_flag[i,j,k] = 0
                    elif visibility_flag[i-1,j+1,k+1]==0:
                        visibility_flag[i,j,k] = 0
                    elif visibility_flag[i+1,j+1,k-1]==0:
                        visibility_flag[i,j,k] = 0
                    elif visibility_flag[i+1,j+1,k+1]==0:
                        visibility_flag[i,j,k] = 0

    #y axis, top down
    for i in range(dimx):
        for k in range(dimz):
            p = 2
            for j in range(dimy):
                if voxels[i,j,k]>0:
                    if p==0:
                        p = 1
                        ray_y1[j] += 1
                    if p==2:
                        p = 1
                else:
                    if p==1:
                        p = 0
            p = 2
            for j in range(dimy-1,-1,-1):
                if voxels[i,j,k]>0:
                    if p==0:
                        p = 1
                        ray_y2[j] += 1
                    if p==2:
                        p = 1
                else:
                    if p==1:
                        p = 0

    #x axis, front back

    for j in range(dimy):
        for k in range(dimz):
            p = 2
            for i in range(dimx):
                if voxels[i,j,k]>0:
                    if p==0:
                        p = 1
                        if visibility_flag[i-1,j,k]==0:
                            ray_x1[i] += 1
                    if p==2:
                        p = 1
                else:
                    if p==1:
                        p = 0
            p = 2
            for i in range(dimx-1,-1,-1):
                if voxels[i,j,k]>0:
                    if p==0:
                        p = 1
                        if visibility_flag[i+1,j,k]==0:
                            ray_x2[i] += 1
                    if p==2:
                        p = 1
                else:
                    if p==1:
                        p = 0


    #special treatment for z axis, the symmetry one

    for i in range(dimx):
        for j in range(dimy):
            p = 2
            for k in range(dimz):
                if voxels[i,j,k]>0:
                    if p==0:
                        p = 1
                        if visibility_flag[i,j,k-1]==0:
                            ray_z1[k] += 1
                    if p==2:
                        p = 1
                else:
                    if p==1:
                        p = 0
                if k==dimz2:
                    if p==0:
                        p = 2
            p = 2
            for k in range(dimz-1,-1,-1):
                if voxels[i,j,k]>0:
                    if p==0:
                        p = 1
                        if visibility_flag[i,j,k+1]==0:
                            ray_z2[k] += 1
                    if p==2:
                        p = 1
                else:
                    if p==1:
                        p = 0
                if k==dimz2:
                    if p==0:
                        p = 2



