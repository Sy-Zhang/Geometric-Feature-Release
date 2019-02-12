import os
import h5py
import numpy as np
from scipy.spatial.distance import pdist
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize
# version for GPD


def read_skeleton_file(file_path):
    skeleton_file = open(file_path)
    lines = skeleton_file.readlines()
    frame_count = len(lines)
    start = int(lines[0].split(',')[0])
    end = int(lines[frame_count-1].split(',')[0])
    pos = np.zeros((frame_count,30,3))
    for i in range(len(lines)):
        pos[i] = np.array(map(float, lines[i].split(',')[1:])).reshape(30,3)
    return pos, start, end, frame_count

def joint_coordinate(joints):
    j_c = joints.reshape(-1)
    return j_c

def joint_joint_distance(joints):
    D = pdist(joints)
    D = np.nan_to_num(D)
    return D

def joint_joint_orientation(joints,mask):
    J = np.repeat(joints, 30, axis=0).reshape(30, 30, 3)
    J_T = J.swapaxes(1, 0)
    O = J - J_T
    jj_o = normalize(np.ma.array(O,mask=mask).compressed().reshape(-1,3))
    return jj_o

def get_jj(jj,i,j):
    # jj could be jj_o or jj_d
    j1 = np.minimum(i,j)
    j2 = np.maximum(i,j)
    return jj[30*i-(1+i)*i/2+j-i-1]

def joint_line_distance(joints,lines,mask):

    j1,j2 = joints[lines.swapaxes(1,0)]

    J1 = np.repeat(j1, 30, axis=0).reshape(58,30,3)
    J2 = np.repeat(j2, 30, axis=0).reshape(58,30,3)
    J3 = np.repeat(joints,58,axis=0).reshape(58,30,3,order='F')

    A = np.sqrt(np.sum(np.square(J1-J3),axis=2))
    B = np.sqrt(np.sum(np.square(J2-J3),axis=2))
    C = np.sqrt(np.sum(np.square(J1-J2),axis=2))

    P = (A+B+C)/2
    jl_d = 2*np.sqrt(P*(P-A)*(P-B)*(P-C))/C
    jl_d = np.ma.array(jl_d,mask=mask).compressed()
    jl_d = np.nan_to_num(jl_d)
    return jl_d

def line_line_angle(jj_o,lines,mask):
    A = get_jj(jj_o, lines[:,0], lines[:,1])
    J = np.repeat(A, 58, axis=0).reshape(58,58,3)
    J_T = J.swapaxes(1,0)
    ll_a = np.ma.array(np.sum(J*J_T,axis=2),mask=mask).compressed()
    ll_a[np.where(ll_a > 1)] = 1
    ll_a[np.where(ll_a < -1)] = -1
    ll_a = np.arccos(ll_a)
    return ll_a


def joint_plane_distance(jj_d,jj_o,planes,index):
    D = get_jj(jj_d,index[:,:,0],index[:,:,1])
    A = get_jj(jj_o,index[:,:,0],index[:,:,1])
    B = get_jj(jj_o,planes[:,0],planes[:,1])
    C = get_jj(jj_o,planes[:,0],planes[:,2])
    jp_d = D*np.sum(A*np.cross(B,C),axis=2)
    return jp_d.flatten()


def line_plane_angle(jj_o,planes,index):
    A = get_jj(jj_o,index[:,:,0],index[:,:,1])
    B = get_jj(jj_o,planes[:,0],planes[:,1])
    C = get_jj(jj_o,planes[:,0],planes[:,2])
    lp_a = np.sum(A*np.cross(B,C),axis=2)
    lp_a[np.where(lp_a > 1)] = 1
    lp_a[np.where(lp_a < -1)] = -1
    lp_a = np.arccos(lp_a)
    return lp_a.flatten()

def plane_plane_angle(jj_o,planes):
    B = get_jj(jj_o,planes[:,0],planes[:,1])
    C = get_jj(jj_o,planes[:,0],planes[:,2])
    pp_a = pdist(np.cross(B,C), lambda u, v: np.dot(u,v))
    pp_a[np.where(pp_a > 1)] = 1
    pp_a[np.where(pp_a < -1)] = -1
    pp_a = np.arccos(pp_a)
    return pp_a

if __name__ == '__main__':

    f = h5py.File("SBUKinectGPD.hdf5", 'a')

    skeleton_root = '/home/zsy/data/SBU'
    index = 0

    # J1 and J2 directly adjacent in the kinectic chain
    lines = np.array([[1,2],[2,3],[2,4],[2,7],[4,5],[5,6],[7,8],
        [8,9],[3,10],[3,13],[10,11],[11,12],[13,14],[14,15]])

    # J1 is end site, J2 is two steps away
    lines = np.append(lines,[[1,3],[4,6],[7,9],[10,12],[13,15]],axis=0)

    # Both J1 and J2 are end site
    lines = np.append(lines, [[1,6],[1,9],[1,12],[1,15],[6,9],[6,12],[6,15],[9,12],[9,15],[12,15]], axis=0)

    # person B
    lines = np.append(lines, lines+15, axis=0) - 1

    planes = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
    planes = np.append(planes, planes+15, axis=0) - 1

    jj_mask = np.full((30,30,3), False, dtype=bool)
    for i in xrange(0,30):
        jj_mask[i,0:i+1,:] = True

    jl_mask = np.full((58,30), False, dtype=bool)
    for i in xrange(0,58):
        jl_mask[i][lines[i]] = True

    ll_mask = np.full((58,58), False, dtype=bool)
    for i in xrange(0,58):
        ll_mask[i][0:i+1] = True


    jp_index = np.zeros((27,10,2),dtype='int')
    for i in xrange(0,10):
        gen = (k for k in xrange(0,30) if k not in planes[i])
        j = 0
        for k in gen:
            jp_index[j,i,0] = min(k,planes[i,0])
            jp_index[j,i,1] = max(k,planes[i,0])
            j = j+1

    lp_index = np.zeros((55,10,2),dtype='int')
    for i in xrange(0,10):
        j = 0
        for x,y in lines:
            if x in planes[i] and y in planes[i]:
                continue
            lp_index[j,i,:] = [x,y]
            j = j+1

    for root, dirs, files in os.walk(skeleton_root):
        for file in sorted(files):
            if file != 'skeleton_pos.txt':
                continue
            file_path = os.path.join(root, file)
            basename, action_id, tries_id = file_path.split('/')[5:8]
            grp = f.require_group(basename+'/'+action_id+'/'+tries_id)
            pos, start, end, frame_count = read_skeleton_file(
                file_path)

            grp.attrs.modify('frame_count', frame_count)
            grp.attrs.modify('start', start)
            grp.attrs.modify('end', end)

            J_C = grp.require_dataset("J_c",shape=(frame_count,90), dtype='float32', chunks=True)
            JJ_D = grp.require_dataset("JJ_d",shape=(frame_count,435), dtype='float32', chunks=True)
            JJ_O = grp.require_dataset("JJ_o",shape=(frame_count,1305), dtype='float32', chunks=True)
            JL_D = grp.require_dataset("JL_d",shape=(frame_count,1624), dtype='float32', chunks=True)
            LL_A = grp.require_dataset("LL_a",shape=(frame_count,1653), dtype='float32', chunks=True)
            JP_D = grp.require_dataset("JP_d",shape=(frame_count,270), dtype='float32', chunks=True)
            LP_A = grp.require_dataset("LP_a",shape=(frame_count,550), dtype='float32', chunks=True)
            PP_A = grp.require_dataset("PP_a",shape=(frame_count,45), dtype='float32', chunks=True)

            J_C[:] = np.array(Parallel(n_jobs=24)(delayed(joint_coordinate)(pos[i]) for i in range(frame_count)))
            JJ_D[:] = np.array(Parallel(n_jobs=24)(delayed(joint_joint_distance)(pos[i]) for i in range(frame_count)))
            jj_o = np.array(Parallel(n_jobs=24)(delayed(joint_joint_orientation)(pos[i],jj_mask) for i in range(frame_count)))
            JL_D[:] = np.array(Parallel(n_jobs=24)(delayed(joint_line_distance)(pos[i],lines,jl_mask) for i in range(frame_count)))
            LL_A[:] = np.array(Parallel(n_jobs=24)(delayed(line_line_angle)(jj_o[i],lines,ll_mask) for i in range(frame_count)))
            JP_D[:] = np.array(Parallel(n_jobs=24)(delayed(joint_plane_distance)(JJ_D[i],jj_o[i],planes,jp_index) for i in range(frame_count)))
            LP_A[:] = np.array(Parallel(n_jobs=24)(delayed(line_plane_angle)(jj_o[i],planes,lp_index) for i in range(frame_count)))
            PP_A[:] = np.array(Parallel(n_jobs=24)(delayed(plane_plane_angle)(jj_o[i],planes) for i in range(frame_count)))
            JJ_O[:] = jj_o.reshape(frame_count,1305)

            index = index + 1
            print index,'/',len(files),basename+'/'+action_id+'/'+tries_id