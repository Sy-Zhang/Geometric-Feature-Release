import os
import h5py
import numpy as np
from scipy.spatial.distance import pdist
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize
# version for GPD


def read_skeleton_file(file_path):
    import scipy.io
    pos = scipy.io.loadmat(file_path)['skel_jnt']
    pos = np.reshape(pos,(-1,35,3))
    frame_count = len(pos)
    return pos, frame_count

def rotation(joints):
    j_c = joints[:] - joints[0]
    x = joints[8]-joints[15]
    y = joints[3]-joints[0]
    z = np.cross(x,y)
    R = np.array(normalize([x,y,z]))
    return R

def length(joints,lines):
    j1,j2 = joints[lines.swapaxes(1,0)]
    L = np.sum(np.sqrt(np.sum(np.square(j1-j2),axis=1)))
    return L

def joint_coordinate(joints,rotation, L):
    J = np.dot(joints-joints[0],rotation)
    J = joints-joints[0]
    J = J/L
    J = np.nan_to_num(J)
    j_c = np.append(joints[0][1],J[1:])
    j_c = j_c.flatten()
    return j_c

def joint_joint_distance(joints):
    D = pdist(joints)
    D = np.nan_to_num(D)
    return D

def joint_joint_orientation(joints,mask):
    J = np.repeat(joints, 35, axis=0).reshape(35, 35, 3)
    J_T = J.swapaxes(1, 0)
    O = J - J_T
    jj_o = normalize(np.ma.array(O,mask=mask).compressed().reshape(-1,3))
    return jj_o

def get_jj(jj,i,j):
    # jj could be jj_o or jj_d
    j1 = np.minimum(i,j)
    j2 = np.maximum(i,j)
    return jj[35*i-(1+i)*i/2+j-i-1]

def joint_line_distance(joints,lines,mask):

    j1,j2 = joints[lines.swapaxes(1,0)]

    J1 = np.repeat(j1, 35, axis=0).reshape(47,35,3)
    J2 = np.repeat(j2, 35, axis=0).reshape(47,35,3)
    J3 = np.repeat(joints,47,axis=0).reshape(47,35,3,order='F')

    A = np.sqrt(np.sum(np.square(J1-J3),axis=2))
    B = np.sqrt(np.sum(np.square(J2-J3),axis=2))
    C = np.sqrt(np.sum(np.square(J1-J2),axis=2))

    P = (A+B+C)/2
    M = P*(P-A)*(P-B)*(P-C)
    M[M < 0] = 0
    jl_d = 2*np.sqrt(M)/(C)
    jl_d = np.ma.array(jl_d,mask=mask).compressed()
    jl_d = np.nan_to_num(jl_d)
    jl_d
    return jl_d

def line_line_angle(jj_o,lines,mask):
    A = get_jj(jj_o, lines[:,0], lines[:,1])
    J = np.repeat(A, 47, axis=0).reshape(47,47,3)
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


def line_plane_angle(jj_o,lines,planes,mask):
    A = get_jj(jj_o,lines[:,0],lines[:,1])
    A = np.repeat(A,5,axis=0).reshape(47,5,3)
    B = get_jj(jj_o,planes[:,0],planes[:,1])
    C = get_jj(jj_o,planes[:,0],planes[:,2])
    lp_a = np.ma.array(np.sum(A*np.cross(B,C),axis=2),mask=mask).compressed()
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

def line_joint_ratio(joints,lines,mask):

    j1,j2 = joints[lines.swapaxes(1,0)]

    J1 = np.repeat(j1, 35, axis=0).reshape(47,35,3)
    J2 = np.repeat(j2, 35, axis=0).reshape(47,35,3)
    J3 = np.repeat(joints,47,axis=0).reshape(47,35,3,order='F')

    A = np.sqrt(np.sum(np.square(J1-J3),axis=2))
    B = np.sqrt(np.sum(np.square(J2-J3),axis=2))
    C = np.sqrt(np.sum(np.square(J1-J2),axis=2))

    P = (A+B+C)/2
    jl_d = 2*np.sqrt(P*(P-A)*(P-B)*(P-C))/C#???
    lj_r = np.sqrt(np.square(A)-np.square(jl_d))/C
    lj_r = np.nan_to_num(lj_r)
    lj_r = np.ma.array(lj_r,mask=mask).compressed()
    return lj_r

if __name__ == '__main__':

    f = h5py.File("BerkeleyMHAD-J_c-103.hdf5", 'a')

    skeleton_root = '/home/zsy/data/BerkeleyMHAD'
    index = 0

    # J1 and J2 directly adjacent in the kinectic chain
    lines = np.array([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],
        [4,8],[8,9],[9,10],[10,11],[11,12],[12,13],
        [4,15],[15,16],[16,17],[17,18],[18,19],[19,20],
        [1,22],[22,23],[23,24],[24,25],[25,26],[26,27],[27,28],
        [1,29],[29,30],[30,31],[31,32],[32,33],[33,34],[34,35]])

    # J1 is end site, J2 is two steps away
    lines = np.append(lines,[[3,7],[9,14],[16,21],[22,26],[29,33]],axis=0)

    # Both J1 and J2 are end site
    lines = np.append(lines, [[7,14],[7,21],[7,26],[7,33],[14,21],[14,26],[14,33],
        [21,26],[21,33],[26,33]], axis=0)
    lines = lines - 1

    planes = np.array([[3,5,7],[9,11,14],[16,18,21],[22,24,26],[29,31,33]])-1

    jj_mask = np.full((35,35,3), False, dtype=bool)
    for i in xrange(0,35):
        jj_mask[i,0:i+1,:] = True

    jl_mask = np.full((47,35), False, dtype=bool)
    for i in xrange(0,47):
        jl_mask[i][lines[i]] = True

    ll_mask = np.full((47,47), False, dtype=bool)
    for i in xrange(0,47):
        ll_mask[i][0:i+1] = True


    jp_index = np.zeros((32,5,2),dtype='int')
    for i in xrange(0,5):
        gen = (k for k in xrange(0,35) if k not in planes[i])
        j = 0
        for k in gen:
            jp_index[j,i,0] = min(k,planes[i,0])
            jp_index[j,i,1] = max(k,planes[i,0])
            j = j+1

    lp_mask = np.full((47,5), False, dtype=bool)
    for i in xrange(0,5):
        j = 0
        for x,y in lines:
            if x in planes[i] and y in planes[i]:
                lp_mask[j,i] = True
            j = j+1

    for root, dirs, files in os.walk(skeleton_root):
        for file in sorted(files):
            basename = file.split('.')[0]
            grp = f.require_group(basename)
            file_path = os.path.join(root, file)
            pos, frame_count = read_skeleton_file(file_path)
            # noise = np.random.normal(0,64,pos.size).reshape(pos.shape)
            # pos = pos + noise
            grp.attrs.modify('frame_count', frame_count)

            J_C = grp.require_dataset("J_c",shape=(frame_count,103), dtype='float32', chunks=True)
            JJ_D = grp.require_dataset("JJ_d",shape=(frame_count,595), dtype='float32', chunks=True)
            JJ_O = grp.require_dataset("JJ_o",shape=(frame_count,1785), dtype='float32', chunks=True)
            JL_D = grp.require_dataset("JL_d",shape=(frame_count,1551), dtype='float32', chunks=True)
            LL_A = grp.require_dataset("LL_a",shape=(frame_count,1081), dtype='float32', chunks=True)
            JP_D = grp.require_dataset("JP_d",shape=(frame_count,160), dtype='float32', chunks=True)
            LP_A = grp.require_dataset("LP_a",shape=(frame_count,230), dtype='float32', chunks=True)
            PP_A = grp.require_dataset("PP_a",shape=(frame_count,10), dtype='float32', chunks=True)
            # LJ_r = grp.require_dataset("LJ_r",shape=(frame_count,1127), dtype='float32', chunks=True)

            R = np.array(Parallel(n_jobs=24)(delayed(rotation)(pos[i]) for i in range(frame_count)))
            L = np.array(Parallel(n_jobs=24)(delayed(length)(pos[i],lines[0:32]) for i in range(frame_count)))
            J_C[:] = np.array(Parallel(n_jobs=24)(delayed(joint_coordinate)(pos[i], R[i], L[i]) for i in range(frame_count)))
            JJ_D[:] = np.array(Parallel(n_jobs=24)(delayed(joint_joint_distance)(pos[i]) for i in range(frame_count)))
            jj_o = np.array(Parallel(n_jobs=24)(delayed(joint_joint_orientation)(pos[i],jj_mask) for i in range(frame_count)))
            JL_D[:] = np.array(Parallel(n_jobs=24)(delayed(joint_line_distance)(pos[i],lines,jl_mask) for i in range(frame_count)))
            LL_A[:] = np.array(Parallel(n_jobs=24)(delayed(line_line_angle)(jj_o[i],lines,ll_mask) for i in range(frame_count)))
            JP_D[:] = np.array(Parallel(n_jobs=24)(delayed(joint_plane_distance)(JJ_D[i],jj_o[i],planes,jp_index) for i in range(frame_count)))
            LP_A[:] = np.array(Parallel(n_jobs=24)(delayed(line_plane_angle)(jj_o[i],lines,planes,lp_mask) for i in range(frame_count)))
            PP_A[:] = np.array(Parallel(n_jobs=24)(delayed(plane_plane_angle)(jj_o[i],planes) for i in range(frame_count)))
            JJ_O[:] = np.array(Parallel(n_jobs=24)(delayed(np.dot)(jj_o[i],R[i]) for i in range(frame_count))).reshape(frame_count,1785)
            # LJ_r[:] = np.array(Parallel(n_jobs=24)(delayed(line_joint_ratio)(pos[i],lines,jl_mask) for i in range(frame_count)))
            index = index + 1
            print index,'/',len(files),file