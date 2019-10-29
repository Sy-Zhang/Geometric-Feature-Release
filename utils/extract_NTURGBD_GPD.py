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
    frame_count = int(lines[0])
    positions = {}
    starts = {}
    ends = {}
    ind = 1
    for f in xrange(0, frame_count):
        body_count = int(lines[ind])
        ind = ind + 1
        for b in xrange(0, body_count):
            body_id = lines[ind].split(' ')[0]
            trackingState = lines[ind].split(' ')[9]

            if body_id not in positions:
                starts[body_id] = f

                positions[body_id] = np.zeros((frame_count, 25, 3))
            joint_count = int(lines[ind + 1])
            ind = ind + 2
            for j in xrange(0, joint_count):
                line = lines[ind + j]
                positions[body_id][f, j] = [
                    float(w) for w in line.split(' ')[0:3]]
            ends[body_id] = f
            ind = ind + joint_count

    body_var = {}

    if bool(positions):
        for key, value in positions.iteritems():
            # if frame_count != len(np.trim_zeros(value[:, 1, 0])):
            # print file, key, frame_count, len(np.trim_zeros(value[:, 1, 0]))
            body_var[key] = 0
            for j in xrange(0, 25):
                body_var[key] += np.var(value[starts[key]:ends[key], j, 0]) + \
                    np.var(value[starts[key]:ends[key], j, 1]) + \
                    np.var(value[starts[key]:ends[key], j, 2])
        import operator
        chosen_body_id = max(body_var.iteritems(),
                             key=operator.itemgetter(1))[0]
        pos = positions[chosen_body_id]
        start = starts[chosen_body_id]
        end = ends[chosen_body_id] + 1
        return pos[start:end], start, end, frame_count, chosen_body_id
    else:
        return None, None, None, frame_count, None

def rotation(joints):
    j_c = joints[:] - joints[1]
    x = joints[4]-joints[8]
    y = joints[20]-joints[0]
    z = np.cross(x,y)
    R = np.array(normalize([x,y,z]))
    return R

def length(joints,lines):
    j1,j2 = joints[lines.swapaxes(1,0)]
    L = np.sum(np.sqrt(np.sum(np.square(j1-j2),axis=1)))
    return L

def joint_coordinate(joints,rotation, L):
    J = np.dot(joints-joints[1],rotation)
    J = J/L
    J = np.nan_to_num(J)
    j_c = np.append(joints[1][1],np.append([J[0]],J[2:]))
    return j_c

def joint_joint_distance(joints):
    D = pdist(joints)
    D = np.nan_to_num(D)
    return D

def joint_joint_orientation(joints,mask):
    J = np.repeat(joints, 25, axis=0).reshape(25, 25, 3)
    J_T = J.swapaxes(1, 0)
    O = J - J_T
    jj_o = normalize(np.ma.array(O,mask=mask).compressed().reshape(-1,3))
    return jj_o

def get_jj(jj,i,j):
    # jj could be jj_o or jj_d
    j1 = np.minimum(i,j)
    j2 = np.maximum(i,j)
    return jj[25*i-(1+i)*i/2+j-i-1]

def joint_line_distance(joints,lines,mask):

    j1,j2 = joints[lines.swapaxes(1,0)]

    J1 = np.repeat(j1, 25, axis=0).reshape(39,25,3)
    J2 = np.repeat(j2, 25, axis=0).reshape(39,25,3)
    J3 = np.repeat(joints,39,axis=0).reshape(39,25,3,order='F')

    A = np.sqrt(np.sum(np.square(J1-J3),axis=2))
    B = np.sqrt(np.sum(np.square(J2-J3),axis=2))
    C = np.sqrt(np.sum(np.square(J1-J2),axis=2))

    P = (A+B+C)/2
    jl_d = 2*np.sqrt(P*(P-A)*(P-B)*(P-C))/(C)
    jl_d = np.ma.array(jl_d,mask=mask).compressed()
    jl_d = np.nan_to_num(jl_d)
    return jl_d

def line_line_angle(jj_o,lines,mask):
    A = get_jj(jj_o, lines[:,0], lines[:,1])
    J = np.repeat(A, 39, axis=0).reshape(39,39,3)
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

def line_joint_ratio(joints,lines,mask):

    j1,j2 = joints[lines.swapaxes(1,0)]

    J1 = np.repeat(j1, 25, axis=0).reshape(39,25,3)
    J2 = np.repeat(j2, 25, axis=0).reshape(39,25,3)
    J3 = np.repeat(joints,39,axis=0).reshape(39,25,3,order='F')

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

    f = h5py.File("nturgbdTest.hdf5", 'a')

    skeleton_root = '/home/zsy/data/AllSkeletonFiles_remove_nan_nolabel'
    index = 0

    # J1 and J2 directly adjacent in the kinectic chain
    lines = np.array([[1,2],[2,21],[21,9],[21,5],[21,3],[3,4],
        [5,6],[6,7],[7,8],[8,22],[8,23],[9,10],[10,11],[11,12],
        [12,25],[12,24],[1,17],[17,18],[18,19],[19,20],[1,13],[13,14],[14,15],
        [15,16]])

    # J1 is end site, J2 is two steps away
    lines = np.append(lines,[[4,21],[11,9],[7,5],[13,15],[17,19]],axis=0)

    # Both J1 and J2 are end site
    lines = np.append(lines, [[4,11],[4,7],[4,19],[4,15],
        [11,19],[11,15],[11,7],[7,15],[7,19],[15,19]], axis=0)
    lines = lines - 1

    planes = np.array([[3,4,21],[9,10,11],[5,6,7],[17,18,19],[13,14,15]])-1

    jj_mask = np.full((25,25,3), False, dtype=bool)
    for i in xrange(0,25):
        jj_mask[i,0:i+1,:] = True

    jl_mask = np.full((39,25), False, dtype=bool)
    for i in xrange(0,39):
        jl_mask[i][lines[i]] = True

    ll_mask = np.full((39,39), False, dtype=bool)
    for i in xrange(0,39):
        ll_mask[i][0:i+1] = True


    jp_index = np.zeros((22,5,2),dtype='int')
    for i in xrange(0,5):
        gen = (k for k in xrange(0,25) if k not in planes[i])
        j = 0
        for k in gen:
            jp_index[j,i,0] = min(k,planes[i,0])
            jp_index[j,i,1] = max(k,planes[i,0])
            j = j+1

    lp_index = np.zeros((36,5,2),dtype='int')
    for i in xrange(0,5):
        j = 0
        for x,y in lines:
            if x in planes[i] and y in planes[i]:
                continue
            lp_index[j,i,:] = [x,y]
            j = j+1

    for root, dirs, files in os.walk(skeleton_root):
        for file in sorted(files):
            basename = file.split('.')[0]
            grp = f.require_group(basename)
            file_path = os.path.join(root, file)
            pos, start, end, frame_count, body_id = read_skeleton_file(
                file_path)

            grp.attrs.modify('frame_count', frame_count)
            if body_id is None:
                index = index + 1
                continue
            grp.attrs.modify('body_id', body_id)
            grp.attrs.modify('start', start)
            grp.attrs.modify('end', end)

            J_C = grp.require_dataset("J_c",shape=(end-start,73), dtype='float32', chunks=True)
            JJ_D = grp.require_dataset("JJ_d",shape=(end-start,300), dtype='float32', chunks=True)
            JJ_O = grp.require_dataset("JJ_o",shape=(end-start,900), dtype='float32', chunks=True)
            JL_D = grp.require_dataset("JL_d",shape=(end-start,897), dtype='float32', chunks=True)
            LL_A = grp.require_dataset("LL_a",shape=(end-start,741), dtype='float32', chunks=True)
            JP_D = grp.require_dataset("JP_d",shape=(end-start,110), dtype='float32', chunks=True)
            LP_A = grp.require_dataset("LP_a",shape=(end-start,180), dtype='float32', chunks=True)
            PP_A = grp.require_dataset("PP_a",shape=(end-start,10), dtype='float32', chunks=True)
            # LJ_r = grp.require_dataset("LJ_r",shape=(end-start,1127), dtype='float32', chunks=True)

            R = np.array(Parallel(n_jobs=24)(delayed(rotation)(pos[i]) for i in range(end-start)))
            L = np.array(Parallel(n_jobs=24)(delayed(length)(pos[i],lines[0:24]) for i in range(end-start)))
            J_C[:] = np.array(Parallel(n_jobs=24)(delayed(joint_coordinate)(pos[i], R[i], L[i]) for i in range(end-start)))
            JJ_D[:] = np.array(Parallel(n_jobs=24)(delayed(joint_joint_distance)(pos[i]) for i in range(end-start)))
            jj_o = np.array(Parallel(n_jobs=24)(delayed(joint_joint_orientation)(pos[i],jj_mask) for i in range(end-start)))
            JL_D[:] = np.array(Parallel(n_jobs=24)(delayed(joint_line_distance)(pos[i],lines,jl_mask) for i in range(end-start)))
            LL_A[:] = np.array(Parallel(n_jobs=24)(delayed(line_line_angle)(jj_o[i],lines,ll_mask) for i in range(end-start)))
            JP_D[:] = np.array(Parallel(n_jobs=24)(delayed(joint_plane_distance)(JJ_D[i],jj_o[i],planes,jp_index) for i in range(end-start)))
            LP_A[:] = np.array(Parallel(n_jobs=24)(delayed(line_plane_angle)(jj_o[i],planes,lp_index) for i in range(end-start)))
            PP_A[:] = np.array(Parallel(n_jobs=24)(delayed(plane_plane_angle)(jj_o[i],planes) for i in range(end-start)))
            JJ_O[:] = np.array(Parallel(n_jobs=24)(delayed(np.dot)(jj_o[i],R[i]) for i in range(end-start))).reshape(start-end,900)
            # LJ_r[:] = np.array(Parallel(n_jobs=24)(delayed(line_joint_ratio)(pos[i],lines,jl_mask) for i in range(end-start)))

            index = index + 1
            print index,'/',len(files),file