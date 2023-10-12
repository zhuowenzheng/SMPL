from task5 import shape_skin, load_skeleton, load_hierarchy, load_quaternions, generate_skel
from task1 import save_obj
from scipy.spatial.transform import Rotation as R
import numpy as np

beta0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
beta1 = np.array(
    [-1.711935, 2.352964, 2.285835, -0.073122, 1.501402, -1.790568, -0.391194, 2.078678, 1.461037, 2.297462])
beta2 = np.array(
    [1.573618, 2.028960, -1.865066, 2.066879, 0.661796, -2.012298, -1.107509, 0.234408, 2.287534, 2.324443])

DATA_DIR = '../input'
hierarchy_map = load_hierarchy('../input/smpl_hierarchy.txt')

quaternion_data_7516 = load_quaternions('../input/smpl_quaternions_mosh_cmu_7516.txt')
quaternion_data_8806 = load_quaternions('../input/smpl_quaternions_mosh_cmu_8806.txt')

mocap_b0_matrices_7516, mocap_b0_matrices_inv_7516 = load_skeleton('../input/smpl_skel00.txt')
mocap_b1_matrices_7516, mocap_b1_matrices_inv_7516 = generate_skel(mocap_b0_matrices_7516, mocap_b0_matrices_inv_7516,
                                                                   beta1, 160, 24)
mocap_b2_matrices_7516, mocap_b2_matrices_inv_7516 = generate_skel(mocap_b0_matrices_7516, mocap_b0_matrices_inv_7516,
                                                                   beta2, 160, 24)

# # 0-94
# for k in range(95):
#     # animating and skinning using quaternion_data_7516[k]
#     segment_7516_b0 = shape_skin.linear_blended_skinning(k, mocap_b0_matrices_7516, mocap_b0_matrices_inv_7516,
#                                                          '../output1/frame000.obj')
#     tuple_segment_7516_b0 = [(segment_7516_b0[i], segment_7516_b0[i + 1], segment_7516_b0[i + 2]) for i in
#                              range(0, len(segment_7516_b0), 3)]
#     save_obj('../output6/frame{:03d}.obj'.format(k), tuple_segment_7516_b0)


base_7516 = [mocap_b0_matrices_7516[94]]

b1_7516 = [mocap_b1_matrices_7516[94]]

base_rotation_7516 = R.from_matrix([base_7516[0][i][:3, :3] for i in range(24)])
b1_rotation_7516 = R.from_matrix([b1_7516[0][i][:3, :3] for i in range(24)])

base_quaternion_7516 = base_rotation_7516.as_quat()  # xyzw
b1_quaternion_7516 = b1_rotation_7516.as_quat()  # xyzw
# print(base_quaternion_7516)

# 95-144
for k in range(50):
    t = k / 49  # t goes from 0 to 1
    print(t)
    # interpolate the quaternions
    quat_b0b1 = (1 - t) * base_quaternion_7516 + t * b1_quaternion_7516
    # quat_b0b1 = quat_b0b1 / np.linalg.norm(quat_b0b1) # normalize
    print(quat_b0b1)




    # interpolate_7516_b0b1, interpolate_7516_b0b1_inv = generate_skel(base, base_inv, beta, frame_count=1, bone_count=24)
    # segment_7516_b0b1 = shape_skin.linear_blended_skinning(0, interpolate_7516_b0b1, interpolate_7516_b0b1_inv,
    #                                                        '../output1/frame000.obj')
    # tuple_segment_7516_b0b1 = [(segment_7516_b0b1[i], segment_7516_b0b1[i + 1], segment_7516_b0b1[i + 2]) for i in
    #                            range(0, len(segment_7516_b0b1), 3)]
    #
    # save_obj('../output6/frame{:03d}.obj'.format(k + 95), tuple_segment_7516_b0b1)

    # save_obj(f'../output6/frame{str(k + 95).zfill(3)}.obj', tuple_vertices_b0)
