from task5 import shape_skin, load_skeleton, load_hierarchy, load_quaternions, generate_skel
from task1 import save_obj
import numpy as np

beta1 = np.array([-1.711935, 2.352964, 2.285835, -0.073122, 1.501402, -1.790568, -0.391194, 2.078678, 1.461037, 2.297462])
beta2 = np.array([1.573618, 2.028960, -1.865066, 2.066879, 0.661796, -2.012298, -1.107509, 0.234408, 2.287534, 2.324443])

DATA_DIR = '../input'
hierarchy_map = load_hierarchy('../input/smpl_hierarchy.txt')

quaternion_data_7516 = load_quaternions('../input/smpl_quaternions_mosh_cmu_7516.txt')
quaternion_data_8806 = load_quaternions('../input/smpl_quaternions_mosh_cmu_8806.txt')

mocap_b0_matrices_7516, mocap_b0_matrices_inv_7516 = load_skeleton('../input/smpl_skel00.txt', quaternion_data_7516)
mocap_b1_matrices_7516, mocap_b1_matrices_inv_7516 = generate_skel(mocap_b0_matrices_7516, mocap_b0_matrices_inv_7516, beta1, frame_count=160, bone_count=24)
mocap_b2_matrices_7516, mocap_b2_matrices_inv_7516 = generate_skel(mocap_b0_matrices_7516, mocap_b0_matrices_inv_7516, beta2, frame_count=160, bone_count=24)

mocap_b0_matrices_8806, mocap_b0_matrices_inv_8806 = load_skeleton('../input/smpl_skel00.txt', quaternion_data_8806)
mocap_b1_matrices_8806, mocap_b1_matrices_inv_8806 = generate_skel(mocap_b0_matrices_8806, mocap_b0_matrices_inv_8806, beta1, frame_count=160, bone_count=24)
mocap_b2_matrices_8806, mocap_b2_matrices_inv_8806 = generate_skel(mocap_b0_matrices_8806, mocap_b0_matrices_inv_8806, beta2, frame_count=160, bone_count=24)

# # 0-94
# for k in range(95):
#     # animating and skinning using quaternion_data_7516[k]
#     segment_7516_b0 = shape_skin.linear_blended_skinning(k, mocap_b0_matrices_7516, mocap_b0_matrices_inv_7516, '../output1/frame000.obj')
#     tuple_segment_7516_b0 = [(segment_7516_b0[i], segment_7516_b0[i + 1], segment_7516_b0[i + 2]) for i in range(0, len(segment_7516_b0), 3)]
#
#     save_obj('../output6/frame{:03d}.obj'.format(k), tuple_segment_7516_b0)
base = [mocap_b0_matrices_7516[94]]
base_inv = [mocap_b0_matrices_inv_7516[94]]
b1 = [mocap_b1_matrices_7516[94]]
b1_inv = [mocap_b1_matrices_inv_7516[94]]

# 95-144
for k in range(50):
    t = k / 49  # t goes from 0 to 1
    # interpolate_7516_b0b1, interpolate_7516_b0b1_inv = generate_skel(base, base_inv, beta, frame_count=1, bone_count=24)
    segment_7516_b0 = shape_skin.linear_blended_skinning(0, base, base_inv, '../output1/frame000.obj')
    segment_7516_b1 = shape_skin.linear_blended_skinning(0, b1, b1_inv, '../output1/frame001.obj')
    segment_7516_b0b1 = segment_7516_b0 * (1 - t) + segment_7516_b1 * t
    tuple_segment_7516_b0b1 = [(segment_7516_b0b1[i], segment_7516_b0b1[i + 1], segment_7516_b0b1[i + 2]) for i in range(0, len(segment_7516_b0b1), 3)]

    save_obj('../output6/frame{:03d}.obj'.format(k + 95), tuple_segment_7516_b0b1)



    #save_obj(f'../output6/frame{str(k + 95).zfill(3)}.obj', tuple_vertices_b0)

#145-209
# for k in range(65):
#     # finish playing the last 65 frames of mocap_b1
#     segment_7516_b1 = shape_skin.linear_blended_skinning(k+95, mocap_b1_matrices_7516, mocap_b1_matrices_inv_7516, '../output1/frame001.obj')
#     tuple_segment_7516_b1 = [(segment_7516_b1[i], segment_7516_b1[i + 1], segment_7516_b1[i + 2]) for i in range(0, len(segment_7516_b1), 3)]
#
#     save_obj('../output6/frame{:03d}.obj'.format(k + 145), tuple_segment_7516_b1)
#

#Once mocap data #7516 finishes, use 50 frames to linearly interpolate the rotations (quaternions) between the last frame of #7516 and the first frame of #8806.
#210-259
last_7516 = [mocap_b1_matrices_7516[94]]



for k in range(50):
    #last frame of #7516
