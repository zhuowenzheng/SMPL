from task5 import shape_skin, load_skeleton, load_hierarchy, load_quaternions, generate_skel
from task1 import save_obj
import numpy as np

DATA_DIR = '../input'
hierarchy_map = load_hierarchy('../input/smpl_hierarchy.txt')

quaternion_data_7516 = load_quaternions('../input/smpl_quaternions_mosh_cmu_7516.txt')
quaternion_data_8806 = load_quaternions('../input/smpl_quaternions_mosh_cmu_8806.txt')

mocap_b0_matrices_7516, mocap_b0_matrices_inv_7516 = load_skeleton('../input/smpl_skel00.txt')

# # 0-94
# for k in range(95):
#     # animating and skinning using quaternion_data_7516[k]
#     segment_7516_b0 = shape_skin.linear_blended_skinning(k, mocap_b0_matrices_7516, mocap_b0_matrices_inv_7516, '../output1/frame000.obj')
#     tuple_segment_7516_b0 = [(segment_7516_b0[i], segment_7516_b0[i + 1], segment_7516_b0[i + 2]) for i in range(0, len(segment_7516_b0), 3)]
#
#     save_obj('../output6/frame{:03d}.obj'.format(k), tuple_segment_7516_b0)


beta0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
beta1 = np.array([-1.711935, 2.352964, 2.285835, -0.073122, 1.501402, -1.790568, -0.391194, 2.078678, 1.461037, 2.297462])
beta2 = np.array([1.573618, 2.028960, -1.865066, 2.066879, 0.661796, -2.012298, -1.107509, 0.234408, 2.287534, 2.324443])

# 95-144
for k in range(50):
    t = k / 49  # t goes from 0 to 1
    beta = beta0 * (1 - t) + beta1 * t
    base = [mocap_b0_matrices_7516[94]]
    base_inv = [mocap_b0_matrices_inv_7516[94]]

    interpolate_7516_b0b1, interpolate_7516_b0b1_inv = generate_skel(base, base_inv, beta, frame_count=1, bone_count=24)
    segment_7516_b0b1 = shape_skin.linear_blended_skinning(0, interpolate_7516_b0b1, interpolate_7516_b0b1_inv, '../output1/frame000.obj')
    tuple_segment_7516_b0b1 = [(segment_7516_b0b1[i], segment_7516_b0b1[i + 1], segment_7516_b0b1[i + 2]) for i in range(0, len(segment_7516_b0b1), 3)]

    save_obj('../output6/frame{:03d}.obj'.format(k + 95), tuple_segment_7516_b0b1)



    #save_obj(f'../output6/frame{str(k + 95).zfill(3)}.obj', tuple_vertices_b0)
