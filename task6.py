import copy

from task5 import ShapeSkin, load_skeleton, load_hierarchy, load_quaternions, generate_skel
from task1 import save_obj
from scipy.spatial.transform import Rotation as R
import numpy as np

shape_skin = ShapeSkin()
shape_skin.load_attachment('smpl_skin.txt')

beta0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
beta1 = np.array(
    [-1.711935, 2.352964, 2.285835, -0.073122, 1.501402, -1.790568, -0.391194, 2.078678, 1.461037, 2.297462])
beta2 = np.array(
    [1.573618, 2.028960, -1.865066, 2.066879, 0.661796, -2.012298, -1.107509, 0.234408, 2.287534, 2.324443])

DATA_DIR = '../input'
hierarchy_map = load_hierarchy('../input/smpl_hierarchy.txt')

print("Loading Quaternion Data...")
quaternion_data_7516 = load_quaternions('../input/smpl_quaternions_mosh_cmu_7516.txt')
quaternion_data_8806 = load_quaternions('../input/smpl_quaternions_mosh_cmu_8806.txt')

print("Loading and generating skels for 7516...")
mocap_b0_matrices_7516, mocap_b0_matrices_inv_7516 = load_skeleton('../input/smpl_skel00.txt', quaternion_data_7516)
mocap_b1_matrices_7516, mocap_b1_matrices_inv_7516 = generate_skel(mocap_b0_matrices_7516, mocap_b0_matrices_inv_7516,
                                                                   beta1, quaternion_data_7516, frame_count=160,
                                                                   bone_count=24)
mocap_b2_matrices_7516, mocap_b2_matrices_inv_7516 = generate_skel(mocap_b0_matrices_7516, mocap_b0_matrices_inv_7516,
                                                                   beta2, quaternion_data_7516, frame_count=160,
                                                                   bone_count=24)

print("Loading and generating skels for 8806...")
mocap_b0_matrices_8806, mocap_b0_matrices_inv_8806 = load_skeleton('../input/smpl_skel00.txt', quaternion_data_8806)
mocap_b1_matrices_8806, mocap_b1_matrices_inv_8806 = generate_skel(mocap_b0_matrices_8806, mocap_b0_matrices_inv_8806,
                                                                   beta1, quaternion_data_8806, frame_count=160,
                                                                   bone_count=24)
mocap_b2_matrices_8806, mocap_b2_matrices_inv_8806 = generate_skel(mocap_b0_matrices_8806, mocap_b0_matrices_inv_8806,
                                                                   beta2, quaternion_data_8806, frame_count=160,
                                                                   bone_count=24)

print("Generating frame 0-94...")
# 0-94
for k in range(95):
    # animating and skinning using quaternion_data_7516[k]
    segment_7516_b0 = shape_skin.linear_blended_skinning(k, mocap_b0_matrices_7516, mocap_b0_matrices_inv_7516,
                                                         '../output1/frame000.obj')
    tuple_segment_7516_b0 = [(segment_7516_b0[i], segment_7516_b0[i + 1], segment_7516_b0[i + 2]) for i in
                             range(0, len(segment_7516_b0), 3)]

    save_obj('../output6/frame{:03d}.obj'.format(k), tuple_segment_7516_b0)

print("Generating frame 95-144...")
# 95-144
for k in range(50):
    t = k / 49  # t goes from 0 to 1
    beta_intp1 = (1 - t) * beta0 + t * beta1
    # animating and skinning using beta_intp1
    b0b1_7516, b0b1_7516_inv = generate_skel([mocap_b0_matrices_7516[94]],
                                             [mocap_b0_matrices_inv_7516[94]],
                                             beta_intp1, quaternion_data_7516, frame_count=1,
                                             bone_count=24)
    segment_7516_b0b1 = shape_skin.linear_blended_skinning_intp(0, b0b1_7516, b0b1_7516_inv,
                                                                beta_intp1)
    tuple_segment_7516_b0b1 = [(segment_7516_b0b1[i], segment_7516_b0b1[i + 1], segment_7516_b0b1[i + 2]) for i in
                               range(0, len(segment_7516_b0b1), 3)]

    save_obj('../output6/frame{:03d}.obj'.format(k + 95), tuple_segment_7516_b0b1)

# save_obj(f'../output6/frame{str(k + 95).zfill(3)}.obj', tuple_vertices_b0)
print("Generating frame 145-209...")
# 145-209
for k in range(65):
    # finish playing the last 65 frames of mocap_b1
    segment_7516_b1 = shape_skin.linear_blended_skinning(k + 95, mocap_b1_matrices_7516, mocap_b1_matrices_inv_7516,
                                                         '../output1/frame001.obj')
    tuple_segment_7516_b1 = [(segment_7516_b1[i], segment_7516_b1[i + 1], segment_7516_b1[i + 2]) for i in
                             range(0, len(segment_7516_b1), 3)]

    save_obj('../output6/frame{:03d}.obj'.format(k + 145), tuple_segment_7516_b1)

# Once mocap data #7516 finishes, use 50 frames to linearly interpolate the rotations (quaternions) between the last
# frame of #7516 and the first frame of #8806. 210-259 last frame of #7516
last_7516 = mocap_b1_matrices_7516[159]
last_7516_inv = mocap_b1_matrices_inv_7516[159]
# first frame of #8806
first_8806 = mocap_b0_matrices_8806[0]
first_8806_inv = mocap_b0_matrices_inv_8806[0]
# print("last_7516:\n", [last_7516])
# print("first_8806:\n", [first_8806])
# print("last_7516_inv:\n", [mocap_b1_matrices_inv_7516[159]])
# print("first_8806_inv:\n", [mocap_b0_matrices_inv_8806[0]])
translations_last_7516 = [mat[:3, 3] for mat in last_7516]
translations_first_8806 = [mat[:3, 3] for mat in first_8806]
translations_last_inv_7516 = [mat[:3, 3] for mat in last_7516_inv]
translations_first_inv_8806 = [mat[:3, 3] for mat in first_8806_inv]

# negating if dot product < 0
quaternion_last_7516 = quaternion_data_7516[159]  # xyzw
quaternion_first_8806 = quaternion_data_8806[0]  # xyzw

quaternion_last_7516 = np.array(quaternion_last_7516)
quaternion_first_8806 = np.array(quaternion_first_8806)

for i in range(24):
    dot_product = np.dot(quaternion_last_7516[i], quaternion_first_8806[i])
    if dot_product < 0:
        # Negate all elements in both quaternions
        quaternion_last_7516[i] = -quaternion_last_7516[i]
        quaternion_first_8806[i] = -quaternion_first_8806[i]

# 210-259
print("Generating frame 210-259...")
for k in range(50):
    a = k / 49  # t goes from 0 to 1
    quaternion_7516_8806 = [quaternion_last_7516[i][j] * (1 - a) + quaternion_first_8806[i][j] * a for i in range(24)
                            for
                            j in range(4)]
    quaternion_7516_8806_reshaped = [quaternion_7516_8806[i:i + 4] for i in range(0, len(quaternion_7516_8806), 4)]
    # Normalize each of the quaternions
    normalized_quaternions = [q / np.linalg.norm(q) for q in quaternion_7516_8806_reshaped]
    rotation_mat_7516_8806 = [R.from_quat(q).as_matrix() for q in normalized_quaternions]

    # translation
    # interpolated_translations = [(1 - a) * translations_last_7516[i] + a * translations_first_8806[i] for i in
    #                             range(24)]
    # interpolated_translations_inv = [(1 - a) * translations_last_inv_7516[i] + a * translations_first_inv_8806[i] for i
    #                                 in range(24)]

    transformation_mat_7516_8806 = [copy.deepcopy(mocap_b1_matrices_7516[159])]
    transformation_mat_7516_8806_inv = [copy.deepcopy(mocap_b1_matrices_inv_7516[159])]
    for idx, rotation_mat in enumerate(rotation_mat_7516_8806):
        transformation_mat_7516_8806[0][idx][:3, :3] = rotation_mat
        # transformation_mat_7516_8806[0][idx][:3, 3] = interpolated_translations[idx]
        # transformation_mat_7516_8806_inv[0][idx][:3, 3] = -interpolated_translations[idx]
    # if k == 0 or k == 49:
    #     print("k:", k, "\n", transformation_mat_7516_8806[0])
    #     print("k_inv:", k, "\n", transformation_mat_7516_8806_inv[0])
    segment_7516_8806 = shape_skin.linear_blended_skinning(0, transformation_mat_7516_8806,
                                                           transformation_mat_7516_8806_inv, '../output1/frame001.obj')
    tuple_segment_7516_8806 = [(segment_7516_8806[i], segment_7516_8806[i + 1], segment_7516_8806[i + 2]) for i in
                               range(0, len(segment_7516_8806), 3)]

    save_obj('../output6/frame{:03d}.obj'.format(k + 210), tuple_segment_7516_8806)

# 260-339
print("Generating frame 260-339...")

for k in range(80):
    # play the first 80 frames of 8806
    segment_8806_b1 = shape_skin.linear_blended_skinning(k, mocap_b1_matrices_8806, mocap_b1_matrices_inv_8806,
                                                         '../output1/frame001.obj')
    tuple_segment_8806_b1 = [(segment_8806_b1[i], segment_8806_b1[i + 1], segment_8806_b1[i + 2]) for i in
                             range(0, len(segment_8806_b1), 3)]

    save_obj('../output6/frame{:03d}.obj'.format(k + 260), tuple_segment_8806_b1)

# 340-399
print("Generating frame 340-399...")
for k in range(50):
    t = k / 49  # t goes from 0 to 1
    beta_intp2 = (1 - t) * beta1 + t * beta2
    # animating and skinning using beta_intp1
    b1b2_8806, b1b2_8806_inv = generate_skel([mocap_b0_matrices_8806[79]],
                                             [mocap_b0_matrices_inv_8806[79]],
                                             beta_intp2, quaternion_data_8806, frame_count=1,
                                             bone_count=24)
    segment_8806_b1b2 = shape_skin.linear_blended_skinning_intp(0, b1b2_8806, b1b2_8806_inv,
                                                                beta_intp2)
    tuple_segment_8806_b1b2 = [(segment_8806_b1b2[i], segment_8806_b1b2[i + 1], segment_8806_b1b2[i + 2]) for i in
                               range(0, len(segment_8806_b1b2), 3)]

    save_obj('../output6/frame{:03d}.obj'.format(k + 340), tuple_segment_8806_b1b2)

# 400-479
print("Generating frame 400-479...")
for k in range(80):
    # play the last 80 frames of 8806 in b2
    segment_8806_b2 = shape_skin.linear_blended_skinning(k + 80, mocap_b2_matrices_8806, mocap_b2_matrices_inv_8806,
                                                         '../output1/frame002.obj')
    tuple_segment_8806_b2 = [(segment_8806_b2[i], segment_8806_b2[i + 1], segment_8806_b2[i + 2]) for i in
                             range(0, len(segment_8806_b2), 3)]

    save_obj('../output6/frame{:03d}.obj'.format(k + 400), tuple_segment_8806_b2)
