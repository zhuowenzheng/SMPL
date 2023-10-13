from task5 import ShapeSkin, load_skeleton, load_hierarchy, load_quaternions, generate_skel
from task1 import save_obj

import numpy as np

if __name__ == "__main__":
    try:
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
        mocap_b0_matrices_7516, mocap_b0_matrices_inv_7516 = load_skeleton('../input/smpl_skel00.txt',
                                                                           quaternion_data_7516)
        mocap_b1_matrices_7516, mocap_b1_matrices_inv_7516 = generate_skel(mocap_b0_matrices_7516,
                                                                           mocap_b0_matrices_inv_7516,
                                                                           beta1, quaternion_data_7516, frame_count=160,
                                                                           bone_count=24)
        mocap_b2_matrices_7516, mocap_b2_matrices_inv_7516 = generate_skel(mocap_b0_matrices_7516,
                                                                           mocap_b0_matrices_inv_7516,
                                                                           beta2, quaternion_data_7516, frame_count=160,
                                                                           bone_count=24)

        print("Loading and generating skels for 8806...")
        mocap_b0_matrices_8806, mocap_b0_matrices_inv_8806 = load_skeleton('../input/smpl_skel00.txt',
                                                                           quaternion_data_8806)
        mocap_b1_matrices_8806, mocap_b1_matrices_inv_8806 = generate_skel(mocap_b0_matrices_8806,
                                                                           mocap_b0_matrices_inv_8806,
                                                                           beta1, quaternion_data_8806, frame_count=160,
                                                                           bone_count=24)
        mocap_b2_matrices_8806, mocap_b2_matrices_inv_8806 = generate_skel(mocap_b0_matrices_8806,
                                                                           mocap_b0_matrices_inv_8806,
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
        vertices_7516_b0_94 = shape_skin.linear_blended_skinning(94, mocap_b0_matrices_7516, mocap_b0_matrices_inv_7516,
                                                                 '../output1/frame000.obj')
        tuple_vertices_7516_b0_94 = [(vertices_7516_b0_94[i], vertices_7516_b0_94[i + 1], vertices_7516_b0_94[i + 2])
                                     for i
                                     in
                                     range(0, len(vertices_7516_b0_94), 3)]
        vertices_7516_b1_94 = shape_skin.linear_blended_skinning(94, mocap_b1_matrices_7516, mocap_b1_matrices_inv_7516,
                                                                 '../output1/frame001.obj')
        tuple_vertices_7516_b1_94 = [(vertices_7516_b1_94[i], vertices_7516_b1_94[i + 1], vertices_7516_b1_94[i + 2])
                                     for i
                                     in
                                     range(0, len(vertices_7516_b1_94), 3)]

        for k in range(50):
            t = k / 49  # t goes from 0 to 1
            vertices_intp1 = []
            for i in range(len(tuple_vertices_7516_b0_94)):
                v_temp = [0.0, 0.0, 0.0]
                for j in range(3):
                    v_temp[j] = (1 - t) * tuple_vertices_7516_b0_94[i][j] + t * tuple_vertices_7516_b1_94[i][j]
                vertices_intp1.append(v_temp)
            tuple_vertices_intp1 = [(vertices_intp1[i][0], vertices_intp1[i][1], vertices_intp1[i][2]) for i in
                                    range(0, len(vertices_intp1))]
            save_obj('../output6/frame{:03d}.obj'.format(k + 95), tuple_vertices_intp1)

        print("Generating frame 145-209...")
        # 145-209
        for k in range(65):
            # finish playing the last 65 frames of mocap_b1
            segment_7516_b1 = shape_skin.linear_blended_skinning(k + 95, mocap_b1_matrices_7516,
                                                                 mocap_b1_matrices_inv_7516,
                                                                 '../output1/frame001.obj')
            tuple_segment_7516_b1 = [(segment_7516_b1[i], segment_7516_b1[i + 1], segment_7516_b1[i + 2]) for i in
                                     range(0, len(segment_7516_b1), 3)]

            save_obj('../output6/frame{:03d}.obj'.format(k + 145), tuple_segment_7516_b1)

        # negating if dot product < 0
        quaternion_last_7516 = quaternion_data_7516[159]  # xyzw
        quaternion_first_8806 = quaternion_data_8806[0]  # xyzw

        quaternion_last_7516 = np.array(quaternion_last_7516)
        quaternion_first_8806 = np.array(quaternion_first_8806)
        # long-short problem
        for i in range(24):
            dot_product = np.dot(quaternion_last_7516[i], quaternion_first_8806[i])
            if dot_product < 0:
                # Negate all elements in both quaternions
                quaternion_last_7516[i] = -quaternion_last_7516[i]
                quaternion_first_8806[i] = -quaternion_first_8806[i]

        # 210-259
        # generate quaternions for 50 frames
        quaternions_7516_8806 = []
        for k in range(50):
            t = k / 49  # t goes from 0 to 1
            frame_quaternions = []
            for i in range(24):
                quat = [quaternion_last_7516[i][j] * (1 - t) + quaternion_first_8806[i][j] * t for j in range(4)]
                # Normalize the quaternion
                norm = np.linalg.norm(quat)
                if norm != 0:
                    quat = [q / norm for q in quat]
                frame_quaternions.append(quat)

            quaternions_7516_8806.append(frame_quaternions)
        # rotation_mat_7516_8806 = [R.from_quat(q).as_matrix() for q in normalized_quaternions]

        intpq_7516_8806, intpq_7516_8806_inv = load_skeleton('../input/smpl_skel00.txt', quaternions_7516_8806,
                                                             frame_=50)
        b1_intpq_7516_8806, b1_intpq_7516_8806_inv = generate_skel(intpq_7516_8806, intpq_7516_8806_inv, beta1,
                                                                   quaternions_7516_8806, frame_count=50, bone_count=24,
                                                                   frame_=50)

        print("Generating frame 210-259...")
        for k in range(50):
            segment_7516_8806 = shape_skin.linear_blended_skinning(k, b1_intpq_7516_8806, b1_intpq_7516_8806_inv,
                                                                   '../output1/frame001.obj')
            tuple_segment_7516_8806 = [(segment_7516_8806[i], segment_7516_8806[i + 1], segment_7516_8806[i + 2]) for i
                                       in
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

        # 340-389
        print("Generating frame 340-399...")
        # Similar to 95-144
        vertices_8806_b1_79 = shape_skin.linear_blended_skinning(79, mocap_b1_matrices_8806, mocap_b1_matrices_inv_8806,
                                                                 '../output1/frame001.obj')
        tuple_vertices_8806_b1_79 = [(vertices_8806_b1_79[i], vertices_8806_b1_79[i + 1], vertices_8806_b1_79[i + 2])
                                     for i
                                     in
                                     range(0, len(vertices_8806_b1_79), 3)]
        vertices_8806_b2_79 = shape_skin.linear_blended_skinning(79, mocap_b2_matrices_8806, mocap_b2_matrices_inv_8806,
                                                                 '../output1/frame002.obj')
        tuple_vertices_8806_b2_79 = [(vertices_8806_b2_79[i], vertices_8806_b2_79[i + 1], vertices_8806_b2_79[i + 2])
                                     for i
                                     in
                                     range(0, len(vertices_8806_b2_79), 3)]

        for k in range(50):
            t = k / 49  # t goes from 0 to 1
            vertices_intp2 = []
            for i in range(len(tuple_vertices_8806_b1_79)):
                v_temp = [0.0, 0.0, 0.0]
                for j in range(3):
                    v_temp[j] = (1 - t) * tuple_vertices_8806_b1_79[i][j] + t * tuple_vertices_8806_b2_79[i][j]
                vertices_intp2.append(v_temp)
            tuple_vertices_intp2 = [(vertices_intp2[i][0], vertices_intp2[i][1], vertices_intp2[i][2]) for i in
                                    range(0, len(vertices_intp2))]
            save_obj('../output6/frame{:03d}.obj'.format(k + 340), tuple_vertices_intp2)

        # 390-479
        print("Generating frame 390-469...")
        for k in range(80):
            # play the last 80 frames of 8806 in b2
            segment_8806_b2 = shape_skin.linear_blended_skinning(k + 80, mocap_b2_matrices_8806,
                                                                 mocap_b2_matrices_inv_8806,
                                                                 '../output1/frame002.obj')
            tuple_segment_8806_b2 = [(segment_8806_b2[i], segment_8806_b2[i + 1], segment_8806_b2[i + 2]) for i in
                                     range(0, len(segment_8806_b2), 3)]

            save_obj('../output6/frame{:03d}.obj'.format(k + 390), tuple_segment_8806_b2)
        print("Task 6 done.")
    except Exception as e:
        print(f"An error occurred: {e}")