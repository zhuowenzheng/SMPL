import copy

import numpy as np
import os
from task1 import save_obj
from task4 import load_hierarchy, BoneTransform, beta1, beta2, ShapeSkin
from scipy.spatial.transform import Rotation as Rt
from MatrixStack import MatrixStack as m_stack

matrix_stack = m_stack()

DATA_DIR = '../input'

hierarchy_map = load_hierarchy('../input/smpl_hierarchy.txt')


def load_quaternions(file_path):
    quaternion_frames = []
    try:
        with open(file_path, 'r') as file:
            # Skip the first three lines starting with '#'
            for _ in range(4):
                next(file)
            frame_count, bone_count = next(file).strip().split()

            for line in file:
                values = line.strip().split()
                root_translation = list(map(float, values[:3]))  # Capture root translation values
                quaternions = [float(val) for val in values[3:]]  # Capture quaternion values
                # Organize quaternions into groups of 4 (x, y, z, w) for each bone
                quaternions = [quaternions[i:i + 4] for i in range(0, len(quaternions), 4)]
                quaternion_frames.append(quaternions)
    except FileNotFoundError:
        print(f"Cannot read {file_path}")
        return None
    return quaternion_frames


def load_skeleton(file_name, quat_data, frame_=160, bone_=24):
    filename = os.path.join(DATA_DIR, file_name)
    _translation = np.array([0, 0, 0])  # Initialize translation here
    try:
        with open(filename, 'r') as file:
            # print(f"Loading {filename}")

            for _ in range(3):  # Skip the first three lines
                next(file)

            frame_count, bone_count = map(int, next(file).split())
            frame_count += 1  # 0 indexing

            _bone_transforms = [[BoneTransform(0, 0, 0, 0, 0, 0, 0) for _ in range(bone_count)] for _ in
                                range(frame_count)]
            _bone_matrices = [[np.eye(4) for _ in range(bone_count)] for _ in range(frame_count)]
            _bone_matrices_inv = [[np.eye(4) for _ in range(bone_count)] for _ in range(frame_count)]

            for frame in range(frame_count):
                line = next(file)
                values = map(float, line.split())
                for bone in range(bone_count):
                    _bone_transforms[frame][bone] = BoneTransform(*[next(values) for _ in range(7)])

                    _translation = np.array([_bone_transforms[frame][bone].px,
                                             _bone_transforms[frame][bone].py,
                                             _bone_transforms[frame][bone].pz])
                    _rotation = Rt.from_quat([_bone_transforms[frame][bone].qx,
                                              _bone_transforms[frame][bone].qy,
                                              _bone_transforms[frame][bone].qz,
                                              _bone_transforms[frame][bone].qw])
                    M = np.eye(4)
                    M[:3, :3] = _rotation.as_matrix()
                    M[:, 3] = np.append(_translation, 1)
                    _bone_matrices_inv[frame][bone] = np.linalg.inv(M)
                    _bone_matrices[frame][bone] = M
                    # if root, absolute transformation = relative transformation
                    if hierarchy_map[bone].parent_index == -1:
                        hierarchy_map[bone].T = _bone_matrices[frame][bone]
                    else:
                        # absolute translation = child translation - parent translation
                        hierarchy_map[bone].T = _bone_matrices[frame][bone] - _bone_matrices[frame][
                            hierarchy_map[bone].parent_index] + np.eye(4)

            animation(quat_data, hierarchy_map, _bone_matrices, _bone_matrices_inv, frame_, bone_)
            # print("Loading Complete.")
            return _bone_matrices, _bone_matrices_inv
    except FileNotFoundError:
        print(f"Cannot read {filename}")


def accumulate_matrix_to_stack(hierarchy_map, bone_index, matrix_stack):
    """Recursively push matrices to the stack starting from the root to the given bone."""
    if bone_index == -1:  # If the bone has no parent, simply return
        return

    # First, handle the parent's matrices
    parent_bone = hierarchy_map[bone_index].parent_index
    accumulate_matrix_to_stack(hierarchy_map, parent_bone, matrix_stack)

    # Combine the current bone's matrices and then push onto the stack
    current_bone = hierarchy_map[bone_index]
    combined_matrix = np.dot(current_bone.T, current_bone.R)
    matrix_stack.push(np.dot(matrix_stack.top(), combined_matrix))


def update_matrices(frame, hierarchy_map, _bone_matrices, _bone_matrices_inv, child_bone_index):
    matrix_stack.push(np.eye(4))

    # Accumulate matrices onto the stack from the root to the current child
    accumulate_matrix_to_stack(hierarchy_map, child_bone_index, matrix_stack)

    # At this point, matrix_stack.top() contains the accumulated transformation for the child bone
    _bone_matrices[frame][child_bone_index] = matrix_stack.top()
    # Continue traversal for the children of the current bone
    for key, value in hierarchy_map.items():
        if value.parent_index == child_bone_index:
            update_matrices(frame, hierarchy_map, _bone_matrices, _bone_matrices_inv, key)

    # Pop the accumulated transformation before moving to the next sibling bone
    matrix_stack.pop()


# rotate all bones in each frame using quaternions to generate animation mocap matrices
def animation(quat_data, hierarchy_map, _bone_matrices, _bone_matrices_inv, frame_count, bone_count):
    for frame in range(frame_count):
        _bone_matrices.append([np.eye(4) for _ in range(bone_count)])
        _bone_matrices_inv.append([np.eye(4) for _ in range(bone_count)])
        for bone in range(bone_count):
            matrix_stack.push(np.eye(4))
            q = [quat_data[frame][bone][3], quat_data[frame][bone][0],
                 quat_data[frame][bone][1], quat_data[frame][bone][2]]
            matrix_stack.rotate(q)
            hierarchy_map[bone].R = matrix_stack.top()
        update_matrices(frame, hierarchy_map, _bone_matrices, _bone_matrices_inv, 0)


def generate_skel(base_bone_matrices, base_bone_matrices_inv, betas, quat_data, frame_count, bone_count, frame_=160):
    # Creating deep copies to ensure the original matrices are not modified
    new_bone_matrices = copy.deepcopy(base_bone_matrices)
    new_bone_matrices_inv = copy.deepcopy(base_bone_matrices_inv)

    b_mats = [load_skeleton('smpl_skel{:02d}.txt'.format(i), quat_data, frame_)[0] for i in range(1, 11)]
    b_mats_inv = [load_skeleton('smpl_skel{:02d}.txt'.format(i), quat_data, frame_)[1] for i in range(1, 11)]

    for b in range(10):
        for frame in range(frame_count):
            b_mats[b].append([np.eye(4) for _ in range(bone_count)])
            b_mats_inv[b].append([np.eye(4) for _ in range(bone_count)])

    for frame in range(frame_count):
        for bone in range(bone_count):
            delta_p_sum = np.zeros(4)
            delta_p_inv_sum = np.zeros(4)
            for b in range(10):
                dpb = b_mats[b][frame][bone][:, 3] - base_bone_matrices[frame][bone][:, 3]
                dpb_inv = b_mats_inv[b][frame][bone][:, 3] - base_bone_matrices_inv[frame][bone][:, 3]
                delta_p_sum += betas[b] * dpb
                delta_p_inv_sum += betas[b] * dpb_inv
            new_bone_matrices[frame][bone][:, 3] = base_bone_matrices[frame][bone][:, 3] + delta_p_sum
            new_bone_matrices_inv[frame][bone][:, 3] = base_bone_matrices_inv[frame][bone][:, 3] + delta_p_inv_sum

    return new_bone_matrices, new_bone_matrices_inv


if __name__ == "__main__":
    file_path = '../input/smpl_quaternions_mosh_cmu_7516.txt'
    quat_data = load_quaternions(file_path)

    mocap_b0_matrices, mocap_b0_matrices_inv = load_skeleton('smpl_skel00.txt', quat_data)

    mocap_b1_matrices, mocap_b1_matrices_inv = generate_skel(mocap_b0_matrices, mocap_b0_matrices_inv, beta1, quat_data,
                                                             frame_count=160, bone_count=24)
    mocap_b2_matrices, mocap_b2_matrices_inv = generate_skel(mocap_b0_matrices, mocap_b0_matrices_inv, beta2, quat_data,
                                                             frame_count=160, bone_count=24)

    shape_skin = ShapeSkin()
    shape_skin.load_attachment('smpl_skin.txt')

    for k in range(160):
        vertices_b0 = shape_skin.linear_blended_skinning(k, mocap_b0_matrices, mocap_b0_matrices_inv,
                                                         '../output1/frame000.obj')
        vertices_b1 = shape_skin.linear_blended_skinning(k, mocap_b1_matrices, mocap_b1_matrices_inv,
                                                         '../output1/frame001.obj')
        vertices_b2 = shape_skin.linear_blended_skinning(k, mocap_b2_matrices, mocap_b2_matrices_inv,
                                                         '../output1/frame002.obj')

        tuple_vertices_b0 = [(vertices_b0[i], vertices_b0[i + 1], vertices_b0[i + 2]) for i in
                             range(0, len(vertices_b0), 3)]
        tuple_vertices_b1 = [(vertices_b1[i], vertices_b1[i + 1], vertices_b1[i + 2]) for i in
                             range(0, len(vertices_b1), 3)]
        tuple_vertices_b2 = [(vertices_b2[i], vertices_b2[i + 1], vertices_b2[i + 2]) for i in
                             range(0, len(vertices_b2), 3)]

        save_obj('../output5/frame{:03d}.obj'.format(k), tuple_vertices_b0)
        save_obj('../output5/frame{:03d}.obj'.format(k + 160), tuple_vertices_b1)
        save_obj('../output5/frame{:03d}.obj'.format(k + 320), tuple_vertices_b2)
