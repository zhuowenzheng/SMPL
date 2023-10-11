
import copy
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from task1 import save_obj
from task1 import load_obj
from MatrixStack import MatrixStack as m_stack

matrix_stack = m_stack()
class Bone:
    def __init__(self, joint_index, parent_index, rotation_order, joint_name):
        self.joint_index = joint_index
        self.parent_index = parent_index
        self.rotation_order = rotation_order
        self.joint_name = joint_name
        self.T = np.eye(4)
        self.R = np.eye(4)

class BoneTransform:
    def __init__(self, qx, qy, qz, qw, px, py, pz):
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw
        self.px = px
        self.py = py
        self.pz = pz


def load_hierarchy(file_path):
    hierarchy_map = {}
    try:
        with open(file_path, 'r') as file:
            # Skip the first three lines starting with '#'
            for _ in range(3):
                next(file)
            bone_count = int(next(file).strip())  # Read the bone count from the fourth line
            for line in file:
                parts = line.strip().split()
                joint_index, parent_index, rotation_order, joint_name = list(map(int, parts[:2])) + parts[2:]
                bone = Bone(joint_index, parent_index, rotation_order, joint_name)
                hierarchy_map[joint_index] = bone
        return hierarchy_map
    except FileNotFoundError:
        print(f"Cannot read {file_path}")
        return None

# print the hierarchy map

def load_skeleton(file_name):
    filename = os.path.join(DATA_DIR, file_name)
    _translation = np.array([0, 0, 0])  # Initialize translation here
    try:
        with open(filename, 'r') as file:
            print(f"Loading {filename}")

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
                    _rotation = R.from_quat([_bone_transforms[frame][bone].qx,
                                             _bone_transforms[frame][bone].qy,
                                             _bone_transforms[frame][bone].qz,
                                             _bone_transforms[frame][bone].qw])
                    M = np.eye(4)
                    M[:3, :3] = _rotation.as_matrix()
                    M[:, 3] = np.append(_translation, 1)
                    _bone_matrices_inv[frame][bone] = np.linalg.inv(M)

                    M = np.eye(4)
                    M[:3, :3] = _rotation.as_matrix()
                    M[:, 3] = np.append(_translation, 1)
                    _bone_matrices[frame][bone] = M

                    if bone == 16:  # Check if the bone is the left shoulder at index 16

                        # Push the current bone matrix onto the matrix stack
                        matrix_stack.push(_bone_matrices[frame][bone])
                        print("before")
                        print(_bone_matrices[frame][bone])
                        # Apply the rotation using the provided quaternion(w, x, y, z)
                        quaternion = [0.9239, 0, 0, -0.3827]
                        matrix_stack.rotate(quaternion)

                        # Update the bone matrix with the transformed matrix from the matrix stack
                        _bone_matrices[frame][bone] = matrix_stack.top()
                        print("after")
                        print(_bone_matrices[frame][bone])
                        # Pop the current matrix off the matrix stack
                        matrix_stack.pop()

                        print("bone " + str(bone) + " rotated")
                transform_child(frame, hierarchy_map, _bone_matrices, _bone_matrices_inv)

    except FileNotFoundError:
        print(f"Cannot read {filename}")

    return _bone_matrices, _bone_matrices_inv


def transform_child(frame, hierarchy_map, _bone_matrices, _bone_matrices_inv, parent_bone_index=16):
    for key, value in hierarchy_map.items():
        if value.parent_index == parent_bone_index:

            # Push the current bone matrix onto the stack
            matrix_stack.push(_bone_matrices[frame][key])

            # Apply the rotation
            q = [0.9239, 0, 0, -0.3827]
            matrix_stack.rotate(q)

            # Update the bone matrix with the modified matrix from the stack
            _bone_matrices[frame][key] = matrix_stack.top()

            print("bone " + str(value.joint_index) + " rotated")
            print("after ", key)
            print(_bone_matrices[frame][key])
            # Recursively call transform_child on the current bone to transform its children
            transform_child(frame, hierarchy_map, _bone_matrices, _bone_matrices_inv,
                            parent_bone_index=value.joint_index)

            # Pop the current bone matrix off the stack, reverting back to the parent's state
            matrix_stack.pop()


# Load hierarchy data
hierarchy_map = load_hierarchy('../input/smpl_hierarchy.txt')

# Compute absolute transforms
# compute_absolute_transforms(hierarchy_map)


def generate_skel(base_bone_matrices, base_bone_matrices_inv, betas, frame_count=1, bone_count=24):
    # Creating deep copies to ensure the original matrices are not modified
    new_bone_matrices = copy.deepcopy(base_bone_matrices)
    new_bone_matrices_inv = copy.deepcopy(base_bone_matrices_inv)

    b_mats = [load_skeleton('smpl_skel{:02d}.txt'.format(i),)[0] for i in range(1, 11)]
    b_mats_inv = [load_skeleton('smpl_skel{:02d}.txt'.format(i))[1] for i in range(1, 11)]

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


DATA_DIR = "../input/"

# smpl_00 is the base mesh
# smpl_01 to smpl_10 are the delta blendshapes

# Load the skeleton data
bone_matrices, bone_matrices_inv = load_skeleton('smpl_skel00.txt')

# Generate New Meshes
beta1 = [-1.711935, 2.352964, 2.285835, -0.073122, 1.501402, -1.790568, -0.391194, 2.078678, 1.461037, 2.297462]
beta2 = [1.573618, 2.028960, -1.865066, 2.066879, 0.661796, -2.012298, -1.107509, 0.234408, 2.287534, 2.324443]

b1_bone_matrices, b1_bone_matrices_inv = generate_skel(bone_matrices, bone_matrices_inv, beta1)
b2_bone_matrices, b2_bone_matrices_inv = generate_skel(bone_matrices, bone_matrices_inv, beta2)


# Apply Skinning
class ShapeSkin:
    def __init__(self):
        self.boneCount = 0
        self.vertexCount = 0
        self.maxInfluences = 0
        self.boneIndices = []
        self.skinningWeights = []

    def load_attachment(self, filename):
        filepath = os.path.join("../input", filename)
        try:
            with open(filepath, 'r') as file:
                print(f"Loading {filepath}")

                # Skip the first four lines
                for _ in range(4):
                    next(file)

                # Read the initial numbers
                vertCount, boneCount, maxInfluences = map(int, next(file).split())

                self.boneCount = boneCount
                self.vertexCount = vertCount
                self.maxInfluences = maxInfluences

                print(f"vertCount: {vertCount}")
                print(f"boneCount: {boneCount}")

                # Resize the vectors to the correct size

                self.boneIndices = [[0 for _ in range(maxInfluences)] for _ in range(vertCount)]
                self.skinningWeights = [[0.0 for _ in range(maxInfluences)] for _ in range(vertCount)]

                # Read the skinning weights and bone indices
                for i in range(vertCount):
                    line = next(file)
                    values = iter(line.split())
                    for j in range(maxInfluences):
                        self.boneIndices[i][j] = int(next(values))
                        self.skinningWeights[i][j] = float(next(values))

        except FileNotFoundError:
            print(f"Cannot read {filepath}")

    def linear_blended_skinning(self, k, bone_mat, bone_mat_inv, mesh_dir):
        # CPU skinning calculations
        # Linear Blend Skinning

        global xi_k
        M_product = []
        # pos_buf is a numpy array ,size = 3 * vertexCount
        pos_buf = load_obj(mesh_dir)
        newPosBuf = np.zeros(len(pos_buf) * 3)

        for j in range(self.boneCount):
            Mjk = bone_mat[k][j]
            Mj0_inv = bone_mat_inv[0][j]
            M_product.append(np.dot(Mjk, Mj0_inv))

        # Iterate through all related bones
        for i, vertex in enumerate(pos_buf):
            x, y, z = vertex

            xi_0 = np.array([x, y, z, 1])
            xi_k = np.zeros(4)

            J = self.boneIndices[i]
            w = self.skinningWeights[i]

            for n in range(self.maxInfluences):
                xi_k += w[n] * np.dot(M_product[J[n]], xi_0)

            # send xi_k to posbuf
            newPosBuf[i * 3] = xi_k[0]
            newPosBuf[i * 3 + 1] = xi_k[1]
            newPosBuf[i * 3 + 2] = xi_k[2]

        return newPosBuf


shape_skin = ShapeSkin()
shape_skin.load_attachment('smpl_skin.txt')

# Compose posBuf into a list of vertices
vertices_b0 = shape_skin.linear_blended_skinning(0, bone_matrices, bone_matrices_inv, '../output1/frame000.obj')
vertices_b1 = shape_skin.linear_blended_skinning(0, b1_bone_matrices, b1_bone_matrices_inv, '../output1/frame001.obj')
vertices_b2 = shape_skin.linear_blended_skinning(0, b2_bone_matrices, b2_bone_matrices_inv, '../output1/frame002.obj')

tuple_vertices_b0 = [(vertices_b0[i], vertices_b0[i + 1], vertices_b0[i + 2]) for i in range(0, len(vertices_b0), 3)]
tuple_vertices_b1 = [(vertices_b1[i], vertices_b1[i + 1], vertices_b1[i + 2]) for i in range(0, len(vertices_b1), 3)]
tuple_vertices_b2 = [(vertices_b2[i], vertices_b2[i + 1], vertices_b2[i + 2]) for i in range(0, len(vertices_b2), 3)]

# Save the original and new meshes
save_obj('../output4/frame000.obj', tuple_vertices_b0)
save_obj('../output4/frame001.obj', tuple_vertices_b1)
save_obj('../output4/frame002.obj', tuple_vertices_b2)


