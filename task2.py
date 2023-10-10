import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from task1 import save_obj
from task1 import load_obj


class BoneTransform:
    def __init__(self, qx, qy, qz, qw, px, py, pz):
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw
        self.px = px
        self.py = py
        self.pz = pz


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
            # print(frame_count, bone_count)

            _bone_transforms = [[BoneTransform(0, 0, 0, 0, 0, 0, 0) for _ in range(bone_count)] for _ in
                                range(frame_count)]
            _bone_matrices = [[np.eye(4) for _ in range(bone_count)] for _ in range(frame_count)]
            _bone_matrices_inv = [[np.eye(4) for _ in range(bone_count)] for _ in range(frame_count)]

            for frame in range(frame_count):
                line = next(file)
                values = map(float, line.split())
                for bone in range(bone_count):
                    _bone_transforms[frame][bone] = BoneTransform(*[next(values) for _ in range(7)])
                    if bone == 18:  # Check if the bone is the one at index 18
                        _translation = np.array([_bone_transforms[frame][bone].px,
                                                 _bone_transforms[frame][bone].py + 0.2,  # Add 0.2 to the y-coordinate
                                                 _bone_transforms[frame][bone].pz])
                        print("translated bone 18")
                        print(_translation)
                    else:
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
                    _bone_matrices[frame][bone] = M
                    _bone_matrices_inv[frame][bone] = np.linalg.inv(M)

    except FileNotFoundError:
        print(f"Cannot read {filename}")

    return _translation, _bone_matrices, _bone_matrices_inv


def generate_skel(base_translation, translations_list, betas):
    new_translations = []
    for i in range(len(base_translation)):
        delta_sum = np.array([0.0, 0.0, 0.0])
        for beta, translation_list in zip(betas, translations_list):
            delta_t = np.array(translation_list[i])  # Get the corresponding translation from each translation list
            delta_sum += beta * (delta_t - np.array(base_translation[i]))

        new_translation = tuple(delta_sum + np.array(base_translation[i]))
        new_translations.append(new_translation)
    return new_translations


DATA_DIR = "../input/"
# bone_transforms, bone_matrices, bone_matrices_inv = load_skeleton()
# smpl_00 is the base mesh
# smpl_01 to smpl_10 are the delta blendshapes

# Load the skeleton data
translation, bone_matrices, bone_matrices_inv = load_skeleton('smpl_skel00.txt')


translations_list = [tuple(translation)]
for i in range(1, 11):
    _translations, _, __ = load_skeleton(f'smpl_skel{i:02d}.txt')
    translations_list.append(tuple(_translations))

# Generate New Meshes
beta1 = [-1.711935, 2.352964, 2.285835, -0.073122, 1.501402, -1.790568, -0.391194, 2.078678, 1.461037, 2.297462]
beta2 = [1.573618, 2.028960, -1.865066, 2.066879, 0.661796, -2.012298, -1.107509, 0.234408, 2.287534, 2.324443]

skel_b1 = generate_skel(translation, translations_list, beta1)
skel_b2 = generate_skel(translation, translations_list, beta2)


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


    def linear_blended_skinning(self, k, boneMatrices, boneMatricesInv):
        # CPU skinning calculations
        # Linear Blend Skinning

        global xi_k
        M_product = []
        # pos_buf is a numpy array ,size = 3 * vertexCount
        pos_buf = load_obj('../input/smpl_00.obj')
        newPosBuf = np.zeros(len(pos_buf)*3)

        for j in range(self.boneCount):
            Mjk = boneMatrices[k][j]
            Mj0_inv = boneMatricesInv[0][j]
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
vertices = shape_skin.linear_blended_skinning(0, bone_matrices, bone_matrices_inv)
tuple_vertices = [(vertices[i], vertices[i + 1], vertices[i + 2]) for i in range(0, len(vertices), 3)]

# Save the original and new meshes
save_obj('../output2/frame000.obj', tuple_vertices)
save_obj('../output2/frame001.obj', skel_b1)
save_obj('../output2/frame002.obj', skel_b2)
