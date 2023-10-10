import os

import pywavefront


def load_obj(filename):
    scene = pywavefront.Wavefront(filename, collect_faces=True)
    vertices = scene.vertices
    return vertices


# Compute Delta Blendshapes
base_vertices = load_obj('../input/smpl_00.obj')


delta_blendshapes = [load_obj('../input/smpl_{:02d}.obj'.format(i)) for i in range(1, 11)]
for i, blendshape in enumerate(delta_blendshapes):
    delta_blendshapes[i] = [(blendshape_v[0] - base_v[0], blendshape_v[1] - base_v[1], blendshape_v[2] - base_v[2]) for
                            blendshape_v, base_v in zip(blendshape, base_vertices)]

# Generate New Meshes

beta1 = [-1.711935, 2.352964, 2.285835, -0.073122, 1.501402, -1.790568, -0.391194, 2.078678, 1.461037, 2.297462]
beta2 = [1.573618, 2.028960, -1.865066, 2.066879, 0.661796, -2.012298, -1.107509, 0.234408, 2.287534, 2.324443]


def generate_mesh(base_vertices, delta_blendshapes, betas):
    new_vertices = []
    for i, base_v in enumerate(base_vertices):
        delta_sum = [0, 0, 0]
        for beta, blendshape in zip(betas, delta_blendshapes):
            delta_v = blendshape[i]  # Get the corresponding vertex from each blendshape

            delta_sum[0] += beta * delta_v[0]
            delta_sum[1] += beta * delta_v[1]
            delta_sum[2] += beta * delta_v[2]
        new_vertex = (
            base_v[0] + delta_sum[0],
            base_v[1] + delta_sum[1],
            base_v[2] + delta_sum[2]
        )
        new_vertices.append(new_vertex)
    return new_vertices

mesh1_vertices = generate_mesh(base_vertices, delta_blendshapes, beta1)
mesh2_vertices = generate_mesh(base_vertices, delta_blendshapes, beta2)


# Export New Meshes

def save_obj(filename, vertices):
    # Extract directory name from the filename
    directory = os.path.dirname(filename)

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    with open(filename, 'w') as file:
        for vertex in vertices:
            file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')


# Save the original and new meshes
save_obj('../output1/frame000.obj', base_vertices)
save_obj('../output1/frame001.obj', mesh1_vertices)
save_obj('../output1/frame002.obj', mesh2_vertices)
