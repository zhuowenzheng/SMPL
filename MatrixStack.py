import numpy as np

class MatrixStack:
    def __init__(self):
        # Initialize with an identity matrix
        self.stack = [np.eye(4)]

    def push(self, matrix=None):
        # Push a copy of the current matrix or the provided matrix onto the stack
        if matrix is None:
            self.stack.append(self.stack[-1].copy())
        else:
            self.stack.append(matrix)

    def pop(self):
        # Pop the top matrix from the stack
        if len(self.stack) > 1:  # Always leave at least one matrix (identity)
            return self.stack.pop()
        raise ValueError("Cannot pop the base identity matrix")

    def top(self):
        # Get the top matrix from the stack without popping
        return self.stack[-1]

    def multiply(self, matrix):
        # Multiply the top matrix with the provided matrix
        self.stack[-1] = np.dot(self.stack[-1], matrix)

    def translate(self, tx, ty, tz):
        translation = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])
        self.multiply(translation)

    def scale(self, sx, sy, sz):
        scaling = np.array([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ])
        self.multiply(scaling)

    def quaternion_to_matrix(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
            [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w), 0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2), 0],
            [0, 0, 0, 1]
        ])

    def rotate(self, q):
        rotation_matrix = self.quaternion_to_matrix(q)
        self.multiply(rotation_matrix)




