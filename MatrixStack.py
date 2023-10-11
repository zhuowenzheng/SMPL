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

    def rotate(self, angle, x, y, z):
        # Rotation using the Rodrigues' rotation formula
        rad_angle = np.radians(angle)
        axis = np.array([x, y, z])
        axis = axis / np.linalg.norm(axis)
        cosA = np.cos(rad_angle)
        sinA = np.sin(rad_angle)
        rotation = np.eye(4)
        rotation[:3, :3] = cosA * np.eye(3) + (1 - cosA) * np.outer(axis, axis) + sinA * np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        self.multiply(rotation)


# Example usage:
stack = MatrixStack()

stack.push()  # Push the identity matrix
stack.translate(1, 2, 3)
print(stack.top())

stack.push()  # Push the current matrix
stack.rotate(45, 0, 1, 0)
print(stack.top())

stack.pop()
print(stack.top())
