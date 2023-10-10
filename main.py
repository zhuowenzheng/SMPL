import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Load OBJ file
def load_obj(filename):
    vertices = []
    with open(filename) as file:
        for line in file:
            if line.startswith('v '):
                vertex = list(map(float, line.strip().split()[1:]))
                vertices.append(vertex)
    return vertices

# Initialize Pygame
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

# Setup OpenGL
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

# Load the OBJ data
vertices = load_obj('../output2/frame000.obj')

# Run the event loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Render the OBJ
    glBegin(GL_TRIANGLES)
    for vertex in vertices:
        glVertex3fv(vertex)
    glEnd()

    # Swap buffers
    pygame.display.flip()
    pygame.time.wait(10)

pygame.quit()
