import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

pygame.init()
screen = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
glEnable(GL_DEPTH_TEST)

# Set up the 3D camera
gluPerspective(45, 800/600, 0.1, 50.0)
glTranslatef(0, 0, -5)

# Cube vertices
cube_vertices = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)
]

# Cube faces
cube_faces = [
    (0, 1, 2, 3), (3, 2, 6, 7), (7, 6, 5, 4), (4, 5, 1, 0),
    (0, 3, 7, 4), (1, 5, 6, 2)
]

def draw_cube():
    glBegin(GL_QUADS)
    for face in cube_faces:
        for vertex in face:
            glVertex3fv(cube_vertices[vertex])
    glEnd()

def get_ray_from_mouse(mx, my):
    """ Converts 2D mouse position into a 3D ray. """
    viewport = glGetIntegerv(GL_VIEWPORT)
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    win_x = float(mx)
    win_y = float(viewport[3] - my)  # Flip y
    win_z = 0.0
    near_point = gluUnProject(win_x, win_y, win_z, modelview, projection, viewport)
    win_z = 1.0
    far_point = gluUnProject(win_x, win_y, win_z, modelview, projection, viewport)
    return near_point, far_point

def ray_intersects_cube(ray_origin, ray_direction):
    """ Checks if the ray intersects the cube's bounding box. """
    min_bounds = [-1, -1, -1]
    max_bounds = [1, 1, 1]

    for i in range(3):
        if ray_direction[i] == 0:
            if ray_origin[i] < min_bounds[i] or ray_origin[i] > max_bounds[i]:
                return False
        else:
            t1 = (min_bounds[i] - ray_origin[i]) / ray_direction[i]
            t2 = (max_bounds[i] - ray_origin[i]) / ray_direction[i]
            if t1 > t2:
                t1, t2 = t2, t1
            if t2 < 0:
                return False
    return True

selected = False
running = True
while running:
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            near_point, far_point = get_ray_from_mouse(mx, my)
            direction = np.subtract(far_point, near_point)
            direction /= np.linalg.norm(direction)
            if ray_intersects_cube(near_point, direction):
                selected = not selected  # Toggle selection

    glColor3f(1, 0, 0) if selected else glColor3f(0, 1, 0)
    draw_cube()

    pygame.display.flip()
    pygame.time.wait(10)

pygame.quit()
