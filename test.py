from pythreejs import * 
import numpy as np


from ipywidgets.embed import embed_minimal_html
import webbrowser
import os


## Geometry
def draw_plane(origin=[0, 0, 0], normal=[0, 1, 0], size=10, color='gray', opacity=0.5):
    """Draw a plane with a given normal and size."""
    # Define the plane geometry
    plane_geometry = PlaneGeometry(width=size, height=size)
    plane_material = MeshStandardMaterial(
        color=color, 
        transparent=True, 
        opacity=opacity, 
        side='DoubleSide')
    plane = Mesh(geometry=plane_geometry, material=plane_material)
    plane.receiveShadow = True
    
    # Rotate the plane to the normal vector
    default_normal = np.array([0, 0, 1])  # PlaneGeometry default normal
    axis = np.cross(default_normal, normal)
    angle = np.arccos(np.dot(default_normal, normal))  # Angle between vectors
    
    if np.linalg.norm(axis) > 1e-6:  # Avoid division by zero if vectors are parallel
        axis = axis / np.linalg.norm(axis)
        plane.quaternion = list(np.append(axis * np.sin(angle / 2), np.cos(angle / 2)))  # Quaternion rotation 
    
    plane.position = origin
    return plane

def draw_axis():
    axis_vertices = np.array([
    [0, 0, 0], [1, 0, 0],  # X-axis (Red)
    [0, 0, 0], [0, 1, 0],  # Y-axis (Green)
    [0, 0, 0], [0, 0, 1]   # Z-axis (Blue)
    ], dtype=np.float32)

    axis_colors = np.array([
        [1, 0, 0], [1, 0, 0],  # Red for X
        [0, 1, 0], [0, 1, 0],  # Green for Y
        [0, 0, 1], [0, 0, 1]   # Blue for Z
    ], dtype=np.float32)

    # Create BufferGeometry for axes
    axis_geometry = BufferGeometry(
        attributes={
            'position': BufferAttribute(array=axis_vertices, normalized=False),
            'color': BufferAttribute(array=axis_colors, normalized=False),
        }
    )

    # Create material (Use vertex colors)
    axis_material = LineBasicMaterial(vertexColors='VertexColors', depthTest=True)

    # Create Line object
    axis_lines = LineSegments(geometry=axis_geometry, material=axis_material)
    return axis_lines

def generate_cube(width=1.0, height=1.0, depth=1.0):
    """Generate vertices and faces for a box."""
    vertices = np.array([
        [-width/2, -height/2, -depth/2],  # 0 - Back Bottom Left
        [ width/2, -height/2, -depth/2],  # 1 - Back Bottom Right
        [ width/2,  height/2, -depth/2],  # 2 - Back Top Right
        [-width/2,  height/2, -depth/2],  # 3 - Back Top Left
        [-width/2, -height/2,  depth/2],  # 4 - Front Bottom Left
        [ width/2, -height/2,  depth/2],  # 5 - Front Bottom Right
        [ width/2,  height/2,  depth/2],  # 6 - Front Top Right
        [-width/2,  height/2,  depth/2]   # 7 - Front Top Left
    ], dtype=np.float32)

    # Define 12 triangles (6 faces x 2 triangles per face)
    faces = np.array([
        [4, 5, 6], [4, 6, 7],  # Front (+Z)
        [1, 0, 3], [1, 3, 2],  # Back (-Z)
        [5, 1, 2], [5, 2, 6],  # Right (+X)
        [0, 4, 7], [0, 7, 3],  # Left (-X)
        [3, 7, 6], [3, 6, 2],  # Top (+Y)
        [0, 1, 5], [0, 5, 4]   # Bottom (-Y)
    ], dtype=np.uint32)
    
    # Define normals per triangle face (each normal repeats for 2 triangles per face)
    normals = np.array([
        [-0.577, -0.577, -0.577],  # 0 - Averaged from Back, Bottom, Left
        [ 0.577, -0.577, -0.577],  # 1 - Averaged from Back, Bottom, Right
        [ 0.577,  0.577, -0.577],  # 2 - Averaged from Back, Top, Right
        [-0.577,  0.577, -0.577],  # 3 - Averaged from Back, Top, Left
        [-0.577, -0.577,  0.577],  # 4 - Averaged from Front, Bottom, Left
        [ 0.577, -0.577,  0.577],  # 5 - Averaged from Front, Bottom, Right
        [ 0.577,  0.577,  0.577],  # 6 - Averaged from Front, Top, Right
        [-0.577,  0.577,  0.577]   # 7 - Averaged from Front, Top, Left
    ], dtype=np.float32)
    return vertices, faces, normals

def draw_cube(width=1.0, height=1.0, depth=1.0, color='green', position=[0, 0, 0]):
    """Draw a cube with a given size and color."""
    vertices, faces, normals = generate_cube(width, height, depth)
    geometry = BufferGeometry(
        attributes={
            'position': BufferAttribute(array=vertices.copy(), normalized=False),
            'normal': BufferAttribute(array=normals.copy(), normalized=True),
        },
        index=BufferAttribute(array=faces.copy().ravel(), normalized=False)
    )
    material = MeshStandardMaterial(
        color=color,  # RGB color of the material
        flatShading=True,
    )
    cube = Mesh(geometry=geometry, material=material)
    cube.castShadow = True
    
    cube.geometry.attributes['position'].array += np.array(position, dtype=np.float32)
    return cube
    




if __name__ == "__main__":
    # Create a 3D scene
    scene = Scene(background='black')  # Background color of the scene
    camera = PerspectiveCamera(
        position=[5, 5, 10], 
        lookAt=[0, 0, 0]
        )
    
    ambient_light = AmbientLight(color='white', intensity=1.0)  # Soft overall light
    point_light = PointLight(color='white', intensity=1.0, position=[2, 2, 2])  # Spot source near origin
    directional_light = DirectionalLight(color='white', intensity=1.0, position=[5, 5, -5])  # Sun-like
    spot_light = SpotLight(
        color='white',
        intensity=3.0,        # Increase brightness
        position=[0, 5, 0],   # Place the light above the cube
        distance=10,          # Light range (covers cube)
        angle=0.5,            # Cone angle (medium focus)
        penumbra=0.3,         # Soft shadow edges
        decay=2,              # Realistic falloff
    )
    spot_light.castShadow = True  # Enable shadow casting
    
    scene.add(ambient_light)
    # scene.add(point_light)
    # scene.add(directional_light)
    scene.add(spot_light)

    # Create a cube
    cube1 = draw_cube(width=1.0, height=1.0, depth=1.0, color='green', position=[1, 1, 0])
    def axis_angle_to_quaternion(axis, angle):
        """Convert an axis-angle rotation to a quaternion"""
        axis = np.array(axis, dtype=np.float32)
        axis = axis / np.linalg.norm(axis)  # Normalize axis
        sin_half_angle = np.sin(angle / 2)
        
        return tuple(list(axis * sin_half_angle) + [np.cos(angle / 2)])
    q_y = axis_angle_to_quaternion([0, 1, 0], np.pi / 4)
    # Apply quaternion to cube
    cube1.quaternion = q_y
    cube2 = draw_cube(width=1.0, height=1.0, depth=1.0, color='red', position=[0, 1, 0])
    
    axis = draw_axis()
    plane = draw_plane(normal=[0, 1, 0], size=10, color='gray', opacity=0.5)
    # Add to scene
    scene.add(cube1)
    scene.add(cube2)
    scene.add(axis)
    scene.add(plane)



    # Create a renderer
    renderer = Renderer(
        scene=scene, 
        camera=camera, 
        controls=[OrbitControls(controlling=camera)],
        width=800,
        height=800, 
        )
    renderer.shadowMap.enabled = True
    renderer.shadowMap.type = 'PCFSoftShadowMap'

    # Save as an HTML file in the workspace folder
    html_file = "index.html"
    embed_minimal_html(html_file, views=[renderer], title="Live Server Example")

    print(f"HTML file saved as {html_file}. Open with Live Server in VSCode.")

