import os
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import trimesh
import xml.etree.ElementTree as ET
import time
from concurrent.futures import ThreadPoolExecutor

def euler_to_matrix(rpy):
    roll, pitch, yaw = rpy
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    R = Rz.dot(Ry.dot(Rx))
    return R

class GURDF:
    
    def __init__(self, urdf_file, mesh_path=None):
        self.gpu_buffers = {}  # Store OpenGL VBOs
        
        self.links = {}  
        self.joints = {} 
        self.meshes = {}  
        self.joint_states = {} 
        self.display_lists = {}  
        
        # Thread pool for parallel mesh loading
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        self.urdf_path = urdf_file
        print(f"Loading URDF: {urdf_file}")
        self.mesh_path = mesh_path or os.path.join(os.path.dirname(urdf_file), "meshes")
        print(f"Mesh path: {self.mesh_path}")
        
        self._parse_urdf()
        self._create_display_lists()
        self._create_gpu_buffers()
        
        for joint in self.joints.values():
            self.joint_states[joint["name"]] = 0.0

    def _parse_urdf(self):
        """Parse URDF file"""
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        
        # Parse links
        for link in root.findall("link"):
            link_name = link.get("name")
            self.links[link_name] = {
                "name": link_name,
                "visual": [],
                "parent_joint": None,
                "child_joints": []
            }
            
            # Parse visual
            for visual in link.findall("visual"):
                vis_data = {"origin": np.eye(4)}
                
                # Parse origin
                origin = visual.find("origin")
                if origin is not None:
                    xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
                    rpy = [float(r) for r in origin.get("rpy", "0 0 0").split()]
                    vis_data["origin"][:3,3] = xyz
                    vis_data["origin"][:3,:3] = euler_to_matrix(rpy)
                
                # Parse geometry 
                geometry = visual.find("geometry")
                if geometry is not None:
                    mesh = geometry.find("mesh")
                    if mesh is not None:
                        filename = mesh.get("filename")
                        if filename.startswith("package://"):
                            filename = filename.split("package://")[1].split("/")[-1]
                        filename = os.path.join(self.mesh_path, filename)
                        vis_data["mesh"] = filename
                        
                self.links[link_name]["visual"].append(vis_data)
                
        # Parse joints
        for joint in root.findall("joint"):
            joint_name = joint.get("name")
            joint_type = joint.get("type")
            
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            
            origin = joint.find("origin")
            transform = np.eye(4)
            if origin is not None:
                xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
                rpy = [float(r) for r in origin.get("rpy", "0 0 0").split()]
                transform[:3,3] = xyz
                transform[:3,:3] = euler_to_matrix(rpy)
                
            axis = [1, 0, 0] # Default rotation axis
            axis_elem = joint.find("axis")
            if axis_elem is not None:
                axis = [float(x) for x in axis_elem.get("xyz").split()]
                
            joint_data = {
                "name": joint_name,
                "type": joint_type,
                "parent": parent,
                "child": child,
                "origin": transform,
                "axis": np.array(axis)
            }
            
            self.joints[joint_name] = joint_data
            
            # Update link's joint reference
            self.links[parent]["child_joints"].append(joint_name)
            self.links[child]["parent_joint"] = joint_name

    def _create_display_lists(self):
        """Create OpenGL display lists for each mesh"""
        for link_name, link_data in self.links.items():
            visuals = link_data["visual"]
            
            if not visuals:
                continue
                
            display_list = glGenLists(1)
            glNewList(display_list, GL_COMPILE)
            
            for visual in visuals:
                if "mesh" in visual:
                    mesh_file = visual["mesh"]
                    print(f"Loading mesh: {mesh_file}")
                    
                    if not os.path.exists(mesh_file):
                        print(f"Warning: Mesh file does not exist: {mesh_file}")
                        continue
                        
                    if mesh_file not in self.meshes:
                        mesh = self._load_mesh_async(mesh_file)
                        if mesh is not None:
                            self.meshes[mesh_file] = mesh
                        else:
                            continue
                            
                    mesh = self.meshes[mesh_file]
                    
                    # Apply visual element transform
                    transformed_vertices = np.array(mesh.vertices)
                    if visual["origin"] is not None:
                        transform = visual["origin"]
                        transformed_vertices = np.dot(transformed_vertices, 
                                                    transform[:3,:3].T) + transform[:3,3]
                    
                    # Draw mesh
                    glBegin(GL_TRIANGLES)
                    for face in mesh.faces:
                        # Compute face normal for lighting
                        v1 = transformed_vertices[face[0]]
                        v2 = transformed_vertices[face[1]]
                        v3 = transformed_vertices[face[2]]
                        normal = np.cross(v2 - v1, v3 - v1)
                        normal = normal / np.linalg.norm(normal)
                        
                        glNormal3f(*normal)
                        for vertex_idx in face:
                            vertex = transformed_vertices[vertex_idx]
                            glVertex3f(*vertex)
                    glEnd()
                    
            glEndList()
            self.display_lists[link_name] = display_list

    def _load_mesh_async(self, mesh_file):
        """Asynchronously load mesh file"""
        try:
            # Use trimesh to load mesh
            loaded = trimesh.load(mesh_file)
            if isinstance(loaded, trimesh.Scene):
                # Handle Scene type
                meshes = []
                for name, geom in loaded.geometry.items():
                    if isinstance(geom, trimesh.Trimesh):
                        meshes.append(geom)
                if meshes:
                    # Combine all meshes
                    combined = meshes[0]
                    for m in meshes[1:]:
                        combined = trimesh.util.concatenate([combined, m])
                    return combined
            elif isinstance(loaded, trimesh.Trimesh):
                return loaded
            return None
        except Exception as e:
            print(f"Failed to load mesh: {mesh_file}")
            print(f"Error: {str(e)}")
            return None

    def _create_gpu_buffers(self):
        """Create GPU buffers for OpenGL rendering"""
        for link_name, link_data in self.links.items():
            visuals = link_data["visual"]
            if not visuals:
                continue
                
            for visual in visuals:
                if "mesh" not in visual:
                    continue
                    
                mesh_file = visual["mesh"]
                mesh = self.meshes.get(mesh_file)
                if mesh is None:
                    continue
                
                # Create VBO
                vertices = np.array(mesh.vertices, dtype=np.float32)
                faces = np.array(mesh.faces, dtype=np.uint32)
                
                vbo = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, vbo)
                glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
                
                ibo = glGenBuffers(1)
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)
                
                self.gpu_buffers[mesh_file] = (vbo, ibo, len(faces) * 3)

    def compute_transform(self, joint_name, joint_value):
        """Compute joint transform matrix"""
        joint = self.joints[joint_name]
        
        if joint["type"] == "fixed":
            return joint["origin"]
            
        elif joint["type"] == "revolute" or ["type"] == "continuous":
            axis = joint["axis"]
            c = np.cos(joint_value)
            s = np.sin(joint_value)
            
            if np.allclose(axis, [1, 0, 0]):
                R = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
            elif np.allclose(axis, [0, 1, 0]):
                R = np.array([[c, 0, s],
                            [0, 1, 0],
                            [-s, 0, c]])
            elif np.allclose(axis, [0, 0, 1]):
                R = np.array([[c, -s, 0],
                            [s, c, 0],
                            [0, 0, 1]])
            else:
                K = np.array([[0, -axis[2], axis[1]],
                            [axis[2], 0, -axis[0]],
                            [-axis[1], axis[0], 0]])
                R = np.eye(3) + s * K + (1 - c) * (K @ K)
                
            transform = np.eye(4)
            transform[:3,:3] = R
            return joint["origin"] @ transform
            
        elif joint["type"] == "prismatic":
            # Compute translation transform
            transform = np.eye(4)
            transform[:3,3] = joint["axis"] * joint_value
            return joint["origin"] @ transform
            
        return np.eye(4)

    def update_joint_state(self, joint_name, position):
        """Update joint state"""
        if joint_name in self.joint_states:
            self.joint_states[joint_name] = position

    def compute_link_transforms(self):
        """Compute global transforms for all links"""
        transforms = {}
        
        def get_link_transform(link_name, parent_transform=np.eye(4)):
            # Compute current link's global transform
            current_transform = parent_transform
            
            # If link has parent joint, compute joint transform first
            joint_name = self.links[link_name]["parent_joint"]
            if joint_name:
                joint_transform = self.compute_transform(
                    joint_name, 
                    self.joint_states[joint_name]
                )
                current_transform = current_transform @ joint_transform
                
            transforms[link_name] = current_transform
            
            # Recursively process all child joints and links
            for child_joint in self.links[link_name]["child_joints"]:
                child_link = self.joints[child_joint]["child"]
                get_link_transform(child_link, current_transform)
                
        # Start computing from root link
        root_link = next(link for link, data in self.links.items() 
                        if data["parent_joint"] is None)
        get_link_transform(root_link)
        
        return transforms

    def draw(self):
        """Draw using OpenGL display lists"""
        transforms = self.compute_link_transforms()
        
        # Draw each link
        for link_name, transform in transforms.items():
            if link_name not in self.display_lists:
                continue
                
            glPushMatrix()
            # OpenGL uses column-major order, need to transpose
            glMultMatrixf(transform.T.astype(np.float32))
            
            # Call display list to draw mesh
            glCallList(self.display_lists[link_name])
            
            glPopMatrix()

    def cleanup(self):
        """Clean up resources"""
        for vbo, ibo, _ in self.gpu_buffers.values():
            glDeleteBuffers(1, [vbo])
            glDeleteBuffers(1, [ibo])
            
        # Shutdown thread pool
        self.thread_pool.shutdown()

if __name__ == "__main__":
    import sys
    import pygame
    from pygame.locals import *
    
    pygame.init()
    window_size = (800, 600)
    screen = pygame.display.set_mode(window_size, HWSURFACE | OPENGL | DOUBLEBUF)
    pygame.display.set_caption("URDF Viewer")
    
    glClearColor(0.2, 0.2, 0.2, 1.0)  
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (5.0, 5.0, 5.0, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))
    glEnable(GL_COLOR_MATERIAL) # Enable material color
    
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (window_size[0] / window_size[1]), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    
    # Create URDF renderer
    urdf_path = sys.argv[1] if len(sys.argv) > 1 else "gviz/urdf/go1.urdf"
    mesh_path = sys.argv[2] if len(sys.argv) > 2 else "gviz/urdf/go1/meshes"
    renderer = GURDF(urdf_path, mesh_path)
    
    # Camera control parameters
    camera_distance = 3.0
    camera_rotation = [0.0, 0.0]  
    mouse_prev = None
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1: 
                    mouse_prev = pygame.mouse.get_pos()
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1: 
                    mouse_prev = None
            elif event.type == MOUSEMOTION:
                if mouse_prev is not None:  
                    mouse_pos = pygame.mouse.get_pos()
                    dx = mouse_pos[0] - mouse_prev[0]
                    dy = mouse_pos[1] - mouse_prev[1]
                    camera_rotation[0] += dy * 0.5
                    camera_rotation[1] += dx * 0.5
                    mouse_prev = mouse_pos
            elif event.type == MOUSEWHEEL:  
                camera_distance = max(1.0, min(10.0, camera_distance - event.y * 0.3))
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        glTranslatef(0, 0, -camera_distance)
        glRotatef(camera_rotation[0], 1, 0, 0)
        glRotatef(camera_rotation[1], 0, 1, 0)
        
        glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        for i in range(-5, 6):
            glColor3f(0.3, 0.3, 0.3)
            glVertex3f(i, -5, 0)
            glVertex3f(i, 5, 0)
            glVertex3f(-5, i, 0)
            glVertex3f(5, i, 0)
        glEnd()
        
        # Draw axes
        glBegin(GL_LINES)

        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(1, 0, 0)

        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 1, 0)

        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 1)
        glEnd()
        
        # Enable lighting to draw model
        glEnable(GL_LIGHTING)
        glColor3f(0.8, 0.8, 0.8)
        renderer.draw()
        
        pygame.display.flip()
        pygame.time.wait(10)
    
    # Clean up resources
    renderer.cleanup()
    pygame.quit()