import trimesh
import numpy as np
import os
import pymeshlab
from heapq import heappush, heappop
import scipy.sparse as sparse

def compute_quadric(vertices, faces, face_normals):
    """Calculate quadric error matrices for each vertex"""
    V = len(vertices)
    quadrics = np.zeros((V, 4, 4))
    
    for face, normal in zip(faces, face_normals):
        # Calculate plane equation ax + by + cz + d = 0
        a, b, c = normal
        d = -np.dot(normal, vertices[face[0]])
        
        # Build quadric matrix for the plane
        p = np.array([a, b, c, d])
        q = np.outer(p, p)
        
        # Add quadric matrix to all vertices of the face
        for v_idx in face:
            quadrics[v_idx] += q
            
    return quadrics

def compute_edge_cost(v1, v2, q1, q2):
    """Calculate edge collapse cost and optimal position"""
    Q = q1 + q2
    # Build equation system
    A = Q[:3, :3]
    b = Q[:3, 3]
    
    try:
        # Solve for optimal position
        v_opt = np.linalg.solve(A, -b)
        cost = float(np.dot(np.dot(v_opt, Q[:3, :3]), v_opt) + 
                    2 * np.dot(Q[:3, 3], v_opt) + Q[3, 3])
    except np.linalg.LinAlgError:
        # If equation is unsolvable, use edge midpoint
        v_opt = (v1 + v2) / 2
        cost = float(np.dot(np.dot(v_opt, Q[:3, :3]), v_opt) + 
                    2 * np.dot(Q[:3, 3], v_opt) + Q[3, 3])
    
    return cost, v_opt

def simplify_quadric_decimation(mesh, target_faces):
    """
    Mesh simplification using quadric error metrics
    
    Args:
        mesh: trimesh.Trimesh object
        target_faces: target number of faces
    Returns:
        simplified_mesh: simplified mesh
    """
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    
    # Calculate quadric matrices for each vertex
    quadrics = compute_quadric(vertices, faces, mesh.face_normals)
    
    # Build edge list and cost priority queue
    edges = set()
    cost_queue = []
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i+1)%3]
            if v1 > v2:
                v1, v2 = v2, v1
            if (v1, v2) not in edges:
                edges.add((v1, v2))
                cost, v_new = compute_edge_cost(vertices[v1], vertices[v2],
                                              quadrics[v1], quadrics[v2])
                heappush(cost_queue, (cost, v1, v2, v_new))
    
    # Track valid vertices and faces
    valid_vertices = np.ones(len(vertices), dtype=bool)
    valid_faces = np.ones(len(faces), dtype=bool)
    
    # Perform edge collapses until target face count is reached
    while len(faces[valid_faces]) > target_faces and cost_queue:
        cost, v1, v2, v_new = heappop(cost_queue)
        
        if not valid_vertices[v1] or not valid_vertices[v2]:
            continue
            
        # Update mesh
        vertices[v1] = v_new
        valid_vertices[v2] = False
        
        # Update faces
        v2_faces = np.where(faces == v2)
        faces[v2_faces[0], v2_faces[1]] = v1
        
        # Remove degenerate faces
        degenerate = np.any(faces == faces[:, [1, 2, 0]], axis=1)
        valid_faces[degenerate] = False
        
    # Build simplified mesh
    new_vertices = vertices[valid_vertices]
    new_faces = faces[valid_faces]
    
    # Remap face indices
    vertex_map = np.cumsum(valid_vertices) - 1
    new_faces = vertex_map[new_faces]
    
    return trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

def get_mesh_size_mb(filepath: str) -> float:
    """Get mesh file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)

def compress_mesh(input_path: str, output_path: str, target_size_mb: float) -> None:
    """
    Compress 3D mesh file to target size
    
    Args:
        input_path: Input mesh file path (.stl or .dae)
        output_path: Output mesh file path
        target_size_mb: Target file size in MB
    """
    try:
        # Use pymeshlab for large file processing
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_path)
        
        # Get original file size
        original_size = os.path.getsize(input_path) / (1024 * 1024)
        original_faces = ms.current_mesh().face_number()
        
        # Calculate target face count
        ratio = target_size_mb / original_size
        if ratio >= 1:
            print(f"Target size ({target_size_mb:.2f}MB) is larger than original size ({original_size:.2f}MB), no compression needed")
            ms.save_current_mesh(output_path)
            return
            
        target_faces = int(original_faces * ratio)
        target_faces = max(target_faces, 100)  # Ensure at least 100 faces are kept
        
        # Use pymeshlab's simplification algorithm
        print(f"Simplifying mesh from {original_faces} faces to {target_faces} faces...")
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preserveboundary=True,
            preservenormal=True,
            planarquadric=True
        )
        
        # Clean and optimize mesh
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_repair_non_manifold_edges()
        
        # Save result
        ms.save_current_mesh(output_path)
        
        # Print results
        final_size = os.path.getsize(output_path) / (1024 * 1024)
        final_faces = ms.current_mesh().face_number()
        print(f"Original file size: {original_size:.2f} MB")
        print(f"Compressed file size: {final_size:.2f} MB")
        print(f"Face count: {original_faces} -> {final_faces}")
        
    except Exception as e:
        print(f"Compression error: {str(e)}")
        raise

if __name__ == "__main__":

    input_file = "/home/wx/warp/warp_sensor/gviz/urdf/go1/meshes/thigh.dae"
    output_file = "/home/wx/warp/warp_sensor/gviz/urdf/go1/meshes/thigh.dae"
    target_size = 1.0  # target size (MB)
    
    compress_mesh(input_file, output_file, target_size)
