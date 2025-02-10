import numpy as np


class PBDSimulation:
    def __init__(self, 
                 time_step: float = 0.0333,  
                 substeps: int = 1,
                 gravity: list = None,
                 dynamic_collision: bool = False):
        """
        Initializes the PBD simulation environment.
        
        Args:
            time_step (float): The simulation time step.
            substeps (int): Number of substeps per time step for stability.
            gravity (list): Gravity vector (default: [0, -9.8, 0]).
        """
        self.dynamic_bodies: list[PhysicalObject] = []
        self.static_bodies: list[PhysicalObject] = []
        self.constraints: list[Constraint] = []

        self.dynamic_collisions = dynamic_collision
    
        self.gravity = np.array(gravity if gravity is not None else [0, -9.8, 0], dtype=np.float32)
        self.time_step = time_step
        self.substeps = max(1, substeps)
        self.h = self.time_step / self.substeps

    def add_constraint(self, constraint: Constraint):
        """Adds a constraint to the simulation."""
        self.constraints.append(constraint)

    def add_body(self, body):
        if isinstance(body, list):
            for b in body:
                self._add_single_body(b)
        else:
            self._add_single_body(body)

    def _add_single_body(self, body):
        if body.is_static:
            self.static_bodies.append(body)
        else:
            self.dynamic_bodies.append(body)        
    
    def step(self):
        collisions = self.check_collisions()
        
        for _ in range(self.substeps):
            contacts = self.check_contacts(collisions)
            self.integrate()
            self.solve_positions(contacts)
            self.update_velocities()
            self.solve_velocities(contacts)

    def check_collisions(self):
        if not self.dynamic_collisions:
            return []
        ## OBB - SAT
        potential_collisions = []
        
        # Get all cube vertices
        cube_positions = np.array([cube.currPos for cube in self.dynamic_bodies])  # Shape (N, 8, 3)

        # Compute face normals for each cube
        cube_faces = np.array([cube.faces for cube in self.dynamic_bodies])  # Shape (N, 12, 3)
        cube_faces = cube_faces[:, ::2] # Shape (N, 6, 3) : two faces have same normal vector
        
        q1, q2, q3 = cube_positions[:, cube_faces[:, :, 0]], cube_positions[:, cube_faces[:, :, 1]], cube_positions[:, cube_faces[:, :, 2]]
        face_normals = np.cross(q2 - q1, q3 - q1)  # Shape (N, 6, 3)
        face_normals /= np.linalg.norm(face_normals, axis=2, keepdims=True)

        # Compute edge directions for each cube
        cube_edges = np.array([cube.edges for cube in self.dynamic_bodies])  # Shape (N, 12, 2)
        edge_dirs = cube_positions[:, cube_edges[:, :, 1]] - cube_positions[:, cube_edges[:, :, 0]]
        edge_dirs /= np.linalg.norm(edge_dirs, axis=2, keepdims=True)

        # Compute cross products (Edge1 x Edge2) for all cube pairs
        cross_axes = np.cross(edge_dirs[:, :, None, :], edge_dirs[None, :, :, :])  # Shape (N, N, 12, 12, 3)
        valid_mask = np.linalg.norm(cross_axes, axis=4) > 1e-6  # Avoid zero vectors
        cross_axes[~valid_mask] = 0
        cross_axes /= np.linalg.norm(cross_axes, axis=4, keepdims=True, where=valid_mask)

        # Combine separating axes (face normals + cross products)
        separating_axes = np.concatenate((face_normals[:, None, :, :], face_normals[None, :, :, :], cross_axes), axis=2)  # Shape (N, N, 15, 3)

        # Project all cube vertices onto separating axes
        proj = np.einsum('nij, nmj->nim', cube_positions, separating_axes)  # Shape (N, N, 8, 15)

        # Get min and max projections
        min_proj, max_proj = np.min(proj, axis=2), np.max(proj, axis=2)  # Shape (N, N, 15)

        # Check for separating axis (vectorized SAT test)
        is_separated = (max_proj[:, :, None, :] < min_proj[None, :, :, :]) | (max_proj[None, :, :, :] < min_proj[:, :, None, :])
        collision_mask = ~np.any(is_separated, axis=-1)

        # Extract colliding pairs
        colliding_indices = np.column_stack(np.where(collision_mask))
        
        for i, j in colliding_indices:
            if i < j:
                potential_collisions.append((self.dynamic_bodies[i], self.dynamic_bodies[j]))

        return potential_collisions


    def check_contacts(self, collisions):
        contacts = []

        # Body-Body Collision
        if self.dynamic_collisions:
            for body1, body2 in collisions:
                contact_type, p1, p2, normal, penetration_depth = self.compute_dynamic_contacts(body1, body2)
                if penetration_depth <= 0:
                    continue  # No penetration → No contact constraint
                
                # **Dynamic-Dynamic Collision**
                if isinstance(body1, Cube) and isinstance(body2, Cube):
                    if contact_type == "vertex-face":
                        contacts.append(PointPlaneDistanceConstraint(body1, p1, body2, p2, normal, penetration_depth))
                    elif contact_type == "edge-edge":
                        contacts.append(EdgeEdgeDistanceConstraint(body1, p1, body2, p2, normal, penetration_depth))

        
        
        # Ground Collision
        plane = self.static_bodies[0].mesh
        for i, body in enumerate(self.dynamic_bodies):
            coll_idx = np.where(np.dot(body.currPos - plane.origin, plane.normal) < 0)[0]
            for i in coll_idx:
                contacts.append(GroundCollisionConstraint(body, i, plane,)) 
        return contacts
    
    
    def compute_dynamic_contacts(self, body1, body2):
        """Detects vertex-face and edge-edge contacts between two bodies."""
        
        ### **Step 1: Detect Vertex-Face Collisions**
        vertices = body1.currPos
        faces = body2.faces

        q1, q2, q3 = body2.currPos[faces[:, 0]], body2.currPos[faces[:, 1]], body2.currPos[faces[:, 2]]

        # Compute face normals
        e1, e2 = q2 - q1, q3 - q1
        face_normals = np.cross(e1, e2)
        face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)

        # Compute signed distances (vertex penetration)
        v_minus_q1 = vertices[:, None, :] - q1[None, :, :]
        signed_distances = np.einsum('ijk,jk->ij', v_minus_q1, face_normals)

        # Find penetrating vertices
        penetration_mask = signed_distances < 0
        vertex_indices, face_indices = np.where(penetration_mask)

        if len(vertex_indices) > 0:
            # Select the deepest penetration
            min_penetration_idx = np.argmin(signed_distances[vertex_indices, face_indices])
            v_idx = vertex_indices[min_penetration_idx]
            f_idx = face_indices[min_penetration_idx]
            normal = face_normals[f_idx]
            penetration_depth = -signed_distances[v_idx, f_idx]

            return "vertex-face", v_idx, faces[f_idx], normal, penetration_depth

        ### **Step 2: Detect Edge-Edge Collisions**
        edges1, edges2 = body1.edges, body2.edges
        p1, p2 = body1.currPos[edges1[:, 0]], body1.currPos[edges1[:, 1]]
        q1, q2 = body2.currPos[edges2[:, 0]], body2.currPos[edges2[:, 1]]

        d1, d2 = p2 - p1, q2 - q1

        # Compute closest points between edges
        a = np.einsum('ij,ij->i', d1, d1)
        b = np.einsum('ij,ij->i', d1, d2)
        c = np.einsum('ij,ij->i', d2, d2)
        d = np.einsum('ij,ij->i', q1 - p1, d1)
        e = np.einsum('ij,ij->i', q1 - p1, d2)

        det = a * c - b * b
        s = np.clip((b * e - c * d) / det, 0, 1)
        t = np.clip((a * e - b * d) / det, 0, 1)

        closest_p1 = p1 + s[:, None] * d1
        closest_q1 = q1 + t[:, None] * d2
        distances = np.linalg.norm(closest_p1 - closest_q1, axis=1)

        # Find closest edge pair
        min_distance_idx = np.argmin(distances)
        penetration_depth = -distances[min_distance_idx]
        
        if penetration_depth < 0:
            return "edge-edge", edges1[min_distance_idx], edges2[min_distance_idx], (closest_p1[min_distance_idx] - closest_q1[min_distance_idx]) / np.linalg.norm(closest_p1[min_distance_idx] - closest_q1[min_distance_idx]), penetration_depth

        return None, None, None, None, None
    
    
    def integrate(self):
        for body in self.dynamic_bodies:
            body.prevPos = body.currPos.copy()
            body.currVel += self.gravity * self.h
            body.currPos += body.currVel * self.h


    def solve_positions(self, contacts):
        for constraint in self.constraints:
            constraint.solve(self.h)
        
        for contact in contacts:
            contact.solve(self.h)


    def update_velocities(self):
        for body in self.dynamic_bodies:
            body.currVel = (body.currPos - body.prevPos) / self.h


    def solve_velocities(self, contacts):
        for contact in contacts:
            contact.solve_velocity(self.h)
        
    
    def reset(self):
        for body in self.dynamic_bodies:
            body.reset()
        for constraint in self.constraints:
            constraint.lambda_ = 0.0
            
            
    def update(self):
        for body in self.dynamic_bodies:
            body.update_mesh()