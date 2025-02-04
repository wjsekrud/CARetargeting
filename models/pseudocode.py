class ContactAwareMotionRetargeting:
    def __init__(self):
        self.encoder = RNNEncoder()
        self.decoder = RNNDecoder()
        self.pointnet = PointNetEncoder()
        self.fk_layer = ForwardKinematicsLayer()
        self.skinning_layer = SkinningLayer()
        
        # State for maintaining temporal information
        self.h_enc = None  # Encoder hidden state
        self.h_dec = None  # Decoder hidden state
        self.prev_local_motion = None
        self.prev_root_velocity = None
        
    def process_single_frame(self, source_frame, target_skeleton, target_geometry):
        """Process a single frame of motion data"""
        # Encode source frame, maintaining temporal state
        motion_features, self.h_enc = self.encoder(
            source_frame, 
            hidden_state=self.h_enc
        )
        
        # Get geometry features (only needed once per sequence)
        if not hasattr(self, 'geometry_features'):
            self.geometry_features = self.pointnet(target_geometry)
        
        # Decode frame with previous motion state
        local_rotations, root_velocity, self.h_dec = self.decoder(
            motion_features,
            self.geometry_features,
            target_skeleton,
            prev_local_motion=self.prev_local_motion,
            prev_root_velocity=self.prev_root_velocity,
            hidden_state=self.h_dec
        )
        
        # Update previous motion states
        self.prev_local_motion = local_rotations
        self.prev_root_velocity = root_velocity
        
        # Apply forward kinematics for current frame
        joint_positions = self.fk_layer(local_rotations)
        global_positions = joint_positions + root_velocity
        
        # Apply skinning for current frame
        vertex_positions = self.skinning_layer(
            global_positions,
            target_geometry
        )
        
        return local_rotations, global_positions, vertex_positions
    
    def reset_states(self):
        """Reset temporal states between sequences"""
        self.h_enc = None
        self.h_dec = None
        self.prev_local_motion = None
        self.prev_root_velocity = None
        if hasattr(self, 'geometry_features'):
            del self.geometry_features
    
    def detect_frame_contacts(self, frame_vertices):
        """Detect contacts for a single frame"""
        frame_contacts = detect_mesh_intersections(frame_vertices)
        return filter_by_velocity_and_distance(frame_contacts)
    
    def compute_frame_energy(self, frame_output, frame_contacts):
        """Compute energy function for a single frame"""
        E_geo = self.geometry_energy(frame_output, frame_contacts)
        E_skel = self.skeleton_energy(frame_output)
        return E_geo + E_skel
    
    def optimize_frame(self, frame_output, frame_contacts, num_steps=30):
        """Optimize encoder hidden state for current frame"""
        h_enc_frame = self.h_enc
        
        for _ in range(num_steps):
            energy = self.compute_frame_energy(frame_output, frame_contacts)
            grad = torch.autograd.grad(energy, h_enc_frame)
            h_enc_frame = h_enc_frame - learning_rate * grad
            
            # Update frame output with new hidden state
            frame_output = self.decoder(
                h_enc_frame,
                self.geometry_features,
                hidden_state=self.h_dec
            )
            
        return frame_output

def train_model():
    model = ContactAwareMotionRetargeting()
    
    for epoch in range(num_epochs):
        for motion_sequence in dataloader:
            model.reset_states()
            
            for frame_idx in range(len(motion_sequence)):
                source_frame = motion_sequence[frame_idx]
                target_skeleton = motion_sequence.target_skeleton
                target_geometry = motion_sequence.target_geometry
                
                # Process single frame
                frame_output = model.process_single_frame(
                    source_frame, 
                    target_skeleton, 
                    target_geometry
                )
                
                # Detect contacts for current frame
                frame_contacts = model.detect_frame_contacts(frame_output)
                
                # Compute loss for current frame
                frame_energy = model.compute_frame_energy(
                    frame_output, 
                    frame_contacts
                )
                
                # Backward pass and optimize
                optimizer.zero_grad()
                frame_energy.backward()
                optimizer.step()
                
                # Frame-wise encoder space optimization
                if epoch > warmup_epochs:
                    frame_output = model.optimize_frame(
                        frame_output,
                        frame_contacts
                    )