import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pathlib import Path
class CUDABVHLib:
    def __init__(self):
        with open(Path("../cpp/bvh_cuda.cu"), 'r') as f:
            cuda_code = f.read()
        self.mod = SourceModule(cuda_code)
        
        # CUDA 함수 가져오기
        self.build_bvh = self.mod.get_function("buildBVHKernel")
        self.detect_contacts = self.mod.get_function("detectContactsKernel")
        self.compute_skinning = self.mod.get_function("computeSkinningKernel")
        
        # BVH 노드 구조체 크기
        self.node_size = 32  # float3(12) * 2 + int(4) * 2 = 32 bytes
        
    def create_bvh(self, positions, margin=0.1):
        num_vertices = len(positions)
        
        # CPU 메모리 준비
        indices = np.arange(num_vertices, dtype=np.int32)
        nodes = np.zeros(2 * num_vertices - 1, dtype=[
            ('min', np.float32, 3),
            ('max', np.float32, 3),
            ('leftFirst', np.int32),
            ('triCount', np.int32)
        ])
        
        # GPU 메모리 할당
        d_positions = cuda.mem_alloc(positions.nbytes)
        d_indices = cuda.mem_alloc(indices.nbytes)
        d_nodes = cuda.mem_alloc(nodes.nbytes)
        
        # 데이터 복사
        cuda.memcpy_htod(d_positions, positions)
        cuda.memcpy_htod(d_indices, indices)
        
        # BVH 구축
        block_size = 256
        grid_size = (num_vertices + block_size - 1) // block_size
        self.build_bvh(
            d_positions,
            d_indices,
            d_nodes,
            np.int32(num_vertices),
            np.float32(margin),
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )
        
        cuda.memcpy_dtoh(nodes, d_nodes)
        return nodes, d_nodes
    
    def detect_vertex_contacts(self, hand_bvh, other_bvh,
                             hand_pos, hand_vel,
                             other_pos, other_vel,
                             max_contacts=10000):
        # GPU 메모리 할당
        d_hand_pos = cuda.mem_alloc(hand_pos.nbytes)
        d_hand_vel = cuda.mem_alloc(hand_vel.nbytes)
        d_other_pos = cuda.mem_alloc(other_pos.nbytes)
        d_other_vel = cuda.mem_alloc(other_vel.nbytes)
        
        contacts = np.zeros(max_contacts, dtype=np.int32)
        distances = np.zeros(max_contacts, dtype=np.float32)
        contact_count = np.zeros(1, dtype=np.int32)
        
        d_contacts = cuda.mem_alloc(contacts.nbytes)
        d_distances = cuda.mem_alloc(distances.nbytes)
        d_contact_count = cuda.mem_alloc(contact_count.nbytes)
        
        # 데이터 복사
        cuda.memcpy_htod(d_hand_pos, hand_pos)
        cuda.memcpy_htod(d_hand_vel, hand_vel)
        cuda.memcpy_htod(d_other_pos, other_pos)
        cuda.memcpy_htod(d_other_vel, other_vel)
        
        # Contact detection 실행
        block_size = 256
        grid_size = (max_contacts + block_size - 1) // block_size
        
        self.detect_contacts(
            hand_bvh,
            other_bvh,
            d_hand_pos,
            d_hand_vel,
            d_other_pos,
            d_other_vel,
            d_contacts,
            d_distances,
            d_contact_count,
            np.int32(max_contacts),
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )
        
        # 결과 복사
        cuda.memcpy_dtoh(contacts, d_contacts)
        cuda.memcpy_dtoh(distances, d_distances)
        cuda.memcpy_dtoh(contact_count, d_contact_count)
        
        return contacts[:contact_count[0]], distances[:contact_count[0]]
        
    def compute_skinning(self, rest_positions, bone_matrices, bone_indices, bone_weights):
        num_vertices = len(rest_positions)
        
        # GPU 메모리 할당
        d_rest_positions = cuda.mem_alloc(rest_positions.nbytes)
        d_bone_matrices = cuda.mem_alloc(bone_matrices.nbytes)
        d_bone_indices = cuda.mem_alloc(bone_indices.nbytes)
        d_bone_weights = cuda.mem_alloc(bone_weights.nbytes)
        
        skinned_positions = np.zeros_like(rest_positions)
        d_skinned_positions = cuda.mem_alloc(skinned_positions.nbytes)
        
        # 데이터 복사
        cuda.memcpy_htod(d_rest_positions, rest_positions)
        cuda.memcpy_htod(d_bone_matrices, bone_matrices)
        cuda.memcpy_htod(d_bone_indices, bone_indices)
        cuda.memcpy_htod(d_bone_weights, bone_weights)
        
        # 스키닝 실행
        block_size = 256
        grid_size = (num_vertices + block_size - 1) // block_size
        
        self.compute_skinning(
            d_rest_positions,
            d_bone_matrices,
            d_bone_indices,
            d_bone_weights,
            d_skinned_positions,
            np.int32(num_vertices),
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )
        
        # 결과 복사
        cuda.memcpy_dtoh(skinned_positions, d_skinned_positions)
        
        return skinned_positions