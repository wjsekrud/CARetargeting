from pathlib import Path
import sys
sys.path.append("../")
from utils import BaseContactDetector
from utils.CUDAkernels import cuda_contact_kernel, cuda_transform_kernel
from utils.newvertex import NewVertex

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree





class CPUContactDetector(BaseContactDetector):
    def __init__(self, vertex_file_path, bvh_file_path):
        super.init(vertex_file_path, bvh_file_path)
                
    def create_vertex_groups(self):
        """NewVertex의 vertex_group 정보를 기반으로 그룹화"""
        groups = {}
        for i, vertex in enumerate(self.vertices):
            group_id = vertex.vertex_group
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(i)
        return groups

    def create_bmesh_for_group(self, group_indices, frame):
        """특정 프레임에서의 vertex group에 대한 bmesh 생성"""
        bm = bmesh.new()
        
        # 현재 프레임의 vertex 위치 계산
        vertex_positions = self.calculate_vertex_positions(group_indices, frame)
        
        # vertices 추가
        bm_verts = [bm.verts.new(pos) for pos in vertex_positions]
        bm.verts.ensure_lookup_table()
        
        # triangles 추가
        for triangle in self.triangles:
            vertices = triangle['indices']
            # 모든 vertex가 현재 그룹에 있는 경우에만 triangle 추가
            if all(v in group_indices for v in vertices):
                try:
                    bm.faces.new([bm_verts[group_indices.index(v)] for v in vertices])
                except Exception:
                    continue  # 잘못된 topology 무시
                    
        bm.faces.ensure_lookup_table()
        return bm
        
    def calculate_vertex_positions(self, vertex_indices, frame):
        """특정 프레임에서의 vertex positions 계산"""
        bpy.context.scene.frame_set(frame)
        positions = []
        
        for idx in vertex_indices:
            vertex = self.vertices[idx]
            final_position = Vector(vertex.position)
            
            # bone weights 적용
            for weight_info in vertex.bone_weights:
                bone_idx = weight_info['bone_index']
                weight = weight_info['weight']
                
                # 해당 bone의 world matrix 가져오기
                bone = self.armature.pose.bones[bone_idx]
                bone_matrix = self.armature.matrix_world @ bone.matrix
                
                # vertex position 업데이트
                transformed_position = bone_matrix @ final_position
                final_position = final_position.lerp(transformed_position, weight)
                
            positions.append(final_position)
            
        return positions
        
    def detect_contacts_for_frame(self, frame):
        """특정 프레임에서의 contact detection 수행"""
        print(f"start check for frame {frame}")
        # 손 그룹 식별
        hand_groups = {}
        
        # 각 vertex group에 대해 검사
        for group_id, vertex_indices in self.vertex_groups.items():

            if self.is_hand_bone(group_id):
                hand_groups[group_id] = vertex_indices

        contact_pairs = []

        # 각 손 그룹에 대해 검사
        for hand_id, hand_vertices in hand_groups.items():
            
            hand_bmesh = self.create_bmesh_for_group(hand_vertices, frame)
            hand_bvh = BVHTree.FromBMesh(hand_bmesh)
            
            # 다른 모든 그룹과 검사
            for other_id, other_vertices in self.vertex_groups.items():
                if other_id == hand_id:
                    continue
                    
                other_bmesh = self.create_bmesh_for_group(other_vertices, frame)
                other_bvh = BVHTree.FromBMesh(other_bmesh)
                
                # BVH intersection 체크
                intersect = hand_bvh.overlap(other_bvh)
                
                if intersect:
                    # velocity check
                    if self.check_contact_conditions(hand_vertices, other_vertices, frame):
                        closest_pairs = self.find_closest_pairs(
                            hand_vertices, other_vertices, frame)
                        contact_pairs.extend(closest_pairs[:3])
                        #print(contact_pairs, other_id)
                        #validate_contacts.validate_contact(frame, self.vertices, self.create_contact_struct(contact_pairs, frame), Path("../dataset/"))
                        
                other_bmesh.free()
            hand_bmesh.free()
            
        return self.create_contact_struct(contact_pairs, frame)
        
    def is_hand_bone(self, bone_index):
        """본 인덱스가 손에 해당하는지 확인"""
        #print(bone_index)
        bone = self.armature.pose.bones[bone_index]
        return 'hand' in bone.name.lower()
        
    def check_contact_conditions(self, vertices1, vertices2, frame):
        """접촉 조건 확인"""
        # 현재 프레임과 다음 프레임의 위치 계산
        pos1_current = self.calculate_vertex_positions(vertices1, frame)
        pos1_next = self.calculate_vertex_positions(vertices1, frame + 1)
        pos2_current = self.calculate_vertex_positions(vertices2, frame)
        pos2_next = self.calculate_vertex_positions(vertices2, frame + 1)
        
        # velocity vectors 계산
        vel1 = [(n - c) for c, n in zip(pos1_current, pos1_next)]
        vel2 = [(n - c) for c, n in zip(pos2_current, pos2_next)]
        
        # average velocities
        avg_vel1 = sum(vel1, Vector()) / len(vel1)
        avg_vel2 = sum(vel2, Vector()) / len(vel2)
        
        # distance check (0.2cm threshold)
        min_distance = float('inf')
        for p1 in pos1_current:
            for p2 in pos2_current:
                dist = (p1 - p2).length
                
                min_distance = min(min_distance, dist)
        print(f"mind: {min_distance}")
        # cosine similarity check
        if avg_vel1.length and avg_vel2.length:
            cos_sim = avg_vel1.dot(avg_vel2) / (avg_vel1.length * avg_vel2.length)
            print(f"cos-sim : {cos_sim}")
            if cos_sim > 0.9:
                return True
                
        
       
        
        return min_distance < 2  # 0.2cm = 0.002m
        
    def find_closest_pairs(self, vertices1, vertices2, frame):
        """가장 가까운 vertex pairs 찾기"""
        pairs = []
        pos1 = self.calculate_vertex_positions(vertices1, frame)
        pos2 = self.calculate_vertex_positions(vertices2, frame)
        
        for i, p1 in enumerate(pos1):
            for j, p2 in enumerate(pos2):
                distance = (p1 - p2).length
                pairs.append((distance, (vertices1[i], vertices2[j])))
        
        pairs.sort(key=lambda x: x[0])
        return [(p[1][0], p[1][1]) for p in pairs]
        

class CUDAContactDetector(BaseContactDetector):
    def __init__(self, vertex_file_path, bvh_file_path):
        # Initialize CUDA
        self.mod_transform = SourceModule(cuda_transform_kernel)
        self.mod_contact = SourceModule(cuda_contact_kernel)
        self.transform_kernel = self.mod_transform.get_function("transform_vertices")
        self.contact_kernel = self.mod_contact.get_function("detect_contacts")
        
        # Load vertices and animation data
        self.vertices, self.triangles = self.load_vertex_data(vertex_file_path)
        self.load_animation(bvh_file_path)
        
        # Prepare CUDA memory
        self.prepare_cuda_memory()
        
    def prepare_cuda_memory(self):
        # Allocate memory for vertex positions
        vertex_positions = np.array([v.position for v in self.vertices], dtype=np.float32)
        self.d_positions = cuda.mem_alloc(vertex_positions.nbytes)
        cuda.memcpy_htod(self.d_positions, vertex_positions)
        
        # Allocate memory for bone weights and indices
        max_weights = max(len(v.bone_weights) for v in self.vertices)
        bone_indices = np.zeros((len(self.vertices), max_weights), dtype=np.int32)
        weights = np.zeros((len(self.vertices), max_weights), dtype=np.float32)
        
        for i, vertex in enumerate(self.vertices):
            for j, weight_info in enumerate(vertex.bone_weights):
                bone_indices[i, j] = weight_info['bone_index']
                weights[i, j] = weight_info['weight']
        
        self.d_bone_indices = cuda.mem_alloc(bone_indices.nbytes)
        self.d_weights = cuda.mem_alloc(weights.nbytes)
        cuda.memcpy_htod(self.d_bone_indices, bone_indices)
        cuda.memcpy_htod(self.d_weights, weights)
        
        # Allocate memory for transformed positions
        self.d_transformed = cuda.mem_alloc(vertex_positions.nbytes)
        
    def detect_contacts_for_frame(self, frame):
        # Update bone matrices for current frame
        bone_matrices = self.get_bone_matrices(frame)
        d_bone_matrices = cuda.mem_alloc(bone_matrices.nbytes)
        cuda.memcpy_htod(d_bone_matrices, bone_matrices)
        
        # Transform vertices
        block_size = 256
        grid_size = (len(self.vertices) + block_size - 1) // block_size
        
        self.transform_kernel(
            self.d_positions,
            d_bone_matrices,
            self.d_bone_indices,
            self.d_weights,
            self.d_transformed,
            np.int32(len(self.vertices)),
            np.int32(1),
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )
        
        # Get hand and non-hand vertices
        hand_indices = self.get_hand_vertices()
        other_indices = self.get_non_hand_vertices()
        
        # Prepare contact detection
        max_contacts = 1000  # Adjust based on your needs
        contact_pairs = np.zeros(max_contacts * 2 + 1, dtype=np.int32)
        distances = np.zeros(max_contacts, dtype=np.float32)
        
        d_contact_pairs = cuda.mem_alloc(contact_pairs.nbytes)
        d_distances = cuda.mem_alloc(distances.nbytes)
        
        # Launch contact detection kernel
        block_size = (16, 16, 1)
        grid_size = (
            (len(hand_indices) + block_size[0] - 1) // block_size[0],
            (len(other_indices) + block_size[1] - 1) // block_size[1]
        )
        
        self.contact_kernel(
            self.d_transformed,
            self.d_transformed,
            np.float32(0.02),  # distance threshold (2cm)
            np.float32(0.9),   # velocity threshold
            block=block_size,
            grid=grid_size
        )
        
        # Get results
        cuda.memcpy_dtoh(contact_pairs, d_contact_pairs)
        cuda.memcpy_dtoh(distances, d_distances)
        
        return self.process_contact_results(frame, contact_pairs, distances)
    
    def get_bone_matrices(self, frame):
        """현재 프레임의 bone matrices를 가져옴"""
        bpy.context.scene.frame_set(frame)
        matrices = []
        
        for bone in self.armature.pose.bones:
            # World space에서의 bone matrix 계산
            matrix = self.armature.matrix_world @ bone.matrix
            # numpy array로 변환
            matrix_array = np.array(matrix.to_4x4(), dtype=np.float32)
            matrices.append(matrix_array)
            
        return np.array(matrices)
    
    def get_hand_vertices(self):
        """손에 해당하는 vertex indices 반환"""
        hand_vertices = []
        for i, vertex in enumerate(self.vertices):
            # vertex의 주요 영향을 받는 bone이 hand bone인 경우
            max_weight = 0
            main_bone = None
            for weight_info in vertex.bone_weights:
                if weight_info['weight'] > max_weight:
                    max_weight = weight_info['weight']
                    main_bone = self.armature.pose.bones[weight_info['bone_index']]
            
            if main_bone and 'hand' in main_bone.name.lower():
                hand_vertices.append(i)
        return hand_vertices
    
    def get_non_hand_vertices(self):
        """손이 아닌 부분의 vertex indices 반환"""
        all_vertices = set(range(len(self.vertices)))
        hand_vertices = set(self.get_hand_vertices())
        return list(all_vertices - hand_vertices)
    
    def calculate_velocities(self, positions_current, positions_next):
        """현재 프레임과 다음 프레임 사이의 속도 계산"""
        velocities = positions_next - positions_current
        return velocities
    
    def process_contact_results(self, frame, contact_pairs, distances):
        """CUDA 계산 결과를 contact struct 형식으로 변환"""
        num_contacts = contact_pairs[0]  # 첫 번째 요소는 접촉 쌍의 개수
        
        processed_pairs = []
        for i in range(num_contacts):
            hand_idx = contact_pairs[i * 2 + 1]
            other_idx = contact_pairs[i * 2 + 2]
            distance = distances[i]
            
            if distance <= 0.02:  # 2cm threshold
                processed_pairs.append((hand_idx, other_idx))
        
        return {
            'frame': frame,
            'contactPolygonPairs': [Vector((p[0], p[1])) for p in processed_pairs],
            'localDistanceField': [],  # 필요한 경우 구현
            'geodesicDistance': []     # 필요한 경우 구현
        }
    
    def load_vertex_data(self, vertex_file_path):
        """vertex 파일에서 데이터 로드"""
        return NewVertex.load_from_simple_txt(vertex_file_path)
        
    def create_vertex_groups(self):
        """vertex group 정보 생성"""
        groups = {}
        for i, vertex in enumerate(self.vertices):
            group_id = vertex.vertex_group
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(i)
        return groups


    def process_all_frames(self, output_dir, index):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        contact_data = []
        for frame in range(self.frame_start, self.frame_end + 1):
            data = self.detect_contacts_for_frame(frame)
            contact_data.append(data)
            
            # Save results
            output_file = output_path / f"{index}.txt"
            self.save_contact_data(contact_data, output_file)
            
def main():

    characters = ["Remy"]

    # CUDA 사용 여부 결정
    use_cuda = True  # 환경 변수나 설정 파일에서 읽어올 수 있음
    
    for char_name in characters:
        index = 0
        bvh_files = list(Path(f"../dataset/Animations/bvhs/{char_name}").glob("*.bvh"))
        vertex_file = Path(f"../dataset/vertexes/{char_name}_clean_vertices.txt")
        detector_class = CUDAContactDetector if use_cuda else CPUContactDetector

        for animation in bvh_files:
            print(f"\nProcessing character: {char_name}")
            
            detector = detector_class(vertex_file, animation)
            
            output_dir = Path(f"../dataset/Contacts/{char_name}")
            detector.process_all_frames(output_dir, index)
            index += 1

if __name__ == "__main__":
    main()