from pathlib import Path
import sys
sys.path.append("../")
from utils.newvertex import NewVertex
from utils.bvhParser import BVHAnimation
from utils.bvh_accelerator import CUDABVHLib
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from mathutils import Vector


class CUDAContactDetector:
    def __init__(self, vertex_file_path, bvh_file_path):
        self.vertices, self.triangles = self.load_vertex_data(vertex_file_path)
        self.bvh_parser = BVHAnimation(bvh_file_path)
        print("BVHParsing complete")
        self.vertex_groups = self.create_vertex_groups()
        self.cuda_lib = CUDABVHLib()
        self.frame_start = 0
        self.frame_end = self.bvh_parser.num_frames - 1
        self.hand_bvh = None
        self.other_bvh = None

        # Initialize CUDA

        # Load vertices and animation data
        self.vertices, self.triangles = self.load_vertex_data(vertex_file_path)
        self.bvh_parser.load(bvh_file_path)
        
        # Prepare CUDA memory
        print("preparing cuda memory")
        self.prepare_cuda_memory()
        
    def prepare_cuda_memory(self):
        # Allocate memory for vertex positions
        self.vertex_positions = np.array([v.position for v in self.vertices], dtype=np.float32)
        self.d_positions = cuda.mem_alloc(self.vertex_positions.nbytes)
        cuda.memcpy_htod(self.d_positions, self.vertex_positions)
        
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
        self.d_transformed = cuda.mem_alloc(self.vertex_positions.nbytes)
        
    def detect_contacts_for_frame(self, frame):
        """특정 프레임에서의 contact detection 수행"""
        print(f"Processing frame {frame}")

        block_size = 256
        grid_size = (len(self.vertices) + block_size - 1) // block_size
        
        # 1. 현재 프레임의 bone matrices 가져오기
        bone_matrices = self.get_bone_matrices(frame).reshape(-1)
        d_bone_matrices = cuda.mem_alloc(bone_matrices.nbytes)
        cuda.memcpy_htod(d_bone_matrices, bone_matrices)

        # 2. Hand와 other vertices 분리
        hand_indices = self.get_hand_vertices()
        other_indices = self.get_non_hand_vertices()

        max_weights = max(len(v.bone_weights) for v in self.vertices) #본마다 할당된 가중치 본 최대 개수는 1

        bone_indices = np.zeros((len(self.vertices)), dtype=np.int32)
        weights = np.zeros((len(self.vertices)), dtype=np.float32)
        for i, vertex in enumerate(self.vertices):
            for j, weight_info in enumerate(vertex.bone_weights):
                bone_indices[i] = weight_info['bone_index']
                weights[i] = weight_info['weight']
        
        # 3. Vertex positions 계산 (스키닝)
        current_positions = np.zeros_like(self.vertex_positions).reshape(-1)
        vertex_positions_flat = self.vertex_positions.copy().reshape(-1)

        print(current_positions)
        self.cuda_lib.compute_skinning(
            vertex_positions_flat,
            bone_matrices,
            bone_indices,
            weights,
            current_positions,
            np.int32(len(self.vertex_positions)),
            block= (256,1,1),
            grid = (grid_size,1)
        )
        
        print("passed skinninga")
        '''
        next_positions = np.zeros_like(self.vertex_positions)
        # 4. 현재 프레임과 다음 프레임의 위치 계산 (velocity 계산용)
        next_frame = min(frame + 1, self.frame_end)
        next_bone_matrices = self.get_bone_matrices(next_frame)
        next_positions = self.cuda_lib.compute_skinning(
            self.vertex_positions,
            next_bone_matrices,
            bone_indices,
            weights,
            next_positions,
            len(self.vertices),
            block=(256,1,1)
        )
        #'''
        # Velocities 계산
        velocities = next_positions - current_positions
        
        # 5. Hand와 other vertices의 위치와 속도 데이터 준비
        hand_positions = current_positions[hand_indices]
        hand_velocities = velocities[hand_indices]
        other_positions = current_positions[other_indices]
        other_velocities = velocities[other_indices]
        
        # 6. BVH 구축 (첫 프레임이거나 필요한 경우)
        if self.hand_bvh is None:
            self.hand_bvh, d_hand_bvh = self.cuda_lib.create_bvh(hand_positions)
        if self.other_bvh is None:
            self.other_bvh, d_other_bvh = self.cuda_lib.create_bvh(other_positions)
        
        # 7. Contact detection 실행
        contact_pairs = []
        distances = []
        contactCount = [0]
        contacts, distances = self.cuda_lib.detect_vertex_contacts(
            d_hand_bvh,
            d_other_bvh,
            hand_positions,
            hand_velocities,
            other_positions,
            other_velocities,
            contact_pairs,
            distances,
            contactCount,
            1
        )
        
        # 8. 결과 변환
        contact_pairs = []
        for i in range(len(contacts)):
            hand_idx = hand_indices[contacts[i][0]]
            other_idx = other_indices[contacts[i][1]]
            if distances[i] <= 0.02:  # 2cm threshold
                contact_pairs.append((hand_idx, other_idx))
        
        return self.process_contact_results(frame, contact_pairs)
    
    def get_bone_matrices(self, frame):
        """현재 프레임의 bone matrices를 가져옴"""
        return np.array(self.bvh_parser.get_joint_matrices(frame))
    
    def get_hand_vertices(self):
        """
        손에 해당하는 vertex indices를 반환
        BVH의 joint 이름을 기반으로 hand bone을 식별하고, 
        해당 bone에 가장 큰 영향을 받는 vertex들을 찾음
        """
        hand_vertices = []
        hand_joint_keywords = ['hand', 'finger', 'thumb', 'index', 'middle', 'ring', 'pinky']
        
        # BVH에서 hand 관련 joint들의 인덱스를 찾음
        hand_bone_indices = set()
        joint_names = self.bvh_parser.get_joint_names()
        
        for i, name in enumerate(joint_names):
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in hand_joint_keywords):
                hand_bone_indices.add(i)
        
        # 각 vertex에 대해 hand bone의 영향력 확인
        for i, vertex in enumerate(self.vertices):
            #print(f"i, vertex: {i}, {vertex}")
            max_weight = 0
            main_bone_idx = None
            
            # vertex의 bone weights를 확인
            for weight_info in vertex.bone_weights:
                bone_idx = weight_info['bone_index']
                weight = weight_info['weight']
                #print(bone_idx, weight)
                if weight > max_weight:
                    max_weight = weight
                    main_bone_idx = bone_idx
            
            # 주요 영향을 주는 bone이 hand bone인 경우
            if main_bone_idx in hand_bone_indices:
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
            print(data)
            contact_data.append(data)
            
            # Save results
            output_file = output_path / f"{index}.txt"
            self.save_contact_data(contact_data, output_file)
    
    def save_contact_data(self, contact_data, output_file):
        """Contact data를 파일로 저장"""
        with open(output_file, 'w') as f:
            for data in contact_data:
                if len(data['contactPolygonPairs']) > 1:
                    f.write(f"{data['frame']}|")
                    for pair in data['contactPolygonPairs']:
                        f.write(f"{int(pair[0])},{int(pair[1])}|")
                    f.write("\n")

def main():

    characters = ["Remy"]

    for char_name in characters:
        index = 0
        bvh_files = list(Path(f"../dataset/Animations/bvhs/{char_name}").glob("*.bvh"))
        vertex_file = Path(f"../dataset/vertexes/{char_name}_clean_vertices.txt")
        detector_class = CUDAContactDetector
        for animation in bvh_files:
            print(f"\nProcessing character: {char_name}")
            
            detector = detector_class(vertex_file, animation)
            
            output_dir = Path(f"../dataset/Contacts/{char_name}")
            detector.process_all_frames(output_dir, index)
            index += 1

if __name__ == "__main__":
    main()