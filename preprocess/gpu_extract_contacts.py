from pathlib import Path
import sys
sys.path.append("../")
from utils.CUDAkernels import cuda_contact_kernel, cuda_transform_kernel
from utils.newvertex import NewVertex
from utils.bvhParser import BVHAnimation

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
        print("CreateVertexGroup complete")
        self.frame_start = 0
        self.frame_end = self.bvh_parser.num_frames - 1

        # Initialize CUDA
        print("INIT CUDA : transform")
        self.mod_transform = SourceModule(cuda_transform_kernel)
        print("INIT CUDA : contact")
        self.mod_contact = SourceModule(cuda_contact_kernel)
        self.transform_kernel = self.mod_transform.get_function("transform_vertices")
        self.contact_kernel = self.mod_contact.get_function("detect_contacts")
        
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
        
        # 1. 현재 프레임의 bone matrices 가져오기
        bone_matrices = self.get_bone_matrices(frame)
        d_bone_matrices = cuda.mem_alloc(bone_matrices.nbytes)
        cuda.memcpy_htod(d_bone_matrices, bone_matrices)
        
        # 2. Vertex 변환 수행
        block_size = 256
        grid_size = (len(self.vertices) + block_size - 1) // block_size
        
        self.transform_kernel(
            self.d_positions,
            d_bone_matrices,
            self.d_bone_indices,
            self.d_weights,
            self.d_transformed,
            np.int32(len(self.vertices)),
            np.int32(max(len(v.bone_weights) for v in self.vertices)),
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )
        
        # 3. 손과 비손 vertex 구분
        hand_indices = np.array(self.get_hand_vertices(), dtype=np.int32)
        other_indices = np.array(self.get_non_hand_vertices(), dtype=np.int32)
        #print(hand_indices)
        
        # 4. 현재 프레임과 다음 프레임의 위치 계산 (velocity 계산용)
        if frame < self.frame_end:
            next_bone_matrices = self.get_bone_matrices(frame + 1)
            d_next_bone_matrices = cuda.mem_alloc(next_bone_matrices.nbytes)
            cuda.memcpy_htod(d_next_bone_matrices, next_bone_matrices)
            
            d_next_transformed = cuda.mem_alloc(self.vertex_positions.nbytes)
            self.transform_kernel(
                self.d_positions,
                d_next_bone_matrices,
                self.d_bone_indices,
                self.d_weights,
                d_next_transformed,
                np.int32(len(self.vertices)),
                np.int32(max(len(v.bone_weights) for v in self.vertices)),
                block=(block_size, 1, 1),
                grid=(grid_size, 1)
            )
        else:
            # 마지막 프레임의 경우 현재 위치를 다음 위치로도 사용
            d_next_transformed = self.d_transformed
        
        # 5. Hand와 other vertices의 위치와 속도 데이터 준비
        positions = np.zeros((len(self.vertices), 3), dtype=np.float32)
        cuda.memcpy_dtoh(positions, self.d_transformed)
        
        next_positions = np.zeros((len(self.vertices), 3), dtype=np.float32)
        cuda.memcpy_dtoh(next_positions, d_next_transformed)
        
        velocities = (next_positions - positions) / self.bvh_parser.frame_time
        #print(positions[hand_indices])
        # Hand vertices 데이터
        hand_positions = positions[hand_indices]
        hand_velocities = velocities[hand_indices]
        
        # Other vertices 데이터
        other_positions = positions[other_indices]
        other_velocities = velocities[other_indices]
        
        # GPU 메모리 할당
        d_hand_positions = cuda.mem_alloc(hand_positions.nbytes)
        d_hand_velocities = cuda.mem_alloc(hand_velocities.nbytes)
        d_other_positions = cuda.mem_alloc(other_positions.nbytes)
        d_other_velocities = cuda.mem_alloc(other_velocities.nbytes)
        
        # 데이터 전송
        cuda.memcpy_htod(d_hand_positions, hand_positions)
        cuda.memcpy_htod(d_hand_velocities, hand_velocities)
        cuda.memcpy_htod(d_other_positions, other_positions)
        cuda.memcpy_htod(d_other_velocities, other_velocities)
        
        # 6. Contact detection 결과를 저장할 메모리 준비
        max_contacts = min(1000, len(hand_indices) * len(other_indices))  # 최대 접촉 쌍 수 제한
        contact_pairs = np.zeros(max_contacts * 2 + 1, dtype=np.int32)  # +1 for count
        distances = np.zeros(max_contacts, dtype=np.float32)
        
        d_contact_pairs = cuda.mem_alloc(contact_pairs.nbytes)
        d_distances = cuda.mem_alloc(distances.nbytes)
        
        cuda.memcpy_htod(d_contact_pairs, contact_pairs)
        cuda.memcpy_htod(d_distances, distances)
        
        # 7. Contact detection 커널 실행
        block_size = (16, 16, 1)
        grid_size = (
            (len(hand_indices) + block_size[0] - 1) // block_size[0],
            (len(other_indices) + block_size[1] - 1) // block_size[1]
        )
        
        self.contact_kernel(
            d_hand_positions,
            d_other_positions,
            d_hand_velocities,
            d_other_velocities,
            d_contact_pairs,
            d_distances,
            np.int32(len(hand_indices)),
            np.int32(len(other_indices)),
            np.float32(0.02),  # distance threshold (2cm)
            np.float32(0.9),   # velocity threshold (cosine similarity)

        )
        
        # 8. 결과 가져오기
        cuda.memcpy_dtoh(contact_pairs, d_contact_pairs)
        cuda.memcpy_dtoh(distances, d_distances)
        print(f"distances: {distances}")
        print(f"pairs: {contact_pairs}")
        # 9. 메모리 정리
        if frame < self.frame_end:
            d_next_bone_matrices.free()
            if d_next_transformed != self.d_transformed:
                d_next_transformed.free()
        d_bone_matrices.free()
        d_hand_positions.free()
        d_hand_velocities.free()
        d_other_positions.free()
        d_other_velocities.free()
        d_contact_pairs.free()
        d_distances.free()
        
        # 10. 결과 처리 및 반환
        return self.process_contact_results(frame, contact_pairs, distances)
    
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

    # CUDA 사용 여부 결정
    use_cuda = True  # 환경 변수나 설정 파일에서 읽어올 수 있음
    
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