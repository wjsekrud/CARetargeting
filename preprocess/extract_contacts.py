import bpy
import bmesh
from pathlib import Path
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import math
import sys
sys.path.append("../")

from utils import validate_contacts

class NewVertex:
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.vertex_group = 0
        self.bone_weights = []

    @classmethod
    def load_from_simple_txt(cls, file_path):
        """단순 텍스트 파일에서 NewVertex 객체들을 로드"""
        vertices = []
        triangles = []

        with open(file_path, 'r') as f:
            lines = f.readlines()
            vertex_count = int(lines[0].split(":")[1])
            triangle_count = int(lines[1].split(":")[1])

            vertex_lines = lines[3:3 + vertex_count]
            for line in vertex_lines:
                position, bone_weight, vertex_group = line.strip().split('|')
                new_vertex = cls()
                new_vertex.position = list(map(float, position.split(',')))
                new_vertex.vertex_group = int(vertex_group)
                new_vertex.bone_weights = [
                    {
                        'bone_index': int(bw.split(':')[0]),
                        'weight': float(bw.split(':')[1]),
                    } for bw in bone_weight.split(',') if bw
                ]
                vertices.append(new_vertex)

            triangle_lines = lines[4 + vertex_count:]
            for line in triangle_lines:
                triangles.append({
                    'indices': list(map(int, line.split(','))),
                })

        return vertices, triangles
    
class ContactDetector:
    def __init__(self, vertex_file_path, bvh_file_path):
        self.vertices, self.triangles = NewVertex.load_from_simple_txt(vertex_file_path)
        self.load_animation(bvh_file_path)
        self.vertex_groups = self.create_vertex_groups()
        
    def load_animation(self, bvh_file_path):
        """BVH 파일 로드 및 애니메이션 데이터 추출"""
        # 기존 데이터 초기화
        bpy.ops.wm.read_factory_settings(use_empty=True)
        
        # BVH 파일 import
        bpy.ops.import_anim.bvh(
            filepath=str(bvh_file_path),
            rotate_mode='NATIVE',
            axis_forward='-Z',
            axis_up='Y',
            update_scene_fps=True,
            use_fps_scale=False
        )
        
        # 애니메이션 정보 저장
        self.armature = bpy.context.selected_objects[0]
        self.frame_start = bpy.context.scene.frame_start
        self.frame_end = bpy.context.scene.frame_end
        
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
                        print(contact_pairs, other_id)
                        
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
        
    def create_contact_struct(self, contact_pairs, frame):
        """Contact Struct 생성"""
        print(contact_pairs)
        return {
            'frame': frame,
            'contactPolygonPairs': [Vector((p[0], p[1])) 
                                  for p in contact_pairs],
            'localDistanceField': [],  # 추후 구현
            'geodesicDistance': []     # 추후 구현
        }
        
    def process_all_frames(self, output_dir, index):
        """모든 프레임에 대해 contact detection 수행 및 결과 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        contact_data = []
        data = {}
        for frame in range(self.frame_start, self.frame_end + 1):
            data = self.detect_contacts_for_frame(frame)
            contact_data.append(data)
            if data['contactPolygonPairs'] != []:
                print("validatecontacts")
                #validate_contacts.validate_contact(frame, self.vertices, contact_data[-1], Path("../dataset/"))
            # 결과를 파일로 저장
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

    characters = ["Y_bot"]

    for char_name in characters:
        index=0
        bvh_files = list(Path(f"../dataset/Animations/bvhs/{char_name}").glob("*.bvh"))
        vertex_file = Path(f"../dataset/vertexes/{char_name}_clean_vertices.txt")
        for animation in bvh_files:
            print(f"\nProcessing character: {char_name}")
            detector = ContactDetector(vertex_file, animation)
            output_dir = Path(f"../dataset/Contacts/{char_name}")
            detector.process_all_frames(output_dir, index)
            index += 1

    
if __name__ == "__main__":
    main()