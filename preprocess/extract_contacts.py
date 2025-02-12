import bpy
import bmesh
from pathlib import Path
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import math
import sys
sys.path.append("../")

from utils.NewVertex import NewVertex

    
class CachedBVHTree:
    def __init__(self, max_cache_size=100):
        self.cache = {}  # (group_id, frame) -> (bvh_tree, last_access_time, bmesh)
        self.max_cache_size = max_cache_size
        self.access_count = 0

    def get(self, group_id, frame, creator_func):
        key = (group_id, frame)
        self.access_count += 1
        
        if key in self.cache:
            bvh_tree, _, bmesh = self.cache[key]
            self.cache[key] = (bvh_tree, self.access_count, bmesh)
            return bvh_tree, bmesh
            
        if len(self.cache) >= self.max_cache_size:
            self._evict_least_recently_used()
            
        bmesh = creator_func()
        bvh_tree = BVHTree.FromBMesh(bmesh)
        self.cache[key] = (bvh_tree, self.access_count, bmesh)
        return bvh_tree, bmesh

    def _evict_least_recently_used(self):
        if not self.cache:
            return
            
        lru_key = min(self.cache.keys(), 
                     key=lambda k: self.cache[k][1])
        _, _, bmesh = self.cache[lru_key]
        bmesh.free()
        del self.cache[lru_key]

    def clear(self):
        for _, _, bmesh in self.cache.values():
            bmesh.free()
        self.cache.clear()

class ContactDetector:
    def __init__(self, vertex_file_path, bvh_file_path):
        self.vertices, self.triangles ,self.height= NewVertex.load_from_simple_txt(vertex_file_path)
        self.load_animation(bvh_file_path)
        self.vertex_groups = self.create_vertex_groups()
        self.bvh_cache = CachedBVHTree()
        
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
            use_fps_scale=True
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
                print(f"newid: {group_id}")
            groups[group_id].append(i)
        return groups

    def create_bmesh_for_group(self, group_indices, frame):
        bm = bmesh.new()
        
        # 메시를 한 번에 생성할 수 있도록 데이터 준비
        vertex_positions = self.calculate_vertex_positions(group_indices, frame)
        vertex_lookup = {old_idx: new_idx for new_idx, old_idx in enumerate(group_indices)}
        
        # vertices를 한 번에 추가
        bm_verts = [bm.verts.new(pos) for pos in vertex_positions]
        bm.verts.ensure_lookup_table()
        
        # 현재 group에 포함된 triangle들만 미리 필터링
        valid_triangles = [
            [vertex_lookup[v] for v in triangle['indices']]
            for triangle in self.triangles
            if all(v in vertex_lookup for v in triangle['indices'])
        ]
        
        # faces를 한 번에 추가
        for triangle_indices in valid_triangles:
            try:
                bm.faces.new([bm_verts[i] for i in triangle_indices])
            except Exception:
                continue
                
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
        print(f"start check for frame {frame}")
        hand_groups = {}
        
        for group_id, vertex_indices in self.vertex_groups.items():
            if self.is_hand_bone(group_id):
                hand_groups[group_id] = vertex_indices

        contact_pairs = []

        for hand_id, hand_vertices in hand_groups.items():
            creator_func = lambda: self.create_bmesh_for_group(hand_vertices, frame)
            hand_bvh, hand_bmesh = self.bvh_cache.get(hand_id, frame, creator_func)
            
            for other_id, other_vertices in self.vertex_groups.items():
                if other_id == hand_id:
                    continue
                    
                creator_func = lambda: self.create_bmesh_for_group(other_vertices, frame)
                other_bvh, other_bmesh = self.bvh_cache.get(other_id, frame, creator_func)
                
                intersect = hand_bvh.overlap(other_bvh)
                
                if intersect:
                    if self.check_contact_conditions(hand_vertices, other_vertices, frame):
                        closest_pairs = self.find_closest_pairs(hand_vertices, other_vertices, frame)
                        contact_pairs.extend(closest_pairs[:3])
                        print(contact_pairs, other_id)
            
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
        
        # cosine similarity check
        if avg_vel1.length and avg_vel2.length:
            cos_sim = avg_vel1.dot(avg_vel2) / (avg_vel1.length * avg_vel2.length)
            print(f"cos-sim : {cos_sim}")
            if cos_sim > 0.9:
                return True
            
        # distance check (0.2cm threshold)
        min_distance = float('inf')
        for p1 in pos1_current:
            for p2 in pos2_current:
                dist = (p1 - p2).length
                min_distance = min(min_distance, dist)
                
        print(f"mind: {min_distance}")
        
                
        Hthreshold = self.height * 0.54
       
        
        return min_distance < Hthreshold  # 0.2cm = 0.002m
        
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
            #'localDistanceField': [],  # 추후 구현
            #'geodesicDistance': []     # 추후 구현
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
    
    def __del__(self):
        if hasattr(self, 'bvh_cache'):
            self.bvh_cache.clear()


            
def main():

    characters = [ "Y_bot", "Remy", "ch14_nonPBR"]

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