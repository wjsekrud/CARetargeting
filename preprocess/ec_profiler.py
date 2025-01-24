import bpy
import bmesh
from pathlib import Path
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import math
import sys
import cProfile
import pstats
from functools import wraps
import time

sys.path.append("../")
from utils import validate_contacts

def profile_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        wrapper.total_time = getattr(wrapper, 'total_time', 0) + (end_time - start_time)
        wrapper.call_count = getattr(wrapper, 'call_count', 0) + 1
        return result
    return wrapper

class GeometryCacheEntry:
    @classmethod
    @profile_decorator
    def __init__(self, bmesh, bvh_tree, vertex_positions):
        self.bmesh = bmesh
        self.bvh_tree = bvh_tree
        self.vertex_positions = vertex_positions
        self.last_access = 0

class GeometryCache:
    @classmethod
    @profile_decorator
    def __init__(self, max_size=100):
        self.cache = {}  # (group_id, frame) -> GeometryCacheEntry
        self.max_size = max_size
        self.access_counter = 0
    
    def get_or_create(self, key, creation_func):
        self.access_counter += 1
        
        if key in self.cache:
            entry = self.cache[key]
            entry.last_access = self.access_counter
            return entry
            
        if len(self.cache) >= self.max_size:
            self._evict_lru()
            
        bmesh_obj, positions = creation_func()
        bvh_tree = BVHTree.FromBMesh(bmesh_obj)
        entry = GeometryCacheEntry(bmesh_obj, bvh_tree, positions)
        entry.last_access = self.access_counter
        self.cache[key] = entry
        return entry
    
    def _evict_lru(self):
        if not self.cache:
            return
        lru_key = min(self.cache.keys(), 
                     key=lambda k: self.cache[k].last_access)
        self.cache[lru_key].bmesh.free()
        del self.cache[lru_key]
    
    def clear(self):
        for entry in self.cache.values():
            entry.bmesh.free()
        self.cache.clear()

class CachedBVHTree:
    @classmethod
    @profile_decorator
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

class NewVertex:
    @classmethod
    @profile_decorator
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
        self.bvh_cache = CachedBVHTree()
        self.geometry_cache = GeometryCache()
        
    @profile_decorator
    def load_animation(self, bvh_file_path):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_anim.bvh(
            filepath=str(bvh_file_path),
            rotate_mode='NATIVE',
            axis_forward='-Z',
            axis_up='Y',
            update_scene_fps=True,
            use_fps_scale=False
        )
        self.armature = bpy.context.selected_objects[0]
        self.frame_start = bpy.context.scene.frame_start
        self.frame_end = bpy.context.scene.frame_end
    
    @profile_decorator
    def create_vertex_groups(self):
        groups = {}
        for i, vertex in enumerate(self.vertices):
            group_id = vertex.vertex_group
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(i)
        return groups
    
    @profile_decorator
    def calculate_vertex_positions(self, vertex_indices, frame):
        bpy.context.scene.frame_set(frame)
        positions = []
        
        for idx in vertex_indices:
            vertex = self.vertices[idx]
            final_position = Vector(vertex.position)
            
            for weight_info in vertex.bone_weights:
                bone_idx = weight_info['bone_index']
                weight = weight_info['weight']
                bone = self.armature.pose.bones[bone_idx]
                bone_matrix = self.armature.matrix_world @ bone.matrix
                transformed_position = bone_matrix @ final_position
                final_position = final_position.lerp(transformed_position, weight)
                
            positions.append(final_position)
            
        return positions
    
    @profile_decorator
    def create_geometry(self, vertex_indices, frame):
        vertex_positions = self.calculate_vertex_positions(vertex_indices, frame)
        bm = bmesh.new()
        vertex_lookup = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_indices)}
        
        bm_verts = [bm.verts.new(pos) for pos in vertex_positions]
        bm.verts.ensure_lookup_table()
        
        valid_triangles = [
            [vertex_lookup[v] for v in triangle['indices']]
            for triangle in self.triangles
            if all(v in vertex_lookup for v in triangle['indices'])
        ]
        
        for triangle_indices in valid_triangles:
            try:
                bm.faces.new([bm_verts[i] for i in triangle_indices])
            except Exception:
                continue
                
        bm.faces.ensure_lookup_table()
        #print(len(vertex_positions))
        return bm, vertex_positions
    
    @profile_decorator
    def find_closest_pairs_cached(self, vertices1, vertices2, pos1, pos2):
        pairs = []
        print(len(vertices1), len(pos1))
        for i, p1 in enumerate(pos1):
            for j, p2 in enumerate(pos2):
                distance = (p1 - p2).length
                #print(i)
                pairs.append((distance, (vertices1[i], vertices2[j])))
        pairs.sort(key=lambda x: x[0])
        return [(p[1][0], p[1][1]) for p in pairs]
    
    @profile_decorator
    def detect_contacts_for_frame(self, frame):
        hand_groups = {group_id: vertices 
                      for group_id, vertices in self.vertex_groups.items() 
                      if self.is_hand_bone(group_id)}
        contact_pairs = []

        for hand_id, hand_vertices in hand_groups.items():
            hand_geo = self.geometry_cache.get_or_create(
                (hand_id, frame),
                lambda: self.create_geometry(hand_vertices, frame)
            )
            
            
            for other_id, other_vertices in self.vertex_groups.items():
                if other_id == hand_id:
                    continue
                    
                other_geo = self.geometry_cache.get_or_create(
                    (other_id, frame),
                    lambda: self.create_geometry(other_vertices, frame)
                )
                print("handverteices:")
                print(hand_vertices.__len__(),  hand_geo.vertex_positions.__len__())    
                if hand_geo.bvh_tree.overlap(other_geo.bvh_tree):
                    min_distance = float('inf')
                    for p1 in hand_geo.vertex_positions:
                        for p2 in other_geo.vertex_positions:
                            dist = (p1 - p2).length
                            min_distance = min(min_distance, dist)
                    
                    if min_distance < 2:
                        closest_pairs = self.find_closest_pairs_cached(
                            hand_vertices, other_vertices,
                            hand_geo.vertex_positions, other_geo.vertex_positions
                        )
                        contact_pairs.extend(closest_pairs[:3])
        
        return self.create_contact_struct(contact_pairs, frame)

    
    @profile_decorator
    def is_hand_bone(self, bone_index):
        bone = self.armature.pose.bones[bone_index]
        return 'hand' in bone.name.lower()
    
    @profile_decorator
    def check_contact_conditions(self, vertices1, vertices2, frame):
        pos1_current = self.calculate_vertex_positions(vertices1, frame)
        pos1_next = self.calculate_vertex_positions(vertices1, frame + 1)
        pos2_current = self.calculate_vertex_positions(vertices2, frame)
        pos2_next = self.calculate_vertex_positions(vertices2, frame + 1)
        
        vel1 = [(n - c) for c, n in zip(pos1_current, pos1_next)]
        vel2 = [(n - c) for c, n in zip(pos2_current, pos2_next)]
        
        avg_vel1 = sum(vel1, Vector()) / len(vel1)
        avg_vel2 = sum(vel2, Vector()) / len(vel2)
        
        min_distance = float('inf')
        for p1 in pos1_current:
            for p2 in pos2_current:
                dist = (p1 - p2).length
                min_distance = min(min_distance, dist)
                
        print(f"mind: {min_distance}")
        
        if avg_vel1.length and avg_vel2.length:
            cos_sim = avg_vel1.dot(avg_vel2) / (avg_vel1.length * avg_vel2.length)
            print(f"cos-sim : {cos_sim}")
            if cos_sim > 0.9:
                return True
                
        return min_distance < 2
    
    @profile_decorator
    def find_closest_pairs(self, vertices1, vertices2, frame):
        pairs = []
        pos1 = self.calculate_vertex_positions(vertices1, frame)
        pos2 = self.calculate_vertex_positions(vertices2, frame)
        
        for i, p1 in enumerate(pos1):
            for j, p2 in enumerate(pos2):
                distance = (p1 - p2).length
                pairs.append((distance, (vertices1[i], vertices2[j])))
        
        pairs.sort(key=lambda x: x[0])
        return [(p[1][0], p[1][1]) for p in pairs]
    
    @profile_decorator
    def create_contact_struct(self, contact_pairs, frame):
        print(contact_pairs)
        return {
            'frame': frame,
            'contactPolygonPairs': [Vector((p[0], p[1])) for p in contact_pairs],
            'localDistanceField': [],
            'geodesicDistance': []
        }
    
    @profile_decorator
    def process_all_frames(self, output_dir, index):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        contact_data = []
        
        for frame in range(10):
            data = self.detect_contacts_for_frame(frame)
            contact_data.append(data)
            if data['contactPolygonPairs'] != []:
                print("validatecontacts")
            
            output_file = output_path / f"{index}.txt"
            self.save_contact_data(contact_data, output_file)

    
    @profile_decorator
    def save_contact_data(self, contact_data, output_file):
        with open(output_file, 'w') as f:
            for data in contact_data:
                if len(data['contactPolygonPairs']) > 1:
                    f.write(f"{data['frame']}|")
                    for pair in data['contactPolygonPairs']:
                        f.write(f"{int(pair[0])},{int(pair[1])}|")
                    f.write("\n")

def print_profile_stats():
    print("\nFunction Profiling Results:")
    print("-" * 80)
    print(f"{'Function Name':<40} {'Total Time':<15} {'Calls':<10} {'Avg Time/Call'}")
    print("-" * 80)
    
    for item in dir(ContactDetector):
        func = getattr(ContactDetector, item)
        if hasattr(func, 'total_time'):
            name = func.__name__
            total_time = func.total_time
            calls = func.call_count
            avg_time = total_time / calls if calls else 0
            print(f"{name:<40} {total_time:<15.4f} {calls:<10} {avg_time:.4f}")

def main():
    characters = ["Y_bot"]
    
    profiler = cProfile.Profile()
    profiler.enable()

    for char_name in characters:
        index = 0
        bvh_files = list(Path(f"../dataset/Animations/bvhs/{char_name}").glob("*.bvh"))
        vertex_file = Path(f"../dataset/vertexes/{char_name}_clean_vertices.txt")
        for animation in bvh_files:
            print(f"\nProcessing character: {char_name}")
            detector = ContactDetector(vertex_file, animation)
            output_dir = Path(f"../dataset/Contacts/{char_name}")
            detector.process_all_frames(output_dir, index)
            index += 1

            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats()
            break
            
    print_profile_stats()

if __name__ == "__main__":
    main()