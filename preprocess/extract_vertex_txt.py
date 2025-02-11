import bpy
import bmesh
import heapq
from collections import defaultdict
from pathlib import Path
import numpy as np
from mathutils import Vector

class NewVertexExporter:
    def __init__(self, mesh_obj, scale=0.01):
        self.mesh_obj = mesh_obj
        self.vertex_groups = mesh_obj.vertex_groups
        self.mesh = mesh_obj.data
        self.global_scale = scale
        self.end_effectors = ['mixamorig:Head', 'mixamorig:LeftHand', 'mixamorig:RightHand', 'mixamorig:LeftToeBase', 'mixamorig:RightToeBase']

    def calculate_geodesic_distance(self):
        # bmesh 생성
        bm = bmesh.new()
        bm.from_mesh(self.mesh)
        bm.edges.ensure_lookup_table()
        
        # 엔드 이펙터 버텍스 그룹 매핑
        end_effector_vertices = defaultdict(list)
        for vertex in self.mesh.vertices:
            for group in vertex.groups:
                group_name = self.vertex_groups[group.group].name
                if group_name in self.end_effectors and group.weight > 0.5:
                    end_effector_vertices[group_name].append(vertex.index)

        def dijkstra(start_idx, bm):
            distances = {v.index: float('inf') for v in bm.verts}
            distances[start_idx] = 0
            pq = [(0, start_idx)]
            
            while pq:
                current_distance, current_vertex = heapq.heappop(pq)

                if current_distance > distances[current_vertex]:
                    continue
                    

                for edge in bm.verts[current_vertex].link_edges:
                    next_vertex = edge.other_vert(bm.verts[current_vertex])
                    distance = edge.calc_length() * self.global_scale
                    new_distance = distances[current_vertex] + distance
                    
                    if new_distance < distances[next_vertex.index]:
                        distances[next_vertex.index] = new_distance
                        heapq.heappush(pq, (new_distance, next_vertex.index))

            
            return distances

        # 모든 엔드 이펙터 쌍 간의 최대 거리 계산
        max_geodesic_distance = 0

        representative_vertices = {}
        max_geodesic_distance = 0
        processed_pairs = set()

        bm.verts.ensure_lookup_table()
        for i, (name1, vertices1) in enumerate(end_effector_vertices.items()):
            for name2, vertices2 in list(end_effector_vertices.items())[i+1:]:
                if (name1, name2) in processed_pairs:
                    continue
                
                rep1 = rep2 = None

                if name1 in representative_vertices:
                    vertices1 = [representative_vertices[name1]]
                if name2 in representative_vertices:
                    vertices2 = [representative_vertices[name2]]

                for v1 in vertices1:
                    distances = dijkstra(v1, bm)
                    for v2 in vertices2:
                        if distances[v2] != float('inf'):
                            max_geodesic_distance = max(max_geodesic_distance, distances[v2])
                            rep1 = v1
                            rep2 = v2

                if name1 not in representative_vertices:
                    representative_vertices[name1] = rep1
                if name2 not in representative_vertices:
                    representative_vertices[name2] = rep2

                processed_pairs.add((name1, name2))
                processed_pairs.add((name2, name1))
                print(f"processed : {name1, name2}, distance: {max_geodesic_distance}")


        bm.free()
        return max_geodesic_distance
    
    def assign_vertex_groups(self):
        """버텍스 그룹이 없는 경우, 본 인덱스 기반 그룹 생성"""
        bone_map = {}
        for vertex_idx, vertex in enumerate(self.mesh.vertices):
            max_weight = 0
            primary_bone = None

            for group in vertex.groups:
                weight = group.weight
                if weight > max_weight:
                    max_weight = weight
                    primary_bone = group.group

            #if primary_bone and max_weight >= 0.5:
            if primary_bone and int(str(primary_bone)) < 22:
                if primary_bone not in bone_map:
                    new_group = self.mesh_obj.vertex_groups.new(name=str(primary_bone))
                    bone_map[primary_bone] = new_group
                bone_map[primary_bone].add([vertex_idx], max_weight, 'REPLACE')

    def get_mesh_height(self):
        # 월드 공간에서의 바운딩 박스 좌표
        world_bbox = [self.mesh_obj.matrix_world @ Vector(corner) for corner in self.mesh_obj.bound_box]
        
        # z축 최대/최소값 차이로 높이 계산
        min_z = min(corner.z for corner in world_bbox)
        max_z = max(corner.z for corner in world_bbox)
        return (max_z - min_z) * self.global_scale
    
    def export_to_file(self, output_path):
        # 버텍스 그룹 생성
        self.assign_vertex_groups()

        # 메시를 삼각형화
        bpy.context.view_layer.objects.active = self.mesh_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris()
        bpy.ops.object.mode_set(mode='OBJECT')

        # 데이터 수집
        vertices = []
        triangles = []

        # 버텍스 데이터 수집
        for vertex_idx, vertex in enumerate(self.mesh.vertices):
            vertex_data = {
                'position': [coord * self.global_scale for coord in vertex.co], #list(vertex.co),  # [x, y, z]
                'bone_weights': self._get_vertex_weights(vertex_idx),
                'vertex_group': self._get_primary_group(vertex_idx)
            }
            vertices.append(vertex_data)

        # 삼각형 데이터 수집 (CW 순서 보장)
        for poly in self.mesh.polygons:
            if len(poly.vertices) == 3:  # 삼각형인지 확인
                triangles.append({
                    'indices': list(poly.vertices),
                })

        # 데이터를 단순 텍스트 형식으로 저장
        self._save_to_simple_txt(output_path, vertices, triangles)

    def _get_vertex_weights(self, vertex_idx):
        """해당 버텍스에 영향을 주는 본 가중치 정보 수집"""
        weights = []
        for group in self.mesh.vertices[vertex_idx].groups:

            if group.group < 22:
                weight = group.weight
                #if weight > 0.5:
                weights.append({
                    'bone_index': group.group,
                    'weight': weight
                })
        return weights

    def _get_primary_group(self, vertex_idx):
        """가장 큰 가중치를 가진 버텍스 그룹 반환"""
        max_weight = -1
        primary_group = 0

        for group in self.mesh.vertices[vertex_idx].groups:
            weight = group.weight
            if weight > max_weight:
                max_weight = weight
                primary_group = group.group

        return primary_group

    def _save_to_simple_txt(self, output_path, vertices, triangles):
        """단순 텍스트 형식으로 파일 저장"""
        with open(output_path, 'w') as f:
            max_geodesic = self.calculate_geodesic_distance()

            f.write(f"Height: {self.get_mesh_height()}\n")
            f.write(f"MaxGeodesicDistance: {max_geodesic}\n")
            f.write(f"VertexCount: {len(vertices)}\n")
            f.write(f"TriangleCount: {len(triangles)}\n")

            f.write("Vertices:\n")
            for vertex in vertices:
                position = ",".join(map(str, vertex['position']))
                bone_weight = ",".join(f"{w['bone_index']}:{w['weight']}" for w in vertex['bone_weights'])
                f.write(f"{position}|{bone_weight}|{vertex['vertex_group']}\n")

            f.write("Triangles:\n")
            for triangle in triangles:
                indices = ",".join(map(str, triangle['indices']))
                f.write(f"{indices}\n")

class NewVertex:
    """다른 모듈에서 사용할 NewVertex 클래스"""
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.allocated_bone = 0
        self.vertex_group = 0
        self.bone_weights = []

    @classmethod
    def load_from_simple_txt(cls, file_path):
        """단순 텍스트 파일에서 NewVertex 객체들을 로드"""
        vertices = []
        triangles = []

        with open(file_path, 'r') as f:
            lines = f.readlines()
            height = int(lines[0].split(":")[1])
            vertex_count = int(lines[1].split(":")[1])
            triangle_count = int(lines[2].split(":")[1])

            vertex_lines = lines[4:4 + vertex_count]
            for line in vertex_lines:
                position, bone_weight, vertex_group = line.strip().split('|')
                new_vertex = cls()
                new_vertex.position = list(map(float, position.split(',')))
                new_vertex.vertex_group = int(vertex_group)
                new_vertex.bone_weights = [
                    {
                        'bone_index': int(bw.split(':')[0]),
                        'weight': float(bw.split(':')[1])
                    } for bw in bone_weight.split(',') if bw
                ]
                vertices.append(new_vertex)

            triangle_lines = lines[4 + vertex_count:]
            for line in triangle_lines:
                indices = line.strip().split('|')
                triangles.append({
                    'indices': list(map(int, indices.split(','))),
                })

        return vertices, triangles

def export_all_fbx_to_newvertex(base_path):
    """Characters 폴더 하위의 모든 FBX 파일을 NewVertex 구조체 파일로 변환"""
    characters_path = Path("../dataset/characters/clean")
    output_path = Path("../dataset/vertexes")
    output_path.mkdir(parents=True, exist_ok=True)
    
    fbx_files = characters_path.glob("**/*.fbx")
    
    for fbx_file in fbx_files:
        bpy.ops.wm.read_factory_settings(use_empty=True)  # 기존 씬 초기화
        bpy.ops.import_scene.fbx(filepath=str(fbx_file))  # FBX 임포트
        
        # 메시 오브젝트 찾기
        mesh_obj = None
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                mesh_obj = obj
                break
        
        if mesh_obj is None:
            print(f"No mesh found in {fbx_file}")
            continue
        
        # NewVertex 구조체로 변환 및 저장
        output_file = output_path / (fbx_file.stem + "_vertices.txt")
        exporter = NewVertexExporter(mesh_obj)
        exporter.export_to_file(output_file)
        print(f"Exported: {output_file}")

# 사용 예시
if __name__ == "__main__":
    base_path = Path(__file__).parent  # 현재 파일의 디렉토리
    export_all_fbx_to_newvertex(base_path)