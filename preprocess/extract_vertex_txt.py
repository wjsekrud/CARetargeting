import bpy
import struct
import json
from pathlib import Path
import numpy as np

class NewVertexExporter:
    def __init__(self, mesh_obj):
        self.mesh_obj = mesh_obj
        self.vertex_groups = mesh_obj.vertex_groups
        self.mesh = mesh_obj.data

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

            if primary_bone and max_weight >= 0.5:
                if primary_bone not in bone_map:
                    new_group = self.mesh_obj.vertex_groups.new(name=str(primary_bone))
                    bone_map[primary_bone] = new_group
                bone_map[primary_bone].add([vertex_idx], max_weight, 'REPLACE')

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
                'position': list(vertex.co),  # [x, y, z]
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
            weight = group.weight
            if weight > 0.5:
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