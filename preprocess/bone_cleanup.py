import bpy
import os
from pathlib import Path
import mathutils
from mathutils import Vector
import bmesh

def read_bone_names(filepath):
    """본 이름 목록이 있는 텍스트 파일을 읽어옵니다."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def clean_bone_name(name):
    """본 이름에서 'mixamorig:' 접두어를 제거합니다."""
    prefix = "mixamorig:"
    return name[len(prefix):] if name.startswith(prefix) else name

def should_keep_bone(bone_name, keep_bones):
    """본을 유지해야 하는지 확인합니다."""
    cleaned_name = clean_bone_name(bone_name)
    return cleaned_name in keep_bones

def find_closest_keeper_parent(bone, keep_bones):
    """유지할 본 목록에 있는 가장 가까운 부모 본을 찾습니다."""
    current = bone
    while current and current.parent:
        
        current = current.parent
        if should_keep_bone(current.name, keep_bones):
            return current
        
    return None

def find_deformed_mesh(armature):
    """아마추어에 의해 변형되는 메시 오브젝트를 찾습니다."""
    meshes =[]
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.find_armature() == armature:
            meshes.append(obj)
    return meshes

def transfer_weights_to_parent(armature, bone_name, keep_bones):
    """삭제될 본에 할당된 가중치를 유지할 가장 가까운 부모 본으로 전달합니다."""
    # 스키닝된 메시 찾기
    mesh_objs = find_deformed_mesh(armature)
    if mesh_objs == []:
        print(f"Warning: No deformed mesh found for armature")
        return
    
    bone = armature.data.bones.get(bone_name)
    if not bone:
        return
        
    # 유지할 가장 가까운 부모 본 찾기
    keeper_parent = find_closest_keeper_parent(bone, keep_bones)
    if not keeper_parent:
        print(f"Warning: No keeper parent found for bone {bone_name}")
        return
        
    parent_name = keeper_parent.name
    
    for mesh_obj in mesh_objs:
        vertex_indices_to_update = []
        weights_to_update = []

        for vertex in mesh_obj.data.vertices:
            total_weight = 0
            for group in vertex.groups: 
                if mesh_obj.vertex_groups[group.group].name == bone_name:
                    total_weight += group.weight
                    group.weight = 0  # 현재 본의 가중치를 0으로 설정
                    
            if total_weight > 0:
                vertex_indices_to_update.append(vertex.index)
                weights_to_update.append(total_weight)
        
        if vertex_indices_to_update:
            # 부모 본의 버텍스 그룹을 찾거나 생성
            parent_group = mesh_obj.vertex_groups.get(parent_name)
            if not parent_group:
                parent_group = mesh_obj.vertex_groups.new(name=parent_name)
            
            # 가중치를 한 번에 전달
            for idx, weight in zip(vertex_indices_to_update, weights_to_update):
                parent_group.add([idx], weight, 'ADD')

def normalize_armature_joints(armature_name):
    """
    아마추어의 모든 본들의 회전값을 정규화합니다.
    본의 형태는 유지한 채로 회전값만 초기화합니다.
    
    Args:
        armature_name (str): 정규화할 아마추어 오브젝트의 이름
    """
    # 아마추어 오브젝트 가져오기
    if armature_name not in bpy.data.objects:
        raise ValueError(f"Armature '{armature_name}' not found")
    
    armature = bpy.data.objects[armature_name]
    if armature.type != 'ARMATURE':
        raise ValueError(f"Object '{armature_name}' is not an armature")
    
    # 현재 모드 저장
    current_mode = bpy.context.object.mode
    
    # 포즈 모드로 전환하여 회전값 초기화
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    # 각 포즈 본의 회전값 초기화
    for pose_bone in armature.pose.bones:
        # 현재 본의 월드 매트릭스 저장
        world_matrix = pose_bone.matrix.copy()
        
        # 회전값 초기화
        pose_bone.rotation_euler = mathutils.Euler((0, 0, 0))
        pose_bone.rotation_quaternion = mathutils.Quaternion((1, 0, 0, 0))
        
        # 본의 제약 조건들도 초기화
        for constraint in pose_bone.constraints:
            if hasattr(constraint, 'influence'):
                constraint.influence = 0
    
    # 이전 모드로 복귀
    bpy.ops.object.mode_set(mode=current_mode)

def apply_rest_pose(armature_name):
    """
    현재 포즈를 레스트 포즈로 적용합니다.
    
    Args:
        armature_name (str): 레스트 포즈를 적용할 아마추어 오브젝트의 이름
    """
    armature = bpy.data.objects[armature_name]
    bpy.context.view_layer.objects.active = armature
    
    # 현재 포즈를 레스트 포즈로 적용
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.armature_apply()
    
    # 오브젝트 모드로 복귀
    bpy.ops.object.mode_set(mode='OBJECT')

def apply_transforms_to_armature_meshes(armature_name):
    """
    지정된 아마추어에 연결된 모든 메시들의 트랜스폼을 적용합니다.
    
    Args:
        armature_name (str): 아마추어 오브젝트의 이름
    """
    # 아마추어 확인
    if armature_name not in bpy.data.objects:
        raise ValueError(f"Armature '{armature_name}' not found")
        
    armature = bpy.data.objects[armature_name]
    if armature.type != 'ARMATURE':
        raise ValueError(f"Object '{armature_name}' is not an armature")
    
    # 현재 선택된 객체들 저장
    original_selection = bpy.context.selected_objects
    original_active = bpy.context.active_object
    
    # 아마추어와 연결된 메시 찾기
    meshes_to_process = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            # 아마추어 모디파이어가 있는지 확인
            for modifier in obj.modifiers:
                if modifier.type == 'ARMATURE' and modifier.object == armature:
                    meshes_to_process.append(obj)
                    break
    
    if not meshes_to_process:
        print(f"No meshes found connected to armature '{armature_name}'")
        return
    
    # 모든 객체 선택 해제
    bpy.ops.object.select_all(action='DESELECT')
    
    # 각 메시에 대해 transform 적용
    for mesh in meshes_to_process:
        # 메시 선택 및 활성화
        mesh.select_set(True)
        bpy.context.view_layer.objects.active = mesh
        
        # 모든 transform 적용
        bpy.ops.object.transform_apply(
            location=True, 
            rotation=True, 
            scale=True
        )
        
        # 선택 해제
        mesh.select_set(False)
        print(f"Applied transforms to mesh: {mesh.name}")
    
    # 원래 선택 상태 복원
    for obj in original_selection:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = original_active
    
    print(f"Successfully applied transforms to all meshes connected to {armature_name}")


def cleanup_and_quadriflow_remesh(obj_name, target_faces=3000):
    """
    메시를 정리하고 QuadriFlow Remesh를 적용합니다.
    분리된 부분들을 제거하고 가장 큰 연결된 부분만 남깁니다.
    
    Args:
        obj_name (str): 리메시할 오브젝트의 이름
        target_faces (int): 목표 페이스 수 (기본값: 3000)
    """
    # 오브젝트 선택
    if obj_name not in bpy.data.objects:
        raise ValueError(f"Object '{obj_name}' not found")
        
    obj = bpy.data.objects[obj_name]
    if obj.type != 'MESH':
        raise ValueError(f"Object '{obj_name}' is not a mesh")
    
    # 활성 오브젝트로 설정
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # 편집 모드로 전환
    bpy.ops.object.mode_set(mode='EDIT')
    
    # bmesh 생성
    bm = bmesh.from_edit_mesh(obj.data)
    
    # 분리된 부분들 찾기
    separate_parts = []
    tagged_verts = set()
    
    def flood_select(start_vert):
        """시작 버텍스에서 연결된 모든 버텍스를 찾습니다"""
        connected_verts = set()
        to_process = {start_vert}
        
        while to_process:
            vert = to_process.pop()
            if vert not in connected_verts:
                connected_verts.add(vert)
                # 연결된 엣지를 통해 이웃한 버텍스들 찾기
                for edge in vert.link_edges:
                    other_vert = edge.other_vert(vert)
                    if other_vert not in connected_verts:
                        to_process.add(other_vert)
        
        return connected_verts
    
    # 모든 분리된 부분 찾기
    for vert in bm.verts:
        if vert not in tagged_verts:
            connected = flood_select(vert)
            separate_parts.append(connected)
            tagged_verts.update(connected)
    
    # 가장 큰 부분 찾기
    largest_part = max(separate_parts, key=len)
    
    # 가장 큰 부분이 아닌 버텍스들 선택
    for vert in bm.verts:
        vert.select = vert not in largest_part
    
    # 선택된 버텍스(작은 부분들) 삭제
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.delete(type='VERT')
    
    # 오브젝트 모드로 돌아가기
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # QuadriFlow Remesh 적용
    bpy.ops.object.quadriflow_remesh(
        target_faces=target_faces,
        use_mesh_symmetry=True,
    )
    
    print(f"Successfully cleaned up and remeshed {obj_name} with {target_faces} target faces")


def cleanup_armature(fbx_path, bones_list_path, output_path, import_scale=1.0, export_scale=1):
    """메인 처리 함수"""
    # 기존 씬 초기화
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # FBX 파일 임포트
    bpy.ops.import_scene.fbx(filepath=fbx_path, global_scale=import_scale)
    
    # 아마추어 찾기
    armature = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break
    
    if not armature:
        raise Exception("아마추어를 찾을 수 없습니다.")
    
    # 유지할 본 이름 목록 읽기
    keep_bones = set(read_bone_names(bones_list_path))
    
    # 편집 모드로 전환
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    
    # 삭제할 본 목록 생성 (계층 구조 순서대로 처리하기 위해 리프 노드부터)
    bones_to_remove = []
    def add_removable_bones(bone):
        # 자식 본들을 먼저 처리
        for child in bone.children:
            add_removable_bones(child)
        # 현재 본이 삭제 대상이면 추가
        if not should_keep_bone(bone.name, keep_bones):
            bones_to_remove.append(bone.name)
    
    # 루트 본부터 시작하여 삭제할 본 목록 생성
    for bone in [b for b in armature.data.edit_bones if not b.parent]:
        add_removable_bones(bone)
    
    # 포즈 모드로 전환하여 가중치 전달
    bpy.ops.object.mode_set(mode='POSE')
    for bone_name in bones_to_remove:
        transfer_weights_to_parent(armature, bone_name, keep_bones)
    
    # 다시 편집 모드로 전환하여 본 삭제
    bpy.ops.object.mode_set(mode='EDIT')
    for bone_name in bones_to_remove:
        bone = armature.data.edit_bones.get(bone_name)
        if bone:
            armature.data.edit_bones.remove(bone)
    
    # 오브젝트 모드로 복귀
    bpy.ops.object.mode_set(mode='OBJECT')

    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            normalize_armature_joints(obj.name)
            apply_rest_pose(obj.name)
            break

    # 모든 메쉬 선택 후 부모 해제
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        else:
            obj.select_set(False)
    
    
    
    # 메쉬 병합
    bpy.ops.object.join()

    

    
    # Remesh 수정자 적용
    active_obj = bpy.context.active_object
    remesh = active_obj.modifiers.new(name="Remesh", type='REMESH')
    remesh.mode = 'VOXEL'
    #cleanup_and_quadriflow_remesh(active_obj.name, target_faces=3000)
    # 캐릭터 크기 계산
    bbox_corners = [active_obj.matrix_world @ Vector(corner) for corner in active_obj.bound_box]
    bbox_size = (max(v.x for v in bbox_corners) - min(v.x for v in bbox_corners),
                max(v.y for v in bbox_corners) - min(v.y for v in bbox_corners),
                max(v.z for v in bbox_corners) - min(v.z for v in bbox_corners))
    character_height = bbox_size[2]  # Z축 높이

    # 캐릭터 높이 기준 voxel size 계산 (기본 2m 캐릭터 기준 2.5cm로 설정)
    BASE_HEIGHT = 2.0  # 기준 캐릭터 높이(m)
    BASE_VOXEL_SIZE = 2  # 기준 voxel size(m)
    remesh.voxel_size = (character_height / BASE_HEIGHT) * BASE_VOXEL_SIZE
    bpy.ops.object.modifier_apply(modifier=remesh.name)
    print(character_height)

    
    # 자동 스키닝
    active_obj.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    
    # 결과 FBX 저장
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=False,
        use_armature_deform_only=True,
        add_leaf_bones=False,
        bake_anim=True,
        global_scale=export_scale
    )
    
# 스크립트 실행 예시
if __name__ == "__main__":
    characters_path = Path("../dataset/characters")
    output_path = Path("../dataset/characters/clean")
    output_path.mkdir(parents=True, exist_ok=True)
    bones_list_path = "./bones_to_keep.txt"  # 유지할 본 목록이 있는 텍스트 파일 경로

    fbx_files = characters_path.glob("*.fbx")

    for fbx_file in fbx_files:
        bpy.ops.wm.read_factory_settings(use_empty=True)  # 기존 씬 초기화
        bpy.ops.import_scene.fbx(filepath=str(fbx_file))  # FBX 임포트
        
        # NewVertex 구조체로 변환 및 저장
        output_file = output_path / (fbx_file.stem + "_clean.fbx")
        cleanup_armature(str(fbx_file), bones_list_path, str(output_file))
        #print(f"Exported: {output_file}")
    