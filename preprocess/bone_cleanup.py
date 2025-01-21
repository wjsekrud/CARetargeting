import bpy
import os
from pathlib import Path



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
                    print(bone_name)
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

def cleanup_armature(fbx_path, bones_list_path, output_path):
    """메인 처리 함수"""
    # 기존 씬 초기화
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # FBX 파일 임포트
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    
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
    
    # 결과 FBX 저장
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=False,
        use_armature_deform_only=True,
        add_leaf_bones=False,
        bake_anim=True
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
    