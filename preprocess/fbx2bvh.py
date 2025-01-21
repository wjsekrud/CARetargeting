import bpy
import os
from pathlib import Path

class AnimationProcessor:
    def __init__(self, char_name):
        self.char_name = char_name
        self.base_path = Path(bpy.path.abspath("//"))  # Blender 프로젝트 경로
        self.anim_source_path = Path(f"../dataset/Animations/fbx/{char_name}")
        self.char_clean_path = Path(f"../dataset/characters/clean/{char_name}_clean.fbx")
        self.output_path = Path(f"../dataset/Animations/bvhs/{char_name}")
        
    def process_animations(self):
        """캐릭터의 모든 애니메이션 처리"""
        # 출력 폴더 생성
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 애니메이션 폴더에서 모든 FBX 파일 찾기
        fbx_files = list(self.anim_source_path.glob("*.fbx"))
        
        if not fbx_files:
            print(f"No animation files found for character: {self.char_name}")
            return
            
        for fbx_file in fbx_files:
            self.process_single_animation(fbx_file)
            
    def process_single_animation(self, anim_file):
        """단일 애니메이션 클립 처리"""
        print(f"Processing animation: {anim_file.name}")
        
        # 씬 초기화
        bpy.ops.wm.read_factory_settings(use_empty=True)
        
        # 캐릭터 기본 스켈레톤 로드 ([CharName].fbx)
        bpy.ops.import_scene.fbx(filepath=str(self.char_clean_path))
        target_armature = self._find_armature()
        if not target_armature:
            print("No armature found in character base file")
            return
            
        # 애니메이션 FBX 로드
        bpy.ops.import_scene.fbx(filepath=str(anim_file))
        anim_armature = self._find_armature(exclude=target_armature)
        if not anim_armature:
            print("No armature found in animation file")
            return
            
        # 본 로테이션 복사
        self._copy_bone_animation(source=anim_armature, target=target_armature)
        
        # 애니메이션 armature 삭제 (메모리 정리)
        bpy.data.objects.remove(anim_armature, do_unlink=True)
        
        # BVH로 내보내기
        output_file = self.output_path / f"{anim_file.stem}.bvh"
        
        # 타겟 아마추어 선택
        bpy.context.view_layer.objects.active = target_armature
        target_armature.select_set(True)
        
        bpy.ops.export_anim.bvh(
            filepath=str(output_file),
            global_scale=0.01,
            frame_start=bpy.context.scene.frame_start,
            frame_end=bpy.context.scene.frame_end,
            rotate_mode='NATIVE',
            root_transform_only=False
        )
        
        print(f"Saved BVH: {output_file.name}")
        
    def _find_armature(self, exclude=None):
        """씬에서 아마추어 오브젝트 찾기"""
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE' and obj != exclude:
                return obj
        return None
        
    def _copy_bone_animation(self, source, target):
        """소스 아마추어에서 타겟 아마추어로 본 애니메이션 복사"""
        # 프레임 범위 설정
        bpy.context.scene.frame_start = int(source.animation_data.action.frame_range[0])
        bpy.context.scene.frame_end = int(source.animation_data.action.frame_range[1])
        
        # 타겟 아마추어에 새로운 액션 생성
        if target.animation_data is None:
            target.animation_data_create()
        
        action_name = f"BVH_{source.animation_data.action.name}"
        target_action = bpy.data.actions.new(name=action_name)
        target.animation_data.action = target_action
        
        # 타겟 아마추어의 본 구조를 기준으로 애니메이션 복사
        for target_bone in target.pose.bones:
            if target_bone.name in source.pose.bones:
                source_bone = source.pose.bones[target_bone.name]
                
                # 모든 키프레임에 대해
                for fcurve in source.animation_data.action.fcurves:
                    if fcurve.data_path.startswith(f'pose.bones["{source_bone.name}"]'):
                        data_path = fcurve.data_path
                        array_index = fcurve.array_index
                        
                        # 기존 fcurve가 있다면 제거
                        existing_fcurves = [fc for fc in target.animation_data.action.fcurves 
                                          if fc.data_path == data_path and fc.array_index == array_index]
                        for fc in existing_fcurves:
                            target.animation_data.action.fcurves.remove(fc)
                        
                        # 새 fcurve 생성 및 키프레임 복사
                        new_fcurve = target.animation_data.action.fcurves.new(
                            data_path=data_path,
                            index=array_index,
                            action_group=target_bone.name
                        )
                        
                        # 키프레임 값 복사
                        for keyframe in fcurve.keyframe_points:
                            new_fcurve.keyframe_points.insert(
                                keyframe.co[0],
                                keyframe.co[1],
                                options={'FAST'}
                            )

def main():
    # 처리할 캐릭터 이름 목록
    characters = ["Y_bot"]  # 실제 캐릭터 이름으로 수정
    
    for char_name in characters:
        print(f"\nProcessing character: {char_name}")
        processor = AnimationProcessor(char_name)
        processor.process_animations()

if __name__ == "__main__":
    main()