from abc import ABC, abstractmethod
import bpy
from pathlib import Path
from mathutils import Vector, Matrix

class BaseContactDetector(ABC):
    """Contact detector의 기본 인터페이스를 정의하는 추상 클래스"""
    def __init__(self, vertex_file_path, bvh_file_path):
        self.vertices, self.triangles = self.load_vertex_data(vertex_file_path)
        self.load_animation(bvh_file_path)
        self.vertex_groups = self.create_vertex_groups()
        
    @abstractmethod
    def detect_contacts_for_frame(self, frame):
        """특정 프레임에서의 contact detection을 수행"""
        pass
        
    def load_animation(self, bvh_file_path):
        """BVH 파일 로드 및 애니메이션 데이터 추출"""
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
        for frame in range(self.frame_start, self.frame_end + 1):
            contact_data.append(self.detect_contacts_for_frame(frame))
                
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

    