import bpy
import bmesh
import copy
from mathutils import Vector, Matrix
import os

class ContactVisualizer:
    def __init__(self):
        self.contact_material = self.create_contact_material()
        self.setup_scene()
        
    def create_contact_material(self):
        """접촉점 표시를 위한 material 생성"""
        mat = bpy.data.materials.new(name="ContactMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        
        # 기존 노드 제거
        nodes.clear()
        
        # Emission 노드 추가
        emit = nodes.new(type='ShaderNodeEmission')
        emit.inputs[0].default_value = (1, 0, 0, 1)  # 빨간색
        emit.inputs[1].default_value = 5.0  # 강도
        
        # Material Output 노드
        output = nodes.new(type='ShaderNodeOutputMaterial')
        
        # 노드 연결
        mat.node_tree.links.new(emit.outputs[0], output.inputs[0])
        
        return mat
        
    def setup_scene(self):
        """기본 씬 설정 (조명, 카메라 등)"""
        # 기본 큐브, 카메라, 광원 삭제
        #bpy.ops.object.select_all(action='SELECT')
        #bpy.ops.object.delete()

        # 기본 카메라 추가
        bpy.ops.object.camera_add(location=(0, -5, 2.8), rotation=(90, 0, 0))
        camera = bpy.context.active_object
        
        # 여러 방향의 조명 추가
        # 정면 조명
        front_light = bpy.data.lights.new(name="front_light", type='POINT')
        front_light_obj = bpy.data.objects.new(name="Front Light", object_data=front_light)
        front_light_obj.location = (0, -1, 3)
        front_light_obj.rotation_euler = (1, 0, 0)
        bpy.context.scene.collection.objects.link(front_light_obj)
        
        # 측면 조명
        side_light = bpy.data.lights.new(name="side_light", type='POINT')
        side_light_obj = bpy.data.objects.new(name="Side Light", object_data=side_light)
        side_light_obj.location = (1, 0, 3)
        side_light_obj.rotation_euler = (1, 0, -1.5)
        bpy.context.scene.collection.objects.link(side_light_obj)
        
        # 탑 조명
        top_light = bpy.data.lights.new(name="top_light", type='POINT')
        top_light_obj = bpy.data.objects.new(name="Top Light", object_data=top_light)
        top_light_obj.location = (0, 0, 1)
        top_light_obj.rotation_euler = (0, 0, 0)
        bpy.context.scene.collection.objects.link(top_light_obj)
        
        # 조명 세기 조정
        front_light.energy = 10
        side_light.energy = 10
        top_light.energy = 10
        
        # 렌더링 설정
        bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'  # 더 빠른 렌더링을 위해 Eevee 사용

        # 카메라를 활성 카메라로 설정
        bpy.context.scene.camera = camera
        
        #return camera
        
    def visualize_contacts_at_frame(self, frame, vertices, contacts, output_dir):
        """특정 프레임에서의 접촉 시각화"""
        # 현재 프레임으로 이동
        bpy.context.scene.frame_set(frame)
        
        
        if not contacts['contactPolygonPairs']:
            return  # 접촉이 없는 경우 스킵
        
        # 접촉점 표시용 오브젝트 생성
        self.create_contact_markers(contacts, vertices)
        
        # 여러 각도에서 렌더링
        angles = [(0, 0, 0), (0, 0, 90), (90, 0, 0)]
        for i, angle in enumerate(angles):
            self.render_view(frame, angle, output_dir, i)
        
        # 마커 제거
        self.cleanup_markers()
        
    def create_contact_markers(self, contacts, vertices):
        """접촉점을 표시하는 구체 생성"""
        print("contactmarker")
        for pair in contacts['contactPolygonPairs']:
            # 첫 번째 점
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.02)
            sphere1 = bpy.context.active_object
            sphere1.name = f"contact_marker_{len(bpy.data.objects)}"
            sphere1.location = (vertices[int(pair[0])].position[0], vertices[int(pair[0])].position[1], vertices[int(pair[0])].position[2])
            sphere1.data.materials.append(self.contact_material)
            
            # 두 번째 점
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.02)
            sphere2 = bpy.context.active_object
            sphere2.name = f"contact_marker_{len(bpy.data.objects)}"
            sphere2.location = (vertices[int(pair[1])].position[0], vertices[int(pair[1])].position[1], vertices[int(pair[1])].position[2])
            sphere2.data.materials.append(self.contact_material)
            
            # 점들을 연결하는 선
            bpy.ops.mesh.primitive_cylinder_add(radius=0.005)
            cylinder = bpy.context.active_object
            cylinder.name = f"contact_line_{len(bpy.data.objects)}"
            
            # 실린더를 두 점 사이에 위치시키기
            dir = sphere2.location - sphere1.location
            loc = sphere1.location + dir/2
            cylinder.location = loc
            
            # 실린더의 방향과 크기 조정
            cylinder.scale.z = dir.length
            cylinder.rotation_euler = dir.to_track_quat('-Z', 'Y').to_euler()
            cylinder.data.materials.append(self.contact_material)
    
    def render_view(self, frame, angle, output_dir, angle_index):
        """특정 각도에서 뷰 렌더링"""
        cam = bpy.context.scene.camera
        #cam.rotation_euler = [a * 3.14159 / 180 for a in angle]
        
        # 렌더링 설정
        scene = bpy.context.scene
        #scene.camera = bpy.context.scene.camera
        bpy.context.scene.render.filepath = os.path.join(
            output_dir, 
            f'contact_frame_{frame:04d}_angle_{angle_index}.png'
        )
        
        # 렌더링 실행
        bpy.ops.render.render(write_still=True)
    
    def cleanup_markers(self):
        """생성된 마커 오브젝트들 제거"""
        for obj in bpy.data.objects:
            if obj.name.startswith(("contact_marker_", "contact_line_")):
                bpy.data.objects.remove(obj, do_unlink=True)
    
    def visualize_all_contact_frames(self, frame, vertices, contacts,  output_dir):
        """접촉이 발생한 모든 프레임 시각화"""
        
        os.makedirs(output_dir, exist_ok=True)
        print("MKDIR")
        if contacts['contactPolygonPairs']:
            print("visualize_contacts_at_frame")
            self.visualize_contacts_at_frame(frame, vertices, contacts, output_dir)

# 사용 예시
def validate_contact(frame, vertices, contacts, output_dir):

    visualizer = ContactVisualizer()
    data = copy.deepcopy(contacts)
    visualizer.visualize_all_contact_frames(frame, vertices, data, output_dir)