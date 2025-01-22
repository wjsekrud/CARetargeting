
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