def python_exec():

    import os

    import bpy

    try:

        # 2.92 and older

        path = bpy.app.binary_path_python

    except AttributeError:

        # 2.93 and later

        import sys

        path = sys.executable

    return os.path.abspath(path)

print(python_exec())