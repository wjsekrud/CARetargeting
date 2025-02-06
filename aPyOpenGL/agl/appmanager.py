import glfw
from OpenGL.GL import *

from .app    import App
from .render import Render, RenderMode
from .const  import SHADOW_MAP_SIZE

class AppManager:
    _app = None

    @staticmethod
    def start(app: App):
        AppManager._app = app
        AppManager._render_loop()

    @staticmethod
    def _render_loop():
        if AppManager._app is None:
            raise Exception("AppManager.app is empty")
        
        app = AppManager._app

        # start
        app.start()
        app.initialize_ui()

        # main loop
        glfw.set_time(0)
        while not glfw.window_should_close(app.window):
            # update window size
            width, height = glfw.get_window_size(app.window)
            glViewport(0, 0, width, height)

            # process inputs for ui
            app.process_inputs()
            
            # sky color
            sky_color = Render.sky_color()
            glClearColor(sky_color.x, sky_color.y, sky_color.z, sky_color.a)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # update
            app.update()

            # update camera & light
            Render.update_render_view(app, width, height)

            # render shadow
            Render.set_render_mode(RenderMode.eSHADOW)
            glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)
            glClear(GL_DEPTH_BUFFER_BIT)
            app.render()

            # render scene
            Render.set_render_mode(RenderMode.eDRAW)
            glViewport(0, 0, width, height)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            app.render()

            # render text
            Render.set_render_mode(RenderMode.eTEXT)
            glViewport(0, 0, width, height)
            app.render_text()
            
            # render xray
            Render.set_render_mode(RenderMode.eDRAW)
            glClear(GL_DEPTH_BUFFER_BIT)
            app.render_xray()

            # render ui
            app._render_ui()

            # late update
            app.late_update()

            # event
            glfw.poll_events()
            glfw.swap_buffers(app.window)

        app.terminate_ui()
        glfw.destroy_window(app.window)
        glfw.terminate()
        app.terminate()
        Render.vao_cache = {}