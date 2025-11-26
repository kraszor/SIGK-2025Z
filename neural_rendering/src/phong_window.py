import os.path
import json
import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44, Vector4

from base_window import BaseWindow


class PhongWindow(BaseWindow):

    def __init__(self, **kwargs):
        super(PhongWindow, self).__init__(**kwargs)
        self.frame = 0
        self.params_list = []

    def init_shaders_variables(self):
        self.model_view_projection = self.program["model_view_projection"]
        self.model_matrix = self.program["model_matrix"]
        self.material_diffuse = self.program["material_diffuse"]
        self.material_shininess = self.program["material_shininess"]
        self.light_position = self.program["light_position"]
        self.camera_position = self.program["camera_position"]

    def is_visible(self, mvp):
        pos_clip = mvp * Vector4([0.0, 0.0, 0.0, 1.0])
        pos_clip = np.array(pos_clip)
        if pos_clip[3] > 0.0:
            ndc = pos_clip[:3] / pos_clip[3]
            return all(abs(coord) <= 1.0 for coord in ndc)
        return False

    def on_render(self, time: float, frame_time: float):
        if self.frame == self.max_frames:
            print(f"Generated {self.frame} frames")
            with open(os.path.join(self.output_path, "params.json"), "w") as f:
                json.dump(self.params_list, f, indent=4)
            self.wnd.close()
            return
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        model_translation = np.random.randint(-20, 20, 3)

        camera_position = [5.0, 5.0, 15.0]
        model_matrix = Matrix44.from_translation(model_translation)
        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            camera_position,
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )

        model_view_projection = proj * lookat * model_matrix

        if not self.is_visible(model_view_projection):
            return

        material_diffuse = np.random.randint(0, 255, 3) / 255.0
        material_shininess = np.random.randint(3, 20)
        light_position = np.random.randint(-20, 20, 3)

        model_relative_translation = np.array(model_translation) - np.array(camera_position)
        light_relative_position = np.array(light_position) - np.array(camera_position)

        self.model_view_projection.write(model_view_projection.astype('f4').tobytes())
        self.model_matrix.write(model_matrix.astype('f4').tobytes())
        self.material_diffuse.write(np.array(material_diffuse, dtype='f4').tobytes())
        self.material_shininess.write(np.array([material_shininess], dtype='f4').tobytes())
        self.light_position.write(np.array(light_position, dtype='f4').tobytes())
        self.camera_position.write(np.array(camera_position, dtype='f4').tobytes())

        self.vao.render()
        if self.output_path:
            img = (
                Image.frombuffer('RGBA', self.wnd.size, self.wnd.fbo.read(components=4))
                     .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            )
            image_path = f'images/image_{self.frame:04}.png'
            img.save(os.path.join(self.output_path, image_path))
            params = {
                "image_filename": image_path.split('/')[-1],
                "model_translation_relative": model_relative_translation.tolist(),
                "model_translation": model_translation.tolist(),
                "material_diffuse": material_diffuse.tolist(),
                "material_shininess": material_shininess,
                "light_position_relative": light_relative_position.tolist(),
                "light_position": light_position.tolist(),
                "camera_position": camera_position,
                "frame": self.frame
            }
            self.params_list.append(params)
            if self.frame == 0:
                print(params)
            self.frame += 1
