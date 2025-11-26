from collections import namedtuple
from enum import Enum
from pathlib import Path
import moderngl_window

from phong_window import PhongWindow

Task = namedtuple('Task', ['window_args', 'window_cls'])


class TaskType(Enum):
    @property
    def window_args(self):
        return self.value.window_args

    @property
    def window_cls(self):
        return self.value.window_cls

    PHONG = Task(
        [
            f"--shaders_dir_path={str(Path(__file__).parent.parent / 'resources/shaders/phong')}",
            "--shader_name=phong",
            "--model_name=sphere.obj",
            f"--output_path={str(Path(__file__).parent.parent / 'output')}"
        ],
        PhongWindow
    )


if __name__ == '__main__':
    task = TaskType.PHONG
    moderngl_window.run_window_config(task.window_cls, args=task.window_args)
