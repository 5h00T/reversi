import time

import pyxel
import scene_manager


class App:
    def __init__(self):
        self.scene_manager = scene_manager.SceneManager()
        pyxel.init(72, 82, title="reversi", fps=30)
        pyxel.mouse(True)
        pyxel.run(self.update, self.draw)

    def update(self):
        self.scene_manager.update()

    def draw(self):
        pyxel.cls(6)
        self.scene_manager.draw()


if __name__ == "__main__":
    App()
