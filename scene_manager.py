import pyxel
from scene import Scene
import menu
import reversi_manager
import game_settings

class SceneManager():

    def __init__(self):
        self.scene = menu.Menu()

    def update(self):
        scene_transition = self.scene.update()
        if scene_transition[0] == Scene.MENU:
            self.scene = menu.Menu()
        elif scene_transition[0] == Scene.QUIT:
            pyxel.quit()
        elif scene_transition[0] == Scene.GAME:
            self.scene = reversi_manager.ReversiManager(scene_transition[1][0], scene_transition[1][1])
        elif scene_transition[0] == Scene.GAME_SETTINGS:
            self.scene = game_settings.GameSettings(cursor_position=0)

    def draw(self):
        self.scene.draw()
