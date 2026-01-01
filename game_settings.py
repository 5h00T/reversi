import pyxel
from scene import Scene
import player
import cplayer

# AI_TYPE = player.MCTSPlayer(2000, 5)
# AI_TYPE = cplayer.MinMaxAI(5)
AI_TYPE = player.MinMaxAI(5)


class GameSettings:

    def __init__(self, cursor_position):
        self.cursor = cursor_position
        self.match_player_text = [
            "human vs human",
            "human vs COM",
            "COM vs human",
            "COM vs COM",
        ]
        self.match_player = [
            (player.Human(), player.Human()),
            (player.Human(), AI_TYPE),
            (AI_TYPE, player.Human()),
            (AI_TYPE, AI_TYPE),
        ]
        self.is_active = True

    def update(self):
        # カーソルが要素外を示さないように制限
        if pyxel.btnp(pyxel.KEY_UP, 30, 20):
            self.cursor = max(0, self.cursor - 1)
        elif pyxel.btnp(pyxel.KEY_DOWN, 30, 20):
            self.cursor = min(len(self.match_player) - 1, self.cursor + 1)

        if pyxel.btnp(pyxel.KEY_Z, 10, 10):
            for idx in range(len(self.match_player)):
                if self.cursor == idx:
                    return Scene.GAME, self.match_player[idx]

        if pyxel.btn(pyxel.KEY_X):
            return Scene.MENU, 0

        return Scene.NO_SCENE_CHANGE, 0

    def draw(self):
        pyxel.text(10, 120, self.match_player_text[0], 8)
        pyxel.text(10, 140, self.match_player_text[1], 5)

        for i in range(0, len(self.match_player)):
            if i == self.cursor:
                pyxel.text(5, 10 + i * 10, self.match_player_text[i], 8)
            else:
                pyxel.text(5, 10 + i * 10, self.match_player_text[i], 5)
