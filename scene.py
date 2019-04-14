import enum


class Scene(enum.IntEnum):
    NO_SCENE_CHANGE = 0
    MENU = 1
    GAME = 2
    QUIT = 3
    GAME_SETTINGS = 4
