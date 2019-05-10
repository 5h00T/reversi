import enum


class Stone(enum.IntEnum):
    UNDEFINED = 0  # 盤面外
    BLACK = 1
    WHITE = 2
    NONE = 3  # 何も置かれてないマス
