import reversi

class ReversiManager():
    """
    ゲームを管理する
    """

    def __init__(self, black_player, white_player):
        self.reversi = reversi.Reversi(black_player, white_player)

    def update(self):
        result = self.reversi.update()

        return result, 0

    def draw(self):
        self.reversi.draw()
