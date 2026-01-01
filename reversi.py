import pyxel

import player
from scene import Scene
import concurrent.futures
import stone
import turn


ROW_CELLS = 8


class Reversi:
    def __init__(self, black_player, white_player):
        self.row_cells = ROW_CELLS

        self.board = [[0] * (ROW_CELLS + 2) for i in range(ROW_CELLS + 2)]
        self.win_player = None
        self.now_turn = turn.Turn.BLACK  # 最初は黒のターン
        self.legal_move_count_list = None
        self.black_player = black_player
        self.white_player = white_player

        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        self.black_player_think_result = None
        self.white_player_think_result = None
        self.black_player_name = None
        self.white_player_name = None
        self.legal_move_count_list, self.black_stones, self.white_stones = (
            None,
            None,
            None,
        )

        self.init_game()

    def init_game(self):
        self.board = [[0] * (ROW_CELLS + 2) for i in range(ROW_CELLS + 2)]
        for i in range(1, ROW_CELLS + 1):
            for j in range(1, ROW_CELLS + 1):
                self.board[i][j] = stone.Stone.NONE
        self.board[4][4] = self.board[5][5] = stone.Stone.WHITE
        self.board[4][5] = self.board[5][4] = stone.Stone.BLACK
        self.win_player = None
        self.now_turn = turn.Turn.BLACK  # 最初は黒のターン
        self.legal_move_count_list = []
        self.black_player_name = self.black_player.__class__.__name__
        self.white_player_name = self.white_player.__class__.__name__
        print(self.black_player_name, self.white_player_name)
        # 黒のターンを開始
        _board = [board_1d[1:9] for board_1d in self.board[1:9]]
        self.black_player_think_result = self.executor.submit(
            self.black_player.thinking, _board, self.now_turn
        )
        self.white_player_think_result = None
        self.legal_move_count_list, self.black_stones, self.white_stones = (
            self.exist_legal_move_and_count_stones(self.now_turn)
        )

    def update(self):

        if self.win_player is None:
            result_x = result_y = None
            if self.now_turn == turn.Turn.BLACK:
                if self.black_player_name == "Human":
                    _board = [board_1d[1:9] for board_1d in self.board[1:9]]
                    result_y, result_x = self.black_player.thinking(
                        _board, self.now_turn
                    )
                elif (
                    self.black_player_think_result is not None
                    and self.black_player_think_result.done()
                ):
                    result_y, result_x = self.black_player_think_result.result()
                    self.black_player_think_result = None
            elif self.now_turn == turn.Turn.WHITE:
                if self.white_player_name == "Human":
                    _board = [board_1d[1:9] for board_1d in self.board[1:9]]
                    result_y, result_x = self.white_player.thinking(
                        _board, self.now_turn
                    )
                elif (
                    self.white_player_think_result is not None
                    and self.white_player_think_result.done()
                ):
                    result_y, result_x = self.white_player_think_result.result()
                    self.white_player_think_result = None

            if not (result_x == result_y is None):
                legal_move_list = [tuple(i[0:2]) for i in self.legal_move_count_list]
                if (
                    result_x is not None
                    and result_x < 9
                    and result_y < 9
                    and (result_y, result_x) in legal_move_list
                    and self.board[result_y][result_x] is not stone.Stone.UNDEFINED
                ):
                    if self.put_stone(result_y, result_x, self.now_turn):
                        # ターン変更
                        self.now_turn = (
                            turn.Turn.BLACK
                            if self.now_turn == turn.Turn.WHITE
                            else turn.Turn.WHITE
                        )
                        (
                            self.legal_move_count_list,
                            self.black_stones,
                            self.white_stones,
                        ) = self.exist_legal_move_and_count_stones(self.now_turn)
                        legal_move_list = [
                            tuple(i[0:2]) for i in self.legal_move_count_list
                        ]

                        # 打つ手がない場合
                        if len(legal_move_list) == 0:
                            # print("cannot put, pass", self.now_turn, "turn")
                            # ターンを渡す
                            self.now_turn = (
                                turn.Turn.BLACK
                                if self.now_turn == turn.Turn.WHITE
                                else turn.Turn.WHITE
                            )
                            (
                                self.legal_move_count_list,
                                self.black_stones,
                                self.white_stones,
                            ) = self.exist_legal_move_and_count_stones(self.now_turn)
                            legal_move_list = [
                                tuple(i[0:2]) for i in self.legal_move_count_list
                            ]
                            # 渡されても打つ手がない場合(両者打つ手がない)
                            # :盤面がすべて埋まる
                            # :どちらかが全滅する
                            # :両者置く場所がない
                            if len(legal_move_list) == 0:
                                # ゲーム終了
                                if self.white_stones == self.black_stones:
                                    self.win_player = "DRAW"
                                elif self.white_stones > self.black_stones:
                                    self.win_player = "WHITE"
                                elif self.white_stones < self.black_stones:
                                    self.win_player = "BLACK"

                                # print("cannot continue game\n{} win".format(self.win_player))
                                return Scene.NO_SCENE_CHANGE
                        _board = [board_1d[1:9] for board_1d in self.board[1:9]]
                        if (
                            self.now_turn == turn.Turn.BLACK
                            and not self.black_player_name == "Human"
                        ):
                            self.executor = concurrent.futures.ProcessPoolExecutor(
                                max_workers=1
                            )
                            self.black_player_think_result = self.executor.submit(
                                self.black_player.thinking, _board, self.now_turn
                            )
                        elif (
                            self.now_turn == turn.Turn.WHITE
                            and not self.white_player_name == "Human"
                        ):
                            self.executor = concurrent.futures.ProcessPoolExecutor(
                                max_workers=1
                            )
                            self.white_player_think_result = self.executor.submit(
                                self.white_player.thinking, _board, self.now_turn
                            )
        else:
            if pyxel.btnp(pyxel.KEY_R, 60, 60):
                self.init_game()
            elif pyxel.btnp(pyxel.KEY_Q, 60, 60):
                return Scene.MENU

        return Scene.NO_SCENE_CHANGE

    def draw(self):
        # 状態の表示
        if self.win_player is not None:
            if self.win_player == "DRAW":
                pyxel.text(19, 72, "DRAW", 14)
            elif self.win_player == "WHITE":
                pyxel.text(18, 72, "WHITE", 7)
                pyxel.text(44, 72, "WIN", 9)
            elif self.win_player == "BLACK":
                pyxel.text(18, 72, "BLACK", 5)
                pyxel.text(44, 72, "WIN", 9)
        elif self.now_turn == turn.Turn.BLACK:
            pyxel.text(16, 72, "BLACK TURN", 5)
        elif self.now_turn == turn.Turn.WHITE:
            pyxel.text(16, 72, "WHITE TURN", 7)

        # 石の数の表示
        pyxel.text(6, 72, str(self.black_stones), 5)
        pyxel.text(60, 72, str(self.white_stones), 7)

        # 石の表示
        for y in range(1, 9):
            for x in range(1, 9):
                if self.board[y][x] == stone.Stone.BLACK:
                    pyxel.circ(x * 8, y * 8, 3, 5)
                elif self.board[y][x] == stone.Stone.WHITE:
                    pyxel.circ(x * 8, y * 8, 3, 7)

        # 線の表示
        for x in range(0, 8 * 9, 8):
            pyxel.line(x + 4, 0 + 4, x + 4, 8 * 8 + 4, 0)
        for y in range(0, 8 * 9, 8):
            pyxel.line(0 + 4, y + 4, 8 * 8 + 4, y + 4, 0)

        # 置ける場所の表示
        legal_move_list = [tuple(i[0:2]) for i in self.legal_move_count_list]
        for y, x in legal_move_list:
            pyxel.circ(x * 8, y * 8, 2, 9)

        pyxel.circ(pyxel.mouse_x, pyxel.mouse_y, 2, 2)

    def print_board(self, board):
        for y in range(len(board)):
            for x in range(len(board)):
                print(int(board[y][x]), end="")
            print()
        print()

    def put_stone(self, y, x, player):
        """
        石を置く
        """
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                if i == j == 0:
                    continue
                count = self.count_turn_over(player, y, x, i, j)
                for k in range(1, count + 1):
                    self.board[y + k * i][x + k * j] = player

        self.board[y][x] = player

        return True

    def count_turn_over(self, player, y, x, i, j):
        """
        (y,x)の(i,j)方向の石をいくつひっくり返せるかを返す
        """
        # 相手の石の色
        opponent = turn.Turn.BLACK if player == turn.Turn.WHITE else turn.Turn.WHITE
        k = 1
        while self.board[y + k * i][x + k * j] == opponent:
            k += 1

        if self.board[y + k * i][x + k * j] == player:
            return k - 1
        else:
            return 0

    def is_legal_move(self, player, y, x):
        """
        (y,x)にplayerの石が置けるか調べておける場合にはひっくり返せる個数を返す
        """

        if self.board[y][x] != stone.Stone.NONE:
            return False

        if x < 1 or x > 8 or y < 1 or y > 8:
            return False

        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                turn_over_count = self.count_turn_over(player, y, x, i, j)
                if not (i == 0 and j == 0) and turn_over_count:
                    return turn_over_count

        return False

    def exist_legal_move_and_count_stones(self, player):
        """
        playerが置ける石の場所と取れる個数と白黒それぞれの石の個数を求めて返す
        """
        legal_move_list = []
        white_stones = 0
        black_stones = 0
        for i in range(1, 9):
            for j in range(1, 9):
                if self.board[i][j] == stone.Stone.WHITE:
                    white_stones += 1
                elif self.board[i][j] == stone.Stone.BLACK:
                    black_stones += 1

                turn_over_count = self.is_legal_move(player, i, j)
                if turn_over_count:
                    legal_move_list.append((i, j, turn_over_count))

        return legal_move_list, black_stones, white_stones
