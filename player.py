import copy
import math
import pyxel
import random
import time
import stone
import turn


ROW_CELLS = 8


class Player():

    def __init__(self):
        self.player = 0

    def thinking(self, board, turn):
        self.player = turn

    def print_board(self, board):
        for y in range(len(board)):
            for x in range(len(board)):
                print(int(board[y][x]), end="")
            print()
        print()

    def count_turn_over(self, board, player, y, x, i, j):
        """
        (y,x)の(i,j)方向の石をいくつひっくり返せるかを返す
        """
        # 相手の石の色
        opponent = turn.Turn.BLACK if player == turn.Turn.WHITE else turn.Turn.WHITE
        k = 1
        while board[y + k * i][x + k * j] == opponent:
            k += 1

        if board[y + k * i][x + k * j] == player:
            return k - 1
        else:
            return 0

    def is_legal_move(self, board, player, y, x):
        """
        (y,x)にplayerの石が置けるか調べておける場合にはひっくり返せる個数を返す
        """

        if board[y][x] != stone.Stone.NONE:
            return False

        if x < 1 or x > 8 or y < 1 or y > 8:
            return False

        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                turn_over_count = self.count_turn_over(board, player, y, x, i, j)
                if not (i == 0 and j == 0) and turn_over_count:
                    return turn_over_count

        return False

    def exist_legal_move_and_count_stones(self, board, player):
        """
        playerが置ける石の場所と取れる個数と白黒それぞれの石の個数を求めて返す
        """
        legal_move_list = []
        append = legal_move_list.append
        white_stones = 0
        black_stones = 0
        for i in range(1, 9):
            for j in range(1, 9):
                if board[i][j] == stone.Stone.WHITE:
                    white_stones += 1
                elif board[i][j] == stone.Stone.BLACK:
                    black_stones += 1

                turn_over_count = self.is_legal_move(board, player, i, j)
                if turn_over_count:
                    append((i, j, turn_over_count))

        return legal_move_list, black_stones, white_stones

    def put_stone(self, board, y, x, player):
        """
        石を置く
        """
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                if i == j == 0:
                    continue
                count = self.count_turn_over(board, player, y, x, i, j)
                for k in range(1, count + 1):
                    board[y + k * i][x + k * j] = player

        board[y][x] = player

class Human(Player):

    def __init__(self):
        super().__init__()

    def thinking(self, board, turn):
        super().thinking(board, turn)
        _board = [[0] * (ROW_CELLS + 2) for _ in range(ROW_CELLS + 2)]
        _board[1:9][1:9] = board
        for i in range(1, 9):
            for j in range(1, 9):
                _board[i][j] = board[i - 1][j - 1]
        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            _board, turn)
        # self.print_board(_board)
        legal_move_list = [tuple(i[0:2]) for i in legal_move_count_list]
        # print(legal_move_list)

        if pyxel.btnp(pyxel.MOUSE_LEFT_BUTTON, 60, 60):
            # クリックされたウィンドウの座標を盤面の座標に変換
            x = round(pyxel.mouse_x / 8)
            y = round(pyxel.mouse_y / 8)
            if (y, x) in legal_move_list:
                return y, x

        return None, None


class RandomChoiceAI(Player):

    def __init__(self):
        super().__init__()

        self.board_gain = ((30, -30, 10, 5, 5, 10, -30, 30),
                           (-30, -30, -5, -5, -5, -5, -30, -30),
                           (10, -5, 5, 1, 1, 5, -5, 10),
                           (5, -5, 1, 0, 0, 1, -5, 5),
                           (5, -5, 1, 0, 0, 1, -5, 5),
                           (10, -5, 5, 1, 1, 5, -5, 10),
                           (-30, -30, -5, -5, -5, -5, -30, -30),
                           (30, -30, 10, 5, 5, 10, -30, 30))

    def thinking(self, board, turn):
        _board = [[0] * (ROW_CELLS + 2) for i in range(ROW_CELLS + 2)]
        _board[1:9][1:9] = board
        for i in range(1, 9):
            for j in range(1, 9):
                _board[i][j] = board[i - 1][j - 1]

        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            _board, turn)
        legal_move_list = [tuple(i[0:2]) for i in legal_move_count_list]
        x, y = random.choice(legal_move_list)

        return x, y


class MinMaxAI(Player):

    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.board_gain = ((50, -20, 10, 5, 5, 10, -20, 50),
                           (-20, -30, -5, -5, -5, -5, -30, -20),
                           (10, -5, 5, 3, 3, 5, -5, 10),
                           (5, -5, 3, 3, 3, 3, -5, 5),
                           (5, -5, 3, 3, 3, 3, -5, 5),
                           (10, -5, 5, 3, 3, 5, -5, 10),
                           (-20, -30, -5, -5, -5, -5, -30, -20),
                           (50, -20, 10, 5, 5, 10, -20, 50))

    def thinking(self, board, turn):
        super().thinking(board, turn)
        _board = [[0] * (ROW_CELLS + 2) for i in range(ROW_CELLS + 2)]
        for i in range(1, 9):
            for j in range(1, 9):
                _board[i][j] = board[i - 1][j - 1]

        val, y, x = self.mini_max(_board, turn, self.depth)

        return y, x

    def mini_max(self, board, player, depth):

        # 葉の場合、評価値を返す
        if depth == 0:
            val = self.evaluation_function(board, self.player)
            return val

        # 相手の色
        opponent = turn.Turn.BLACK if player == turn.Turn.WHITE else turn.Turn.WHITE

        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            board, player)

        random.shuffle(legal_move_count_list)

        if len(legal_move_count_list) == 0:
            val = self.evaluation_function(board, self.player)
            return val

        if player == self.player:
            best = -math.inf
            best_y = best_x = None
            for y, x, count in legal_move_count_list:
                _board = copy.deepcopy(board)
                self.put_stone(_board, y, x, player)
                min_max_result = self.mini_max(_board, opponent, depth - 1)
                if isinstance(min_max_result, int):
                    val = min_max_result
                else:
                    val = min_max_result[0]
                if val > best:
                    best = val
                    best_y = y
                    best_x = x
            return best, best_y, best_x
        elif player != self.player:
            best = math.inf
            best_y = best_x = None
            for y, x, count in legal_move_count_list:
                _board = copy.deepcopy(board)
                self.put_stone(_board, y, x, player)
                min_max_result = self.mini_max(_board, opponent, depth - 1)
                if isinstance(min_max_result, int):
                    val = min_max_result
                else:
                    val = min_max_result[0]
                if val < best:
                    best = val
                    best_y = y
                    best_x = x
            return best, best_y, best_x

    def evaluation_function(self, board, player):
        evaluation = 0
        for y in range(1, 9):
            for x in range(1, 9):
                if board[y][x] == player:
                    evaluation += self.board_gain[y - 1][x - 1]

        return evaluation


class AlphaBetaAI(Player):

    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.board_gain = ((50, -20, 10, 5, 5, 10, -20, 50),
                           (-20, -30, -5, -5, -5, -5, -30, -20),
                           (10, -5, 5, 3, 3, 5, -5, 10),
                           (5, -5, 3, 3, 3, 3, -5, 5),
                           (5, -5, 3, 3, 3, 3, -5, 5),
                           (10, -5, 5, 3, 3, 5, -5, 10),
                           (-20, -30, -5, -5, -5, -5, -30, -20),
                           (50, -20, 10, 5, 5, 10, -20, 50))

    def thinking(self, board, turn):
        super().thinking(board, turn)
        _board = [[0] * (ROW_CELLS + 2) for i in range(ROW_CELLS + 2)]
        for i in range(1, 9):
            for j in range(1, 9):
                _board[i][j] = board[i - 1][j - 1]

        val, y, x = self.alpha_beta(_board, turn, self.depth)
        return y, x

    def alpha_beta(self, board, player, depth, alpha=-math.inf, beta=math.inf):

        # 葉の場合、評価値を返す
        if depth == 0:
            val = self.evaluation_function(board, self.player)
            return val

        # 相手の色
        opponent = turn.Turn.BLACK if player == turn.Turn.WHITE else turn.Turn.WHITE

        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            board, player)

        random.shuffle(legal_move_count_list)

        if len(legal_move_count_list) == 0:
            val = self.evaluation_function(board, self.player)
            return val

        if player == self.player:
            best = -math.inf
            best_y = best_x = None
            for y, x, count in legal_move_count_list:
                _board = copy.deepcopy(board)
                self.put_stone(_board, y, x, player)
                min_max_result = self.alpha_beta(_board, opponent, depth - 1, alpha, beta)
                if isinstance(min_max_result, int):
                    val = min_max_result
                else:
                    val = min_max_result[0]
                if val > best:
                    alpha = val
                    best = val
                    best_y = y
                    best_x = x

                if best > beta:  # betaカット
                    return best, best_y, best_x

            return best, best_y, best_x
        elif player != self.player:
            best = math.inf
            best_y = best_x = None
            for y, x, count in legal_move_count_list:
                _board = copy.deepcopy(board)
                self.put_stone(_board, y, x, player)
                min_max_result = self.alpha_beta(_board, opponent, depth - 1, alpha, beta)
                if isinstance(min_max_result, int):
                    val = min_max_result
                else:
                    val = min_max_result[0]
                if val < best:
                    beta = val
                    best = val
                    best_y = y
                    best_x = x

                if best < alpha:
                    return best, best_y, best_x

            return best, best_y, best_x

    def evaluation_function(self, board, player):
        evaluation = 0
        for y in range(1, 9):
            for x in range(1, 9):
                if board[y][x] == player:
                    evaluation += self.board_gain[y - 1][x - 1]

        return evaluation


class AlphaBetaSecondAI(Player):

    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.board_gain = ((50, -20, 10, 5, 5, 10, -20, 50),
                           (-20, -30, -5, -5, -5, -5, -30, -20),
                           (10, -5, 5, 3, 3, 5, -5, 10),
                           (5, -5, 3, 3, 3, 3, -5, 5),
                           (5, -5, 3, 3, 3, 3, -5, 5),
                           (10, -5, 5, 3, 3, 5, -5, 10),
                           (-20, -30, -5, -5, -5, -5, -30, -20),
                           (50, -20, 10, 5, 5, 10, -20, 50))

    def thinking(self, board, turn):
        super().thinking(board, turn)
        _board = [[0] * (ROW_CELLS + 2) for i in range(ROW_CELLS + 2)]
        for i in range(1, 9):
            for j in range(1, 9):
                _board[i][j] = board[i - 1][j - 1]

        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            _board, turn)
        if black_stones + white_stones > 52:
            val, y, x = self.alpha_beta_finale(_board, turn, 8)
        else:
            val, y, x = self.alpha_beta(_board, turn, self.depth)

        return y, x

    def alpha_beta(self, board, player, depth, alpha=-math.inf, beta=math.inf):

        # 葉の場合、評価値を返す
        if depth == 0:
            val = self.evaluation_function(board, self.player)
            return val

        # 相手の色
        opponent = turn.Turn.BLACK if player == turn.Turn.WHITE else turn.Turn.WHITE

        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            board, player)

        random.shuffle(legal_move_count_list)

        if len(legal_move_count_list) == 0:
            val = self.evaluation_function(board, self.player)
            return val

        if player == self.player:
            best = -math.inf
            best_y = best_x = None
            for y, x, count in legal_move_count_list:
                _board = copy.deepcopy(board)
                self.put_stone(_board, y, x, player)
                min_max_result = self.alpha_beta(_board, opponent, depth - 1, alpha, beta)
                if isinstance(min_max_result, int):
                    val = min_max_result
                else:
                    val = min_max_result[0]
                if val > best:
                    alpha = val
                    best = val
                    best_y = y
                    best_x = x

                if best > beta:  # betaカット
                    return best, best_y, best_x

            return best, best_y, best_x
        elif player != self.player:
            best = math.inf
            best_y = best_x = None
            for y, x, count in legal_move_count_list:
                _board = copy.deepcopy(board)
                self.put_stone(_board, y, x, player)
                min_max_result = self.alpha_beta(_board, opponent, depth - 1, alpha, beta)
                if isinstance(min_max_result, int):
                    val = min_max_result
                else:
                    val = min_max_result[0]
                if val < best:
                    beta = val
                    best = val
                    best_y = y
                    best_x = x

                if best < alpha:
                    return best, best_y, best_x

            return best, best_y, best_x

    def alpha_beta_finale(self, board, player, depth, alpha=-math.inf, beta=math.inf):

        # 葉の場合、評価値を返す
        if depth == 0:
            val = self.evaluation_function_finale(board, self.player)
            return val

        # 相手の色
        opponent = turn.Turn.BLACK if player == turn.Turn.WHITE else turn.Turn.WHITE

        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            board, player)

        random.shuffle(legal_move_count_list)

        if len(legal_move_count_list) == 0:
            val = self.evaluation_function_finale(board, self.player)
            return val

        if player == self.player:
            best = -math.inf
            best_y = best_x = None
            for y, x, count in legal_move_count_list:
                _board = copy.deepcopy(board)
                self.put_stone(_board, y, x, player)
                min_max_result = self.alpha_beta_finale(_board, opponent, depth - 1, alpha, beta)
                if isinstance(min_max_result, int):
                    val = min_max_result
                else:
                    val = min_max_result[0]
                if val > best:
                    alpha = val
                    best = val
                    best_y = y
                    best_x = x

                if best > beta:  # betaカット
                    return best, best_y, best_x

            return best, best_y, best_x
        elif player != self.player:
            best = math.inf
            best_y = best_x = None
            for y, x, count in legal_move_count_list:
                _board = copy.deepcopy(board)
                self.put_stone(_board, y, x, player)
                min_max_result = self.alpha_beta_finale(_board, opponent, depth - 1, alpha, beta)
                if isinstance(min_max_result, int):
                    val = min_max_result
                else:
                    val = min_max_result[0]
                if val < best:
                    beta = val
                    best = val
                    best_y = y
                    best_x = x

                if best < alpha:
                    return best, best_y, best_x

            return best, best_y, best_x

    def evaluation_function(self, board, player):
        evaluation = 0
        for y in range(1, 9):
            for x in range(1, 9):
                if board[y][x] == player:
                    evaluation += self.board_gain[y - 1][x - 1]

        return evaluation

    def evaluation_function_finale(self, board, player):
        evaluation = 0
        for y in range(1, 9):
            for x in range(1, 9):
                if board[y][x] == player:
                    evaluation += 1

        return evaluation


class AlphaBetaThirdAI(Player):

    def __init__(self, depth):
        super().__init__()
        self.depth = depth

    def thinking(self, board, turn):
        super().thinking(board, turn)
        _board = [[0] * (ROW_CELLS + 2) for i in range(ROW_CELLS + 2)]
        for i in range(1, 9):
            for j in range(1, 9):
                _board[i][j] = board[i - 1][j - 1]

        val, y, x = self.alpha_beta(_board, turn, self.depth)
        return y, x

    def alpha_beta(self, board, player, depth, alpha=-math.inf, beta=math.inf):

        # 葉の場合、評価値を返す
        if depth == 0:
            val = self.evaluation_function(board, self.player)
            return val

        # 相手の色
        opponent = turn.Turn.BLACK if player == turn.Turn.WHITE else turn.Turn.WHITE

        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            board, player)

        random.shuffle(legal_move_count_list)

        if len(legal_move_count_list) == 0:
            val = self.evaluation_function(board, self.player)
            return val

        if player == self.player:
            best = -math.inf
            best_y = best_x = None
            for y, x, count in legal_move_count_list:
                _board = copy.deepcopy(board)
                self.put_stone(_board, y, x, player)
                min_max_result = self.alpha_beta(_board, opponent, depth - 1, alpha, beta)
                if isinstance(min_max_result, int):
                    val = min_max_result
                else:
                    val = min_max_result[0]
                if val > best:
                    alpha = val
                    best = val
                    best_y = y
                    best_x = x

                if best > beta:  # betaカット
                    return best, best_y, best_x

            return best, best_y, best_x
        elif player != self.player:
            best = math.inf
            best_y = best_x = None
            for y, x, count in legal_move_count_list:
                _board = copy.deepcopy(board)
                self.put_stone(_board, y, x, player)
                min_max_result = self.alpha_beta(_board, opponent, depth - 1, alpha, beta)
                if isinstance(min_max_result, int):
                    val = min_max_result
                else:
                    val = min_max_result[0]
                if val < best:
                    beta = val
                    best = val
                    best_y = y
                    best_x = x

                if best < alpha:
                    return best, best_y, best_x

            return best, best_y, best_x

    def evaluation_function(self, board, player):
        """
        自分の打てる手の数 - 相手の打てる手の数
        :param board:
        :param player:
        :return:
        """
        opponent = turn.Turn.BLACK if player == turn.Turn.WHITE else turn.Turn.WHITE
        self_legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            board, player)
        opponent_legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            board, opponent)
        evaluation = len(self_legal_move_count_list) - len(opponent_legal_move_count_list)
        # print(evaluation)
        return evaluation

    def put_stone(self, board, y, x, player):
        """
        石を置く
        :param x:
        :param y:
        :param player:
        :return:
        """
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                if i == j == 0:
                    continue
                count = self.count_turn_over(board, player, y, x, i, j)
                for k in range(1, count + 1):
                    board[y + k * i][x + k * j] = player

        board[y][x] = player

        return


class node():
    def __init__(self, board, te):
        self.child = []
        self.board = board
        self.visit_count = 0
        self.wins = 0
        self.te = te


class MCTSPlayer(Player):
    """
    モンテカルロ木探索
    """
    def __init__(self, playout_num, visit_threshold):
        super().__init__()
        self.playout_num = playout_num
        self.visit_threshold = visit_threshold

    def thinking(self, board, turn):
        super().thinking(board, turn)
        _board = [[0] * (ROW_CELLS + 2) for i in range(ROW_CELLS + 2)]
        for i in range(1, 9):
            for j in range(1, 9):
                _board[i][j] = board[i - 1][j - 1]

        montecarlo_tree = node(_board, None)
        y, x = self.MCTS(_board, turn, montecarlo_tree)

        return y, x

    def MCTS(self, board, turn, tree):
        # 打てる手を全て探し出し
        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            tree.board, turn)
        # 展開
        for y, x, count in legal_move_count_list:
            copy_board = copy.deepcopy(tree.board)
            self.put_stone(copy_board, y, x, turn)
            tree.child.append(node(copy_board, (y, x)))

        for i in range(self.playout_num):
            self.selection(turn, tree)

        print("a", tree.visit_count)
        for child in tree.child:
            print(child.visit_count, child.wins, child.te)
            for c in child.child:
                print("    " + str(c.visit_count), c.wins)
                for d in c.child:
                    print("        " + str(d.visit_count), d.wins)

        s = 0
        t = ()
        for child in tree.child:
            # 0除算回避
            if child.visit_count is not 0 and child.wins / child.visit_count > s:
                s = child.wins / child.visit_count
                t = child.te

        if not t:  # 勝つ手を見つけられなかった
            # 打てる手を全て探し出しランダムに返す
            legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
                tree.board, turn)
            t = random.choice(legal_move_count_list)[:2]
        return t

    def selection(self, player, tree):
        opponent = turn.Turn.BLACK if player == turn.Turn.WHITE else turn.Turn.WHITE
        tree.visit_count += 1

        # 打てる手を全て探し出し
        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            tree.board, player)

        if len(legal_move_count_list) == 0:
            # ターンを渡す
            _legal_move_count_list, black_stones, white_stones = \
                self.exist_legal_move_and_count_stones(tree.board, opponent)
            legal_move_list = [tuple(i[0:2]) for i in _legal_move_count_list]
            # 渡されても打つ手がない場合(両者打つ手がない)
            # :盤面がすべて埋まる
            # :どちらかが全滅する
            # :両者置く場所がない
            if len(legal_move_list) == 0:
                # ゲーム終了
                if self.player == turn.Turn.BLACK:
                    if black_stones > white_stones:
                        return 1
                    else:
                        return 0
                else:
                    if white_stones > black_stones:
                        return 1
                    else:
                        return 0

        if not tree.child:  # 未展開
            if self.visit_threshold < tree.visit_count:  # 閾値より大きいの場合
                # 展開
                for y, x, count in legal_move_count_list:
                    copy_board = copy.deepcopy(tree.board)
                    self.put_stone(copy_board, y, x, player)
                    tree.child.append(node(copy_board, (y, x)))

                if not tree.child:
                    result = self.playout(tree.board, player)  # プレイアウトを行う
                    if result == 1:
                        return 1
                    else:
                        return 0

                selected_child = self.select_child(tree)  # ubc1
                selected_child.visit_count += 1
                result = self.playout(selected_child.board, opponent)
                if result == 1:
                    selected_child.wins += 1
                    return 1
                else:
                    return 0

            result = self.playout(tree.board, player)  # プレイアウトを行う
            if result == 1:
                return 1
            else:
                return 0

        else:
            selected_child = self.select_child(tree)  # ubc1
            result = self.selection(opponent, selected_child)
            if result == 1:
                selected_child.wins += 1
                return 1
            else:
                return 0

    def playout(self, board, player):
        opponent = turn.Turn.BLACK if player == turn.Turn.WHITE else turn.Turn.WHITE

        # 打てる手を全て探し出し
        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            board, player)

        if len(legal_move_count_list) == 0:
            # ターンを渡す
            _legal_move_count_list, black_stones, white_stones = \
                self.exist_legal_move_and_count_stones(board, opponent)
            legal_move_list = [tuple(i[0:2]) for i in _legal_move_count_list]
            # 渡されても打つ手がない場合(両者打つ手がない)
            # :盤面がすべて埋まる
            # :どちらかが全滅する
            # :両者置く場所がない
            if len(legal_move_list) == 0:
                # ゲーム終了
                if self.player == turn.Turn.BLACK:
                    if black_stones > white_stones:
                        return 1
                    else:
                        return 0
                else:
                    if white_stones > black_stones:
                        return 1
                    else:
                        return 0
            else:
                y, x, count = random.choice(_legal_move_count_list)
                copy_board = copy.deepcopy(board)
                self.put_stone(copy_board, y, x, opponent)
                if self.playout(copy_board, opponent) == 1:
                    return 1
                else:
                    return 0

        y, x, count = random.choice(legal_move_count_list)
        copy_board = copy.deepcopy(board)
        self.put_stone(copy_board, y, x, player)
        if self.playout(copy_board, opponent) == 1:
            return 1
        else:
            return 0

    def select_child(self, tree):
        max_ucb1 = 0
        max_ucb_idx = None
        for idx in range(len(tree.child)):
            ubc1_value = self.ucb1(tree, tree.child[idx])
            if ubc1_value > max_ucb1:
                max_ucb1 = ubc1_value
                max_ucb_idx = idx

        return tree.child[max_ucb_idx]

    def ucb1(self, tree, child):
        if child.visit_count == 0:
            return math.inf
        value = (child.wins / child.visit_count) + math.sqrt((2 * math.log(tree.visit_count)) / child.visit_count)
        return value


class PrimitiveMonteCarloPlayer(Player):
    def __init__(self, playout_num):
        super().__init__()
        self.playout_num = playout_num

    def thinking(self, board, turn):
        start = time.time()
        super().thinking(board, turn)
        _board = [[0] * (ROW_CELLS + 2) for i in range(ROW_CELLS + 2)]
        for i in range(1, 9):
            for j in range(1, 9):
                _board[i][j] = board[i - 1][j - 1]

        self.montecarlo_tree = node(_board, None)
        y, x = self.primitive_montecarlo(turn, self.montecarlo_tree)
        print(time.time() - start)

        return y, x

    def primitive_montecarlo(self, turn, tree):
        # 打てる手を全て探し出し
        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            tree.board, turn)
        # 展開
        for y, x, count in legal_move_count_list:
            copy_board = copy.deepcopy(tree.board)
            self.put_stone(copy_board, y, x, turn)
            tree.child.append(node(copy_board, (y, x)))

        for i in range(self.playout_num):
            self.selection(turn, tree)

        print("a", tree.visit_count)
        for child in tree.child:
            print(child.visit_count, child.wins, child.te)
            for c in child.child:
                print("    " + str(c.visit_count), c.wins)
                for d in c.child:
                    print("        " + str(d.visit_count), d.wins)

        s = 0
        t = ()
        for child in tree.child:
            # 0除算回避
            if child.visit_count is not 0 and child.wins / child.visit_count > s:
                s = child.wins / child.visit_count
                t = child.te

        if not t:  # 勝つ手を見つけられなかった
            # 打てる手を全て探し出しランダムに返す
            legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
                tree.board, turn)
            t = random.choice(legal_move_count_list)[:2]
        return t

    def playout(self, board, player):
        opponent = turn.Turn.BLACK if player == turn.Turn.WHITE else turn.Turn.WHITE
        # 打てる手を全て探し出し
        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            board, player)

        if len(legal_move_count_list) == 0:
            # ターンを渡す
            opponent = turn.Turn.BLACK if player == turn.Turn.WHITE else turn.Turn.WHITE
            _legal_move_count_list, black_stones, white_stones = \
                self.exist_legal_move_and_count_stones(board, opponent)
            legal_move_list = []
            for i in _legal_move_count_list:
                legal_move_list.append((i[0:2]))
            # 渡されても打つ手がない場合(両者打つ手がない)
            # :盤面がすべて埋まる
            # :どちらかが全滅する
            # :両者置く場所がない
            if len(legal_move_list) == 0:
                # ゲーム終了
                if self.player == turn.Turn.BLACK:
                    if black_stones > white_stones:
                        return 1
                    else:
                        return 0
                else:
                    if white_stones > black_stones:
                        return 1
                    else:
                        return 0
            else:
                y, x, count = random.choice(_legal_move_count_list)
                copy_board = copy.deepcopy(board)
                self.put_stone(copy_board, y, x, player)
                if self.playout(copy_board, opponent) == 1:
                    return 1
                else:
                    return 0

        y, x, count = random.choice(legal_move_count_list)
        copy_board = copy.deepcopy(board)
        self.put_stone(copy_board, y, x, player)
        if self.playout(copy_board, opponent) == 1:
            return 1
        else:
            return 0

    def selection(self, player, tree):
        tree.visit_count += 1

        # 打てる手を全て探し出し
        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            tree.board, player)

        if len(legal_move_count_list) == 0:
            # ターンを渡す
            opponent = turn.Turn.BLACK if player == turn.Turn.WHITE else turn.Turn.WHITE
            _legal_move_count_list, black_stones, white_stones = \
                self.exist_legal_move_and_count_stones(tree.board, opponent)
            legal_move_list = []
            for i in _legal_move_count_list:
                legal_move_list.append((i[0:2]))
            # 渡されても打つ手がない場合(両者打つ手がない)
            # :盤面がすべて埋まる
            # :どちらかが全滅する
            # :両者置く場所がない
            if len(legal_move_list) == 0:
                # ゲーム終了
                if self.player == turn.Turn.BLACK:
                    if black_stones > white_stones:
                        return 1
                    else:
                        return 0
                else:
                    if white_stones > black_stones:
                        return 1
                    else:
                        return 0

        selected_child = random.choice(tree.child)
        selected_child.visit_count += 1
        result = self.playout(selected_child.board, player)  # プレイアウトを行う
        if result == 1:
            selected_child.wins += 1
            return 1
        else:
            return 0
