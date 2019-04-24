import math
import copy
import random
import time
import cProfile
import pstats
import cython
import _pickle as cPickle


cdef enum:
    UNDEFINED = 0  # 盤面外
    BLACK = 1
    WHITE = 2
    NONE = 3  # 何も置かれてないマス
    ROW_CELLS = 8


cdef class Player():
    cdef int player
    def __init__(self):
        self.player = 0

    @cython.nonecheck(False)
    cpdef void thinking(self, list board, int turn):
        self.player = turn

    @cython.nonecheck(False)
    cdef int count_turn_over(self, list board, int player, int y, int x, int i, int j):
        """
        (y,x)の(i,j)方向の石をいくつひっくり返せるかを返す
        """
        cdef int opponent = BLACK if player == WHITE else WHITE
        cdef int k = 1
        while board[y + k * i][x + k * j] == opponent:
            k += 1

        if board[y + k * i][x + k * j] == player:
            return k - 1
        else:
            return 0

    @cython.nonecheck(False)
    cdef int is_legal_move(self, list board, int player, int y, int x):
        """
        (y,x)にplayerの石が置けるか調べておける場合にはひっくり返せる個数を返す
        """
        cdef int turn_over_count
        cdef int i, j

        if board[y][x] != NONE:
            return False

        if x < 1 or x > 8 or y < 1 or y > 8:
            return False

        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                turn_over_count = self.count_turn_over(board, player, y, x, i, j)
                if not (i == 0 and j == 0) and turn_over_count:
                    return turn_over_count

        return False

    @cython.nonecheck(False)
    cdef tuple exist_legal_move_and_count_stones(self, list board, int player):
        """
        playerが置ける石の場所と取れる個数と白黒それぞれの石の個数を求めて返す
        """
        cdef list legal_move_list = []
        cdef int white_stones = 0
        cdef int black_stones = 0
        cdef int i, j

        for i in range(1, 9):
            for j in range(1, 9):
                if board[i][j] == WHITE:
                    white_stones += 1
                elif board[i][j] == BLACK:
                    black_stones += 1

                turn_over_count = self.is_legal_move(board, player, i, j)
                if turn_over_count:
                    legal_move_list.append((i, j, turn_over_count))

        return legal_move_list, black_stones, white_stones

    @cython.nonecheck(False)
    cdef void put_stone(self, list board, int y, int x, int player):
        """
        boardに石を置く
        """
        cdef i, j, k

        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                if i == j == 0:
                    continue
                count = self.count_turn_over(board, player, y, x, i, j)
                for k in range(1, count + 1):
                    board[y + k * i][x + k * j] = player

        board[y][x] = player


cdef class RandomChoiceAI(Player):
    @cython.nonecheck(False)
    def __init__(self):
        super().__init__()

    @cython.nonecheck(False)
    def thinking(self, list board, int turn):
        cdef int x, y
        cdef int i, j
        cdef list _board = [[0] * (ROW_CELLS + 2) for i in range(ROW_CELLS + 2)]
        cdef list legal_move_count_list
        cdef int black_stones
        cdef int white_stones

        _board[1:9][1:9] = board
        for i in range(1, 9):
            for j in range(1, 9):
                _board[i][j] = board[i - 1][j - 1]

        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            _board, turn)
        x, y = random.choice(legal_move_count_list)[0:2]

        return x, y


cdef class MinMaxAI(Player):
    cdef int depth
    cdef tuple board_gain

    @cython.nonecheck(False)
    def __init__(self, int depth):
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

    @cython.nonecheck(False)
    def thinking(self, list board, int turn):
        cdef int x, y, val
        cdef int i, j

        super().thinking(board, turn)
        cdef list _board = [[0] * (ROW_CELLS + 2) for i in range(ROW_CELLS + 2)]
        for i in range(1, 9):
            for j in range(1, 9):
                _board[i][j] = board[i - 1][j - 1]

        val, y, x = self.mini_max(_board, turn, self.depth)
        print("min_max", val)

        return y, x

    @cython.nonecheck(False)
    cdef mini_max(self, list board, int player, int depth):

        cdef list legal_move_count_list
        cdef int black_stones
        cdef int white_stones
        cdef int val
        cdef list _board

        # 葉の場合、評価値を返す
        if depth == 0:
            val = self.evaluation_function(board, self.player)
            print(player, "val", val)
            return val

        # 相手の色
        cdef opponent = BLACK if player == WHITE else WHITE

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
                _board = cPickle.loads(cPickle.dumps(board, -1))
                # _board = copy.deepcopy(board)
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

    @cython.nonecheck(False)
    cdef int evaluation_function(self, list board, int player):
        cdef int evaluation = 0
        cdef x, y

        for y in range(1, 9):
            for x in range(1, 9):
                if board[y][x] == player:
                    print(y, x, self.board_gain[y - 1][x - 1])
                    evaluation += self.board_gain[y - 1][x - 1]

        return evaluation


cdef class AlphaBetaAI(Player):
    cdef int depth
    cdef tuple board_gain

    @cython.nonecheck(False)
    def __init__(self, int depth):
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

    @cython.nonecheck(False)
    def thinking(self, list board, int turn):
        cdef int x, y, val
        cdef int i, j

        super().thinking(board, turn)
        _board = [[0] * (ROW_CELLS + 2) for i in range(ROW_CELLS + 2)]
        for i in range(1, 9):
            for j in range(1, 9):
                _board[i][j] = board[i - 1][j - 1]

        # prf = cProfile.Profile()
        # prf.enable()
        val, y, x = self.alpha_beta(_board, turn, self.depth)
        # prf.disable()
        # prf.print_stats()

        return y, x

    @cython.nonecheck(False)
    cdef alpha_beta(self, list board, int player, int depth, alpha=-math.inf, beta=math.inf):
        cdef list legal_move_count_list
        cdef int black_stones
        cdef int white_stones
        cdef int val
        cdef list _board

        # 葉の場合、評価値を返す
        if depth == 0:
            val = self.evaluation_function(board, self.player)
            return val

        # 相手の色
        cdef int opponent = BLACK if player == WHITE else WHITE

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
                _board = cPickle.loads(cPickle.dumps(board, -1))
                # _board = copy.deepcopy(board)
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
                _board = cPickle.loads(cPickle.dumps(board, -1))
                # _board = copy.deepcopy(board)
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

    @cython.nonecheck(False)
    cdef int evaluation_function(self, list board, int player):
        cdef int evaluation = 0
        cdef int x, y

        for y in range(1, 9):
            for x in range(1, 9):
                if board[y][x] == player:
                    evaluation += self.board_gain[y - 1][x - 1]

        return evaluation


cdef class SwitchTacticsAlphaBetaAI(Player):
    cdef int depth
    cdef tuple board_gain
    cdef int switch_threshold

    @cython.nonecheck(False)
    def __init__(self, int depth):
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
        self.switch_threshold = 52

    @cython.nonecheck(False)
    def thinking(self, board, turn):
        cdef int x, y, val
        cdef list legal_move_count_list
        cdef int black_stones
        cdef int white_stones
        cdef int i, j

        super().thinking(board, turn)
        _board = [[0] * (ROW_CELLS + 2) for i in range(ROW_CELLS + 2)]
        for i in range(1, 9):
            for j in range(1, 9):
                _board[i][j] = board[i - 1][j - 1]

        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            _board, turn)
        if black_stones + white_stones > self.switch_threshold:
            val, y, x = self.final_stage_alpha_beta(_board, turn, 8)
        else:
            val, y, x = self.alpha_beta(_board, turn, self.depth)

        return y, x

    @cython.nonecheck(False)
    cdef alpha_beta(self, list board, int player, int depth, alpha=-math.inf, beta=math.inf):
        cdef list legal_move_count_list
        cdef int black_stones
        cdef int white_stones
        cdef int val
        cdef list _board

        # 葉の場合、評価値を返す
        if depth == 0:
            val = self.evaluation_function(board, self.player)
            return val

        # 相手の色
        cdef int opponent = BLACK if player == WHITE else WHITE

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
                _board = cPickle.loads(cPickle.dumps(board, -1))
                # _board = copy.deepcopy(board)
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
                _board = cPickle.loads(cPickle.dumps(board, -1))
                # _board = copy.deepcopy(board)
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

    @cython.nonecheck(False)
    cdef final_stage_alpha_beta(self, list board, int player, int depth, alpha=-math.inf, beta=math.inf):
        cdef list legal_move_count_list
        cdef int black_stones
        cdef int white_stones
        cdef int val
        cdef list _board

        # 葉の場合、評価値を返す
        if depth == 0:
            val = self.final_stage_evaluation_function(board, self.player)
            return val

        # 相手の色
        cdef int opponent = BLACK if player == WHITE else WHITE

        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            board, player)

        random.shuffle(legal_move_count_list)

        if len(legal_move_count_list) == 0:
            val = self.final_stage_evaluation_function(board, self.player)
            return val

        if player == self.player:
            best = -math.inf
            best_y = best_x = None
            for y, x, count in legal_move_count_list:
                _board = cPickle.loads(cPickle.dumps(board, -1))
                # _board = copy.deepcopy(board)
                self.put_stone(_board, y, x, player)
                min_max_result = self.final_stage_alpha_beta(_board, opponent, depth - 1, alpha, beta)
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
                _board = cPickle.loads(cPickle.dumps(board, -1))
                # _board = copy.deepcopy(board)
                self.put_stone(_board, y, x, player)
                min_max_result = self.final_stage_alpha_beta(_board, opponent, depth - 1, alpha, beta)
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

    @cython.nonecheck(False)
    cdef int evaluation_function(self, list board, int player):
        cdef int evaluation = 0
        cdef int x, y

        for y in range(1, 9):
            for x in range(1, 9):
                if board[y][x] == player:
                    evaluation += self.board_gain[y - 1][x - 1]

        return evaluation

    @cython.nonecheck(False)
    cdef int final_stage_evaluation_function(self, list board, int player):
        cdef int evaluation = 0
        cdef int x, y

        for y in range(1, 9):
            for x in range(1, 9):
                if board[y][x] == player:
                    evaluation += 1

        return evaluation


cdef class CountMovesAlphaBetaAI(Player):

    cdef int depth

    @cython.nonecheck(False)
    def __init__(self, int depth):
        super().__init__()
        self.depth = depth

    @cython.nonecheck(False)
    def thinking(self, list board, int turn):
        cdef int x, y, val
        cdef int i, j

        super().thinking(board, turn)
        _board = [[0] * (ROW_CELLS + 2) for i in range(ROW_CELLS + 2)]
        for i in range(1, 9):
            for j in range(1, 9):
                _board[i][j] = board[i - 1][j - 1]

        val, y, x = self.alpha_beta(_board, turn, self.depth)
        return y, x

    @cython.nonecheck(False)
    cdef alpha_beta(self, list board, int player, int depth, alpha=-math.inf, beta=math.inf):
        cdef list legal_move_count_list
        cdef int black_stones
        cdef int white_stones
        cdef int val
        cdef list _board

        # 葉の場合、評価値を返す
        if depth == 0:
            val = self.evaluation_function(board, self.player)
            return val

        # 相手の色
        cdef int opponent = BLACK if player == WHITE else WHITE

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
                _board = cPickle.loads(cPickle.dumps(board, -1))
                # _board = copy.deepcopy(board)
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
                _board = cPickle.loads(cPickle.dumps(board, -1))
                # _board = copy.deepcopy(board)
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

    @cython.nonecheck(False)
    cdef int evaluation_function(self, list board, int player):
        """
        自分の打てる手の数 - 相手の打てる手の数
        """
        cdef int opponent = BLACK if player == WHITE else WHITE
        cdef list self_legal_move_count_list
        cdef list opponent_legal_move_count_list
        cdef int evaluation

        self_legal_move_count_list, _, _ = self.exist_legal_move_and_count_stones(
            board, player)
        opponent_legal_move_count_list, _, _ = self.exist_legal_move_and_count_stones(
            board, opponent)
        evaluation = len(self_legal_move_count_list) - len(opponent_legal_move_count_list)

        return evaluation


cdef class node():
    cdef public list child
    cdef public list board
    cdef public int visit_count
    cdef public int wins
    cdef public tuple te
    def __init__(self, list board, tuple te):
        self.child = []
        self.board = board
        self.visit_count = 0
        self.wins = 0
        self.te = te


cdef class PrimitiveMonteCarloPlayer(Player):
    """
    原始モンテカルロ法
    """
    cdef int playout_num
    @cython.nonecheck(False)
    def __init__(self, int playout_num):
        super().__init__()
        self.playout_num = playout_num

    @cython.nonecheck(False)
    def thinking(self, list board, int turn):
        start = time.time()
        cdef list _board
        cdef int x, y
        cdef int i, j

        super().thinking(board, turn)
        _board = [[0] * (ROW_CELLS + 2) for i in range(ROW_CELLS + 2)]
        for i in range(1, 9):
            for j in range(1, 9):
                _board[i][j] = board[i - 1][j - 1]

        cdef node montecarlo_tree = node(_board, None)
        y, x = self.primitive_montecarlo(turn, montecarlo_tree)
        print(time.time() - start)

        return y, x

    @cython.nonecheck(False)
    cdef tuple primitive_montecarlo(self, int turn, node tree):
        cdef list legal_move_count_list
        cdef int black_stones
        cdef int white_stones
        cdef list copy_board
        cdef int i

        # 打てる手を全て探し出し
        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            tree.board, turn)

        # 現在の盤面を展開
        cdef int legal_move_count_list_len = len(legal_move_count_list)
        for i in range(legal_move_count_list_len):
            copy_board = cPickle.loads(cPickle.dumps(tree.board, -1))
            # copy_board = copy.deepcopy(tree.board)
            self.put_stone(copy_board, legal_move_count_list[i][0], legal_move_count_list[i][1], turn)
            tree.child.append(node(copy_board, (legal_move_count_list[i][0], legal_move_count_list[i][1])))

        for i in range(self.playout_num):
            self.selection(turn, tree)

        print("a", tree.visit_count)
        for child in tree.child:
            print(child.visit_count, child.wins, child.te)
            for c in child.child:
                print("    " + str(c.visit_count), c.wins)
                for d in c.child:
                    print("        " + str(d.visit_count), d.wins)

        e = 0  # 期待値
        vest_point = ()
        for child in tree.child:
            # 0除算回避
            if child.visit_count is not 0 and child.wins / child.visit_count > e:
                e = child.wins / child.visit_count
                vest_point = child.te

        if not vest_point:  # 勝つ手を見つけられなかった
            # 打てる手を全て探し出しランダムに返す
            legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
                tree.board, turn)
            vest_point = random.choice(legal_move_count_list)[:2]
        return vest_point

    @cython.nonecheck(False)
    cdef int playout(self, list board, int player):
        cdef int opponent = BLACK if player == WHITE else WHITE
        cdef list legal_move_count_list, _legal_move_count_list
        cdef int black_stones
        cdef int white_stones
        cdef list copy_board
        cdef int y, x, count

        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
        board, player)

        if len(legal_move_count_list) == 0:
            # ターンを渡す
            opponent = BLACK if player == WHITE else WHITE
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
                if self.player == BLACK:
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
                copy_board = cPickle.loads(cPickle.dumps(board, -1))
                # copy_board = copy.deepcopy(board)
                self.put_stone(copy_board, y, x, player)
                if self.playout(copy_board, opponent) == 1:
                    return 1
                else:
                    return 0

        y, x, count = random.choice(legal_move_count_list)
        copy_board = cPickle.loads(cPickle.dumps(board, -1))
        # copy_board = copy.deepcopy(board)
        self.put_stone(copy_board, y, x, player)
        if self.playout(copy_board, opponent) == 1:
            return 1
        else:
            return 0

    @cython.nonecheck(False)
    cdef int selection(self, int player, node tree):
        """
        ルートノードの子ノードから一つ選んでプレイアウト
        """
        cdef list legal_move_count_list, _legal_move_count_list
        cdef int black_stones
        cdef int white_stones
        cdef list copy_board
        cdef int y, x, count
        cdef int opponent

        tree.visit_count += 1

        # 打てる手を全て探し出し
        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            tree.board, player)

        if len(legal_move_count_list) == 0:
            # ターンを渡す
            opponent = BLACK if player == WHITE else WHITE
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
                if self.player == BLACK:
                    if black_stones > white_stones:
                        return 1
                    else:
                        return 0
                else:
                    if white_stones > black_stones:
                        return 1
                    else:
                        return 0

        cdef node selected_child = random.choice(tree.child)
        selected_child.visit_count += 1
        cdef int result = self.playout(selected_child.board, player)  # プレイアウトを行う
        if result == 1:
            selected_child.wins += 1
            return 1
        else:
            return 0


cdef class MCTSPlayer(Player):
    """
    モンテカルロ木探索
    """
    cdef int playout_num
    cdef int visit_threshold

    @cython.nonecheck(False)
    def __init__(self, int playout_num, int visit_threshold):
        super().__init__()
        self.playout_num = playout_num
        self.visit_threshold = visit_threshold

    @cython.nonecheck(False)
    def thinking(self, list board, int turn):
        # start = time.time()
        cdef list _board
        cdef int x, y
        cdef int i, j

        super().thinking(board, turn)
        _board = [[0] * (ROW_CELLS + 2) for i in range(ROW_CELLS + 2)]
        for i in range(1, 9):
            for j in range(1, 9):
                _board[i][j] = board[i - 1][j - 1]

        cdef node montecarlo_tree = node(_board, None)
        # prf = cProfile.Profile()
        # prf.enable()
        y, x = self.MCTS(turn, montecarlo_tree)
        # prf.disable()
        # prf.print_stats()
        # stats = pstats.Stats(prf)
        # stats.sort_stats("time")
        # stats.print_stats()
        # print(time.time() - start)

        return y, x

    @cython.nonecheck(False)
    cdef tuple MCTS(self, int turn, node tree):
        cdef list legal_move_count_list
        cdef int black_stones
        cdef int white_stones
        cdef list copy_board
        cdef int i
        cdef double e = 0  # 期待値
        cdef tuple vest_point = ()

        # 打てる手を全て探し出し
        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            tree.board, turn)

        # 現在の盤面を展開
        cdef int legal_move_count_list_len = len(legal_move_count_list)
        for i in range(legal_move_count_list_len):
            copy_board = cPickle.loads(cPickle.dumps(tree.board, -1))
            # copy_board = copy.deepcopy(tree.board)
            self.put_stone(copy_board, legal_move_count_list[i][0], legal_move_count_list[i][1], turn)
            tree.child.append(node(copy_board, (legal_move_count_list[i][0], legal_move_count_list[i][1])))

        for i in range(self.playout_num):
            self.selection(turn, tree)

        print("a", tree.visit_count)
        for child in tree.child:
            print(child.visit_count, child.wins, child.te)
            for c in child.child:
                print("    " + str(c.visit_count), c.wins)
                for d in c.child:
                    print("        " + str(d.visit_count), d.wins)

        for child in tree.child:
            # 0除算回避
            if child.visit_count is not 0 and child.wins / child.visit_count > e:
                e = child.wins / child.visit_count
                vest_point = child.te

        if not vest_point:  # 勝つ手を見つけられなかった
            # 打てる手を全て探し出しランダムに返す
            legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
                tree.board, turn)
            vest_point = random.choice(legal_move_count_list)[:2]
        return vest_point

    @cython.nonecheck(False)
    cdef int selection(self, int player, node tree):
        """
        ルートノードの子ノードから一つ選んでプレイアウト
        """
        cdef list legal_move_count_list, _legal_move_count_list
        cdef int black_stones
        cdef int white_stones
        cdef list copy_board
        cdef int y, x, count
        cdef int opponent
        cdef node selected_child

        tree.visit_count += 1

        opponent = BLACK if player == WHITE else WHITE

        # 打てる手を全て探し出し
        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
            tree.board, player)

        if len(legal_move_count_list) == 0:
            # ターンを渡す
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
                if self.player == BLACK:
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
            if self.visit_threshold < tree.visit_count:  # 閾値より大きい場合
                # 展開
                for y, x, count in legal_move_count_list:
                    copy_board = cPickle.loads(cPickle.dumps(tree.board, -1))
                    # copy_board = copy.deepcopy(tree.board)
                    self.put_stone(copy_board, y, x, player)
                    tree.child.append(node(copy_board, (y, x)))

                if not tree.child:
                    result = self.playout(tree.board, player)  # プレイアウトを行う
                    if result == 1:
                        return 1
                    else:
                        return 0

                selected_child = self.get_max_ubc1(tree)
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
            selected_child = self.get_max_ubc1(tree)  # ubc1
            result = self.selection(opponent, selected_child)  # 選択
            if result == 1:
                selected_child.wins += 1
                return 1
            else:
                return 0

    @cython.nonecheck(False)
    cdef int playout(self, list board, int player):

        cdef int opponent = BLACK if player == WHITE else WHITE
        # 打てる手を全て探し出し
        cdef list legal_move_count_list, _legal_move_count_list
        cdef int black_stones
        cdef int white_stones
        cdef list copy_board
        cdef int y, x, count

        legal_move_count_list, black_stones, white_stones = self.exist_legal_move_and_count_stones(
        board, player)

        if len(legal_move_count_list) == 0:
            # ターンを渡す
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
                if self.player == BLACK:
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
                copy_board = cPickle.loads(cPickle.dumps(board, -1))
                # copy_board = copy.deepcopy(board)
                self.put_stone(copy_board, y, x, opponent)
                if self.playout(copy_board, opponent) == 1:
                    return 1
                else:
                    return 0

        y, x, count = random.choice(legal_move_count_list)
        copy_board = cPickle.loads(cPickle.dumps(board, -1))
        # copy_board = copy.deepcopy(board)
        self.put_stone(copy_board, y, x, player)
        if self.playout(copy_board, opponent) == 1:
            return 1
        else:
            return 0

    @cython.nonecheck(False)
    cdef node get_max_ubc1(self, node tree):
        cdef int idx

        max_ucb1 = 0
        max_ucb_idx = None
        for idx in range(len(tree.child)):
            ubc1_value = self.ucb1(tree, tree.child[idx])
            if ubc1_value > max_ucb1:
                max_ucb1 = ubc1_value
                max_ucb_idx = idx

        return tree.child[max_ucb_idx]

    @cython.nonecheck(False)
    cdef double ucb1(self, node tree, node child):
        if child.visit_count == 0:
            return math.inf
        value = (child.wins / <double>child.visit_count) + ((2 * math.log(tree.visit_count)) / child.visit_count) ** (1 / 2)

        return value