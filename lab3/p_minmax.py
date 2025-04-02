"""
Author: Katarzyna Nałęcz-Charkiewicz
"""

from board import Board
from player import Player

import random

MAX_VALUE = 999


class MinMaxPlayer(Player):
    def __init__(self, name: str, depth_limit: int):
        super().__init__(name)
        self.depth_limit = depth_limit

    def make_move(self, board: Board, your_side: str):
        empty_indexes = board.empty_indexes()
        return empty_indexes[random.randrange(len(empty_indexes))]


        # TODO
        return None

    def minimax(self, board: Board, side: str, depth: int):
        if depth == 0 or board.who_is_winner() is not None:
            return None, self.evaluate(board, side)
        
        for move in board.empty_indexes():
            new_board = board.clone()
            new_board.register_move(move)
            
        return None, None

    def evaluate(self, board: Board, side: str):
        if winner := board.who_is_winner():
            return MAX_VALUE if winner == side else -1 * MAX_VALUE
        return 0
