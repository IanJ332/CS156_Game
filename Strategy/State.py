from constant import ROWS, COLS


class State:
    def __init__(self, board):
        self.board = board
        self.steps = 0
        self.top = [ROWS] * COLS

    def move(self, board, column, player_symbol):
        if self.top[column] == 0:
            raise Exception("Column is full")
        else:
            board[self.top[column] - 1][column] = player_symbol
            self.top[column] -= 1

    @classmethod
    def get_next_state(self, column, player_symbol):
        nxtState = State(self.board)
        nxtState.steps = self.steps + 1
        try:
            nxtState.board = self.move(column, player_symbol)
        except Exception as e:
            print(f"Error: {e}")
            return None
        return nxtState

