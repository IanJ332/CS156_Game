from Strategy.constant import ROWS, COLS
import copy


class State:
    def __init__(self, board, steps=0):
        self.board = board
        self.steps = steps
        self.player_symbol = "X" if steps % 2 == 0 else "O"
        self.top = self.find_top(board)

    def find_top(self, board):
        top = [ROWS] * COLS
        for row in range(ROWS):
            for col in range(COLS):
                if board[row][col] != " ":
                    top[col] -= 1
        return top

    def move(self, column):
        if self.top[column] == 0:
            print("Column is full", column)
            raise Exception("Column is full")
        else:
            self.board[self.top[column] - 1][column] = self.player_symbol
            self.top[column] -= 1

    def __str__(self):
        return f"State(steps={self.steps}, board={self.board})"


    @classmethod
    def get_next_state(cls, state, column):
        new_board = copy.deepcopy(state.board)
        nxtState = cls(new_board, state.steps + 1)  # Create new independent state
        try:
            nxtState.move(column)
        except Exception as e:
            print(f"Error: {e}")
            return None
        return nxtState

    def is_winning_state(self, player_symbol):
        # Check horizontal
        for row in range(ROWS):
            for col in range(COLS - 3):
                if all(self.board[row][col + i] == player_symbol for i in range(4)):
                    return True

        # Check vertical
        for col in range(COLS):
            for row in range(ROWS - 3):
                if all(self.board[row + i][col] == player_symbol for i in range(4)):
                    return True

        # Check diagonal (bottom-left to top-right)
        for row in range(3, ROWS):
            for col in range(COLS - 3):
                if all(self.board[row - i][col + i] == player_symbol for i in range(4)):
                    return True

        # Check diagonal (top-left to bottom-right)
        for row in range(ROWS - 3):
            for col in range(COLS - 3):
                if all(self.board[row + i][col + i] == player_symbol for i in range(4)):
                    return True

        return False

    def one_step_win(self, player_symbol):
        for col in range(COLS):
            if self.top[col] > 0:
                nxtState = self.get_next_state(State(self.board, self.steps), col)
                if nxtState and nxtState.is_winning_state(player_symbol):
                    return col
            else:
                print("Column is full", col, "top", self.top[col])
        return None
