from Strategy.constant import ROWS, COLS
import copy


def print_board(board):
    """Prints the connect 4 game board."""
    for row in board:
        print("|" + "|".join(row) + "|")
    print("-" * (len(board[0]) * 2 + 1))
    print(" " + " ".join(str(i + 1) for i in range(len(board[0]))))




class State:
    def __init__(self, board, steps=0):
        self.board = board
        self.steps = steps
        self.top = self.find_top(board)

    def find_top(self, board):
        top = [ROWS] * COLS
        for row in range(ROWS):
            for col in range(COLS):
                if board[row][col] != " ":
                    top[col] -= 1
        return top

    def move(self, column, player_symbol):
        if self.top[column] == 0:
            print("Column is full", column)
            raise Exception("Column is full")
        else:
            self.board[self.top[column] - 1][column] = player_symbol
            self.top[column] -= 1

    def __str__(self):
        return f"State(steps={self.steps}, board={self.board})"

    @classmethod
    def get_next_state(self, board,player_symbol,steps, column):
        new_board = copy.deepcopy(board)  # Create a deep copy of the board
        nxtState = State(new_board, steps + 1)  # Create new independent state
        try:
            nxtState.move(column, player_symbol)
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

        # print_board(self.board)
        return False
    def one_step_win(self, player_symbol):
        state = copy.deepcopy(self)
        for col in range(COLS):
            if state.top[col] > 0:
                # print("col", col)
                nxtState = state.get_next_state(state.board, player_symbol, state.steps, col)

                if nxtState and nxtState.is_winning_state(player_symbol):
                    return col
            # else:
            #     print("Column is full", col, "top", self.top[col])
        return None
