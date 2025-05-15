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
        self.rows = len(board)
        self.cols = len(board[0]) if board else COLS
        self.top = self.find_top(board)
    
    def find_top(self, board):
        """Find the top position for each column (where next piece would go)"""
        rows = len(board)
        cols = len(board[0]) if board else COLS
        
        top = [rows] * cols
        for row in range(rows):
            for col in range(cols):
                if board[row][col] != " ":
                    top[col] -= 1
        return top
    
    def move(self, column, player_symbol):
        """Make a move by placing player_symbol in column"""
        if column < 0 or column >= self.cols:
            raise Exception(f"Invalid column: {column}")
            
        if self.top[column] == 0:
            print_board(self.board)
            print("Column is full", column)
            raise Exception("Column is full")
        else:
            self.board[self.top[column] - 1][column] = player_symbol
            self.top[column] -= 1
    
    def __str__(self):
        return f"State(steps={self.steps}, board={self.board})"
    
    @classmethod
    def get_next_state(cls, board, player_symbol, steps, column):
        """Get the next state after making a move"""
        new_board = copy.deepcopy(board)  # Create a deep copy of the board
        nxtState = cls(new_board, steps + 1)  # Create new independent state
        try:
            nxtState.move(column, player_symbol)
        except Exception as e:
            print(f"Error: {e}")
            return None
        return nxtState
    
    def is_winning_state(self, player_symbol):
        """Check if the current state is a win for player_symbol"""
        rows = len(self.board)
        cols = len(self.board[0]) if self.board else COLS
        
        # Check horizontal
        for row in range(rows):
            for col in range(cols - 3):
                if all(self.board[row][col + i] == player_symbol for i in range(4)):
                    return True
        
        # Check vertical
        for col in range(cols):
            for row in range(rows - 3):
                if all(self.board[row + i][col] == player_symbol for i in range(4)):
                    return True
        
        # Check diagonal (bottom-left to top-right)
        for row in range(3, rows):
            for col in range(cols - 3):
                if all(self.board[row - i][col + i] == player_symbol for i in range(4)):
                    return True
        
        # Check diagonal (top-left to bottom-right)
        for row in range(rows - 3):
            for col in range(cols - 3):
                if all(self.board[row + i][col + i] == player_symbol for i in range(4)):
                    return True
        
        return False
    
    def one_step_win(self, player_symbol):
        """Check if there's a move that would immediately win the game"""
        state = copy.deepcopy(self)
        cols = len(state.board[0]) if state.board else COLS
        
        for col in range(cols):
            if state.top[col] > 0:
                nxtState = state.get_next_state(state.board, player_symbol, state.steps, col)
                if nxtState and nxtState.is_winning_state(player_symbol):
                    return col
        
        return None