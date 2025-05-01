from Team2 import init_agent as team3_init, what_is_your_move as team3_move
from Team3 import init_agent as team4_init, what_is_your_move as team4_move

import random

ROWS, COLS = 6, 7

def create_board():
    return [[" " for _ in range(COLS)] for _ in range(ROWS)]

def print_board(board):
    for row in board:
        print("| " + " | ".join(row) + " |")
    print("-" * (4 * COLS + 1))

def is_valid_location(board, col):
    return board[0][col] == " "

def get_next_open_row(board, col):
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == " ":
            return r
    return None

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def check_win(board, piece):
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c + i] == piece for i in range(4)):
                return True

    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[r + i][c] == piece for i in range(4)):
                return True

    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i][c + i] == piece for i in range(4)):
                return True

    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[r - i][c + i] == piece for i in range(4)):
                return True

    return False

def get_valid_columns(board):
    return [c for c in range(COLS) if is_valid_location(board, c)]

def simulate_game():
    board = create_board()
    team3_init("X", ROWS, COLS, board)
    team4_init("O", ROWS, COLS, board)

    print_board(board)
    turn = random.choice(["Team3", "Team4"])
    print(f"First move: {turn}")

    while True:
        if turn == "Team3":
            col = team3_move(board, ROWS, COLS, "X")
            symbol = "X"
        else:
            col = team4_move(board, ROWS, COLS, "O")
            symbol = "O"

        if not isinstance(col, int) or not (0 <= col < COLS) or not is_valid_location(board, col):
            print(f"{turn} made invalid move: {col}")
            print(f"{'Team4' if turn == 'Team3' else 'Team3'} wins by forfeit!")
            break

        row = get_next_open_row(board, col)
        drop_piece(board, row, col, symbol)

        print_board(board)

        if check_win(board, symbol):
            print(f"{turn} ({symbol}) wins!")
            break

        if not get_valid_columns(board):
            print("It's a tie!")
            break

        turn = "Team4" if turn == "Team3" else "Team3"

if __name__ == "__main__":
    simulate_game()
