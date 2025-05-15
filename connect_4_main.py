# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: C:\Users\lpwes\SJSU\CS156 Introduction to AI\Spring 2025\Canvas Material\Project Info\PWD_Connect_4\connect_4_main.py
# Bytecode version: 3.9.0beta5 (3425)
# Source timestamp: 2025-05-01 13:19:42 UTC (1746105582)

import os
import sys
import argparse
import importlib
import inspect
import random
parser = argparse.ArgumentParser(prog='connect_4_main.py', description='Connect 4 Game: This program plays the game of                                  Connect 4 between two opponents. The names of each player                                  must be different', usage='connect_4_main.py <Name of first player>                                   <Name of the second player>')
parser.add_argument('player1_name', type=str, help='The name of Player 1: Required.')
parser.add_argument('player2_name', type=str, help='The name of Player 2: Required.')
parser.add_argument('-f', '--result_file_name', type=str, default='connect_4_result.txt', help='The name of the file to save the results to.')
parser.add_argument('-r', '--rows', default=6, type=int, help='The number of rows in the game. Required.')
parser.add_argument('-c', '--cols', default=7, type=int, help='The number of columns in the game. Required.')
agent_module_name_suffix = '_Connect_4_Agent.py'
agent_move_file_suffix = '_Connect_4_Agent_Moves.txt'
result_file_name = 'connect_4_result.txt'
player1_name = ''
player2_name = ''
player1_symbol = ''
player2_symbol = ''
player1_name = ''
player2_name = ''
player1_module_name = ''
player1_move_file_name = ''
player2_module_name = ''
player2_move_file_name = ''
player1_module = ''
player2_module = ''
game_num_rows = 7 
game_num_cols = 9
who_makes_first_move = ''
who_makes_next_move = ''

def function_exists(module, function_name):
    """Checks if a function exists in a module.
  Args:
    module: The module to check.
    function_name: The name of the function to check for.
  Returns:
    True if the function exists in the module, False otherwise.
  """
    return hasattr(module, function_name) and inspect.isfunction(getattr(module, function_name))

def create_board(rows, cols):
    """ Create the Connect 4 board with the specified number of rows and columns."""
    return [[' ' for _ in range(cols)] for _ in range(rows)]

def print_board(board):
    """ Prints the connect 4 board."""
    for row in board:
        print('|' + '|'.join(row) + '|')
    print('-' * (len(board[0]) * 2 + 1))
    print(' ' + ' '.join((str(i + 1) for i in range(len(board[0])))))

def drop_piece(board, col, piece):
    """Drops the specified piece on the specified column."""
    for row in reversed(board):
        if row[col] == ' ':
            row[col] = piece
            return True
    else:
        return False

def check_win(board, piece):
    """Checks if the current player has won."""
    rows = len(board)
    cols = len(board[0])
    piece = 'X' if piece == player1_name else 'O'
    for r in range(rows):
        for c in range(cols - 3):
            if board[r][c] == board[r][c + 1] == board[r][c + 2] == board[r][c + 3] == piece:
                return True
    for c in range(cols):
        for r in range(rows - 3):
            if board[r][c] == board[r + 1][c] == board[r + 2][c] == board[r + 3][c] == piece:
                return True
    for r in range(rows - 3):
        for c in range(cols - 3):
            if board[r][c] == board[r + 1][c + 1] == board[r + 2][c + 2] == board[r + 3][c + 3] == piece:
                return True
            if board[r + 3][c] == board[r + 2][c + 1] == board[r + 1][c + 2] == board[r][c + 3] == piece:
                return True
    return False

def play_game(board, current_player, second_player):
    """ Play the Connect 4 game."""
    game_over = False
    game_state = 0
    player_col_move = -1
    while not game_over:
        print_board(board)
        try:
            if current_player == player1_name:
                player_col_move = int(player1_module.what_is_your_move(board, game_num_rows, game_num_cols, player1_symbol)) - 1
            else:
                player_col_move = int(player2_module.what_is_your_move(board, game_num_rows, game_num_cols, player2_symbol)) - 1
            if 0 <= player_col_move < game_num_cols:
                current_player_symbol = player1_symbol
                if current_player == player2_name:
                    current_player_symbol = player2_symbol
                if drop_piece(board, player_col_move, current_player_symbol):
                    if check_win(board, current_player):
                        print_board(board)
                        print(f'Player {current_player} wins!')
                        game_over = True
                    elif all((' ' not in row for row in board)):
                        print_board(board)
                        print("It's a draw!")
                        game_over = True
                    else:
                        current_player = player2_name if current_player == player1_name else player1_name
                else:
                    print('Column is full, try again.')
            else:
                print('Invalid column, try again.')
        except ValueError:
            print('Invalid input, enter a number.')
if __name__ == '__main__':
    args = parser.parse_args()
    if len(sys.argv) < 3:
        print('connect_4_main: Incorrect number of arguments passed. At least two are required, ' + str(len(sys.argv) - 1) + ' arguments actually passed. Exiting')
        exit(1)
    player1_name = args.player1_name
    player2_name = args.player2_name
    game_num_rows = args.rows
    game_num_cols = args.cols
    result_file_name = args.result_file_name
    player1_module_name = player1_name + agent_module_name_suffix
    player2_module_name = player2_name + agent_module_name_suffix
    if not os.path.exists(player1_module_name):
        print('connect_4_main: The Python module for ' + player1_name + ', ' + player1_module_name + ' does not exist. Exiting.')
        sys.exit(1)
    if not os.path.exists(player2_module_name):
        print('connect_4_main: The Python module for ' + player2_name + ', ' + player2_module_name + ' does not exist. Exiting.')
        sys.exit(1)
    try:
        player1_module = importlib.import_module(player1_module_name[:-3])
        print('connect_4_main: Module ' + player1_module_name + ' imported.')
    except ModuleNotFoundError:
        print('connect_4_main: Module ' + player1_module_name + ' not imported. Exiting.')
        sys.exit(1)
    except Exception as e:
        print('connect_4_main: An error ' + str(e) + ' occured. Exiting.')
    try:
        player2_module = importlib.import_module(player2_module_name[:-3])
    except ModuleNotFoundError:
        print('connect_4_main: Module ' + player2_module_name + ' not imported. Exiting.')
        sys.exit(1)
    except Exception as e:
        print('connect_4_main: An error ' + str(e) + ' occured. Exiting.')
    if function_exists(player1_module, 'init_agent'):
        print('Function init_agent exists in the module for ' + player1_name + '.')
    else:
        print('Function init_agent does not exist for ' + player1_name + '. Exiting.')
        sys.exit(1)
    if function_exists(player1_module, 'what_is_your_move'):
        print('Function what_is_your_move exists for ' + player1_name + '.')
    else:
        print('Function what_is_your_move does not exist for ' + player1_name + '. Exiting.')
        sys.exit(1)
    if function_exists(player2_module, 'init_agent'):
        print('Function init_agent exists in the module for ' + player2_name + '.')
    else:
        print('Function init_agent does not exist for ' + player2_name + '. Exiting.')
        sys.exit(1)
    if function_exists(player2_module, 'what_is_your_move'):
        print('Function what_is_your_move exists for ' + player2_name + '.')
    else:
        print('Function what_is_your_move does not exist for ' + player2_name + '. Exiting.')
        sys.exit(1)
    board = create_board(game_num_rows, game_num_cols)
    random.seed()
    random_bit = random.randint(0, 1)
    if random_bit == 0:
        who_makes_first_move = player1_name
        who_makes_next_move = player2_name
        player1_symbol = 'X'
        print('connect_4_main: ' + player1_name + ' will be represented as X.')
        player2_symbol = 'O'
        print('connect_4_main: ' + player2_name + ' will be represented as O.')
        current_player = player1_symbol
    else:
        who_makes_first_move = player2_name
        who_makes_next_move = player1_name
        player2_symbol = 'X'
        print('connect_4_main: ' + player2_name + ' will be represented as X.')
        player1_symbol = 'O'
        print('connect_4_main: ' + player1_name + ' will be represented as O.')
        current_player = player2_symbol
    print('cols:', game_num_cols)
    print('rows:', game_num_rows)
    player1_module.init_agent(player1_symbol, game_num_rows, game_num_cols, board)
    player2_module.init_agent(player2_symbol, game_num_rows, game_num_cols, board)
    play_game(board, who_makes_first_move, who_makes_next_move)