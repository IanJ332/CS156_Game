#!/usr/bin/env python3
"""
Connect 4 Main Game Controller

This program plays Connect 4 between two agent modules.
Each agent module should implement:
- init_agent(player_symbol, board_num_rows, board_num_cols, board)
- what_is_your_move(board, game_rows, game_cols, my_game_symbol)
"""

import os
import sys
import argparse
import importlib
import inspect
import random

# Set up argument parser
parser = argparse.ArgumentParser(
    prog='connect_4_main.py',
    description='Connect 4 Game: This program plays the game of Connect 4 between two opponents. '
                'The names of each player must be different',
    usage='connect_4_main.py <Name of first player> <Name of the second player>'
)
parser.add_argument('player1_name', type=str, help='The name of Player 1: Required.')
parser.add_argument('player2_name', type=str, help='The name of Player 2: Required.')
parser.add_argument('-f', '--result_file_name', type=str, default='connect_4_result.txt',
                    help='The name of the file to save the results to.')
parser.add_argument('-r', '--rows', default=6, type=int, help='The number of rows in the game. Required.')
parser.add_argument('-c', '--cols', default=7, type=int, help='The number of columns in the game. Required.')

# Constants
agent_module_name_suffix = '_Connect_4_Agent.py'
agent_move_file_suffix = '_Connect_4_Agent_Moves.txt'
result_file_name = 'connect_4_result.txt'

# Initialize variables
player1_name = ''
player2_name = ''
player1_symbol = ''
player2_symbol = ''
player1_module_name = ''
player1_move_file_name = ''
player2_module_name = ''
player2_move_file_name = ''
player1_module = None
player2_module = None
game_num_rows = 6
game_num_cols = 7
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
    print(' ' + ' '.join(str(i + 1) for i in range(len(board[0]))))

def drop_piece(board, col, piece):
    """Drops the specified piece on the specified column."""
    for row in reversed(board):
        if row[col] == ' ':
            row[col] = piece
            return True
    return False

def check_win(board, piece):
    """Checks if the current player has won."""
    rows = len(board)
    cols = len(board[0])
    
    # Check horizontal
    for r in range(rows):
        for c in range(cols - 3):
            if board[r][c] == board[r][c + 1] == board[r][c + 2] == board[r][c + 3] == piece:
                return True
    
    # Check vertical
    for c in range(cols):
        for r in range(rows - 3):
            if board[r][c] == board[r + 1][c] == board[r + 2][c] == board[r + 3][c] == piece:
                return True
    
    # Check diagonal (positive slope)
    for r in range(rows - 3):
        for c in range(cols - 3):
            if board[r][c] == board[r + 1][c + 1] == board[r + 2][c + 2] == board[r + 3][c + 3] == piece:
                return True
    
    # Check diagonal (negative slope)
    for r in range(3, rows):
        for c in range(cols - 3):
            if board[r][c] == board[r - 1][c + 1] == board[r - 2][c + 2] == board[r - 3][c + 3] == piece:
                return True
    
    return False

def play_game(board, current_player, second_player):
    """ Play the Connect 4 game."""
    game_over = False
    game_state = 0
    player_col_move = -1
    
    # Open result file for writing
    with open(result_file_name, 'w') as f:
        f.write(f"Game between {player1_name} and {player2_name}\n")
        f.write(f"{current_player} goes first\n\n")
    
    while not game_over:
        print_board(board)
        
        try:
            if current_player == player1_name:
                player_col_move = int(player1_module.what_is_your_move(board, game_num_rows, game_num_cols, player1_symbol)) - 1
                current_player_symbol = player1_symbol
            else:
                player_col_move = int(player2_module.what_is_your_move(board, game_num_rows, game_num_cols, player2_symbol)) - 1
                current_player_symbol = player2_symbol
            
            # Log the move
            with open(result_file_name, 'a') as f:
                f.write(f"{current_player} moves in column {player_col_move + 1}\n")
            
            if 0 <= player_col_move < game_num_cols:
                if drop_piece(board, player_col_move, current_player_symbol):
                    if check_win(board, current_player_symbol):
                        print_board(board)
                        print(f'Player {current_player} wins!')
                        with open(result_file_name, 'a') as f:
                            f.write(f"\n{current_player} wins!\n")
                        game_over = True
                    elif all(' ' not in row for row in board):
                        print_board(board)
                        print("It's a draw!")
                        with open(result_file_name, 'a') as f:
                            f.write("\nIt's a draw!\n")
                        game_over = True
                    else:
                        current_player = player2_name if current_player == player1_name else player1_name
                else:
                    print('Column is full, try again.')
                    with open(result_file_name, 'a') as f:
                        f.write(f"{current_player} tried column {player_col_move + 1} but it's full. Forfeit.\n")
                        f.write(f"{second_player} wins by forfeit!\n")
                    print(f"{second_player} wins by forfeit!")
                    game_over = True
            else:
                print('Invalid column, try again.')
                with open(result_file_name, 'a') as f:
                    f.write(f"{current_player} made an invalid move: {player_col_move + 1}. Forfeit.\n")
                    f.write(f"{second_player} wins by forfeit!\n")
                print(f"{second_player} wins by forfeit!")
                game_over = True
        except ValueError:
            print('Invalid input, enter a number.')
            with open(result_file_name, 'a') as f:
                f.write(f"{current_player} made an invalid move (not a number). Forfeit.\n")
                f.write(f"{second_player} wins by forfeit!\n")
            print(f"{second_player} wins by forfeit!")
            game_over = True
    
    # Final board state
    with open(result_file_name, 'a') as f:
        f.write("\nFinal board state:\n")
        for row in board:
            f.write('|' + '|'.join(row) + '|\n')
        f.write('-' * (len(board[0]) * 2 + 1) + '\n')
        f.write(' ' + ' '.join(str(i + 1) for i in range(len(board[0]))) + '\n')

if __name__ == '__main__':
    args = parser.parse_args()
    
    if len(sys.argv) < 3:
        print('connect_4_main: Incorrect number of arguments passed. At least two are required, ' 
              + str(len(sys.argv) - 1) + ' arguments actually passed. Exiting')
        exit(1)
    
    player1_name = args.player1_name
    player2_name = args.player2_name
    game_num_rows = args.rows
    game_num_cols = args.cols
    result_file_name = args.result_file_name
    
    player1_module_name = player1_name + '_Connect_4_Agent.py'
    player2_module_name = player2_name + '_Connect_4_Agent.py'
    
    # Check if module files exist
    if not os.path.exists(player1_module_name):
        print('connect_4_main: The Python module for ' + player1_name + ', ' + player1_module_name + ' does not exist. Exiting.')
        sys.exit(1)
    
    if not os.path.exists(player2_module_name):
        print('connect_4_main: The Python module for ' + player2_name + ', ' + player2_module_name + ' does not exist. Exiting.')
        sys.exit(1)
    
    # Import modules
    try:
        player1_module = importlib.import_module(player1_module_name[:-3])
    except ModuleNotFoundError:
        print('connect_4_main: Module ' + player1_module_name + ' not imported. Exiting.')
        sys.exit(1)
    except Exception as e:
        print('connect_4_main: An error ' + str(e) + ' occurred. Exiting.')
        sys.exit(1)
    
    try:
        player2_module = importlib.import_module(player2_module_name[:-3])
    except ModuleNotFoundError:
        print('connect_4_main: Module ' + player2_module_name + ' not imported. Exiting.')
        sys.exit(1)
    except Exception as e:
        print('connect_4_main: An error ' + str(e) + ' occurred. Exiting.')
        sys.exit(1)
    
    # Check if required functions exist
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
    
    # Create board and select first player
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
    else:
        who_makes_first_move = player2_name
        who_makes_next_move = player1_name
        player2_symbol = 'X'
        print('connect_4_main: ' + player2_name + ' will be represented as X.')
        player1_symbol = 'O'
        print('connect_4_main: ' + player1_name + ' will be represented as O.')
    
    # Initialize agents
    player1_module.init_agent(player1_symbol, game_num_rows, game_num_cols, board)
    player2_module.init_agent(player2_symbol, game_num_rows, game_num_cols, board)
    
    # Play the game
    play_game(board, who_makes_first_move, who_makes_next_move)