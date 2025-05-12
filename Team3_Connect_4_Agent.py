#! /usr/bin/Team3_Connect_4_Agent.py

# IMPORTS
import random
from Strategy.ConnectAgent import ConnectAgent
from Strategy.State import State
from Strategy.constant import ROWS, COLS

# DEFINITIONS
# board = [[' ' for _ in range(cols)] for _ in range(rows)]
agent = None

# HELPER FUNCTIONS
# Print the Board
def print_board(board):
    """Prints the connect 4 game board."""
    for row in board:
        print("|" + "|".join(row) + "|")
    print("-" * (len(board[0]) * 2 + 1))
    print(" " + " ".join(str(i + 1) for i in range(len(board[0]))))


def init_agent(player_symbol, board_num_rows, board_num_cols, board):
    """Inits the agent. Should only need to be called once at the start of a game.
    NOTE NOTE NOTE: Do not expect the values you might save in variables to retain
    their values each time a function in this module is called. Therefore, you might
    want to save the variables to a file and re-read them when each function was called.
    This is not to say you should do that. Rather, just letting you know about the variables
    you might use in this module.
    NOTE NOTE NOTE NOTE: All functions called by connect_4_main.py  module will pass in all
    of the variables that you likely will need. So you can probably skip the 'NOTE NOTE NOTE'
    above."""
    num_rows = int(board_num_rows)
    num_cols = int(board_num_cols)

    # game_board = board
    ConnectAgent.set_agent(num_rows, num_cols, board, player_symbol)

    # my_game_symbol = player_symbol

    return True


def what_is_your_move(board, game_rows, game_cols, my_game_symbol):
    """Decide your move, i.e., which column to drop a disk."""

    # Insert your agent code HERE to decide which column to drop/insert your disk.

    opponent_symbol = "X" if my_game_symbol == "O" else "O"
    ConnectAgent.add_state(State(board,ConnectAgent.get_state()[-1].steps+1))
    state = ConnectAgent.get_state()[-1]
    if (state.one_step_win(my_game_symbol) is not None):
        print("winning", state.one_step_win(my_game_symbol)+1)
        return state.one_step_win(my_game_symbol)+1
    if (state.one_step_win(opponent_symbol) is not None):
        print("blocking", state.one_step_win(opponent_symbol)+1)
        return state.one_step_win(opponent_symbol)+1
    return random.randint(1, game_cols)


#####
# MAKE SURE MODULE IS IMPORTED
if __name__ == "__main__":
    print("Team3_Connect_4_Agent.py  is intended to be imported and not executed.")
else:
    print("Team3_Connect_4_Agent.py  has been imported.")
