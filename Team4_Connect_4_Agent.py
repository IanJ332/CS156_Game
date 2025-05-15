#! /usr/bin/Team5_Connect_4_Agent.py

"""
Connect 4 AI Agent Implementation
CS156 Final Project - Spring 2025

This module serves as the main entry point for the Connect 4 AI agent.
It implements the required interface functions and coordinates the
search, representation, and reasoning components.

Team5 belongs entirely to Zhiyuan Xu.
More changes have been made to its MCT and it is an enhanced version of Team 5.
"""

# IMPORT
import random
import time
from Strategy.ConnectAgent import ConnectAgent
from Strategy.State import State

# DEFINITIONS

# HELPER FUNCTIONS
# Print the Board
def print_board(board):
    """Prints the connect 4 game board."""
    for row in board:
        print("|" + "|".join(row) + "|")
    print("-" * (len(board[0]) * 2 + 1))
    print(" " + " ".join(str(i + 1) for i in range(len(board[0]))))

def init_agent(player_symbol, board_num_rows, board_num_cols, board):
    """Inits the agent. Should only need to be called once at the start of a game."""
    debug_mode = False
    num_rows = int(board_num_rows)
    num_cols = int(board_num_cols)
    
    # Initialize agent with board dimensions and initial state
    ConnectAgent.set_agent(num_rows, num_cols, board)
    ConnectAgent.set_symbols(player_symbol)
    
    if (debug_mode):
        print_board(board)
        print(f"Agent initialized with symbol: {player_symbol}")
    return True

def what_is_your_move(board, game_rows, game_cols, my_game_symbol):
    """Decide your move, i.e., which column to drop a disk."""
    debug_mode = True 
    start_time = time.time()
    opponent_symbol = "X" if my_game_symbol == "O" else "O"
    
    # Update our state with the latest board
    ConnectAgent.add_state(State(board, ConnectAgent.get_state()[-1].steps+1))
    state = ConnectAgent.get_state()[-1]
    
    # First, check for immediate winning move
    winning_move = state.one_step_win(my_game_symbol)
    if (winning_move is not None):
        if (debug_mode):
            print(f"Found winning move: {winning_move+1} (in {time.time() - start_time:.3f}s)")
        return winning_move+1
    
    # Second, check if we need to block opponent's winning move
    blocking_move = state.one_step_win(opponent_symbol)
    if (blocking_move is not None):
        if (debug_mode):
            print(f"Found blocking move: {blocking_move+1} (in {time.time() - start_time:.3f}s)")
        return blocking_move+1
    
    # Use enhanced MCTS to find the best move
    try:
        mcts_move = ConnectAgent.get_mcts_move(board)
        elapsed = time.time() - start_time
        if (debug_mode):
            print(f"MCTS selected move: {mcts_move} (in {elapsed:.3f}s)")
        return mcts_move
    except Exception as e:
        if (debug_mode):
            raise e
        print(f"MCTS error: {e}, falling back to random move")
        # Fallback to random move if MCTS fails
        valid_cols = []
        for col in range(game_cols):
            if state.top[col] > 0:  # If column not full
                valid_cols.append(col + 1)  # +1 for 1-indexed columns
        
        if valid_cols:
            return random.choice(valid_cols)
        return random.randint(1, game_cols)  # Last resort

def connect_4_result(board, winner, looser):
    """The Connect 4 manager calls this function when the game is over."""
    # Check if a draw
    if winner == "Draw":
        print(">>> I am player TEAM3 <<<")
        print(">>> The game resulted in a draw. <<<\n")
        return True
    print(">>> I am player TEAM3 <<<")
    print("The winner is " + winner)
    if winner == "Team5":
        print("YEAH!!  :-)")
    else:
        print("BOO HOO HOO  :~(")
    print("The looser is " + looser)
    print()
    
    # Print final board state
    print("Final board state:")
    print_board(board)
    
    return True

# MAKE SURE MODULE IS IMPORTED
if __name__ == "__main__":
    print("Team5_Connect_4_Agent.py is intended to be imported and not executed.")
else:
    print("Team5_Connect_4_Agent.py has been imported.")