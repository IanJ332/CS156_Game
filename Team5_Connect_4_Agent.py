
#!/usr/bin/env python3
"""
Connect 4 AI Agent Implementation
CS156 Final Project - Spring 2025

This module serves as the main entry point for the Connect 4 AI agent.
It implements the required interface functions and coordinates the
search, representation, and reasoning components.
"""

import random
import time
from search import minimax_search, alpha_beta_search
from representation import GameState
from reasoning import evaluate_board, get_valid_moves

# Configuration
MAX_DEPTH = 5  # Maximum search depth for minimax
USE_ALPHA_BETA = True  # Use alpha-beta pruning for performance
CENTER_PREFERENCE = True  # Prefer center columns in early game
DEBUG_MODE = False  # Set to True to enable debug prints

# Global variables to maintain state between function calls
game_state = None
player_symbol = None
opponent_symbol = None

def print_board(board):
    """Utility function to print the board for debugging"""
    rows = len(board)
    cols = len(board[0])
    
    print("\nCurrent Board State:")
    print("-" * (2 * cols + 1))
    
    for r in range(rows):
        print("|", end="")
        for c in range(cols):
            print(f"{board[r][c]}|", end="")
        print("")
    
    print("-" * (2 * cols + 1))
    print(" ", end="")
    for c in range(cols):
        print(f"{c+1} ", end="")
    print("\n")

def init_agent(player_sym, board_num_rows, board_num_cols, board):
    """
    Initializes the agent. Should only need to be called once at the start of a game.
    
    Args:
        player_sym: The symbol representing this agent ('X' or 'O')
        board_num_rows: Number of rows in the board
        board_num_cols: Number of columns in the board
        board: The initial board state
        
    Returns:
        True indicating successful initialization
    """
    global game_state, player_symbol, opponent_symbol
    
    # Store the player's symbol and determine the opponent's symbol
    player_symbol = player_sym
    opponent_symbol = 'O' if player_sym == 'X' else 'X'
    
    # Initialize the game state representation
    game_state = GameState(
        board=board,
        num_rows=int(board_num_rows),
        num_cols=int(board_num_cols),
        player_symbol=player_symbol,
        opponent_symbol=opponent_symbol
    )
    
    if DEBUG_MODE:
        print(f"Agent initialized as player {player_symbol}")
        print_board(board)
    
    return True

def what_is_your_move(board, game_rows, game_cols, my_game_symbol):
    """
    Decide which column to drop a disk.
    
    Args:
        board: Current state of the board
        game_rows: Number of rows in the board
        game_cols: Number of columns in the board
        my_game_symbol: The symbol representing this agent ('X' or 'O')
        
    Returns:
        An integer representing the column to drop the disk (1 to game_cols)
    """
    global game_state, player_symbol, opponent_symbol
    
    start_time = time.time()
    
    # If game_state is None, reinitialize it (should not happen in normal gameplay)
    if game_state is None:
        init_agent(my_game_symbol, game_rows, game_cols, board)
    else:
        # Update the game state with the current board
        game_state.update_board(board)
    
    # Get valid moves (non-full columns)
    valid_moves = get_valid_moves(board, game_rows, game_cols)
    
    if not valid_moves:
        # No valid moves (should not happen in normal gameplay)
        if DEBUG_MODE:
            print("No valid moves found!")
        return 1  # Return a default move, which will be rejected if invalid
    
    # In the first couple of moves, prefer the center columns if possible
    move_count = sum(1 for r in range(game_rows) for c in range(game_cols) if board[r][c] != ' ')
    
    if CENTER_PREFERENCE and move_count < 4:
        middle_cols = [game_cols // 2 + 1]  # Center column (1-indexed)
        
        # If even number of columns, consider both center columns
        if game_cols % 2 == 0:
            middle_cols = [game_cols // 2, game_cols // 2 + 1]
        
        # Check if any middle columns are valid, if so, choose randomly from them
        valid_middle_cols = [col for col in middle_cols if col in valid_moves]
        if valid_middle_cols:
            chosen_move = random.choice(valid_middle_cols)
            if DEBUG_MODE:
                print(f"Early game center preference: column {chosen_move}")
            return chosen_move
    
    # Use minimax with alpha-beta pruning to find the best move
    if USE_ALPHA_BETA:
        best_move = alpha_beta_search(game_state, MAX_DEPTH)
    else:
        best_move = minimax_search(game_state, MAX_DEPTH)
    
    # Ensure the chosen move is valid
    if best_move not in valid_moves:
        if DEBUG_MODE:
            print(f"Search returned invalid move {best_move}, choosing randomly from: {valid_moves}")
        best_move = random.choice(valid_moves)
    
    end_time = time.time()
    
    if DEBUG_MODE:
        print(f"Chose column {best_move} (took {end_time - start_time:.3f} seconds)")
        print_board(board)
    
    return best_move
def connect_4_result(board, winner, looser):
    """The Connect 4 manager calls this function when the game is over.
    If there is a winner, the team name of the winner and looser are the
    values of the respective argument variables. If there is a draw/tie,
    the values of winner = looser = 'Draw'."""

    # Check if a draw
    if winner == "Draw":
        print(">>> I am player TEAM1 <<<")
        print(">>> The game resulted in a draw. <<<\n")
        return True

    print(">>> I am player TEAM1 <<<")
    print("The winner is " + winner)
    if winner == "Team1":
        print("YEAH!!  :-)")
    else:
        print("BOO HOO HOO  :~(")
    print("The looser is " + looser)
    print()

    # print("The final board is") # Uncomment if you want to print the game board.
    # print(board)  # Uncomment if you want to print the game board.

    # Insert your code HERE to do whatever you like with the arguments.

    return True



# If run directly, the agent can be tested with random gameplay
if __name__ == "__main__":
    # Sample code to test the agent
    rows, cols = 6, 7
    test_board = [[' ' for _ in range(cols)] for _ in range(rows)]
    
    init_agent('X', rows, cols, test_board)
    
    # Simulate a few random moves
    print("Simulating random gameplay to test agent...")
    for _ in range(3):
        column = what_is_your_move(test_board, rows, cols, 'X')
        print(f"Agent chose column: {column}")
        
        # Update board with agent's move
        for r in range(rows-1, -1, -1):
            if test_board[r][column-1] == ' ':
                test_board[r][column-1] = 'X'
                break
        
        # Simulate opponent's random move
        valid_cols = [c+1 for c in range(cols) if test_board[0][c] == ' ']
        if valid_cols:
            opp_col = random.choice(valid_cols)
            for r in range(rows-1, -1, -1):
                if test_board[r][opp_col-1] == ' ':
                    test_board[r][opp_col-1] = 'O'
                    break
            print(f"Opponent chose column: {opp_col}")
        
        print_board(test_board)
