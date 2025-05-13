#!/usr/bin/env python3
"""
Human vs AI Connect 4 Game
This script allows a human player to play against the Connect 4 AI agent.
"""

import sys
import os
from Team5_Connect_4_Agent import init_agent, what_is_your_move

# Constants
ROWS = 6
COLS = 7
HUMAN_SYMBOL = 'O'
AI_SYMBOL = 'X'

def create_board():
    """Create an empty Connect 4 board"""
    return [[' ' for _ in range(COLS)] for _ in range(ROWS)]

def print_board(board):
    """Print the Connect 4 board in a human-friendly format"""
    print("\n  ", end="")
    for c in range(COLS):
        print(f" {c+1} ", end="")
    print("\n")
    
    print("  " + "=" * (3 * COLS + 1))
    
    for r in range(ROWS):
        print(f"{r+1} |", end="")
        for c in range(COLS):
            cell = board[r][c]
            if cell == ' ':
                print("   ", end="")
            else:
                print(f" {cell} ", end="")
        print("|")
    
    print("  " + "=" * (3 * COLS + 1))
    
    print("  ", end="")
    for c in range(COLS):
        print(f" {c+1} ", end="")
    print("\n")

def is_valid_location(board, col):
    """Check if a column has space for a new piece"""
    return board[0][col] == ' '

def get_next_open_row(board, col):
    """Get the next open row in the specified column"""
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == ' ':
            return r
    return None

def drop_piece(board, row, col, piece):
    """Drop a piece at the specified location"""
    board[row][col] = piece

def check_win(board, piece):
    """Check if the player with the given piece has won"""
    # Check horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c + i] == piece for i in range(4)):
                return True

    # Check vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[r + i][c] == piece for i in range(4)):
                return True

    # Check diagonal down
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i][c + i] == piece for i in range(4)):
                return True

    # Check diagonal up
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[r - i][c + i] == piece for i in range(4)):
                return True

    return False

def is_board_full(board):
    """Check if the board is full"""
    return all(cell != ' ' for row in board for cell in row)

def human_turn(board):
    """Handle the human player's turn"""
    valid_move = False
    col = None
    
    while not valid_move:
        try:
            col = int(input(f"Your turn (choose column 1-{COLS}): ")) - 1
            
            # Check if the input is valid
            if col < 0 or col >= COLS:
                print(f"Please enter a number between 1 and {COLS}.")
                continue
                
            # Check if the column is full
            if not is_valid_location(board, col):
                print("That column is full. Choose another.")
                continue
                
            valid_move = True
            
        except ValueError:
            print("Please enter a valid number.")
    
    row = get_next_open_row(board, col)
    drop_piece(board, row, col, HUMAN_SYMBOL)
    return row, col

def ai_turn(board):
    """Handle the AI player's turn"""
    print("AI is thinking...")
    col = what_is_your_move(board, ROWS, COLS, AI_SYMBOL)
    
    # Convert from 1-indexed to 0-indexed
    col = col - 1
    
    row = get_next_open_row(board, col)
    drop_piece(board, row, col, AI_SYMBOL)
    print(f"AI chose column {col + 1}")
    return row, col

def play_game():
    """Main game loop"""
    board = create_board()
    
    # Initialize the AI agent
    init_agent(AI_SYMBOL, ROWS, COLS, board)
    
    game_over = False
    turn = 0  # 0 for human, 1 for AI
    
    print("\nWelcome to Connect 4!")
    print(f"You are playing as '{HUMAN_SYMBOL}' and the AI is playing as '{AI_SYMBOL}'.")
    print("To make a move, enter the column number (1-7).")
    
    print_board(board)
    
    # Decide who goes first
    choice = input("Do you want to go first? (y/n): ").lower()
    if choice != 'y':
        turn = 1
        print("AI will go first.")
    else:
        print("You will go first.")
    
    # Game loop
    while not game_over:
        if turn == 0:  # Human's turn
            row, col = human_turn(board)
            
            print_board(board)
            
            if check_win(board, HUMAN_SYMBOL):
                print("Congratulations! You won!")
                game_over = True
                
        else:  # AI's turn
            row, col = ai_turn(board)
            
            print_board(board)
            
            if check_win(board, AI_SYMBOL):
                print("The AI won! Better luck next time.")
                game_over = True
        
        # Check for a tie
        if not game_over and is_board_full(board):
            print("It's a tie!")
            game_over = True
            
        # Switch turns
        turn = 1 - turn
    
    # Ask to play again
    play_again = input("Do you want to play again? (y/n): ").lower()
    if play_again == 'y':
        play_game()
    else:
        print("Thanks for playing!")

if __name__ == "__main__":
    play_game()