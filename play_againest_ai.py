#!/usr/bin/env python3
"""
Human vs AI Connect 4 Game
This script allows a human player to play against the Connect 4 AI agent.
"""

import sys
import os
import importlib

# Constants
ROWS = 6
COLS = 7
HUMAN_SYMBOL = 'O'
AI_SYMBOL = 'X'

def create_board():
    return [[' ' for _ in range(COLS)] for _ in range(ROWS)]

def print_board(board):
    print("\n  ", end="")
    for c in range(COLS):
        print(f" {c+1} ", end="")
    print("\n")
    
    print("  " + "=" * (3 * COLS + 1))
    
    for r in range(ROWS):
        print(f"{r+1} |", end="")
        for c in range(COLS):
            cell = board[r][c]
            print(f" {cell} " if cell != ' ' else "   ", end="")
        print("|")
    
    print("  " + "=" * (3 * COLS + 1))
    print("  ", end="")
    for c in range(COLS):
        print(f" {c+1} ", end="")
    print("\n")

def is_valid_location(board, col):
    return board[0][col] == ' '

def get_next_open_row(board, col):
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == ' ':
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

def is_board_full(board):
    return all(cell != ' ' for row in board for cell in row)

def human_turn(board):
    while True:
        try:
            col = int(input(f"Your turn (choose column 1-{COLS}): ")) - 1
            if col < 0 or col >= COLS:
                print(f"Please enter a number between 1 and {COLS}.")
                continue
            if not is_valid_location(board, col):
                print("That column is full. Choose another.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    row = get_next_open_row(board, col)
    drop_piece(board, row, col, HUMAN_SYMBOL)
    return row, col

def ai_turn(board, ai_module):
    print("AI is thinking...")
    col = ai_module.what_is_your_move(board, ROWS, COLS, AI_SYMBOL)
    col -= 1
    row = get_next_open_row(board, col)
    drop_piece(board, row, col, AI_SYMBOL)
    print(f"AI chose column {col + 1}")
    return row, col

def choose_ai_team():
    while True:
        team = input("Which team do you want to play against (e.g., Team1, Team2, Team5)? ").strip()
        module_name = f"{team}_Connect_4_Agent"
        try:
            ai_module = importlib.import_module(module_name)
            print(f"Loaded {module_name}.")
            return ai_module
        except ModuleNotFoundError:
            print(f"Could not find module '{module_name}'. Please try again.")

def play_game():
    board = create_board()
    
    ai_module = choose_ai_team()
    ai_module.init_agent(AI_SYMBOL, ROWS, COLS, board)
    
    game_over = False
    turn = 0

    print("\nWelcome to Connect 4!")
    print(f"You are playing as '{HUMAN_SYMBOL}' and the AI is playing as '{AI_SYMBOL}'.")
    print("To make a move, enter the column number (1-7).")

    print_board(board)
    
    if input("Do you want to go first? (y/n): ").lower() != 'y':
        turn = 1
        print("AI will go first.")
    else:
        print("You will go first.")

    while not game_over:
        if turn == 0:
            row, col = human_turn(board)
            print_board(board)
            if check_win(board, HUMAN_SYMBOL):
                print("Congratulations! You won!")
                game_over = True
        else:
            row, col = ai_turn(board, ai_module)
            print_board(board)
            if check_win(board, AI_SYMBOL):
                print("The AI won! Better luck next time.")
                game_over = True

        if not game_over and is_board_full(board):
            print("It's a tie!")
            game_over = True

        turn = 1 - turn

    if input("Do you want to play again? (y/n): ").lower() == 'y':
        play_game()
    else:
        print("Thanks for playing!")

if __name__ == "__main__":
    play_game()
