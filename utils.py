#!/usr/bin/env python3
"""
Connect 4 Utility Functions
CS156 Final Project - Spring 2025

This module provides utility functions for the Connect 4 AI agent.
"""

import random
import copy
import time

def print_board_pretty(board):
    """
    Print the board in a user-friendly format.
    
    Args:
        board: The board to print
    """
    rows = len(board)
    cols = len(board[0])
    
    # Print column numbers
    print("\n  ", end="")
    for c in range(cols):
        print(f" {c+1} ", end="")
    print("\n")
    
    # Print top border
    print("  " + "=" * (3 * cols + 1))
    
    # Print board
    for r in range(rows):
        print(f"{r+1} |", end="")
        for c in range(cols):
            cell = board[r][c]
            if cell == ' ':
                print("   ", end="")
            else:
                print(f" {cell} ", end="")
        print("|")
    
    # Print bottom border
    print("  " + "=" * (3 * cols + 1))
    
    # Print column numbers again
    print("  ", end="")
    for c in range(cols):
        print(f" {c+1} ", end="")
    print("\n")

def simulate_move(board, col, symbol):
    """
    Simulate a move on a copy of the board.
    
    Args:
        board: Current board state
        col: Column to place the piece (0-indexed)
        symbol: Symbol to place ('X' or 'O')
        
    Returns:
        New board with the move applied, or None if the move is invalid
    """
    # Create a copy of the board
    new_board = copy.deepcopy(board)
    rows = len(board)
    
    # Check if column is valid
    if col < 0 or col >= len(board[0]):
        return None
    
    # Check if column is full
    if board[0][col] != ' ':
        return None
    
    # Find the first empty slot from the bottom up
    for row in range(rows - 1, -1, -1):
        if board[row][col] == ' ':
            new_board[row][col] = symbol
            return new_board
    
    return None

def measure_execution_time(func, *args, **kwargs):
    """
    Measure the execution time of a function.
    
    Args:
        func: Function to measure
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Tuple of (result, execution_time)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    return result, end_time - start_time

def get_column_heights(board):
    """
    Calculate the height of each column in the board.
    
    Args:
        board: Current board state
        
    Returns:
        List of heights for each column (number of pieces)
    """
    rows = len(board)
    cols = len(board[0])
    heights = [0] * cols
    
    for c in range(cols):
        for r in range(rows-1, -1, -1):
            if board[r][c] != ' ':
                heights[c] = rows - r
                break
    
    return heights

def is_winning_position(board, symbol):
    """
    Check if the given symbol has a winning position.
    
    Args:
        board: Current board state
        symbol: Symbol to check ('X' or 'O')
        
    Returns:
        True if the symbol has a winning position, False otherwise
    """
    rows = len(board)
    cols = len(board[0])
    
    # Check horizontal
    for row in range(rows):
        for col in range(cols - 3):
            if all(board[row][col+i] == symbol for i in range(4)):
                return True
    
    # Check vertical
    for col in range(cols):
        for row in range(rows - 3):
            if all(board[row+i][col] == symbol for i in range(4)):
                return True
    
    # Check diagonal up
    for row in range(3, rows):
        for col in range(cols - 3):
            if all(board[row-i][col+i] == symbol for i in range(4)):
                return True
    
    # Check diagonal down
    for row in range(rows - 3):
        for col in range(cols - 3):
            if all(board[row+i][col+i] == symbol for i in range(4)):
                return True
    
    return False

def get_possible_moves(board):
    """
    Get a list of columns that have at least one empty cell.
    
    Args:
        board: Current board state
        
    Returns:
        List of valid column indices (0-indexed)
    """
    cols = len(board[0])
    return [col for col in range(cols) if board[0][col] == ' ']

def board_is_full(board):
    """
    Check if the board is full.
    
    Args:
        board: Current board state
        
    Returns:
        True if the board is full, False otherwise
    """
    return all(cell != ' ' for row in board for cell in row)

def get_random_move(board):
    """
    Select a random valid move.
    
    Args:
        board: Current board state
        
    Returns:
        A random valid column index (0-indexed), or None if no valid moves
    """
    valid_moves = get_possible_moves(board)
    if not valid_moves:
        return None
    return random.choice(valid_moves)

def drop_piece(board, col, symbol):
    """
    Drop a piece in the specified column.
    
    Args:
        board: Current board state
        col: Column to drop the piece (0-indexed)
        symbol: Symbol to drop ('X' or 'O')
        
    Returns:
        Row where the piece landed, or -1 if the column is full
    """
    rows = len(board)
    
    for row in range(rows-1, -1, -1):
        if board[row][col] == ' ':
            board[row][col] = symbol
            return row
    
    return -1  # Column is full