#!/usr/bin/env python3
"""
Connect 4 Search Algorithm Implementation
CS156 Final Project - Spring 2025

This module implements Minimax and Alpha-Beta pruning search algorithms
for the Connect 4 game.
"""

import math
from reasoning import evaluate_board, is_terminal_state, get_valid_moves

# Constants for evaluation
INFINITY = float('inf')
NEG_INFINITY = float('-inf')

def minimax_search(game_state, max_depth):
    """
    Implements the Minimax search algorithm to find the best move.
    
    Args:
        game_state: Current state of the game (frame-based representation)
        max_depth: Maximum depth to search
        
    Returns:
        The best column to play (1-indexed)
    """
    best_score = NEG_INFINITY
    best_move = None
    
    valid_moves = get_valid_moves(game_state.board, game_state.num_rows, game_state.num_cols)
    
    # Try each possible move
    for move in valid_moves:
        # Create a copy of the game state for simulation
        new_state = game_state.clone()
        
        # Apply the move
        new_state.make_move(move, new_state.player_symbol)
        
        # Evaluate this move using minimax recursion
        score = min_value(new_state, max_depth - 1)
        
        # Update best move if found
        if score > best_score:
            best_score = score
            best_move = move
    
    # Return the best found move
    return best_move if best_move is not None else valid_moves[0]

def min_value(game_state, depth):
    """
    Minimizing player's turn in minimax.
    
    Args:
        game_state: Current state of the game
        depth: Remaining depth to search
        
    Returns:
        Minimum score of possible moves
    """
    # Check if we're at a terminal state
    if is_terminal_state(game_state.board, game_state.num_rows, game_state.num_cols):
        return evaluate_board(game_state.board, game_state.player_symbol,
                             game_state.opponent_symbol, game_state.num_rows, 
                             game_state.num_cols)
    
    # If depth limit reached, evaluate current state
    if depth <= 0:
        return evaluate_board(game_state.board, game_state.player_symbol,
                             game_state.opponent_symbol, game_state.num_rows, 
                             game_state.num_cols)
    
    value = INFINITY
    valid_moves = get_valid_moves(game_state.board, game_state.num_rows, game_state.num_cols)
    
    # Try each possible move
    for move in valid_moves:
        # Create a copy of the game state for simulation
        new_state = game_state.clone()
        
        # Apply the opponent's move
        new_state.make_move(move, new_state.opponent_symbol)
        
        # Recursively find the minimum value
        value = min(value, max_value(new_state, depth - 1))
    
    return value

def max_value(game_state, depth):
    """
    Maximizing player's turn in minimax.
    
    Args:
        game_state: Current state of the game
        depth: Remaining depth to search
        
    Returns:
        Maximum score of possible moves
    """
    # Check if we're at a terminal state
    if is_terminal_state(game_state.board, game_state.num_rows, game_state.num_cols):
        return evaluate_board(game_state.board, game_state.player_symbol,
                             game_state.opponent_symbol, game_state.num_rows, 
                             game_state.num_cols)
    
    # If depth limit reached, evaluate current state
    if depth <= 0:
        return evaluate_board(game_state.board, game_state.player_symbol,
                             game_state.opponent_symbol, game_state.num_rows, 
                             game_state.num_cols)
    
    value = NEG_INFINITY
    valid_moves = get_valid_moves(game_state.board, game_state.num_rows, game_state.num_cols)
    
    # Try each possible move
    for move in valid_moves:
        # Create a copy of the game state for simulation
        new_state = game_state.clone()
        
        # Apply our move
        new_state.make_move(move, new_state.player_symbol)
        
        # Recursively find the maximum value
        value = max(value, min_value(new_state, depth - 1))
    
    return value

def alpha_beta_search(game_state, max_depth):
    """
    Implements Minimax with Alpha-Beta pruning to find the best move more efficiently.
    
    Args:
        game_state: Current state of the game (frame-based representation)
        max_depth: Maximum depth to search
        
    Returns:
        The best column to play (1-indexed)
    """
    best_score = NEG_INFINITY
    best_move = None
    alpha = NEG_INFINITY
    beta = INFINITY
    
    valid_moves = get_valid_moves(game_state.board, game_state.num_rows, game_state.num_cols)
    
    # Check if center column is available for the first few moves
    move_count = sum(1 for r in range(game_state.num_rows) 
                     for c in range(game_state.num_cols) if game_state.board[r][c] != ' ')
    
    if move_count < 2:
        # Prioritize center column(s) for early moves
        center_col = game_state.num_cols // 2 + 1  # 1-indexed
        if center_col in valid_moves:
            return center_col
    
    # Try each possible move with alpha-beta pruning
    for move in valid_moves:
        # Create a copy of the game state for simulation
        new_state = game_state.clone()
        
        # Apply the move
        new_state.make_move(move, new_state.player_symbol)
        
        # Evaluate with alpha-beta pruning
        score = ab_min_value(new_state, max_depth - 1, alpha, beta)
        
        # Update best move if found
        if score > best_score:
            best_score = score
            best_move = move
        
        # Update alpha value
        alpha = max(alpha, best_score)
    
    # Return the best found move
    return best_move if best_move is not None else valid_moves[0]

def ab_min_value(game_state, depth, alpha, beta):
    """
    Minimizing player's turn in alpha-beta pruning.
    
    Args:
        game_state: Current state of the game
        depth: Remaining depth to search
        alpha: Alpha value for pruning
        beta: Beta value for pruning
        
    Returns:
        Minimum score of possible moves
    """
    # Check if we're at a terminal state
    if is_terminal_state(game_state.board, game_state.num_rows, game_state.num_cols):
        return evaluate_board(game_state.board, game_state.player_symbol,
                             game_state.opponent_symbol, game_state.num_rows, 
                             game_state.num_cols)
    
    # If depth limit reached, evaluate current state
    if depth <= 0:
        return evaluate_board(game_state.board, game_state.player_symbol,
                             game_state.opponent_symbol, game_state.num_rows, 
                             game_state.num_cols)
    
    value = INFINITY
    valid_moves = get_valid_moves(game_state.board, game_state.num_rows, game_state.num_cols)
    
    # Try each possible move with pruning
    for move in valid_moves:
        # Create a copy of the game state for simulation
        new_state = game_state.clone()
        
        # Apply the opponent's move
        new_state.make_move(move, new_state.opponent_symbol)
        
        # Recursively find minimum with pruning
        value = min(value, ab_max_value(new_state, depth - 1, alpha, beta))
        
        # Beta pruning
        if value <= alpha:
            return value
        
        # Update beta value
        beta = min(beta, value)
    
    return value

def ab_max_value(game_state, depth, alpha, beta):
    """
    Maximizing player's turn in alpha-beta pruning.
    
    Args:
        game_state: Current state of the game
        depth: Remaining depth to search
        alpha: Alpha value for pruning
        beta: Beta value for pruning
        
    Returns:
        Maximum score of possible moves
    """
    # Check if we're at a terminal state
    if is_terminal_state(game_state.board, game_state.num_rows, game_state.num_cols):
        return evaluate_board(game_state.board, game_state.player_symbol,
                             game_state.opponent_symbol, game_state.num_rows, 
                             game_state.num_cols)
    
    # If depth limit reached, evaluate current state
    if depth <= 0:
        return evaluate_board(game_state.board, game_state.player_symbol,
                             game_state.opponent_symbol, game_state.num_rows, 
                             game_state.num_cols)
    
    value = NEG_INFINITY
    valid_moves = get_valid_moves(game_state.board, game_state.num_rows, game_state.num_cols)
    
    # Try each possible move with pruning
    for move in valid_moves:
        # Create a copy of the game state for simulation
        new_state = game_state.clone()
        
        # Apply our move
        new_state.make_move(move, new_state.player_symbol)
        
        # Recursively find maximum with pruning
        value = max(value, ab_min_value(new_state, depth - 1, alpha, beta))
        
        # Alpha pruning
        if value >= beta:
            return value
        
        # Update alpha value
        alpha = max(alpha, value)
    
    return value