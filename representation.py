#!/usr/bin/env python3
"""
Connect 4 Reasoning Implementation
CS156 Final Project - Spring 2025

This module implements heuristic evaluation and rule-based inference
for the Connect 4 game.
"""

# Evaluation weights
WEIGHTS = {
    'win': 1000000,             # Winning position
    'lose': -1000000,           # Losing position
    'three_in_row': 100,        # Three in a row with open spot
    'two_in_row': 10,           # Two in a row with open spots
    'center_control': 5,        # Control of center columns
    'defensive_block': 80,      # Blocking opponent's threats
    'defensive_prevent': 60,    # Preventing opponent's setups
    'trap_setup': 70,           # Setting up a trap (multiple threats)
    'potential_wins': 50,       # Number of potential winning paths
}

def get_valid_moves(board, rows, cols):
    """
    Determine the list of valid moves (non-full columns).
    
    Args:
        board: Current board state
        rows: Number of rows
        cols: Number of columns
        
    Returns:
        List of valid column numbers (1-indexed)
    """
    valid_moves = []
    
    for col in range(cols):
        # Check if column is not full
        if board[0][col] == ' ':
            valid_moves.append(col + 1)  # 1-indexed
    
    return valid_moves

def is_terminal_state(board, rows, cols):
    """
    Check if the current state is terminal (game over).
    
    Args:
        board: Current board state
        rows: Number of rows
        cols: Number of columns
        
    Returns:
        True if game is over, False otherwise
    """
    # Check for a win
    if check_win(board, rows, cols, 'X') or check_win(board, rows, cols, 'O'):
        return True
    
    # Check if board is full
    for col in range(cols):
        if board[0][col] == ' ':
            return False
    
    return True

def check_win(board, rows, cols, symbol):
    """
    Check if the given symbol has won.
    
    Args:
        board: Current board state
        rows: Number of rows
        cols: Number of columns
        symbol: Symbol to check for win ('X' or 'O')
        
    Returns:
        True if there's a win, False otherwise
    """
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

def count_consecutive(board, rows, cols, symbol, count):
    """
    Count patterns of the given symbol with the specified count.
    
    Args:
        board: Current board state
        rows: Number of rows
        cols: Number of columns
        symbol: Symbol to count ('X' or 'O')
        count: Number of consecutive symbols to search for
        
    Returns:
        Number of occurrences of the pattern
    """
    pattern_count = 0
    
    # Check horizontal
    for row in range(rows):
        for col in range(cols - 3):
            window = [board[row][col+i] for i in range(4)]
            if window.count(symbol) == count and window.count(' ') == 4 - count:
                pattern_count += 1
    
    # Check vertical
    for col in range(cols):
        for row in range(rows - 3):
            window = [board[row+i][col] for i in range(4)]
            if window.count(symbol) == count and window.count(' ') == 4 - count:
                pattern_count += 1
    
    # Check diagonal up
    for row in range(3, rows):
        for col in range(cols - 3):
            window = [board[row-i][col+i] for i in range(4)]
            if window.count(symbol) == count and window.count(' ') == 4 - count:
                pattern_count += 1
    
    # Check diagonal down
    for row in range(rows - 3):
        for col in range(cols - 3):
            window = [board[row+i][col+i] for i in range(4)]
            if window.count(symbol) == count and window.count(' ') == 4 - count:
                pattern_count += 1
    
    return pattern_count

def count_potential_wins(board, rows, cols, symbol):
    """
    Count the number of potential winning paths for the given symbol.
    
    Args:
        board: Current board state
        rows: Number of rows
        cols: Number of columns
        symbol: Symbol to check for ('X' or 'O')
        
    Returns:
        Number of potential winning paths
    """
    # Count empty spaces that could lead to a win
    potential_wins = 0
    
    # Try placing a piece in each column and see if it creates a win
    for col in range(cols):
        # Find the row where a piece would be placed
        row = -1
        for r in range(rows-1, -1, -1):
            if board[r][col] == ' ':
                row = r
                break
        
        # If column is full, skip
        if row == -1:
            continue
        
        # Create a temporary board with the new piece
        temp_board = [row[:] for row in board]
        temp_board[row][col] = symbol
        
        # Check if this creates a win
        if check_win(temp_board, rows, cols, symbol):
            potential_wins += 1
    
    return potential_wins

def detect_traps(board, rows, cols, symbol):
    """
    Detect if there are trap setups (multiple simultaneous threats).
    
    Args:
        board: Current board state
        rows: Number of rows
        cols: Number of columns
        symbol: Symbol to check for ('X' or 'O')
        
    Returns:
        Number of trap setups detected
    """
    opponent = 'O' if symbol == 'X' else 'X'
    trap_count = 0
    
    # For each column, check if placing a piece creates multiple threats
    for col in range(cols):
        # Find the row where a piece would be placed
        row = -1
        for r in range(rows-1, -1, -1):
            if board[r][col] == ' ':
                row = r
                break
        
        # If column is full, skip
        if row == -1:
            continue
        
        # Create a temporary board with the new piece
        temp_board = [row[:] for row in board]
        temp_board[row][col] = symbol
        
        # Count threats after this move
        threats = 0
        
        # Check if there are multiple ways to win after this move
        for test_col in range(cols):
            if test_col == col:
                continue
                
            # Find the row where the next piece would be placed
            test_row = -1
            for r in range(rows-1, -1, -1):
                if temp_board[r][test_col] == ' ':
                    test_row = r
                    break
            
            # If column is full, skip
            if test_row == -1:
                continue
            
            # Create a second temporary board with another piece
            temp_board2 = [row[:] for row in temp_board]
            temp_board2[test_row][test_col] = symbol
            
            # Check if this creates a win
            if check_win(temp_board2, rows, cols, symbol):
                threats += 1
        
        # If multiple threats are created, it's a trap
        if threats > 1:
            trap_count += 1
    
    return trap_count

def evaluate_board(board, player_symbol, opponent_symbol, rows, cols):
    """
    Evaluate the board state using heuristics.
    
    Args:
        board: Current board state
        player_symbol: Symbol of the AI player
        opponent_symbol: Symbol of the opponent
        rows: Number of rows
        cols: Number of columns
        
    Returns:
        Numeric score representing the board state
    """
    score = 0
    
    # Check for terminal states first
    if check_win(board, rows, cols, player_symbol):
        return WEIGHTS['win']
    
    if check_win(board, rows, cols, opponent_symbol):
        return WEIGHTS['lose']
    
    # Check for patterns
    player_threes = count_consecutive(board, rows, cols, player_symbol, 3)
    player_twos = count_consecutive(board, rows, cols, player_symbol, 2)
    opponent_threes = count_consecutive(board, rows, cols, opponent_symbol, 3)
    opponent_twos = count_consecutive(board, rows, cols, opponent_symbol, 2)
    
    # Score patterns
    score += player_threes * WEIGHTS['three_in_row']
    score += player_twos * WEIGHTS['two_in_row']
    score -= opponent_threes * WEIGHTS['three_in_row']
    score -= opponent_twos * WEIGHTS['two_in_row']
    
    # Score center control (middle columns are more valuable)
    center_col = cols // 2
    center_count = 0
    
    # If even number of columns, consider both center columns
    if cols % 2 == 0:
        for row in range(rows):
            if board[row][center_col] == player_symbol:
                center_count += 1
            if board[row][center_col-1] == player_symbol:
                center_count += 1
    else:
        for row in range(rows):
            if board[row][center_col] == player_symbol:
                center_count += 1
    
    score += center_count * WEIGHTS['center_control']
    
    # Score potential wins
    player_potential_wins = count_potential_wins(board, rows, cols, player_symbol)
    opponent_potential_wins = count_potential_wins(board, rows, cols, opponent_symbol)
    
    score += player_potential_wins * WEIGHTS['potential_wins']
    score -= opponent_potential_wins * WEIGHTS['potential_wins']
    
    # Score trap setups
    player_traps = detect_traps(board, rows, cols, player_symbol)
    opponent_traps = detect_traps(board, rows, cols, opponent_symbol)
    
    score += player_traps * WEIGHTS['trap_setup']
    score -= opponent_traps * WEIGHTS['trap_setup']
    
    # Apply defensive reasoning - block opponent's immediate threats
    if opponent_threes > 0:
        score += WEIGHTS['defensive_block']
    
    # Apply defensive reasoning - prevent opponent's potential traps
    if opponent_twos > 1:
        score += WEIGHTS['defensive_prevent']
    
    return score

def get_immediate_threats(board, rows, cols, symbol):
    """
    Identify columns that contain immediate threats.
    
    Args:
        board: Current board state
        rows: Number of rows
        cols: Number of columns
        symbol: Symbol to check for threats against
        
    Returns:
        List of column numbers (1-indexed) that need to be blocked
    """
    opponent = 'O' if symbol == 'X' else 'X'
    threat_columns = []
    
    # Check each column to see if it's a winning move for the opponent
    for col in range(cols):
        # Find the row where a piece would be placed
        row = -1
        for r in range(rows-1, -1, -1):
            if board[r][col] == ' ':
                row = r
                break
        
        # If column is full, skip
        if row == -1:
            continue
        
        # Create a temporary board with the opponent's piece
        temp_board = [row[:] for row in board]
        temp_board[row][col] = opponent
        
        # Check if this creates a win for the opponent
        if check_win(temp_board, rows, cols, opponent):
            threat_columns.append(col + 1)  # 1-indexed
    
    return threat_columns

def get_winning_move(board, rows, cols, symbol):
    """
    Identify a winning move if it exists.
    
    Args:
        board: Current board state
        rows: Number of rows
        cols: Number of columns
        symbol: Symbol to check for winning move
        
    Returns:
        Column number (1-indexed) for a winning move, or None if none exists
    """
    # Check each column to see if it's a winning move
    for col in range(cols):
        # Find the row where a piece would be placed
        row = -1
        for r in range(rows-1, -1, -1):
            if board[r][col] == ' ':
                row = r
                break
        
        # If column is full, skip
        if row == -1:
            continue
        
        # Create a temporary board with the player's piece
        temp_board = [row[:] for row in board]
        temp_board[row][col] = symbol
        
        # Check if this creates a win
        if check_win(temp_board, rows, cols, symbol):
            return col + 1  # 1-indexed
    
    return None

def select_best_move(board, rows, cols, player_symbol, opponent_symbol, max_depth=5):
    """
    Select the best move using rule-based inference.
    
    This is a simpler alternative to the full minimax search,
    using rule-based logic for quick decision making.
    
    Args:
        board: Current board state
        rows: Number of rows
        cols: Number of columns
        player_symbol: Symbol of the AI player
        opponent_symbol: Symbol of the opponent
        max_depth: Maximum search depth
        
    Returns:
        Column number (1-indexed) for the best move
    """
    # Get valid moves
    valid_moves = get_valid_moves(board, rows, cols)
    
    if not valid_moves:
        return None
    
    # Rule 1: If we can win, do it
    winning_move = get_winning_move(board, rows, cols, player_symbol)
    if winning_move is not None and winning_move in valid_moves:
        return winning_move
    
    # Rule 2: If opponent can win, block it
    threats = get_immediate_threats(board, rows, cols, player_symbol)
    if threats and threats[0] in valid_moves:
        return threats[0]
    
    # Rule 3: Prefer center columns in early game
    move_count = sum(1 for r in range(rows) for c in range(cols) if board[r][c] != ' ')
    
    if move_count < 6:
        center_col = cols // 2 + 1  # 1-indexed
        
        # If even number of columns, consider both center columns
        if cols % 2 == 0:
            center_cols = [center_col, center_col - 1]
            for col in center_cols:
                if col in valid_moves:
                    return col
        else:
            if center_col in valid_moves:
                return center_col
    
    # Rule 4: Find move with best evaluation score
    best_score = float('-inf')
    best_move = valid_moves[0]
    
    for move in valid_moves:
        # Create a temporary board with this move
        temp_board = [row[:] for row in board]
        
        # Find the row where the piece would be placed
        row = -1
        for r in range(rows-1, -1, -1):
            if temp_board[r][move-1] == ' ':  # Convert to 0-indexed
                row = r
                break
        
        # Apply the move
        temp_board[row][move-1] = player_symbol
        
        # Evaluate the resulting board
        score = evaluate_board(temp_board, player_symbol, opponent_symbol, rows, cols)
        
        # Update best move if better
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move