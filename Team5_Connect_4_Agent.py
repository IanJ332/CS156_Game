#!/usr/bin/env python3
"""
Team5 Connect 4 AI Agent with Permanent Brain
CS156 Final Project - Spring 2025

Features:
1. Permanent Brain (opponent-turn thinking) - continues analysis during opponent's turn
2. Enhanced threat detection and prioritization
3. Optimized evaluation function
4. Move ordering for alpha-beta efficiency
5. Pattern recognition and opening book
"""

import random
import time
import copy
import threading
import queue

# Configuration
MAX_DEPTH = 9  # Maximum search depth
TIME_LIMIT = 1.8  # Time limit in seconds for regular search
BACKGROUND_THINKING = True  # Enable permanent brain / opponent-turn thinking
PERMANENT_BRAIN_DEPTH = 12  # Maximum depth for background thinking
USE_OPENING_BOOK = True  # Use pre-computed opening moves
USE_TRANSPOSITION_TABLE = True  # Cache evaluated positions
ITERATIVE_DEEPENING = True  # Use iterative deepening
DEBUG_MODE = False  # Set to True to enable debug prints

# Global variables
game_state = None
player_symbol = None
opponent_symbol = None
transposition_table = {}  # Cache for evaluated positions
permanent_brain_results = {}  # Store results from background thinking
last_board_state = None  # Store the last seen board state
background_thread = None  # Thread for opponent-turn thinking
thinking_active = False  # Flag to control background thinking
search_queue = queue.Queue()  # Queue for communicating with background thread
start_time = 0  # For tracking search time
nodes_evaluated = 0  # For performance statistics

# Opening book - maps board state hash to preferred move
OPENING_BOOK = {
    # Empty board - take center
    "empty": 4,  # (1-indexed: 4 is center column for 7-column board)
    
    # Common early game patterns and responses
    # Format: hash(board) -> best_column
    "center_first": 4,  # If opponent takes center, also take center column
    "side_first": 4,    # If opponent takes side, take center
    "3_3": 4,           # If opponent tries 3-3 opening, block with center
}

# Evaluation weights
WEIGHTS = {
    'win': 1000000,            # Winning position
    'lose': -1000000,          # Losing position
    'three_open': 5000,        # Three in a row with open spot
    'three_blocked': 1000,     # Three in a row that needs blocking
    'two_open': 200,           # Two in a row with two open spots
    'center_control': 30,      # Control of center columns
    'defensive_block': 8000,   # Blocking opponent's threats
    'defensive_prevent': 2000, # Preventing opponent's setups
    'trap_setup': 3000,        # Setting up a trap (multiple threats)
    'double_threat': 10000,    # Creating two threats at once
    'potential_wins': 500,     # Number of potential winning paths
    'bottom_row': 20,          # Bonus for controlling bottom row pieces
    'adjacency': 10,           # Bonus for pieces adjacent to our pieces
}

def print_board(board):
    """Utility function to print the board for debugging"""
    if not DEBUG_MODE:
        return
        
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

def board_to_hash(board):
    """Convert board to a string hash for the transposition table"""
    return ''.join(''.join(row) for row in board)

def init_agent(player_sym, board_num_rows, board_num_cols, board):
    """
    Initializes the agent. Called once at the start of a game.
    
    Args:
        player_sym: The symbol representing this agent ('X' or 'O')
        board_num_rows: Number of rows in the board
        board_num_cols: Number of columns in the board
        board: The initial board state
        
    Returns:
        True indicating successful initialization
    """
    global game_state, player_symbol, opponent_symbol, transposition_table
    global last_board_state, background_thread, thinking_active
    
    # Store the player's symbol and determine the opponent's symbol
    player_symbol = player_sym
    opponent_symbol = 'O' if player_sym == 'X' else 'X'
    
    # Initialize the game state
    game_state = {
        'board': copy.deepcopy(board),
        'num_rows': int(board_num_rows),
        'num_cols': int(board_num_cols),
        'player_symbol': player_symbol,
        'opponent_symbol': opponent_symbol,
        'column_heights': calculate_column_heights(board, int(board_num_rows)),
        'move_count': 0
    }
    
    # Store current board state
    last_board_state = copy.deepcopy(board)
    
    # Clear caches
    transposition_table = {}
    permanent_brain_results = {}
    
    # Initialize background thinking thread if enabled
    if BACKGROUND_THINKING:
        thinking_active = True
        background_thread = threading.Thread(target=permanent_brain_worker, daemon=True)
        background_thread.start()
    
    if DEBUG_MODE:
        print(f"Agent initialized as player {player_symbol}")
        print(f"Permanent Brain: {'Enabled' if BACKGROUND_THINKING else 'Disabled'}")
        print_board(board)
    
    return True

def permanent_brain_worker():
    """
    Worker function for the background thinking thread.
    Continuously processes thinking tasks from the queue.
    """
    global thinking_active, permanent_brain_results
    
    while thinking_active:
        try:
            # Get a board state to analyze from the queue (with timeout)
            task = search_queue.get(timeout=0.1)
            
            if task is None:
                # Special signal to stop thread
                break
                
            board, rows, cols, valid_moves = task
            board_hash = board_to_hash(board)
            
            # Skip if we already have results for this board
            if board_hash in permanent_brain_results:
                search_queue.task_done()
                continue
                
            # Use longer time limit for background thinking
            thinking_start_time = time.time()
            thinking_time_limit = 10.0  # Much longer than regular search
            
            results = {}
            # Perform iterative deepening for all possible opponent moves
            for depth in range(1, PERMANENT_BRAIN_DEPTH + 1):
                # Check if we've been thinking too long
                if time.time() - thinking_start_time > thinking_time_limit:
                    break
                    
                # Analyze each valid move at this depth
                for move in valid_moves:
                    # Create a copy of the board and apply the move
                    new_board = copy.deepcopy(board)
                    row = make_move(new_board, move - 1, player_symbol)
                    
                    if row == -1:  # Invalid move
                        continue
                        
                    # Evaluate position with alpha-beta
                    score = min_value(new_board, rows, cols, depth - 1, float('-inf'), float('inf'),
                                     thinking_start_time, thinking_time_limit)
                    
                    # Store result for this move at this depth
                    if move not in results or results[move]['depth'] < depth:
                        results[move] = {
                            'score': score,
                            'depth': depth
                        }
            
            # Save the final results
            permanent_brain_results[board_hash] = {
                'best_move': max(results.items(), key=lambda x: x[1]['score'])[0] if results else None,
                'all_moves': results,
                'timestamp': time.time()
            }
            
            search_queue.task_done()
            
        except queue.Empty:
            # No tasks in queue, just continue waiting
            pass
        except Exception as e:
            if DEBUG_MODE:
                print(f"Error in background thinking: {e}")
            
            # Continue processing other tasks
            continue

def detect_opponent_move(board):
    """
    Detect what move the opponent made since our last turn.
    This is used to prioritize analysis of the opponent's actual move.
    
    Returns:
        (row, col) of the opponent's move, or None if can't determine
    """
    global last_board_state
    
    if last_board_state is None:
        return None
        
    # Find the difference between current board and last board
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c] != last_board_state[r][c] and board[r][c] == opponent_symbol:
                return (r, c)
                
    return None

def schedule_background_thinking(board, rows, cols):
    """
    Schedule background thinking for the current board state.
    Analyzes all possible moves the opponent might make.
    """
    global search_queue, last_board_state
    
    if not BACKGROUND_THINKING or not thinking_active:
        return
    
    # Get valid moves
    valid_moves = get_valid_moves(board, rows, cols)
    
    # Queue the current board for analysis
    try:
        search_queue.put((copy.deepcopy(board), rows, cols, valid_moves), block=False)
    except queue.Full:
        # Queue is full, just continue
        pass
    
    # Update the last board state
    last_board_state = copy.deepcopy(board)

def calculate_column_heights(board, num_rows):
    """Calculate the height of each column (number of pieces)"""
    heights = [0] * len(board[0])
    
    for col in range(len(board[0])):
        for row in range(num_rows - 1, -1, -1):
            if board[row][col] != ' ':
                heights[col] = num_rows - row
                break
    
    return heights

def get_valid_moves(board, num_rows, num_cols):
    """Get list of valid moves (non-full columns)"""
    valid_moves = []
    
    for col in range(num_cols):
        # Check if column is not full
        if board[0][col] == ' ':
            valid_moves.append(col + 1)  # 1-indexed
    
    return valid_moves

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
    global game_state, player_symbol, opponent_symbol, start_time, nodes_evaluated
    global permanent_brain_results, last_board_state
    
    start_time = time.time()
    nodes_evaluated = 0
    
    # If game_state is None, reinitialize it
    if game_state is None:
        init_agent(my_game_symbol, game_rows, game_cols, board)
    
    # Update the game state with the current board
    game_state['board'] = copy.deepcopy(board)
    game_state['column_heights'] = calculate_column_heights(board, game_rows)
    game_state['move_count'] = sum(1 for r in range(game_rows) for c in range(game_cols) if board[r][c] != ' ')
    
    # Get valid moves
    valid_moves = get_valid_moves(board, game_rows, game_cols)
    
    if not valid_moves:
        if DEBUG_MODE:
            print("No valid moves found!")
        return 1  # Return a default move (will be rejected if invalid)
    
    # Check for results from permanent brain thinking
    board_hash = board_to_hash(board)
    has_permanent_brain_result = False
    permanent_brain_move = None
    
    if BACKGROUND_THINKING and board_hash in permanent_brain_results:
        result = permanent_brain_results[board_hash]
        if result['best_move'] in valid_moves:
            permanent_brain_move = result['best_move']
            has_permanent_brain_result = True
            if DEBUG_MODE:
                print(f"Using permanent brain result: {permanent_brain_move}")
    
    # Check for urgent moves (immediate wins and blocks) regardless of permanent brain
    urgent_move = find_urgent_move(board, game_rows, game_cols)
    if urgent_move is not None and urgent_move in valid_moves:
        if DEBUG_MODE:
            print(f"Found urgent move: {urgent_move}")
        
        # Schedule background thinking for next turn
        schedule_background_thinking(board, game_rows, game_cols)
        
        # Update the last board state
        last_board_state = copy.deepcopy(board)
        
        return urgent_move
    
    # Use permanent brain result if available and not overridden by urgent move
    if has_permanent_brain_result:
        # Schedule background thinking for next turn
        schedule_background_thinking(board, game_rows, game_cols)
        
        # Update the last board state
        last_board_state = copy.deepcopy(board)
        
        return permanent_brain_move
    
    # Check opening book for very early game (first 4 moves)
    if USE_OPENING_BOOK and game_state['move_count'] <= 4:
        book_move = check_opening_book(board, valid_moves)
        if book_move is not None:
            if DEBUG_MODE:
                print(f"Using opening book move: {book_move}")
            
            # Schedule background thinking for next turn
            schedule_background_thinking(board, game_rows, game_cols)
            
            # Update the last board state
            last_board_state = copy.deepcopy(board)
            
            return book_move
    
    # Determine best move using search
    if ITERATIVE_DEEPENING:
        best_move = iterative_deepening_search(board, game_rows, game_cols, valid_moves)
    else:
        best_move = alpha_beta_search(board, game_rows, game_cols, valid_moves, MAX_DEPTH)
    
    # Ensure the chosen move is valid
    if best_move not in valid_moves:
        if DEBUG_MODE:
            print(f"Search returned invalid move {best_move}, choosing randomly from: {valid_moves}")
        best_move = random.choice(valid_moves)
    
    end_time = time.time()
    
    if DEBUG_MODE:
        print(f"Chose column {best_move} (took {end_time - start_time:.3f} seconds)")
        print(f"Nodes evaluated: {nodes_evaluated}")
        print_board(board)
    
    # Schedule background thinking for next turn
    schedule_background_thinking(board, game_rows, game_cols)
    
    # Update the last board state
    last_board_state = copy.deepcopy(board)
    
    return best_move

def find_urgent_move(board, rows, cols):
    """
    Find urgent moves (immediate wins or blocks)
    Return priority: 1) Our win, 2) Block opponent win, 3) Block opponent double threat
    """
    # First check if we can win in one move
    winning_move = find_winning_move(board, rows, cols, player_symbol)
    if winning_move is not None:
        return winning_move
    
    # Then check if we need to block opponent's win
    blocking_move = find_winning_move(board, rows, cols, opponent_symbol)
    if blocking_move is not None:
        return blocking_move
    
    # Check for opponent's double threats (two three-in-a-rows)
    double_threat_block = find_double_threat_block(board, rows, cols)
    if double_threat_block is not None:
        return double_threat_block
    
    # Check for opponent's three-in-a-row setups
    three_in_row_block = find_three_in_row_block(board, rows, cols)
    if three_in_row_block is not None:
        return three_in_row_block
    
    return None

def find_winning_move(board, rows, cols, symbol):
    """Check if there's an immediate winning move"""
    for col in range(cols):
        # Find the row where a piece would land
        row = -1
        for r in range(rows - 1, -1, -1):
            if board[r][col] == ' ':
                row = r
                break
        
        if row == -1:  # Column is full
            continue
        
        # Try this move
        board_copy = copy.deepcopy(board)
        board_copy[row][col] = symbol
        
        # Check if this creates a win
        if check_win(board_copy, rows, cols, symbol):
            return col + 1  # 1-indexed
    
    return None

def find_double_threat_block(board, rows, cols):
    """
    Check if opponent can create a double threat
    (a move that leads to two winning paths)
    """
    valid_moves = get_valid_moves(board, rows, cols)
    
    for move in valid_moves:
        col = move - 1  # Convert to 0-indexed
        
        # Find the row where a piece would land
        row = -1
        for r in range(rows - 1, -1, -1):
            if board[r][col] == ' ':
                row = r
                break
        
        if row == -1:  # Column is full
            continue
        
        # Try this move for the opponent
        board_copy = copy.deepcopy(board)
        board_copy[row][col] = opponent_symbol
        
        # Check if this creates a double threat
        if detect_double_threats(board_copy, rows, cols, opponent_symbol) >= 2:
            return move
    
    return None

def find_three_in_row_block(board, rows, cols):
    """
    Check if there's a critical three-in-a-row to block
    """
    valid_moves = get_valid_moves(board, rows, cols)
    
    for move in valid_moves:
        col = move - 1  # Convert to 0-indexed
        
        # Find the row where a piece would land
        row = -1
        for r in range(rows - 1, -1, -1):
            if board[r][col] == ' ':
                row = r
                break
        
        if row == -1:  # Column is full
            continue
        
        # Try this move
        board_copy = copy.deepcopy(board)
        board_copy[row][col] = player_symbol
        
        # Check if this blocks a three-in-a-row
        if blocks_three_in_row(board, board_copy, rows, cols, row, col):
            return move
    
    return None

def blocks_three_in_row(old_board, new_board, rows, cols, row, col):
    """
    Check if a move blocks a three-in-a-row
    """
    # Check horizontal
    for c in range(max(0, col-3), min(cols-3, col+1)):
        window_old = [old_board[row][c+i] for i in range(4)]
        window_new = [new_board[row][c+i] for i in range(4)]
        if window_old.count(opponent_symbol) == 3 and window_old.count(' ') == 1:
            if window_new.count(opponent_symbol) == 3 and window_new.count(player_symbol) == 1:
                return True
    
    # Check vertical
    if row <= rows - 4:  # Only check if there's room for 4 below
        window_old = [old_board[row+i][col] for i in range(4)]
        window_new = [new_board[row+i][col] for i in range(4)]
        if window_old.count(opponent_symbol) == 3 and window_old.count(' ') == 1:
            if window_new.count(opponent_symbol) == 3 and window_new.count(player_symbol) == 1:
                return True
    
    # Check diagonal down
    for r, c in zip(range(max(0, row-3), min(rows-3, row+1)), 
                   range(max(0, col-3), min(cols-3, col+1))):
        window_old = [old_board[r+i][c+i] for i in range(4)]
        window_new = [new_board[r+i][c+i] for i in range(4)]
        if window_old.count(opponent_symbol) == 3 and window_old.count(' ') == 1:
            if window_new.count(opponent_symbol) == 3 and window_new.count(player_symbol) == 1:
                return True
    
    # Check diagonal up
    for r, c in zip(range(min(rows-1, row+3), max(3, row)-1, -1), 
                   range(max(0, col-3), min(cols-3, col+1))):
        if r >= 3 and r < rows and c >= 0 and c < cols - 3:
            window_old = [old_board[r-i][c+i] for i in range(4)]
            window_new = [new_board[r-i][c+i] for i in range(4)]
            if window_old.count(opponent_symbol) == 3 and window_old.count(' ') == 1:
                if window_new.count(opponent_symbol) == 3 and window_new.count(player_symbol) == 1:
                    return True
    
    return False

def check_opening_book(board, valid_moves):
    """Check if the current board state is in our opening book"""
    # Very early game strategy
    move_count = sum(1 for row in board for cell in row if cell != ' ')
    
    # First move - take center if available
    if move_count == 0:
        center = (len(board[0]) // 2) + 1  # 1-indexed
        if center in valid_moves:
            return center
    
    # Try to match the board state with our opening book
    board_hash = board_to_hash(board)
    if board_hash in OPENING_BOOK and OPENING_BOOK[board_hash] in valid_moves:
        return OPENING_BOOK[board_hash]
    
    return None

def check_win(board, rows, cols, symbol):
    """Check if the given symbol has won"""
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

def iterative_deepening_search(board, rows, cols, valid_moves):
    """Use iterative deepening with time limit"""
    global start_time
    
    best_move = valid_moves[0]  # Default to first valid move
    
    # Start with depth 1, then increase until time runs out
    for depth in range(1, MAX_DEPTH + 1):
        if time.time() - start_time > TIME_LIMIT * 0.8:  # Use 80% of time for search
            if DEBUG_MODE:
                print(f"Time limit reached at depth {depth-1}")
            break
        
        # Run alpha-beta search at the current depth
        move = alpha_beta_search(board, rows, cols, valid_moves, depth)
        
        # Update best move if search completed
        if move is not None:
            best_move = move
        
        if DEBUG_MODE:
            print(f"Depth {depth} search completed, best move: {best_move}")
    
    return best_move

def alpha_beta_search(board, rows, cols, valid_moves, depth):
    """Run alpha-beta search to find best move"""
    best_score = float('-inf')
    best_move = None
    alpha = float('-inf')
    beta = float('inf')
    
    # Sort moves to improve pruning efficiency (center first, then ordered by evaluation)
    ordered_moves = order_moves(board, rows, cols, valid_moves)
    
    for move in ordered_moves:
        # Check for time limit frequently
        if time.time() - start_time > TIME_LIMIT * 0.95:  # 95% of time limit
            if best_move is not None:
                return best_move
            return ordered_moves[0]
            
        # Create a copy of the board
        new_board = copy.deepcopy(board)
        
        # Apply the move
        row = make_move(new_board, move - 1, player_symbol)  # Convert to 0-indexed
        if row == -1:  # Invalid move
            continue
        
        # Evaluate position with alpha-beta
        score = min_value(new_board, rows, cols, depth - 1, alpha, beta, 
                         start_time, TIME_LIMIT * 0.95)
        
        # Update best move
        if score > best_score:
            best_score = score
            best_move = move
        
        # Update alpha
        alpha = max(alpha, best_score)
    
    return best_move

def min_value(board, rows, cols, depth, alpha, beta, start_time_val, time_limit_val):
    """Minimizing player's turn in alpha-beta"""
    global nodes_evaluated
    
    nodes_evaluated += 1
    
    # Check for time limit periodically (less frequently for performance)
    if nodes_evaluated % 100 == 0 and time.time() - start_time_val > time_limit_val:
        return 0  # Return neutral score when out of time
    
    # Check for terminal state
    if check_win(board, rows, cols, player_symbol):
        return WEIGHTS['win']
    if check_win(board, rows, cols, opponent_symbol):
        return WEIGHTS['lose']
    
    # Check if board is full (draw)
    if is_board_full(board):
        return 0
    
    # Depth limit reached
    if depth <= 0:
        return evaluate_board(board, rows, cols)
    
    # Check transposition table
    if USE_TRANSPOSITION_TABLE:
        board_hash = board_to_hash(board)
        if board_hash in transposition_table:
            return transposition_table[board_hash]
    
    value = float('inf')
    valid_moves = get_valid_moves(board, rows, cols)
    
    # Order moves for better pruning
    ordered_moves = order_moves(board, rows, cols, valid_moves)
    
    for move in ordered_moves:
        # Create a copy of the board
        new_board = copy.deepcopy(board)
        
        # Apply the move
        row = make_move(new_board, move - 1, opponent_symbol)  # Convert to 0-indexed
        if row == -1:  # Invalid move
            continue
        
        # Recursively evaluate
        value = min(value, max_value(new_board, rows, cols, depth - 1, alpha, beta, 
                                    start_time_val, time_limit_val))
        
        # Beta pruning
        if value <= alpha:
            return value
        
        beta = min(beta, value)
    
    # Store in transposition table
    if USE_TRANSPOSITION_TABLE:
        transposition_table[board_to_hash(board)] = value
    
    return value

def max_value(board, rows, cols, depth, alpha, beta, start_time_val, time_limit_val):
    """Maximizing player's turn in alpha-beta"""
    global nodes_evaluated
    
    nodes_evaluated += 1
    
    # Check for time limit periodically
    if nodes_evaluated % 100 == 0 and time.time() - start_time_val > time_limit_val:
        return 0  # Return neutral score when out of time
    
    # Check for terminal state
    if check_win(board, rows, cols, player_symbol):
        return WEIGHTS['win']
    if check_win(board, rows, cols, opponent_symbol):
        return WEIGHTS['lose']
    
    # Check if board is full (draw)
    if is_board_full(board):
        return 0
    
    # Depth limit reached
    if depth <= 0:
        return evaluate_board(board, rows, cols)
    
    # Check transposition table
    if USE_TRANSPOSITION_TABLE:
        board_hash = board_to_hash(board)
        if board_hash in transposition_table:
            return transposition_table[board_hash]
    
    value = float('-inf')
    valid_moves = get_valid_moves(board, rows, cols)
    
    # Order moves for better pruning
    ordered_moves = order_moves(board, rows, cols, valid_moves)
    
    for move in ordered_moves:
        # Create a copy of the board
        new_board = copy.deepcopy(board)
        
        # Apply the move
        row = make_move(new_board, move - 1, player_symbol)  # Convert to 0-indexed
        if row == -1:  # Invalid move
            continue
        
        # Recursively evaluate
        value = max(value, min_value(new_board, rows, cols, depth - 1, alpha, beta, 
                                    start_time_val, time_limit_val))
        
        # Alpha pruning
        if value >= beta:
            return value
        
        alpha = max(alpha, value)
    
    # Store in transposition table
    if USE_TRANSPOSITION_TABLE:
        transposition_table[board_to_hash(board)] = value
    
    return value

def make_move(board, col, symbol):
    """Apply a move to the board, return the row where the piece lands or -1 if invalid"""
    if col < 0 or col >= len(board[0]) or board[0][col] != ' ':
        return -1
    
    # Find the lowest empty row
    for row in range(len(board) - 1, -1, -1):
        if board[row][col] == ' ':
            board[row][col] = symbol
            return row
    
    return -1  # Should never reach here if column validation is correct

def is_board_full(board):
    """Check if the board is full (no valid moves)"""
    return all(cell != ' ' for row in board for cell in row)

def order_moves(board, rows, cols, valid_moves):
    """Order moves for better alpha-beta pruning efficiency"""
    # Score each move with a quick evaluation
    move_scores = []
    for move in valid_moves:
        # Create a copy of the board
        new_board = copy.deepcopy(board)
        
        # Apply the move
        row = make_move(new_board, move - 1, player_symbol)  # Convert to 0-indexed
        if row == -1:  # Invalid move
            move_scores.append((move, float('-inf')))
            continue
        
        # Quick evaluation
        score = quick_evaluate(new_board, rows, cols, row, move - 1)
        move_scores.append((move, score))
    
    # Sort by score, descending
    move_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return ordered moves
    return [move for move, _ in move_scores]

def quick_evaluate(board, rows, cols, last_row, last_col):
    """Quick evaluation for move ordering"""
    score = 0
    
    # Prefer center columns
    center_col = cols // 2
    score -= abs(center_col - last_col) * 2
    
    # Check if move creates threats
    new_threats = count_threats(board, rows, cols, player_symbol, last_row, last_col)
    score += new_threats * 10
    
    # Check if move blocks threats
    opponent_threats = count_threats(board, rows, cols, opponent_symbol, last_row, last_col)
    score += opponent_threats * 15  # Higher priority for blocking
    
    # Check for immediate wins or blocks (highest priority)
    temp_board = copy.deepcopy(board)
    # Check if this creates a win
    if check_win(temp_board, rows, cols, player_symbol):
        score += 1000
    
    # Check if this is on the bottom row (stability)
    if last_row == rows - 1:
        score += 5
    
    return score

def count_threats(board, rows, cols, symbol, row, col):
    """Count new threats created by the last move"""
    count = 0
    directions = [
        [(0, 1), (0, -1)],  # Horizontal
        [(1, 0), (-1, 0)],  # Vertical
        [(1, 1), (-1, -1)],  # Diagonal down
        [(1, -1), (-1, 1)]   # Diagonal up
    ]
    
    for dir_pair in directions:
        # Check in both directions
        pieces = 1  # Start with the piece we just placed
        spaces = 0
        
        for dx, dy in dir_pair:
            # Count consecutive pieces and empty spaces
            for i in range(1, 4):
                r, c = row + i*dy, col + i*dx
                if 0 <= r < rows and 0 <= c < cols:
                    if board[r][c] == symbol:
                        pieces += 1
                    elif board[r][c] == ' ':
                        spaces += 1
                        # Don't count spaces above the board (affected by gravity)
                        if r > 0 and board[r-1][c] == ' ':
                            spaces -= 1
                    else:
                        break
                else:
                    break
        
        # Three in a row with space is a threat
        if pieces == 3 and spaces >= 1:
            count += 1
        # Two in a row with spaces is a lesser threat
        elif pieces == 2 and spaces >= 2:
            count += 0.5
    
    return count

def evaluate_board(board, rows, cols):
    """Evaluate the board position with enhanced pattern recognition"""
    score = 0
    
    # Check for wins (should be caught earlier, but just in case)
    if check_win(board, rows, cols, player_symbol):
        return WEIGHTS['win']
    if check_win(board, rows, cols, opponent_symbol):
        return WEIGHTS['lose']
    
    # Center column control
    center_col = cols // 2
    for row in range(rows):
        if board[row][center_col] == player_symbol:
            score += WEIGHTS['center_control']
        elif board[row][center_col] == opponent_symbol:
            score -= WEIGHTS['center_control']
    
    # Evaluate horizontal windows
    for row in range(rows):
        for col in range(cols - 3):
            window = [board[row][col+i] for i in range(4)]
            score += evaluate_window(window)
    
    # Evaluate vertical windows
    for col in range(cols):
        for row in range(rows - 3):
            window = [board[row+i][col] for i in range(4)]
            score += evaluate_window(window)
    
    # Evaluate diagonal down windows
    for row in range(rows - 3):
        for col in range(cols - 3):
            window = [board[row+i][col+i] for i in range(4)]
            score += evaluate_window(window)
    
    # Evaluate diagonal up windows
    for row in range(3, rows):
        for col in range(cols - 3):
            window = [board[row-i][col+i] for i in range(4)]
            score += evaluate_window(window)
    
    # Bottom row control (stable positions)
    for col in range(cols):
        if board[rows-1][col] == player_symbol:
            score += WEIGHTS['bottom_row']
        elif board[rows-1][col] == opponent_symbol:
            score -= WEIGHTS['bottom_row']
    
    # Detect double threats (multiple winning paths)
    player_double_threats = detect_double_threats(board, rows, cols, player_symbol)
    opponent_double_threats = detect_double_threats(board, rows, cols, opponent_symbol)
    
    score += player_double_threats * WEIGHTS['double_threat']
    score -= opponent_double_threats * WEIGHTS['double_threat'] * 1.5  # Extra penalty for opponent double threats
    
    return score

def evaluate_window(window):
    """Enhanced window evaluation with better pattern recognition"""
    score = 0
    
    # Count pieces in the window
    player_count = window.count(player_symbol)
    opponent_count = window.count(opponent_symbol)
    empty_count = window.count(' ')
    
    # Score based on piece counts (significantly improved weights)
    if player_count == 4:
        return WEIGHTS['win']  # Should already be caught, but just in case
    elif player_count == 3 and empty_count == 1:
        score += WEIGHTS['three_open']
    elif player_count == 2 and empty_count == 2:
        score += WEIGHTS['two_open']
    
    # Defensive scoring (higher penalties)
    if opponent_count == 3 and empty_count == 1:
        score -= WEIGHTS['three_open'] * 1.2  # Prioritize defense slightly higher
    elif opponent_count == 2 and empty_count == 2:
        score -= WEIGHTS['two_open'] * 1.1
    
    return score

def detect_double_threats(board, rows, cols, symbol):
    """Enhanced double threat detection"""
    winning_columns = set()
    
    # For each valid move, check if it creates a winning position
    for col in range(cols):
        # Skip if column is full
        if board[0][col] != ' ':
            continue
        
        # Find row where piece would land
        row = -1
        for r in range(rows - 1, -1, -1):
            if board[r][col] == ' ':
                row = r
                break
        
        # Apply move to a copy
        temp_board = copy.deepcopy(board)
        temp_board[row][col] = symbol
        
        # Check if this created a win
        if check_win(temp_board, rows, cols, symbol):
            winning_columns.add(col)
        
        # Also check for potential three-in-a-rows
        elif count_potential_threes(temp_board, rows, cols, row, col, symbol) >= 2:
            winning_columns.add(col)
    
    # Return number of winning columns (double threat indicator)
    return len(winning_columns)

def count_potential_threes(board, rows, cols, row, col, symbol):
    """
    Count potential three-in-a-rows created by a move
    that could lead to a win in the next move
    """
    count = 0
    directions = [
        [(0, 1), (0, -1)],  # Horizontal
        [(1, 0), (-1, 0)],  # Vertical
        [(1, 1), (-1, -1)],  # Diagonal down
        [(1, -1), (-1, 1)]   # Diagonal up
    ]
    
    for dir_pair in directions:
        # Check in both directions
        pieces = 1  # Start with the piece we just placed
        empty_spaces = []
        
        for dx, dy in dir_pair:
            # Count consecutive pieces and note empty spaces
            for i in range(1, 4):
                r, c = row + i*dy, col + i*dx
                if 0 <= r < rows and 0 <= c < cols:
                    if board[r][c] == symbol:
                        pieces += 1
                    elif board[r][c] == ' ':
                        # Check if this space is playable (has support below or is bottom row)
                        is_playable = (r == rows-1) or (board[r+1][c] != ' ')
                        if is_playable:
                            empty_spaces.append((r, c))
                    else:
                        break
                else:
                    break
        
        # If this forms a potential three-in-a-row with a playable space
        if pieces == 3 and len(empty_spaces) == 1:
            count += 1
    
    return count

def cleanup():
    """
    Clean up resources when agent is no longer needed.
    Call this when the game is over.
    """
    global thinking_active, background_thread, search_queue
    
    if BACKGROUND_THINKING and thinking_active:
        thinking_active = False
        try:
            # Signal the thread to stop
            search_queue.put(None)
            # Wait for thread to finish (with timeout)
            if background_thread is not None and background_thread.is_alive():
                background_thread.join(1.0)
        except:
            pass

if __name__ == "__main__":
    # Sample code to test the agent
    print("Team5_Permanent_Brain_Agent.py is intended to be imported and not executed.")
    
    # Ensure cleanup is called when the process exits
    import atexit
    atexit.register(cleanup)