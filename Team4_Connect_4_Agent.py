<<<<<<< HEAD
#!/usr/bin/env python3
"""
Team4 Connect 4 Agent with MCTS and Permanent Brain
Enhanced with:
1. Permanent Brain (opponent-turn thinking)
2. Thread-based concurrent search
3. Enhanced pattern detection
4. Efficient bit-based board representation
5. Trap detection and gateway patterns

This agent combines Monte Carlo Tree Search with UCB1 and permanent brain
to continuously analyze the game tree even during the opponent's turn.
"""
=======

#! /usr/bin/Team3_Connect_4_Agent.py 
>>>>>>> 4ffec52dff4ca5d10c6abe1008fcdaf277aba156

# IMPORTS
import random
<<<<<<< HEAD
import time
import math
import copy
import threading
import queue

# Constants for UCB1 calculation
EXPLORATION_CONSTANT = 1.0  # Lower than standard 2.0 as per Xiao's strategy

# Maximum number of simulations per move
MAX_SIMULATIONS = 50000

# Maximum time per move in seconds (safety limit)
MAX_TIME_PER_MOVE = 4.9

# Maximum time for background thinking in seconds
MAX_BACKGROUND_TIME = 60.0

# Node pool size
NODE_POOL_SIZE = 2000000

# Permanent Brain settings
PERMANENT_BRAIN_ENABLED = True  # Enable/disable permanent brain
MAX_BACKGROUND_SIMULATIONS = 200000  # Simulations to run in background

# Global variables to maintain state between function calls
mcts_root = None
player_symbol = None
opponent_symbol = None
board_rows = 0
board_cols = 0
move_count = 0

# Node pool
node_pool = []
next_node_id = 0

# Permanent Brain variables
background_thread = None
thinking_active = False
search_queue = queue.Queue()
background_results = {}
last_board_state = None

# Debug flag
DEBUG = False

class BitBoard:
    """
    Efficient board representation using bit operations
    """
    def __init__(self, board, rows, cols):
        """Initialize the bitboard from a standard board"""
        self.rows = rows
        self.cols = cols
        self.player_bits = [0] * cols  # Each column is represented as a bit field
        self.opponent_bits = [0] * cols
        self.heights = [0] * cols  # Current height of each column
        
        # Convert standard board to bitboard
        for c in range(cols):
            for r in range(rows-1, -1, -1):
                if board[r][c] == player_symbol:
                    self.player_bits[c] |= (1 << (rows - 1 - r))
                elif board[r][c] == opponent_symbol:
                    self.opponent_bits[c] |= (1 << (rows - 1 - r))
                if board[r][c] != ' ':
                    self.heights[c] += 1
    
    def copy(self):
        """Create a deep copy of the bitboard"""
        new_board = BitBoard.__new__(BitBoard)
        new_board.rows = self.rows
        new_board.cols = self.cols
        new_board.player_bits = self.player_bits.copy()
        new_board.opponent_bits = self.opponent_bits.copy()
        new_board.heights = self.heights.copy()
        return new_board
    
    def get_valid_moves(self):
        """Return a list of valid column indices (0-based)"""
        return [c for c in range(self.cols) if self.heights[c] < self.rows]
    
    def make_move(self, col, is_player):
        """Make a move on the board. Returns row where piece was placed."""
        if self.heights[col] >= self.rows:
            return -1  # Invalid move
        
        row = self.rows - 1 - self.heights[col]
        bit = 1 << (self.rows - 1 - row)
        
        if is_player:
            self.player_bits[col] |= bit
        else:
            self.opponent_bits[col] |= bit
        
        self.heights[col] += 1
        return row
    
    def check_win(self, col, row, is_player):
        """Check if the last move at (col, row) created a win"""
        bits = self.player_bits if is_player else self.opponent_bits
        
        # Check horizontal
        count = 0
        for c in range(max(0, col-3), min(col+4, self.cols)):
            if c == col or (bits[c] & (1 << (self.rows - 1 - row))):
                count += 1
                if count >= 4:
                    return True
            else:
                count = 0
        
        # Check vertical
        if self.heights[col] >= 4:
            count = 0
            mask = 0b1111 << (self.rows - 4 - row)
            if bits[col] & mask == mask:
                return True
        
        # Check diagonal up-right
        count = 0
        for i in range(-3, 4):
            c, r = col + i, row + i
            if 0 <= c < self.cols and 0 <= r < self.rows:
                if c == col and r == row:
                    count += 1
                elif bits[c] & (1 << (self.rows - 1 - r)):
                    count += 1
                else:
                    count = 0
                if count >= 4:
                    return True
            else:
                count = 0
        
        # Check diagonal up-left
        count = 0
        for i in range(-3, 4):
            c, r = col + i, row - i
            if 0 <= c < self.cols and 0 <= r < self.rows:
                if c == col and r == row:
                    count += 1
                elif bits[c] & (1 << (self.rows - 1 - r)):
                    count += 1
                else:
                    count = 0
                if count >= 4:
                    return True
            else:
                count = 0
        
        return False
    
    def is_full(self):
        """Check if the board is full"""
        return all(h >= self.rows for h in self.heights)
    
    def find_winning_move(self, is_player):
        """Find a move that immediately wins the game"""
        for col in self.get_valid_moves():
            board_copy = self.copy()
            row = board_copy.make_move(col, is_player)
            if board_copy.check_win(col, row, is_player):
                return col
        return None
    
    def find_blocking_move(self):
        """Find a move that blocks opponent's winning move"""
        return self.find_winning_move(not True)
    
    def find_future_winning_moves(self, is_player, depth=2):
        """Find moves that lead to a win within depth moves"""
        if depth <= 0:
            return []
        
        # Try each move
        winning_setups = []
        for col in self.get_valid_moves():
            board_copy = self.copy()
            row = board_copy.make_move(col, is_player)
            
            # If this move wins, it's a winning setup
            if board_copy.check_win(col, row, is_player):
                winning_setups.append(col)
                continue
            
            # If we're not at max depth, check if opponent has to block and we can win next move
            if depth > 1:
                # Find all opponent's valid moves
                opponent_valid_moves = board_copy.get_valid_moves()
                
                # Check if for all opponent moves, we have a winning move next turn
                all_lead_to_win = True
                for opp_col in opponent_valid_moves:
                    opp_board = board_copy.copy()
                    opp_row = opp_board.make_move(opp_col, not is_player)
                    
                    # Check if we have a winning move after opponent's move
                    winning_move = opp_board.find_winning_move(is_player)
                    if winning_move is None:
                        # No immediate win, check for trap setups recursively
                        future_wins = opp_board.find_future_winning_moves(is_player, depth-1)
                        if not future_wins:
                            all_lead_to_win = False
                            break
                
                if all_lead_to_win and opponent_valid_moves:
                    winning_setups.append(col)
        
        return winning_setups
    
    def detect_gateway_pattern(self):
        """
        Detect gateway patterns - positions that create multiple threats
        Returns a list of columns that create gateways
        """
        gateway_cols = []
        
        for col in self.get_valid_moves():
            # Make a move and check if it creates multiple threats
            board_copy = self.copy()
            row = board_copy.make_move(col, True)  # Simulate player move
            
            # Count how many threats this creates
            threat_count = 0
            
            # Check for vertical threat (needs 3 in column with space above)
            if board_copy.heights[col] >= 3 and board_copy.heights[col] < self.rows:
                vertical_count = 0
                for r in range(1, 4):  # Check 3 positions below the current move
                    check_row = row + r
                    if check_row < self.rows and board_copy.player_bits[col] & (1 << (self.rows - 1 - check_row)):
                        vertical_count += 1
                if vertical_count >= 2:  # Need at least 2 pieces below to form a threat
                    threat_count += 1
            
            # Check for horizontal threats
            for c_start in range(max(0, col-3), min(col+1, self.cols-3)):
                # Check 4 consecutive positions starting at c_start
                player_count = 0
                empty_count = 0
                empty_cols = []
                
                for c in range(c_start, c_start+4):
                    if c == col:  # The current move
                        player_count += 1
                    elif board_copy.player_bits[c] & (1 << (self.rows - 1 - row)):
                        player_count += 1
                    elif board_copy.heights[c] == self.rows - 1 - row:  # Empty cell at same height
                        empty_count += 1
                        empty_cols.append(c)
                    elif board_copy.heights[c] < self.rows - 1 - row:  # Empty cells above
                        break
                
                if player_count + empty_count == 4 and player_count >= 2 and empty_count > 0:
                    threat_count += 1
            
            # Check for diagonal threats (similar to horizontal but with diagonal positions)
            # Diagonal up-right
            for i_start in range(-3, 1):
                player_count = 0
                empty_count = 0
                valid_diagonal = True
                
                for i in range(i_start, i_start+4):
                    c, r = col + i, row + i
                    if not (0 <= c < self.cols and 0 <= r < self.rows):
                        valid_diagonal = False
                        break
                    
                    if c == col and r == row:  # Current move
                        player_count += 1
                    elif board_copy.player_bits[c] & (1 << (self.rows - 1 - r)):
                        player_count += 1
                    elif board_copy.heights[c] == self.rows - 1 - r:  # Empty at right height
                        empty_count += 1
                    else:
                        valid_diagonal = False
                        break
                
                if valid_diagonal and player_count + empty_count == 4 and player_count >= 2:
                    threat_count += 1
            
            # Diagonal up-left
            for i_start in range(-3, 1):
                player_count = 0
                empty_count = 0
                valid_diagonal = True
                
                for i in range(i_start, i_start+4):
                    c, r = col + i, row - i
                    if not (0 <= c < self.cols and 0 <= r < self.rows):
                        valid_diagonal = False
                        break
                    
                    if c == col and r == row:  # Current move
                        player_count += 1
                    elif board_copy.player_bits[c] & (1 << (self.rows - 1 - r)):
                        player_count += 1
                    elif board_copy.heights[c] == self.rows - 1 - r:  # Empty at right height
                        empty_count += 1
                    else:
                        valid_diagonal = False
                        break
                
                if valid_diagonal and player_count + empty_count == 4 and player_count >= 2:
                    threat_count += 1
            
            # If this move creates multiple threats, it's a gateway
            if threat_count >= 2:
                gateway_cols.append(col)
        
        return gateway_cols
    
    def evaluate_move(self, col):
        """Evaluate how good a move is for simulation weighting"""
        if col < 0 or col >= self.cols:
            return 0
        
        score = 0
        
        # Center preference - higher score for columns closer to center
        center_dist = abs(col - (self.cols - 1) / 2)
        score += (min(col, self.cols - 1 - col) / 3) + 1
        
        # Check for potential threats and blocks
        board_copy = self.copy()
        row = board_copy.make_move(col, True)
        
        # Find future winning moves (traps)
        future_wins = board_copy.find_future_winning_moves(True, 2)
        if col in future_wins:
            score += 10  # Heavily favor trap moves
        
        # Check for gateway patterns
        gateway_cols = board_copy.detect_gateway_pattern()
        if col in gateway_cols:
            score += 5  # Favor gateway moves
        
        # Avoid losing moves - if this move lets opponent win next turn
        for opp_col in board_copy.get_valid_moves():
            opp_board = board_copy.copy()
            opp_row = opp_board.make_move(opp_col, False)
            if opp_board.check_win(opp_col, opp_row, False):
                score -= 10  # Strongly penalize moves that lead to opponent's win
        
        return max(0.1, score)  # Ensure minimum weight is 0.1
    
    def to_string(self):
        """Convert the board to a string representation for hashing"""
        s = ""
        for c in range(self.cols):
            for r in range(self.rows):
                bit = 1 << (self.rows - 1 - r)
                if self.player_bits[c] & bit:
                    s += "P"
                elif self.opponent_bits[c] & bit:
                    s += "O"
                else:
                    s += " "
        return s
    
    def print_board(self):
        """Print the current board state (for debugging)"""
        if not DEBUG:
            return
            
        board = [[' ' for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Fill in player pieces
        for c in range(self.cols):
            for r in range(self.rows):
                bit = 1 << (self.rows - 1 - r)
                if self.player_bits[c] & bit:
                    board[r][c] = player_symbol
                elif self.opponent_bits[c] & bit:
                    board[r][c] = opponent_symbol
        
        print("\nCurrent board state:")
        for r in range(self.rows):
            print("|", end="")
            for c in range(self.cols):
                print(f" {board[r][c]} ", end="")
            print("|")
        print("-" * (self.cols * 4 + 1))
        print(" ", end="")
        for c in range(self.cols):
            print(f" {c+1}  ", end="")
        print("\n")

class MCTSNode:
    """Monte Carlo Tree Search Node"""
    def __init__(self, board, parent=None, move=-1, is_player_turn=True):
        self.board = board
        self.parent = parent
        self.move = move  # Move that led to this state
        self.is_player_turn = is_player_turn
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = board.get_valid_moves()
        self.is_fully_expanded = len(self.untried_moves) == 0
        self.next_child_id = 0
    
    def select_child(self):
        """Select the best child node using UCB1"""
        return max(self.children, key=lambda c: c.wins / c.visits + 
                   EXPLORATION_CONSTANT * math.sqrt(math.log(self.visits) / c.visits))
    
    def add_child(self, move):
        """Add a child node for the given move"""
        global next_node_id
        
        if move not in self.untried_moves:
            return None
        
        # Remove the move from untried moves
        self.untried_moves.remove(move)
        if not self.untried_moves:
            self.is_fully_expanded = True
        
        # Create new board state
        child_board = self.board.copy()
        child_board.make_move(move, self.is_player_turn)
        
        # If node pool is full, start reusing nodes
        if next_node_id >= len(node_pool):
            next_node_id = 0  # Restart from the beginning
        
        # Initialize or reuse node from the pool
        node = node_pool[next_node_id]
        if node is None:
            node = MCTSNode(child_board, self, move, not self.is_player_turn)
            node_pool[next_node_id] = node
        else:
            node.board = child_board
            node.parent = self
            node.move = move
            node.is_player_turn = not self.is_player_turn
            node.children = []
            node.wins = 0
            node.visits = 0
            node.untried_moves = child_board.get_valid_moves()
            node.is_fully_expanded = len(node.untried_moves) == 0
            node.next_child_id = 0
        
        next_node_id += 1
        self.children.append(node)
        return node
    
    def update(self, result):
        """Update node statistics with simulation result"""
        self.visits += 1
        self.wins += result

def simulation_policy(board):
    """Smart simulation policy for the rollout phase"""
    # Check for immediate win
    winning_move = board.find_winning_move(True)
    if winning_move is not None:
        return winning_move
    
    # Check for immediate block
    blocking_move = board.find_blocking_move()
    if blocking_move is not None:
        return blocking_move
    
    # Check for gateway moves
    gateway_moves = board.detect_gateway_pattern()
    if gateway_moves:
        return random.choice(gateway_moves)
    
    # Check for trap setups
    trap_moves = board.find_future_winning_moves(True, 2)
    if trap_moves:
        return random.choice(trap_moves)
    
    # Use weighted random selection
    valid_moves = board.get_valid_moves()
    if not valid_moves:
        return -1
    
    # Evaluate each move
    move_weights = [board.evaluate_move(move) for move in valid_moves]
    
    # Choose a move weighted by score
    return random.choices(valid_moves, weights=move_weights, k=1)[0]

def simulation(node):
    """Perform a simulation from the given node"""
    board = node.board.copy()
    is_player_turn = node.is_player_turn
    
    # Continue until we reach a terminal state
    while True:
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return 0.5  # Draw
        
        # Use smart simulation policy
        move = simulation_policy(board)
        if move < 0:
            return 0.5  # Invalid move, treat as draw
        
        row = board.make_move(move, is_player_turn)
        
        # Check for win
        if board.check_win(move, row, is_player_turn):
            return 1.0 if is_player_turn else 0.0
        
        # Switch player
        is_player_turn = not is_player_turn

def mcts_search(root_node, max_iterations, max_time, early_stopping=True):
    """Perform Monte Carlo Tree Search from the root node"""
    start_time = time.time()
    iterations = 0
    
    while iterations < max_iterations:
        # Check time limit
        if early_stopping and time.time() - start_time > max_time:
            break
        
        # Selection: Find the most promising leaf node
        node = root_node
        while node.is_fully_expanded and node.children:
            node = node.select_child()
        
        # Expansion: If leaf is not terminal and has untried moves, expand it
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            node = node.add_child(move)
        
        # Simulation: Play a random game from the new node
        result = simulation(node)
        
        # Backpropagation: Update statistics up the tree
        while node:
            node.update(result)
            node = node.parent
            if node:
                result = 1.0 - result  # Invert result for opponent
        
        iterations += 1
    
    return iterations

# NEW: Permanent Brain Worker function
def permanent_brain_worker():
    """
    Worker function for the permanent brain (background thinking).
    This continuously processes thinking tasks from the queue.
    """
    global thinking_active, background_results
    
    while thinking_active:
        try:
            # Get a thinking task from the queue (with timeout)
            task = search_queue.get(timeout=0.1)
            
            if task is None:
                # Signal to stop the thread
                break
            
            board, root_node = task
            board_hash = board.to_string()
            
            # Skip if we already have results for this board
            if board_hash in background_results:
                search_queue.task_done()
                continue
            
            # Perform MCTS search in the background
            thinking_start_time = time.time()
            simulations = mcts_search(root_node, MAX_BACKGROUND_SIMULATIONS, MAX_BACKGROUND_TIME, False)
            
            # Analyze the results to find the best move
            best_child = None
            best_visits = -1
            results = {}
            
            if root_node.children:
                for child in root_node.children:
                    results[child.move] = (child.wins, child.visits)
                    if child.visits > best_visits:
                        best_visits = child.visits
                        best_child = child
            
            # Store results for later use
            background_results[board_hash] = {
                'best_move': best_child.move if best_child else -1,
                'results': results,
                'simulations': simulations,
                'timestamp': time.time()
            }
            
            if DEBUG:
                print(f"Background thinking completed: {simulations} simulations")
                if best_child:
                    print(f"Best move: {best_child.move+1} with {best_visits} visits")
            
            search_queue.task_done()
            
        except queue.Empty:
            # No tasks in queue, just wait
            pass
        except Exception as e:
            if DEBUG:
                print(f"Error in background thinking: {e}")
            
            # Continue with other tasks
            continue

def detect_opponent_move(board, last_board):
    """Detect what move the opponent made since our last turn"""
    if last_board is None:
        return None
        
    for col in range(len(board.heights)):
        if board.heights[col] > last_board.heights[col]:
            # Found a column with a new piece
            return col
    
    return None

def schedule_background_thinking(bit_board, root_node):
    """Schedule background thinking for the current board state"""
    global search_queue, last_board_state
    
    if not PERMANENT_BRAIN_ENABLED or not thinking_active:
        return
    
    # Queue the current board for analysis
    try:
        # Create a copy of the board and root node
        board_copy = bit_board.copy()
        
        # For each potential opponent move, schedule a thinking task
        valid_moves = bit_board.get_valid_moves()
        for move in valid_moves:
            # Create a new board with the opponent's move
            future_board = bit_board.copy()
            row = future_board.make_move(move, False)  # Opponent's move
            
            # Create a new root node for this state
            future_node = MCTSNode(future_board, None, -1, True)
            
            # Queue for analysis
            search_queue.put((future_board, future_node))
    except queue.Full:
        # Queue is full, just continue
        pass
    
    # Update the last board state
    last_board_state = bit_board.copy()

def init_agent(player_sym, board_num_rows, board_num_cols, board):
    """
    Initialize the agent. Should only need to be called once at the start of a game.
    """
    global player_symbol, opponent_symbol, board_rows, board_cols, mcts_root, node_pool, move_count
    global thinking_active, background_thread, last_board_state
    
    player_symbol = player_sym
    opponent_symbol = 'O' if player_sym == 'X' else 'X'
    board_rows = int(board_num_rows)
    board_cols = int(board_num_cols)
    move_count = 0
    
    # Initialize the node pool
    node_pool = [None] * NODE_POOL_SIZE
    
    # Initialize the MCTS root with the current board state
    bit_board = BitBoard(board, board_rows, board_cols)
    mcts_root = MCTSNode(bit_board, None, -1, True)
    
    # Store current board state
    last_board_state = bit_board.copy()
    
    # Initialize permanent brain if enabled
    if PERMANENT_BRAIN_ENABLED:
        thinking_active = True
        background_thread = threading.Thread(target=permanent_brain_worker, daemon=True)
        background_thread.start()
        if DEBUG:
            print("Permanent Brain initialized and active")
    
    if DEBUG:
        print(f"Agent initialized as player {player_symbol}")
        bit_board.print_board()
    
    return True

def what_is_your_move(board, game_rows, game_cols, my_game_symbol):
    """
    Decide which column to drop a disk.
    """
    global mcts_root, move_count, last_board_state, background_results
    
    start_time = time.time()
    
    if DEBUG:
        print(f"\nMove #{move_count + 1} - Thinking...")
    
    # Convert the board to our representation
    bit_board = BitBoard(board, game_rows, game_cols)
    
    if DEBUG:
        bit_board.print_board()
    
    # Try to use permanent brain results if available
    board_hash = bit_board.to_string()
    if PERMANENT_BRAIN_ENABLED and board_hash in background_results:
        permanent_brain_result = background_results[board_hash]
        best_move = permanent_brain_result['best_move']
        
        # Verify the move is valid
        valid_moves = bit_board.get_valid_moves()
        if best_move in valid_moves:
            if DEBUG:
                print(f"Using permanent brain result: {best_move+1}")
                print(f"Based on {permanent_brain_result['simulations']} simulations")
            
            # Schedule thinking for the next move
            new_board = bit_board.copy()
            new_board.make_move(best_move, True)
            new_root = MCTSNode(new_board, None, -1, False)
            schedule_background_thinking(new_board, new_root)
            
            # Update move counter
            move_count += 1
            
            return best_move + 1  # Convert to 1-indexed
    
    # If this is not the first move, try to find our previous move in the tree
    if mcts_root is not None and mcts_root.children and move_count > 0:
        # Find the opponent's move in the current board compared to our root board
        opponent_move = detect_opponent_move(bit_board, last_board_state)
        
        # If found, find the corresponding child node and make it our new root
        if opponent_move is not None:
            for child in mcts_root.children:
                if child.move == opponent_move:
                    mcts_root = child
                    mcts_root.parent = None  # Detach from parent
                    if DEBUG:
                        print(f"Tree root moved to opponent's move: column {opponent_move+1}")
                    break
            else:
                # If the opponent's move is not in our tree, reset the root
                if DEBUG:
                    print("Opponent move not found in tree, resetting root")
                mcts_root = MCTSNode(bit_board, None, -1, True)
        else:
            # If no opponent move is found, reset the root
            if DEBUG:
                print("No opponent move detected, resetting root")
            mcts_root = MCTSNode(bit_board, None, -1, True)
    else:
        # First move or reset case
        if DEBUG:
            print("First move or tree reset")
        mcts_root = MCTSNode(bit_board, None, -1, True)
    
    # First, check for immediate win
    winning_move = bit_board.find_winning_move(True)
    if winning_move is not None:
        if DEBUG:
            print(f"Immediate win found at column {winning_move+1}")
        
        # Schedule thinking for the next move
        new_board = bit_board.copy()
        new_board.make_move(winning_move, True)
        new_root = MCTSNode(new_board, None, -1, False)
        schedule_background_thinking(new_board, new_root)
        
        move_count += 1
        return winning_move + 1  # Convert to 1-indexed
    
    # Next, check for immediate block
    blocking_move = bit_board.find_blocking_move()
    if blocking_move is not None:
        if DEBUG:
            print(f"Blocking opponent's win at column {blocking_move+1}")
        
        # Schedule thinking for the next move
        new_board = bit_board.copy()
        new_board.make_move(blocking_move, True)
        new_root = MCTSNode(new_board, None, -1, False)
        schedule_background_thinking(new_board, new_root)
        
        move_count += 1
        return blocking_move + 1  # Convert to 1-indexed
    
    # Check for gateway moves
    gateway_moves = bit_board.detect_gateway_pattern()
    if gateway_moves:
        chosen_gateway = random.choice(gateway_moves)
        if DEBUG:
            print(f"Gateway pattern found at column {chosen_gateway+1}")
        
        # Schedule thinking for the next move
        new_board = bit_board.copy()
        new_board.make_move(chosen_gateway, True)
        new_root = MCTSNode(new_board, None, -1, False)
        schedule_background_thinking(new_board, new_root)
        
        move_count += 1
        return chosen_gateway + 1  # Convert to 1-indexed
    
    # Check for trap setups (moves that lead to a forced win)
    trap_moves = bit_board.find_future_winning_moves(True, 2)
    if trap_moves:
        chosen_trap = random.choice(trap_moves)
        if DEBUG:
            print(f"Trap setup found at column {chosen_trap+1}")
        
        # Schedule thinking for the next move
        new_board = bit_board.copy()
        new_board.make_move(chosen_trap, True)
        new_root = MCTSNode(new_board, None, -1, False)
        schedule_background_thinking(new_board, new_root)
        
        move_count += 1
        return chosen_trap + 1  # Convert to 1-indexed
    
    # Perform MCTS search
    simulations = mcts_search(mcts_root, MAX_SIMULATIONS, MAX_TIME_PER_MOVE)
    if DEBUG:
        print(f"Completed {simulations} MCTS simulations")
    
    # Find the best move from statistics
    best_child = None
    best_value = -float('inf')
    
    if mcts_root.children:
        # Select the most visited child
        for child in mcts_root.children:
            value = child.visits  # Use visits as the primary metric
            if DEBUG:
                win_rate = child.wins / child.visits if child.visits > 0 else 0
                print(f"Column {child.move+1}: {child.visits} visits, {win_rate:.2f} win rate")
            
            if value > best_value:
                best_value = value
                best_child = child
        
        # Make sure we have a valid move
        if best_child is not None:
            # Update the root to the chosen child
            best_move = best_child.move
            if DEBUG:
                print(f"Best move: column {best_move+1}")
        else:
            # Fallback to a random move if no children
            valid_moves = bit_board.get_valid_moves()
            best_move = random.choice(valid_moves)
            if DEBUG:
                print(f"Fallback to random move: column {best_move+1}")
    else:
        # Fallback to a random move if no children
        valid_moves = bit_board.get_valid_moves()
        best_move = random.choice(valid_moves)
        if DEBUG:
            print(f"No children, random move: column {best_move+1}")
    
    # After making our move, update the root for the next turn
    if best_child is not None:
        mcts_root = best_child
        mcts_root.parent = None  # Detach from parent
    
    # Schedule thinking for the next move
    new_board = bit_board.copy()
    new_board.make_move(best_move, True)
    new_root = MCTSNode(new_board, None, -1, False)
    schedule_background_thinking(new_board, new_root)
    
    move_count += 1
    
    if DEBUG:
        elapsed = time.time() - start_time
        print(f"Decision took {elapsed:.3f} seconds")
    
    return best_move + 1  # Convert to 1-indexed

def cleanup():
    """Clean up resources when agent is no longer needed"""
    global thinking_active, background_thread, search_queue
    
    if PERMANENT_BRAIN_ENABLED and thinking_active:
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
    # Test the agent with a simple simulation
    print("Team4_MCTS_Permanent_Brain_Agent.py has been imported or executed directly.")
    
    # Create a sample board for testing
    test_rows, test_cols = 6, 7
    test_board = [[' ' for _ in range(test_cols)] for _ in range(test_rows)]
    
    # Initialize agent
    init_agent('X', test_rows, test_cols, test_board)
    
    # Get a move
    move = what_is_your_move(test_board, test_rows, test_cols, 'X')
    print(f"Agent chose move: {move}")
    
    # Clean up when done
    cleanup()
    
    # Ensure cleanup is called when the program exits
    import atexit
    atexit.register(cleanup)
=======
from Strategy.ConnectAgent import ConnectAgent
from Strategy.State import State

# DEFINITIONS
# board = [[' ' for _ in range(cols)] for _ in range(rows)]

# HELPER FUNCTIONS
# Print the Board
def print_board(board):
    """ Prints the connect 4 game board."""
    for row in board:
        print('|' + '|'.join(row) + '|')
    print("-" * (len(board[0]) * 2 + 1))
    print(' ' + ' '.join(str(i+1) for i in range(len(board[0]))))


def init_agent(player_symbol, board_num_rows, board_num_cols, board):
    """ Inits the agent. Should only need to be called once at the start of a game.
    NOTE NOTE NOTE: Do not expect the values you might save in variables to retain
    their values each time a function in this module is called. Therefore, you might
    want to save the variables to a file and re-read them when each function was called.
    This is not to say you should do that. Rather, just letting you know about the variables
    you might use in this module.
    NOTE NOTE NOTE NOTE: All functions called by connect_4_main.py  module will pass in all
    of the variables that you likely will need. So you can probably skip the 'NOTE NOTE NOTE'
    above. """
    num_rows = int(board_num_rows)
    num_cols = int(board_num_cols)

    # game_board = board
    ConnectAgent(num_rows, num_cols, board, player_symbol)

    # my_game_symbol = player_symbol

    return True

def what_is_your_move(board, game_rows, game_cols, my_game_symbol):
   """ Decide your move, i.e., which column to drop a disk. """

   # Insert your agent code HERE to decide which column to drop/insert your disk.
   ConnectAgent.add_state(State(board,ConnectAgent.state[-1].steps+1))
   if (ConnectAgent.state[-1].one_step_win(my_game_symbol) != None):
      return ConnectAgent.state[-1].one_step_win(my_game_symbol)+1
   return random.randint(1, game_cols)


#####
# MAKE SURE MODULE IS IMPORTED
if __name__ == "__main__":
   print("Team3_Connect_4_Agent.py  is intended to be imported and not executed.") 
else:
   print("Team3_Connect_4_Agent.py  has been imported.")


# #!/usr/bin/env python3
# """
# Connect 4 AI Agent Implementation
# CS156 Final Project - Spring 2025

# This module serves as the main entry point for the Connect 4 AI agent.
# It implements the required interface functions and coordinates the
# search, representation, and reasoning components.
# """

# import random
# import time
# from search import minimax_search, alpha_beta_search
# from representation import GameState
# from reasoning import evaluate_board, get_valid_moves

# # Configuration
# MAX_DEPTH = 5  # Maximum search depth for minimax
# USE_ALPHA_BETA = True  # Use alpha-beta pruning for performance
# CENTER_PREFERENCE = True  # Prefer center columns in early game
# DEBUG_MODE = False  # Set to True to enable debug prints

# # Global variables to maintain state between function calls
# game_state = None
# player_symbol = None
# opponent_symbol = None

# def print_board(board):
#     """Utility function to print the board for debugging"""
#     rows = len(board)
#     cols = len(board[0])
    
#     print("\nCurrent Board State:")
#     print("-" * (2 * cols + 1))
    
#     for r in range(rows):
#         print("|", end="")
#         for c in range(cols):
#             print(f"{board[r][c]}|", end="")
#         print("")
    
#     print("-" * (2 * cols + 1))
#     print(" ", end="")
#     for c in range(cols):
#         print(f"{c+1} ", end="")
#     print("\n")

# def init_agent(player_sym, board_num_rows, board_num_cols, board):
#     """
#     Initializes the agent. Should only need to be called once at the start of a game.
    
#     Args:
#         player_sym: The symbol representing this agent ('X' or 'O')
#         board_num_rows: Number of rows in the board
#         board_num_cols: Number of columns in the board
#         board: The initial board state
        
#     Returns:
#         True indicating successful initialization
#     """
#     global game_state, player_symbol, opponent_symbol
    
#     # Store the player's symbol and determine the opponent's symbol
#     player_symbol = player_sym
#     opponent_symbol = 'O' if player_sym == 'X' else 'X'
    
#     # Initialize the game state representation
#     game_state = GameState(
#         board=board,
#         num_rows=int(board_num_rows),
#         num_cols=int(board_num_cols),
#         player_symbol=player_symbol,
#         opponent_symbol=opponent_symbol
#     )
    
#     if DEBUG_MODE:
#         print(f"Agent initialized as player {player_symbol}")
#         print_board(board)
    
#     return True

# def what_is_your_move(board, game_rows, game_cols, my_game_symbol):
#     """
#     Decide which column to drop a disk.
    
#     Args:
#         board: Current state of the board
#         game_rows: Number of rows in the board
#         game_cols: Number of columns in the board
#         my_game_symbol: The symbol representing this agent ('X' or 'O')
        
#     Returns:
#         An integer representing the column to drop the disk (1 to game_cols)
#     """
#     global game_state, player_symbol, opponent_symbol
    
#     start_time = time.time()
    
#     # If game_state is None, reinitialize it (should not happen in normal gameplay)
#     if game_state is None:
#         init_agent(my_game_symbol, game_rows, game_cols, board)
#     else:
#         # Update the game state with the current board
#         game_state.update_board(board)
    
#     # Get valid moves (non-full columns)
#     valid_moves = get_valid_moves(board, game_rows, game_cols)
    
#     if not valid_moves:
#         # No valid moves (should not happen in normal gameplay)
#         if DEBUG_MODE:
#             print("No valid moves found!")
#         return 1  # Return a default move, which will be rejected if invalid
    
#     # In the first couple of moves, prefer the center columns if possible
#     move_count = sum(1 for r in range(game_rows) for c in range(game_cols) if board[r][c] != ' ')
    
#     if CENTER_PREFERENCE and move_count < 4:
#         middle_cols = [game_cols // 2 + 1]  # Center column (1-indexed)
        
#         # If even number of columns, consider both center columns
#         if game_cols % 2 == 0:
#             middle_cols = [game_cols // 2, game_cols // 2 + 1]
        
#         # Check if any middle columns are valid, if so, choose randomly from them
#         valid_middle_cols = [col for col in middle_cols if col in valid_moves]
#         if valid_middle_cols:
#             chosen_move = random.choice(valid_middle_cols)
#             if DEBUG_MODE:
#                 print(f"Early game center preference: column {chosen_move}")
#             return chosen_move
    
#     # Use minimax with alpha-beta pruning to find the best move
#     if USE_ALPHA_BETA:
#         best_move = alpha_beta_search(game_state, MAX_DEPTH)
#     else:
#         best_move = minimax_search(game_state, MAX_DEPTH)
    
#     # Ensure the chosen move is valid
#     if best_move not in valid_moves:
#         if DEBUG_MODE:
#             print(f"Search returned invalid move {best_move}, choosing randomly from: {valid_moves}")
#         best_move = random.choice(valid_moves)
    
#     end_time = time.time()
    
#     if DEBUG_MODE:
#         print(f"Chose column {best_move} (took {end_time - start_time:.3f} seconds)")
#         print_board(board)
    
#     return best_move

# # If run directly, the agent can be tested with random gameplay
# if __name__ == "__main__":
#     # Sample code to test the agent
#     rows, cols = 6, 7
#     test_board = [[' ' for _ in range(cols)] for _ in range(rows)]
    
#     init_agent('X', rows, cols, test_board)
    
#     # Simulate a few random moves
#     print("Simulating random gameplay to test agent...")
#     for _ in range(3):
#         column = what_is_your_move(test_board, rows, cols, 'X')
#         print(f"Agent chose column: {column}")
        
#         # Update board with agent's move
#         for r in range(rows-1, -1, -1):
#             if test_board[r][column-1] == ' ':
#                 test_board[r][column-1] = 'X'
#                 break
        
#         # Simulate opponent's random move
#         valid_cols = [c+1 for c in range(cols) if test_board[0][c] == ' ']
#         if valid_cols:
#             opp_col = random.choice(valid_cols)
#             for r in range(rows-1, -1, -1):
#                 if test_board[r][opp_col-1] == ' ':
#                     test_board[r][opp_col-1] = 'O'
#                     break
#             print(f"Opponent chose column: {opp_col}")
        
#         print_board(test_board)
>>>>>>> 4ffec52dff4ca5d10c6abe1008fcdaf277aba156
