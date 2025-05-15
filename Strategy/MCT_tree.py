from Strategy.State import State
from Strategy.constant import ROWS, COLS
import random
import math
import copy
import time

def print_board(board):
    """Prints the connect 4 game board."""
    for row in board:
        print("|" + "|".join(row) + "|")
    print("-" * (len(board[0]) * 2 + 1))
    print(" " + " ".join(str(i + 1) for i in range(len(board[0]))))
    
class MCTNode:
    """Monte Carlo Tree Search Node class with advanced features"""
    def __init__(self, state, parent=None, move=None, player_symbol=None, x=None, y=None, rows=ROWS, cols=COLS):
        self.rows = rows
        self.cols = cols
        self.state = state
        self.parent = parent
        self.move = move  # The move that led to this state
        self.player_symbol = player_symbol  # The player who made the move
        self.children = [-1] * self.cols  # Initialize with -1 (no child)
        self.visits = 0
        self.value = 0
        self.x = x  # Row coordinate
        self.y = y  # Column coordinate (move)
    
    def score(self):
        """Calculate UCB score for this node"""
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits
    
    def is_leaf(self):
        """Check if node is a leaf node (has no children)"""
        for child in self.children:
            if child != -1:
                return False
        return True
    
    def reset(self, player_symbol, parent):
        """Reset node values"""
        self.player_symbol = player_symbol
        self.parent = parent
        self.children = [-1] * self.cols
        self.visits = 0
        self.value = 0
    
    def set_point(self, x, y):
        """Set the position coordinates"""
        self.x = x
        self.y = y

class MCTSearch:
    """Enhanced Monte Carlo Tree Search implementation for Connect 4"""
    def __init__(self, initial_state, my_symbol, opponent_symbol, simulation_limit=1000, time_limit=None, rows=ROWS, cols=COLS):
        # Constants from the C++ implementation
        self.UCB_C = 1.414  # Exploration constant
        self.K = 10         # For the RAVE weight parameter

        self.rows = rows
        self.cols = cols
        
        # Enhanced column preference constants
        self.COLUMN_PREFERENCE = self._calculate_column_preference()
        
        # Initialize board states
        self.init_state = initial_state
        self.current_state = copy.deepcopy(initial_state)  # Keep a reference to current state
        self.working_state = copy.deepcopy(initial_state)  # State used during traversal
        
        # Player information
        self.my_symbol = my_symbol
        self.opponent_symbol = opponent_symbol
        self.player_number = {"X": 1, "O": 2}
        self.player_symbol = {1: "X", 2: "O"}
        
        # Search parameters
        self.simulation_limit = simulation_limit
        self.time_limit = time_limit
        self.last_move = -1
        
        # Statistics for RAVE implementation
        self.move_value = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.move_cnt = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        
        # For move selection
        self.score = [0] * self.cols
        self.next_move = [0] * self.cols
        
        # Initialize the tree
        self.nodes = []
        # Use correct player number mapping
        root_player = 1 if my_symbol == "X" else 2
        self.root = self._new_node(root_player, -1,rows=self.rows, cols=self.cols)
        
        # Debugging
        self.debug_mode = False
    
    def _calculate_column_preference(self):
        """Calculate preference weights for columns with stronger middle bias"""
        preferences = [0] * self.cols
        mid = self.cols // 2  # Middle column (3 for standard 7-column board)
        
        for i in range(self.cols):
            # Calculate distance from middle (0 for middle, increases as you move away)
            distance = abs(i - mid)
            
            # Higher score for middle columns, decreasing as you move outward
            # Exponential decay ensures middle is significantly preferred
            preferences[i] = math.exp(-0.5 * distance)
            
        # Normalize to sum to 1.0
        total = sum(preferences)
        preferences = [p/total for p in preferences]
        
        return preferences
    
    def _new_node(self, player, parent, x=None, y=None, rows=ROWS, cols=COLS):
        """Create a new node and add it to the node pool"""
        node = MCTNode(self.working_state, parent, None, self.player_symbol[player], rows=rows, cols=cols)
        self.nodes.append(node)
        return len(self.nodes) - 1
    
    def move_score(self, node_idx):
        """Get RAVE score for a move"""
        x = self.nodes[node_idx].x
        y = self.nodes[node_idx].y
        if x is None or y is None:
            return 0
        return self.move_value[x][y] / (self.move_cnt[x][y] + 0.01)
    
    def best_move(self, node_idx):
        """Select best child move using UCB1 with RAVE and middle column preference"""
        node = self.nodes[node_idx]
        best_idx = -1
        max_score = float('-inf')
        log_n = math.log(node.visits + 0.001)
        beta = math.sqrt(self.K / (3 * node.visits + self.K))
        for i in range(self.cols):
            child_idx = node.children[i]
            if child_idx == -1:
                continue
            
            child = self.nodes[child_idx]
            if child.visits == 0:
                return i
            
            # Combined UCB score with RAVE and column preference
            score = ((1 - beta) * child.score() + 
                     beta * self.move_score(child_idx) +
                     self.UCB_C * math.sqrt(log_n / child.visits))
            
            # Apply column preference multiplier
            score *= (1.0 + self.COLUMN_PREFERENCE[i])
            
            if score > max_score:
                max_score = score
                best_idx = i
                
        return best_idx
    
    def back_up(self, node_idx, value):
        """Backpropagate results through the tree with RAVE"""
        while node_idx != -1:
            node = self.nodes[node_idx]
            x, y = node.x, node.y
            
            # Update node statistics
            node.visits += 1
            node.value += value
            
            # Update RAVE statistics
            if x is not None and y is not None:
                print(f"Updating RAVE for node ({x}, {y})")
                self.move_cnt[x][y] += 1
                self.move_value[x][y] += value
            
            # Negate value for opponent
            value = -value
            node_idx = node.parent
    
    def select(self):
        """Select a promising node to expand using UCB1"""
        node_idx = self.root
        player = self.nodes[self.root].player_symbol
        player_num = self.player_number[player]
        # Reset working state for this traversal
        self.working_state = copy.deepcopy(self.init_state)
        
        while not self.nodes[node_idx].is_leaf():
            move_idx = self.best_move(node_idx)
            if move_idx == -1:
                break
                
            node_idx = self.nodes[node_idx].children[move_idx]
            
            # Apply move to working state
            self.working_state.move(move_idx, player)
            player = "X" if player == "O" else "O"
            player_num = 3 - player_num
            
        return node_idx
    
    def enhanced_center_sample(self, move_indices):
        """Sample moves with strong preference for center columns"""
        if not move_indices:
            return -1
            
        # Calculate weighted probabilities for available moves
        weights = []
        for idx in move_indices:
            weights.append(self.COLUMN_PREFERENCE[idx])
            
        # Normalize weights
        total_weight = sum(weights)
        if total_weight <= 0:
            # Fallback to uniform distribution
            return random.choice(move_indices)
            
        # Convert to probabilities
        probabilities = [w/total_weight for w in weights]
        
        # Weighted random choice
        r = random.random()
        cumulative = 0
        for i, p in enumerate(probabilities):
            cumulative += p
            if r <= cumulative:
                return move_indices[i]
                
        # Fallback - should rarely happen due to floating point precision
        return move_indices[-1]
    
    def score_sample(self, move_num, player_symbol):
        """Sample moves based on heuristic scoring with middle column preference"""
        total_score = 0
        player_num = self.player_number[player_symbol]
        
        for i in range(move_num):
            move = self.next_move[i]
            # Get score based on potential connects
            base_score = self._move_score(move, player_num)
            # Apply column preference multiplier
            self.score[i] = base_score * (1.5 + self.COLUMN_PREFERENCE[move])
            total_score += self.score[i]
            
        if total_score == 0:
            # Fallback to pure column preference
            available_moves = [self.next_move[i] for i in range(move_num)]
            return self.enhanced_center_sample(available_moves)
            
        rand_num = random.random() * total_score
        cumulative = 0
        move = 0
        
        for i in range(move_num):
            cumulative += self.score[i]
            if rand_num <= cumulative:
                move = i
                break
            
        return self.next_move[move]
    
    def _check_open_threes(self, board, row, col, symbol):
        """
        Check if placing a piece at (row, col) creates an open-ended three-in-a-row
        Returns the dangerous column(s) that would complete a 4-in-a-row next turn
        """
        directions = [
            (0, 1),  # horizontal
            (1, 0),  # vertical
            (1, 1),  # diagonal \
            (1, -1)  # diagonal /
        ]
        
        threat_cols = []
        
        # Place the piece temporarily
        board[row][col] = symbol
        
        for dr, dc in directions:
            # Scan in both forward and backward directions
            for factor in [1, -1]:
                count = 1  # Count consecutive pieces (start with the placed one)
                open_ends = []
                
                # Check forward and backward directions within same loop
                for step in range(1, 4):  # Look up to 3 steps in each direction
                    r = row + dr * step * factor
                    c = col + dc * step * factor
                    
                    # Check bounds
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        if board[r][c] == symbol:
                            count += 1
                        elif board[r][c] == ' ':
                            # This is an open end - store the column
                            # Calculate bottom row for this column (where piece would land)
                            bottom_row = r
                            while bottom_row < self.rows - 1 and board[bottom_row + 1][c] == ' ':
                                bottom_row += 1
                            
                            # Only consider this a threat if it's a valid move
                            if bottom_row == r:
                                open_ends.append(c)
                            break  # Stop counting after an empty space
                        else:
                            break  # Stop counting after opponent's piece
            
            # Check if we have a dangerous formation (2 or 3 in a row with an open end)
            if count >= 2 and len(open_ends) > 0:
                for end_col in open_ends:
                    if end_col not in threat_cols:
                        threat_cols.append(end_col)
                        
        # Remove the piece to restore the board
        board[row][col] = ' '
        
        return threat_cols

    def _detect_special_threats(self, board, player_symbol):
        """
        Detect special threat patterns like open-ended threes
        Returns the column to block if a threat is found, -1 otherwise
        """
        opponent_symbol = "O" if player_symbol == "X" else "X"
        
        # First check if I can win next move
        for col in range(self.cols):
            if board[0][col] == ' ':  # If column not full
                # Find the row where the piece would land
                row = 0
                while row < self.rows - 1 and board[row + 1][col] == ' ':
                    row += 1
                    
                # Check if this move wins
                temp_board = copy.deepcopy(board)
                temp_board[row][col] = player_symbol
                if self._check_win(temp_board, row, col, player_symbol):
                    if self.debug_mode:
                        print(f"Found winning move: {col}")
                    return col
        
        # Detect opponent's threats
        threat_columns = {}  # Column -> number of threats
        
        # Check opponent's potential moves for threats
        for col in range(self.cols):
            if board[0][col] == ' ':  # If column not full
                # Find the row where the piece would land
                row = 0
                while row < self.rows - 1 and board[row + 1][col] == ' ':
                    row += 1
                
                # 1. Direct win threat - opponent wins in next move
                temp_board = copy.deepcopy(board)
                temp_board[row][col] = opponent_symbol
                if self._check_win(temp_board, row, col, opponent_symbol):
                    if self.debug_mode:
                        print(f"Found immediate blocking move: {col} (opponent would win)")
                    return col
                
                # 2. Check for open-ended three-in-a-row threats
                threat_cols = self._check_open_threes(board, row, col, opponent_symbol)
                if threat_cols:
                    for tc in threat_cols:
                        if tc in threat_columns:
                            threat_columns[tc] += 1
                        else:
                            threat_columns[tc] = 1
                    
                    if self.debug_mode:
                        print(f"Move {col} creates threats at columns: {threat_cols}")
                    
        # 3. If placing our piece creates a double threat, do it
        for col in range(self.cols):
            if board[0][col] == ' ':  # If column not full
                # Find the row where the piece would land
                row = 0
                while row < self.rows - 1 and board[row + 1][col] == ' ':
                    row += 1
                
                # Check if our move creates multiple winning threats
                my_threats = self._check_open_threes(board, row, col, player_symbol)
                if len(my_threats) >= 2:
                    if self.debug_mode:
                        print(f"Found offensive double threat at column: {col}")
                    return col
        
        # 4. Handle opponent's open-ended three threats
        if threat_columns:
            # Find the most threatening column (one that enables most threats)
            max_threats = 0
            most_dangerous_col = -1
            
            for col, threat_count in threat_columns.items():
                if threat_count > max_threats:
                    max_threats = threat_count
                    most_dangerous_col = col
            
            if most_dangerous_col != -1:
                if self.debug_mode:
                    print(f"Found strategic blocking move: {most_dangerous_col} (blocks {max_threats} threats)")
                return most_dangerous_col
            
        return -1  # No special threats found
    
    def _check_win(self, board, row, col, symbol):
        """Check if there's a win at the given position"""
        # Check horizontal
        count = 0
        for c in range(max(0, col-3), min(col+4, self.cols)):
            if board[row][c] == symbol:
                count += 1
                if count >= 4:
                    return True
            else:
                count = 0
                
        # Check vertical
        count = 0
        for r in range(max(0, row-3), min(row+4, self.rows)):
            if board[r][col] == symbol:
                count += 1
                if count >= 4:
                    return True
            else:
                count = 0
                
        # Check diagonal /
        count = 0
        for i in range(-3, 4):
            r = row - i
            c = col + i
            if 0 <= r < self.rows and 0 <= c < self.cols:
                if board[r][c] == symbol:
                    count += 1
                    if count >= 4:
                        return True
                else:
                    count = 0
                    
        # Check diagonal \
        count = 0
        for i in range(-3, 4):
            r = row + i
            c = col + i
            if 0 <= r < self.rows and 0 <= c < self.cols:
                if board[r][c] == symbol:
                    count += 1
                    if count >= 4:
                        return True
                else:
                    count = 0
                    
        return False

    def _move_score(self, col, player_num):
        """Score a move based on potential to make connects with enhanced threat detection"""
        # Create a temporary state
        temp_state = copy.deepcopy(self.working_state)
        
        # Try the move
        if temp_state.top[col] <= 0:
            return 0
        
        # Simple scoring - check if it creates threats
        row = temp_state.top[col] - 1
        player_symbol = self.player_symbol[player_num]
        opponent_symbol = "O" if player_symbol == "X" else "X"
        
        # Apply move
        temp_board = copy.deepcopy(temp_state.board)
        temp_board[row][col] = player_symbol
        
        # Count potential threats
        score = 1  # Base score
        
        # Apply column preference directly to base score
        score *= (1.0 + 2.0 * self.COLUMN_PREFERENCE[col])
        
        # NEW: Check first if this move blocks an immediate win
        for test_col in range(self.cols):
            if temp_state.top[test_col] > 0:
                test_row = temp_state.top[test_col] - 1
                test_board = copy.deepcopy(temp_state.board)
                test_board[test_row][test_col] = opponent_symbol
                if self._check_win(test_board, test_row, test_col, opponent_symbol):
                    if test_col == col:  # We're blocking this threat
                        score += 50  # Very high priority for blocking immediate wins
                
        # NEW: Check for open-ended three threats by opponent
        opponent_threats = self._check_open_threes(temp_state.board, row, col, opponent_symbol)
        if opponent_threats:
            if col in opponent_threats:  # We're blocking a threat column
                score += 30
            
        # NEW: Check if our move creates a double threat (very powerful)
        my_threats = self._check_open_threes(temp_state.board, row, col, player_symbol)
        if len(my_threats) >= 2:
            score += 40  # Double threats are very strong
            
        # Check horizontals (improved scoring)
        for c in range(max(0, col-3), min(col+1, self.cols-3)):
            window = [temp_board[row][c+i] for i in range(4)]
            player_count = window.count(player_symbol)
            empty_count = window.count(' ')
            if player_count == 3 and empty_count == 1:
                score += 15  # Increased from 10
            elif player_count == 2 and empty_count == 2:
                score += 5   # Increased from 3
                
        # Check verticals (improved scoring)
        if row <= self.rows - 4:
            window = [temp_board[row+i][col] for i in range(4)]
            player_count = window.count(player_symbol)
            empty_count = window.count(' ')
            if player_count == 3 and empty_count == 1:
                score += 15  # Increased from 10
            elif player_count == 2 and empty_count == 2:
                score += 5   # Increased from 3
                
        # Check diagonal / (improved scoring)
        for r, c in zip(range(min(row+3, self.rows-1), max(row-4, -1), -1), 
                       range(max(col-3, 0), min(col+4, self.cols))):
            if 0 <= r-3 < self.rows and 0 <= c+3 < self.cols:
                window = [temp_board[r-i][c+i] for i in range(4)]
                player_count = window.count(player_symbol)
                empty_count = window.count(' ')
                if player_count == 3 and empty_count == 1:
                    score += 15  # Increased from 10
                elif player_count == 2 and empty_count == 2:
                    score += 5   # Increased from 3
                    
        # Check diagonal \ (improved scoring)
        for r, c in zip(range(max(row-3, 0), min(row+4, self.rows)), 
                       range(max(col-3, 0), min(col+4, self.cols))):
            if 0 <= r+3 < self.rows and 0 <= c+3 < self.cols:
                window = [temp_board[r+i][c+i] for i in range(4)]
                player_count = window.count(player_symbol)
                empty_count = window.count(' ')
                if player_count == 3 and empty_count == 1:
                    score += 15  # Increased from 10
                elif player_count == 2 and empty_count == 2:
                    score += 5   # Increased from 3
                    
        return score
    
    def smart_policy(self, state, player_symbol):
        """Choose move based on heuristic scoring with middle column preference"""
        # First check for special threats
        board = state.board
        special_move = self._detect_special_threats(board, player_symbol)
        if special_move != -1:
            return special_move
            
        # Normal move selection via scoring
        move_num = 0
        for i in range(self.cols):
            if state.top[i] > 0:  # If column not full
                self.next_move[move_num] = i
                move_num += 1
                
        if move_num == 0:
            return -1
            
        return self.score_sample(move_num, player_symbol)
    
    def random_policy(self):
        """Choose move with enhanced center preference"""
        move_indices = []
        for i in range(self.cols):
            if self.working_state.top[i] > 0:  # If column not full
                move_indices.append(i)
                
        if not move_indices:
            return -1
            
        return self.enhanced_center_sample(move_indices)
    
    def expand(self, node_idx):
        """Expand a node by adding all possible child nodes with middle column preference"""
        node = self.nodes[node_idx]
        player_symbol = node.player_symbol
        player_num = self.player_number[player_symbol]
        opponent_symbol = "O" if player_symbol == "X" else "X"
        opponent_num = 3 - player_num
        
        # Check for terminal state
        if self._is_terminal(self.working_state):
            return node_idx
            
        # NEW: Check for special threat patterns (winning moves and blocking)
        special_move = self._detect_special_threats(self.working_state.board, player_symbol)
        if special_move != -1:
            if 0 <= special_move < self.cols and self.working_state.top[special_move] > 0:
                row = self.working_state.top[special_move] - 1
                
                # Create child for this urgent move and return it
                if node.children[special_move] == -1:
                    node.children[special_move] = self._new_node(opponent_num, node_idx, row, special_move,rows = self.rows, cols=self.cols)
                
                # Clear other children for focus
                for i in range(self.cols):
                    if i != special_move:
                        node.children[i] = -1
                
                return node.children[special_move]
        
        # Apply middle column preference for non-urgent expansion
        # Prioritize expanding middle columns first
        for i in sorted(range(self.cols), key=lambda x: -self.COLUMN_PREFERENCE[x]):
            if self.working_state.top[i] > 0:  # If column not full
                # Calculate row position where piece would land
                row = self.working_state.top[i] - 1
                
                if node.children[i] == -1:
                    node.children[i] = self._new_node(opponent_num, node_idx, row, i,rows=self.rows, cols=self.cols)
                
                return node.children[i]
        
        # No valid moves
        return node_idx
    
    def _is_terminal(self, state):
        """Check if state is terminal (game over)"""
        # Check for winner
        if self._has_winner(state):
            return True
            
        # Check if board is full
        for col in range(self.cols):
            if state.top[col] > 0:  # If any column has space
                return False
                
        return True  # Board is full
    
    def _has_winner(self, state):
        """Check if state has a winner"""
        return state.is_winning_state("X") or state.is_winning_state("O")
    
    def _is_winning_move(self, col, player_symbol):
        """Check if move would create a win"""
        if self.working_state.top[col] <= 0:
            return False
            
        # Create temporary state
        temp_state = copy.deepcopy(self.working_state)
        
        # Try the move
        temp_state.move(col, player_symbol)
        
        # Check if it's a win
        return temp_state.is_winning_state(player_symbol)
    
    def _is_losing_move(self, col, player_symbol):
        """Check if move would allow opponent to win next turn"""
        if self.working_state.top[col] <= 0:
            return True
            
        # Create temporary state
        temp_state = copy.deepcopy(self.working_state)
        opponent_symbol = "O" if player_symbol == "X" else "X"
        
        # Try the move
        temp_state.move(col, player_symbol)
        
        # Check if opponent would have a winning move
        for opp_col in range(self.cols):
            if temp_state.top[opp_col] > 0:
                temp_state2 = copy.deepcopy(temp_state)
                temp_state2.move(opp_col, opponent_symbol)
                if temp_state2.is_winning_state(opponent_symbol):
                    return True
                    
        return False
    
    def rollout(self, node_idx):
        """Simulate random playout from node with middle column preference"""
        node = self.nodes[node_idx]
        player_symbol = node.player_symbol
        opponent_symbol = "O" if player_symbol == "X" else "X"
        
        # Determine sign for backpropagation
        sgn = 1 if player_symbol == self.my_symbol else -1
        
        # Create copy of working state for simulation
        sim_state = copy.deepcopy(self.working_state)
        current_player = player_symbol
        
        # Simulate until terminal state
        while not self._is_terminal(sim_state):
            # Check for immediate win/loss
            if sim_state.is_winning_state(self.my_symbol):
                return sgn
            if sim_state.is_winning_state(self.opponent_symbol):
                return -sgn
            
            # Use smart policy rather than random
            move = self.smart_policy(sim_state, current_player)
            if move == -1:  # No valid moves
                break
                
            # Apply move
            sim_state.move(move, current_player)
            
            # Switch player
            current_player = opponent_symbol if current_player == player_symbol else player_symbol
        
        return 0  # Draw
    
    def final_decision(self):
        """Make final move decision based on node visits, RAVE scores, and middle column preference"""
        # Check for special threat patterns one last time
        special_move = self._detect_special_threats(self.current_state.board, self.my_symbol)
        if special_move != -1:
            if self.debug_mode:
                print(f"Found critical move in final decision: {special_move}")
            return special_move
            
        best_move = -1
        max_score = float('-inf')
        
        # Calculate beta for RAVE weight
        beta = math.sqrt(self.K / (3 * self.nodes[self.root].visits + self.K))
        log_n = math.log(self.nodes[self.root].visits + 0.001)
        
        # Track move scores and visits for debugging
        move_stats = []
        
        for i in range(self.cols):
            child_idx = self.nodes[self.root].children[i]
            if child_idx == -1:
                continue
                
            child = self.nodes[child_idx]
            
            # Combined score with RAVE and enhanced middle preference
            base_score = ((1 - beta) * child.score() + 
                        beta * self.move_score(child_idx))
            
            # Apply middle column preference
            preference_bonus = self.COLUMN_PREFERENCE[i] * 1.5  # Stronger effect for final decision
            score = base_score + preference_bonus
            
            # For debugging
            move_stats.append((i, score, child.visits, child.value, preference_bonus))
            
            if score > max_score:
                max_score = score
                best_move = i
        
        # Sort and print move statistics
        move_stats.sort(key=lambda x: -x[1])
        # for move, score, visits, value, bonus in move_stats:
        #     print(f"Move: {move} Score: {score:.4f} (bonus: {bonus:.4f}) Visits: {visits} Value: {value}")
        
        # Apply final randomization with strong bias to top moves
        # This prevents the AI from being too predictable
        if random.random() < 0.2 and len(move_stats) > 1:
            # 20% chance to choose second-best move if it's close to best
            second_best = move_stats[1][0]
            best = move_stats[0][0]
            score_diff = move_stats[0][1] - move_stats[1][1]
            
            if score_diff < 0.1:  # If scores are close
                best_move = second_best if random.random() < 0.5 else best
        
        return best_move
    
    def get_best_move(self):
        """Run MCTS and return best move"""
        start_time = time.time()
        
        if self.time_limit:
            # Time-based search
            while (time.time() - start_time) < self.time_limit:
                # Phase 1: Selection
                node_idx = self.select()
                
                # Phase 2: Expansion
                if self.nodes[node_idx].visits > 0:
                    node_idx = self.expand(node_idx)
                
                # Phase 3: Simulation
                result = self.rollout(node_idx)
                
                # Phase 4: Backpropagation
                self.back_up(node_idx, result)
        else:
            # Simulation-count-based search
            for _ in range(self.simulation_limit):
                # Phase 1: Selection
                node_idx = self.select()
                
                # Phase 2: Expansion
                if self.nodes[node_idx].visits > 0:
                    node_idx = self.expand(node_idx)
                
                # Phase 3: Simulation
                result = self.rollout(node_idx)
                
                # Phase 4: Backpropagation
                self.back_up(node_idx, result)
        
        # Print column preferences for clarity
        if (self.debug_mode):
            print("Column Preferences:", [f"{i}: {p:.4f}" for i, p in enumerate(self.COLUMN_PREFERENCE)])
        
        # Make final decision
        self.last_move = self.final_decision()
        return self.last_move