from Strategy.State import State
from Strategy.constant import ROWS, COLS
import random
import math
import copy
import time

class MCTNode:
    """Monte Carlo Tree Search Node class with advanced features"""
    def __init__(self, state, parent=None, move=None, player_symbol=None, x=None, y=None):
        self.state = state
        self.parent = parent
        self.move = move  # The move that led to this state
        self.player_symbol = player_symbol  # The player who made the move
        self.children = [-1] * COLS  # Initialize with -1 (no child)
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
        self.children = [-1] * COLS
        self.visits = 0
        self.value = 0
    
    def set_point(self, x, y):
        """Set the position coordinates"""
        self.x = x
        self.y = y

class MCTSearch:
    """Enhanced Monte Carlo Tree Search implementation for Connect 4"""
    def __init__(self, initial_state, my_symbol, opponent_symbol, simulation_limit=1000, time_limit=None):
        # Constants from the C++ implementation
        self.UCB_C = 1.414  # Exploration constant
        self.K = 10         # For the RAVE weight parameter
        
        # Statistics for RAVE implementation
        self.move_value = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.move_cnt = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        
        # For move selection
        self.score = [0] * COLS
        self.next_move = [0] * COLS
        
        # Initialize the tree
        self.nodes = []
        self.root = self._new_node(3 if my_symbol == "X" else 2, -1)
        
        self.init_state = initial_state
        self.current_state = copy.deepcopy(initial_state)  # Keep a reference to current state
        self.working_state = copy.deepcopy(initial_state)  # State used during traversal
        self.my_symbol = my_symbol
        self.opponent_symbol = opponent_symbol
        self.player_number = {"X": 1, "O": 2}
        self.player_symbol = {1: "X", 2: "O"}
        self.simulation_limit = simulation_limit
        self.time_limit = time_limit
        self.last_move = -1
    
    def _new_node(self, player, parent, x=None, y=None):
        """Create a new node and add it to the node pool"""
        node = MCTNode(self.working_state, parent, None, self.player_symbol[player], x, y)
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
        """Select best child move using UCB1 with RAVE"""
        node = self.nodes[node_idx]
        best_idx = -1
        max_score = float('-inf')
        log_n = math.log(node.visits + 0.001)
        beta = math.sqrt(self.K / (3 * node.visits + self.K))
        
        for i in range(COLS):
            child_idx = node.children[i]
            if child_idx == -1:
                continue
            
            child = self.nodes[child_idx]
            if child.visits == 0:
                return i
            
            # Combined UCB score with RAVE
            score = ((1 - beta) * child.score() + 
                     beta * self.move_score(child_idx) +
                     self.UCB_C * math.sqrt(log_n / child.visits))
            
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
    
    def center_sample(self, move_num):
        """Sample moves with preference for center columns"""
        total_score = 0
        mid = move_num // 2
        
        for i in range(move_num):
            self.score[i] = (i // 2 + 1) if i < mid else ((move_num - 1 - i) // 2 + 1)
            total_score += self.score[i]
            
        if total_score == 0:
            return 0
            
        rand_num = random.randint(0, total_score - 1)
        move = -1
        
        while rand_num >= 0:
            move += 1
            rand_num -= self.score[move]
            
        return move
    
    def score_sample(self, move_num, player_symbol):
        """Sample moves based on heuristic scoring"""
        total_score = 0
        player_num = self.player_number[player_symbol]
        
        for i in range(move_num):
            move = self.next_move[i]
            # Get score based on potential connects
            self.score[i] = self._move_score(move, player_num)
            total_score += self.score[i]
            
        if total_score == 0:
            return 0
            
        rand_num = random.randint(0, total_score - 1)
        move = -1
        
        while rand_num >= 0:
            move += 1
            rand_num -= self.score[move]
            
        return move
    
    def _move_score(self, col, player_num):
        """Score a move based on potential to make connects"""
        # Create a temporary state
        temp_state = copy.deepcopy(self.working_state)
        
        # Try the move
        if temp_state.top[col] <= 0:
            return 0
        
        # Simple scoring - check if it creates threats
        row = temp_state.top[col] - 1
        player_symbol = self.player_symbol[player_num]
        
        # Apply move
        temp_board = copy.deepcopy(temp_state.board)
        temp_board[row][col] = player_symbol
        
        # Count potential threats
        score = 1  # Base score
        
        # Check horizontals
        for c in range(max(0, col-3), min(col+1, COLS-3)):
            window = [temp_board[row][c+i] for i in range(4)]
            player_count = window.count(player_symbol)
            empty_count = window.count(' ')
            if player_count == 3 and empty_count == 1:
                score += 10
            elif player_count == 2 and empty_count == 2:
                score += 3
                
        # Check verticals
        if row <= ROWS - 4:
            window = [temp_board[row+i][col] for i in range(4)]
            player_count = window.count(player_symbol)
            empty_count = window.count(' ')
            if player_count == 3 and empty_count == 1:
                score += 10
            elif player_count == 2 and empty_count == 2:
                score += 3
                
        # Check diagonal /
        for r, c in zip(range(min(row+3, ROWS-1), max(row-4, -1), -1), 
                       range(max(col-3, 0), min(col+4, COLS))):
            if 0 <= r-3 < ROWS and 0 <= c+3 < COLS:
                window = [temp_board[r-i][c+i] for i in range(4)]
                player_count = window.count(player_symbol)
                empty_count = window.count(' ')
                if player_count == 3 and empty_count == 1:
                    score += 10
                elif player_count == 2 and empty_count == 2:
                    score += 3
                    
        # Check diagonal \
        for r, c in zip(range(max(row-3, 0), min(row+4, ROWS)), 
                       range(max(col-3, 0), min(col+4, COLS))):
            if 0 <= r+3 < ROWS and 0 <= c+3 < COLS:
                window = [temp_board[r+i][c+i] for i in range(4)]
                player_count = window.count(player_symbol)
                empty_count = window.count(' ')
                if player_count == 3 and empty_count == 1:
                    score += 10
                elif player_count == 2 and empty_count == 2:
                    score += 3
                    
        return score
    
    def smart_policy(self, player_symbol):
        """Choose move based on heuristic scoring"""
        move_num = 0
        for i in range(COLS):
            if self.working_state.top[i] > 0:  # If column not full
                self.next_move[move_num] = i
                move_num += 1
                
        if move_num == 0:
            return -1
            
        return self.next_move[self.score_sample(move_num, player_symbol)]
    
    def random_policy(self):
        """Choose move with center preference"""
        move_num = 0
        for i in range(COLS):
            if self.working_state.top[i] > 0:  # If column not full
                self.next_move[move_num] = i
                move_num += 1
                
        if move_num == 0:
            return -1
            
        return self.next_move[self.center_sample(move_num)]
    
    def expand(self, node_idx):
        """Expand a node by adding all possible child nodes"""
        node = self.nodes[node_idx]
        player_symbol = node.player_symbol
        player_num = self.player_number[player_symbol]
        opponent_symbol = "O" if player_symbol == "X" else "X"
        opponent_num = 3 - player_num
        
        # Check for terminal state
        if self._is_terminal(self.working_state):
            return node_idx
            
        # Check for urgent moves (winning or blocking)
        urgent = -1
        
        for i in range(COLS):
            if self.working_state.top[i] > 0:  # If column not full
                # Calculate row position where piece would land
                row = self.working_state.top[i] - 1
                
                # Check if winning move
                if self._is_winning_move(i, player_symbol):
                    # Create child node for winning move
                    node.children[i] = self._new_node(opponent_num, node_idx, row, i)
                    # Clear previous children (prune the tree)
                    for j in range(i):
                        node.children[j] = -1
                    return node.children[i]
                
                # If not a losing move, add as child
                if not self._is_losing_move(i, player_symbol):
                    node.children[i] = self._new_node(opponent_num, node_idx, row, i)
                
                # Check if opponent winning move (urgent to block)
                if self._is_winning_move(i, opponent_symbol):
                    urgent = i
        
        # Handle urgent blocking move
        if urgent != -1:
            # Prune earlier moves
            for i in range(urgent):
                node.children[i] = -1
            
            # Calculate row for urgent move
            row = self.working_state.top[urgent] - 1
            
            # Create child for urgent move
            if node.children[urgent] == -1:
                node.children[urgent] = self._new_node(opponent_num, node_idx, row, urgent)
            
            return node.children[urgent]
        
        # Return first valid child
        for i in range(COLS):
            if node.children[i] != -1:
                return node.children[i]
        
        # If no children yet, create first valid one
        for i in range(COLS):
            if self.working_state.top[i] > 0:
                row = self.working_state.top[i] - 1
                node.children[i] = self._new_node(opponent_num, node_idx, row, i)
                return node.children[i]
        
        # No valid moves
        return node_idx
    
    def _is_terminal(self, state):
        """Check if state is terminal (game over)"""
        # Check for winner
        if self._has_winner(state):
            return True
            
        # Check if board is full
        for col in range(COLS):
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
        for opp_col in range(COLS):
            if temp_state.top[opp_col] > 0:
                temp_state2 = copy.deepcopy(temp_state)
                temp_state2.move(opp_col, opponent_symbol)
                if temp_state2.is_winning_state(opponent_symbol):
                    return True
                    
        return False
    
    def rollout(self, node_idx):
        """Simulate random playout from node"""
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
            move = self.smart_policy(current_player)
            if move == -1:  # No valid moves
                break
                
            # Apply move
            sim_state.move(move, current_player)
            
            # Switch player
            current_player = opponent_symbol if current_player == player_symbol else player_symbol
        
        return 0  # Draw
    
    def final_decision(self):
        """Make final move decision based on node visits and RAVE scores"""
        best_move = -1
        max_score = float('-inf')
        
        # Calculate beta for RAVE weight
        beta = math.sqrt(self.K / (3 * self.nodes[self.root].visits + self.K))
        log_n = math.log(self.nodes[self.root].visits + 0.001)
        
        for i in range(COLS):
            child_idx = self.nodes[self.root].children[i]
            if child_idx == -1:
                continue
                
            child = self.nodes[child_idx]
            
            # Combined score with RAVE
            score = ((1 - beta) * child.score() + 
                     beta * self.move_score(child_idx) +
                     self.UCB_C * math.sqrt(log_n / child.visits))
            
            # For debugging
            # print(f"Move: {i} Score: {score} Visits: {child.visits} Value: {child.value}")
            
            if score > max_score:
                max_score = score
                best_move = i
        
        # print(f"Confidence: {max_score}")
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
        
        # Make final decision
        self.last_move = self.final_decision()
        return self.last_move