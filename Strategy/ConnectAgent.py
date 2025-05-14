from Strategy.State import State
from Strategy.constant import ROWS, COLS
from Strategy.MCT_tree import MCTSearch
import copy
import time

class ConnectAgent:
    """Connect Four Agent using Monte Carlo Tree Search"""
    
    # Static attributes
    rows = ROWS
    cols = COLS
    states = []
    my_symbol = None
    opponent_symbol = None
    
    @classmethod
    def set_agent(cls, rows, cols, board):
        """Initialize the agent with board dimensions and initial state"""
        cls.rows = rows
        cls.cols = cols
        initial_state = State(board, 0)
        cls.states = [initial_state]
    
    @classmethod
    def set_symbols(cls, my_symbol):
        """Set player symbols"""
        cls.my_symbol = my_symbol
        cls.opponent_symbol = "X" if my_symbol == "O" else "O"
    
    @classmethod
    def add_state(cls, state):
        """Add a new state to the history"""
        cls.states.append(state)
    
    @classmethod
    def get_state(cls):
        """Get the state history"""
        return cls.states
    
    @classmethod
    def get_mcts_move(cls, board):
        """Use MCTS to get the best move"""
        current_state = copy.deepcopy(cls.states[-1])  # Get the current state
        
        # Configure MCTS parameters - adjust based on performance needs
        simulation_limit = 1000  # Default number of simulations
        time_limit = 1.0        # Default time limit in seconds
        
        try:
            # Create the MCTS search object with the current state
            mcts = MCTSearch(
                initial_state=current_state,
                my_symbol=cls.my_symbol,
                opponent_symbol=cls.opponent_symbol,
                simulation_limit=simulation_limit,
                time_limit=time_limit
            )
            
            # Get the best move (column index is 0-based)
            best_move = mcts.get_best_move()
            
            # Convert to 1-based index for the game interface
            return best_move + 1 if best_move != -1 else cls._fallback_move(current_state)
        except Exception as e:
            print(f"MCTS internal error: {e}")
            # Fallback to a simple heuristic if MCTS fails
            return cls._fallback_move(current_state)
    
    @classmethod
    def _fallback_move(cls, state):
        """Fallback strategy if MCTS fails"""
        # Try middle columns first (better strategic position)
        mid = COLS // 2
        if state.top[mid] > 0:
            return mid + 1
        
        # Try columns near the middle
        for offset in range(1, mid + 1):
            if mid - offset >= 0 and state.top[mid - offset] > 0:
                return (mid - offset) + 1
            if mid + offset < COLS and state.top[mid + offset] > 0:
                return (mid + offset) + 1
        
        # Last resort: any valid column
        for col in range(COLS):
            if state.top[col] > 0:
                return col + 1
        
        return 1  # Shouldn't reach here unless board is full