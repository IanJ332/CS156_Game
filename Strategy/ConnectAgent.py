from Strategy.State import State
from Strategy.MCT_tree import MCTSearch
import random
import time

class ConnectAgent:
    def __init__(self, rows, cols, board):
        self.rows = rows
        self.cols = cols
        self.state = [State(board)]
        self.my_symbol = None
        self.opponent_symbol = None
    
    agent = None
    
    @classmethod
    def set_agent(cls, rows, cols, board):
        cls.agent = ConnectAgent(rows, cols, board)
        return cls.agent
    
    @classmethod
    def add_state(cls, state):
        cls.agent.state.append(state)
        
    @classmethod
    def get_state(cls):
        return cls.agent.state
    
    @classmethod
    def set_symbols(cls, my_symbol):
        cls.agent.my_symbol = my_symbol
        cls.agent.opponent_symbol = "X" if my_symbol == "O" else "O"
    
    @classmethod
    def get_mcts_move(cls, board):
        current_state = cls.get_state()[-1]
        
        # Create MCTS with current state - using time limit instead of simulation limit
        mcts = MCTSearch(
            initial_state=current_state,
            my_symbol=cls.agent.my_symbol,
            opponent_symbol=cls.agent.opponent_symbol,
            simulation_limit=2000,  # Higher simulation count
            time_limit=0.95  # Time limit in seconds (adjust based on game requirements)
        )
        
        # Get the best move from MCTS
        best_move = mcts.get_best_move()
        return best_move + 1  # +1 because the game interface uses 1-based column indexing