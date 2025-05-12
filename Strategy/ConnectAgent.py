from Strategy.State import State

class ConnectAgent:
    def __init__(self, rows, cols, board, player_symbol):
        self.rows = rows
        self.cols = cols
        self.player_symbol = player_symbol
        self.state = [State(board)]
        # print("state", self.state[0])

    agent = None
    
    @classmethod
    def set_agent(self, rows,cols, board, player_symbol):
        self.agent = ConnectAgent(rows, cols, board, player_symbol)
        return self.agent
    
    @classmethod
    def add_state(self, state):
        self.agent.state.append(state)
    
    @classmethod
    def get_state(self):
        return self.agent.state

    def connect(self):
        # Logic to connect the agent
        pass

    def disconnect(self):
        # Logic to disconnect the agent
        pass

    def is_connected(self):
        # Logic to check if the agent is connected
        return True