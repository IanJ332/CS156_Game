from State import State

class ConnectAgent:
    def __init__(self, rows, cols, board, player_symbol):
        self.rows = rows
        self.cols = cols
        self.player_symbol = player_symbol
        self.state = [State(board)]

    def connect(self):
        # Logic to connect the agent
        pass

    def disconnect(self):
        # Logic to disconnect the agent
        pass

    def is_connected(self):
        # Logic to check if the agent is connected
        return True