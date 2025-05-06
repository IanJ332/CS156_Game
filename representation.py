#!/usr/bin/env python3
"""
Connect 4 Frame-Based Representation
CS156 Final Project - Spring 2025

This module implements a frame-based representation of the Connect 4 game state.
"""

import copy

class GameState:
    """
    Frame-based representation of the Connect 4 game state.
    """
    
    def __init__(self, board, num_rows, num_cols, player_symbol, opponent_symbol):
        """
        Initialize the game state representation.
        
        Args:
            board: Current state of the board
            num_rows: Number of rows in the board
            num_cols: Number of columns in the board
            player_symbol: Symbol of the AI player ('X' or 'O')
            opponent_symbol: Symbol of the opponent ('X' or 'O')
        """
        self.board = copy.deepcopy(board)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.player_symbol = player_symbol
        self.opponent_symbol = opponent_symbol
        
        # Additional metadata frames
        self.column_heights = self._calculate_column_heights()
        self.threats = self._detect_threats()
        self.move_history = []
    
    def _calculate_column_heights(self):
        """
        Calculate the height (number of pieces) in each column.
        
        Returns:
            List of heights for each column
        """
        heights = [0] * self.num_cols
        
        for col in range(self.num_cols):
            for row in range(self.num_rows - 1, -1, -1):
                if self.board[row][col] != ' ':
                    heights[col] = self.num_rows - row
                    break
        
        return heights
    
    def _detect_threats(self):
        """
        Detect threats (three in a row with an open spot) for both players.
        
        Returns:
            Dictionary with threats for each player
        """
        threats = {
            self.player_symbol: [],
            self.opponent_symbol: []
        }
        
        # Check for horizontal threats
        for row in range(self.num_rows):
            for col in range(self.num_cols - 3):
                window = [self.board[row][col+i] for i in range(4)]
                self._check_window_for_threats(window, threats, (row, col), 'horizontal')
        
        # Check for vertical threats
        for col in range(self.num_cols):
            for row in range(self.num_rows - 3):
                window = [self.board[row+i][col] for i in range(4)]
                self._check_window_for_threats(window, threats, (row, col), 'vertical')
        
        # Check for diagonal up threats
        for row in range(3, self.num_rows):
            for col in range(self.num_cols - 3):
                window = [self.board[row-i][col+i] for i in range(4)]
                self._check_window_for_threats(window, threats, (row, col), 'diagonal_up')
        
        # Check for diagonal down threats
        for row in range(self.num_rows - 3):
            for col in range(self.num_cols - 3):
                window = [self.board[row+i][col+i] for i in range(4)]
                self._check_window_for_threats(window, threats, (row, col), 'diagonal_down')
        
        return threats
    
    def _check_window_for_threats(self, window, threats, position, direction):
        """
        Check if a window contains a threat for either player.
        
        Args:
            window: The window to check (list of 4 cells)
            threats: Dictionary to update with found threats
            position: Starting position (row, col) of the window
            direction: Direction of the window (horizontal, vertical, etc.)
        """
        # Check for player threats (three in a row with an open spot)
        if window.count(self.player_symbol) == 3 and window.count(' ') == 1:
            empty_idx = window.index(' ')
            if direction == 'horizontal':
                threat_pos = (position[0], position[1] + empty_idx)
            elif direction == 'vertical':
                threat_pos = (position[0] + empty_idx, position[1])
            elif direction == 'diagonal_up':
                threat_pos = (position[0] - empty_idx, position[1] + empty_idx)
            else:  # diagonal_down
                threat_pos = (position[0] + empty_idx, position[1] + empty_idx)
            
            threats[self.player_symbol].append(threat_pos)
        
        # Check for opponent threats
        if window.count(self.opponent_symbol) == 3 and window.count(' ') == 1:
            empty_idx = window.index(' ')
            if direction == 'horizontal':
                threat_pos = (position[0], position[1] + empty_idx)
            elif direction == 'vertical':
                threat_pos = (position[0] + empty_idx, position[1])
            elif direction == 'diagonal_up':
                threat_pos = (position[0] - empty_idx, position[1] + empty_idx)
            else:  # diagonal_down
                threat_pos = (position[0] + empty_idx, position[1] + empty_idx)
            
            threats[self.opponent_symbol].append(threat_pos)
    
    def update_board(self, board):
        """
        Update the board state and recalculate metadata.
        
        Args:
            board: New board state
        """
        self.board = copy.deepcopy(board)
        self.column_heights = self._calculate_column_heights()
        self.threats = self._detect_threats()
    
    def make_move(self, column, symbol):
        """
        Apply a move to the board state.
        
        Args:
            column: Column to place the piece (1-indexed)
            symbol: Symbol to place ('X' or 'O')
            
        Returns:
            True if move was successful, False otherwise
        """
        # Convert to 0-indexed
        col = column - 1
        
        # Check if column is valid
        if col < 0 or col >= self.num_cols:
            return False
        
        # Check if column is full
        if self.board[0][col] != ' ':
            return False
        
        # Find the lowest empty row in this column
        for row in range(self.num_rows - 1, -1, -1):
            if self.board[row][col] == ' ':
                self.board[row][col] = symbol
                self.move_history.append((row, col, symbol))
                self.column_heights = self._calculate_column_heights()
                self.threats = self._detect_threats()
                return True
        
        return False
    
    def clone(self):
        """
        Create a deep copy of the current game state.
        
        Returns:
            New GameState object with same properties
        """
        new_state = GameState(
            self.board,
            self.num_rows,
            self.num_cols,
            self.player_symbol,
            self.opponent_symbol
        )
        new_state.move_history = self.move_history.copy()
        return new_state