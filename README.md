# Connect 4 AI Agent

## Project Overview
This project implements an AI agent capable of playing Connect 4 using artificial intelligence techniques. The agent uses a combination of:
- **Search Algorithm**: Minimax with Alpha-Beta Pruning
- **Representation**: Frame-Based Representation
- **Reasoning Method**: Heuristic Evaluation with Rule-Based Inference

The agent can play as either the first or second player and adapts its strategy accordingly.

## Requirements
- Python 3.9 or later
- No additional libraries required (standard Python libraries only)

## File Structure
- `TeamX_Connect_4_Agent.py` - Main entry point containing required functions
- `search.py` - Implementation of Minimax with Alpha-Beta pruning
- `representation.py` - Frame-based game state representation
- `reasoning.py` - Heuristic evaluation and rule-based inference
- `utils.py` - Utility functions for the agent

## How to Run
1. Place the `connect_4_main.pyc` file (provided by instructor) in the same directory as this code
2. Open a terminal and navigate to the directory
3. Run the following command to see the agent play against another agent:
   ```
   python connect_4_main.pyc TeamX TeamY
   ```
   (Replace TeamX and TeamY with the actual team names)
4. The game results will be written to `connect_4_result.txt`

## Agent Strategy
The agent uses several strategies to play effectively:

1. **Minimax with Alpha-Beta Pruning**:
   - Evaluates multiple future moves to select the optimal one
   - Uses pruning to improve computational efficiency
   - Adapts search depth based on game state

2. **Frame-Based Representation**:
   - Represents the board as a frame with slots for each position
   - Maintains additional metadata about patterns and threats
   - Efficiently updates state with minimal recomputation

3. **Heuristic Evaluation**:
   - Prioritizes center column control
   - Detects and blocks opponent threats
   - Values creating multiple potential winning paths
   - Penalizes moves that enable opponent wins

4. **Adaptive Play**:
   - Adjusts strategy when playing as first or second player
   - More defensive as second player, more aggressive as first player

## Modifying the Agent
To modify the agent's strategy:

1. **Adjust Search Depth**: Change `MAX_DEPTH` in `TeamX_Connect_4_Agent.py` (higher values = stronger play but slower performance)
2. **Modify Heuristics**: Edit weights in `reasoning.py` to prioritize different aspects of gameplay
3. **Add New Rules**: Extend rule-based system in `reasoning.py` with additional pattern detection

## Common Issues and Solutions
- **Performance Issues**: If the agent is too slow, reduce `MAX_DEPTH` in `TeamX_Connect_4_Agent.py`
- **Invalid Move Errors**: The agent should handle these automatically, but check for special case handling in `what_is_your_move`
- **Memory Issues**: The agent uses frame-based representation to minimize memory usage, but for larger search depths, you might need to optimize further

## Notes on Player Order
- **First Player**: The agent plays more aggressively, focusing on center control and creating threats
- **Second Player**: The agent plays more defensively, focusing on blocking opponent threats while building its own opportunities

## Development and Testing
The agent was extensively tested against various opponent strategies to ensure robust performance. It was designed to compete effectively while following all requirements specified in the project guidelines.