# PEAS Description for Connect 4 AI Agent

## Performance Measure
- **Primary Goal**: Win the Connect 4 game against other agents
- **Secondary Goals**:
  - Prevent opponent from winning
  - Control the center of the board to maximize future winning opportunities
  - Set up multi-pronged threats ("traps") that force the opponent into a lose-lose situation
  - Make decisions efficiently within reasonable time constraints
  - Adapt strategy based on whether playing as first or second player

## Environment
- **Board Structure**: 6 rows x 7 columns grid (standard Connect 4 dimensions)
- **Turn-Based**: Alternate moves with opponent
- **Gravity Effect**: Pieces fall to the lowest available position in a column
- **Partially Observable**: Can observe the entire current board state, but cannot predict opponent's future moves with certainty
- **Deterministic**: Results of actions are deterministic (pieces fall predictably)
- **Sequential**: Previous moves affect future states and available actions
- **Static**: The environment doesn't change while the agent is deliberating
- **Discrete**: Fixed number of possible actions and states

## Actuators
- **Column Selection**: Agent selects a column (1-7) in which to drop its piece
- **Action Implementation**: The `what_is_your_move()` function returns the selected column number

## Sensors
- **Board State**: Receives the current state of the game board as a 2D array
- **Player Symbol**: Knows which symbol represents itself ('X' or 'O')
- **Board Dimensions**: Knows the number of rows and columns in the board
- **Input Parameters**: All sensor data is received through the parameters of the `init_agent()` and `what_is_your_move()` functions

## Justification for AI Methods Selected

### Minimax with Alpha-Beta Pruning (Search Algorithm)
We selected Minimax with Alpha-Beta Pruning because:
1. **Perfect Information Game**: Connect 4 is a perfect information game where Minimax can theoretically find optimal solutions.
2. **Efficiency**: Alpha-Beta pruning significantly reduces the search space compared to standard Minimax, allowing deeper search depths.
3. **Adversarial Planning**: This approach explicitly models the opponent's moves, assuming they will make optimal decisions.
4. **Well-Defined Termination**: The game has clear win/loss/draw conditions that can be detected.

### Frame-Based Representation
We selected Frame-Based Representation because:
1. **Pattern Recognition**: The frame structure allows us to store and recognize important patterns (threats, opportunities).
2. **Efficient Updates**: Only affected parts of the representation need to be updated after each move.
3. **Metadata Storage**: Can store derived information like column heights and threat patterns.
4. **Rich Knowledge Representation**: Slots in frames can maintain relationships between game elements.

### Heuristic Evaluation with Rule-Based Inference (Reasoning Method)
We selected Heuristic Evaluation with Rule-Based Inference because:
1. **Strategic Knowledge**: Rules can encode domain-specific Connect 4 strategies.
2. **Priority Ordering**: Can implement clear hierarchical decision making (e.g., win if possible, block if necessary).
3. **Computationally Efficient**: Rule-based systems can make fast decisions when full search is too expensive.
4. **Human-like Reasoning**: The approach mimics how human players evaluate positions and decide moves.
5. **Complementary to Search**: Rules can guide the search, while heuristics evaluate positions at search leaf nodes.

The integration of these three methods creates a strong AI agent that balances computational efficiency with strategic depth, able to both plan ahead (search) and recognize important patterns (representation + reasoning).