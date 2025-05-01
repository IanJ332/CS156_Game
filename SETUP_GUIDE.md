# Connect 4 AI Agent - Setup Guide

This guide will walk you through setting up and running the Connect 4 AI Agent for the CS156 project.

## Prerequisites

- Python 3.9 or later
- The `connect_4_main.pyc` file provided by your instructor

## Installation Steps

1. **Set up a Python environment (optional but recommended)**

   If you have multiple Python projects, it's good practice to use a virtual environment:

   ```bash
   # Create a virtual environment
   python -m venv connect4_env
   
   # Activate the virtual environment
   # On Windows:
   connect4_env\Scripts\activate
   
   # On macOS/Linux:
   source connect4_env/bin/activate
   ```

2. **Install the requirements**

   The project uses only standard Python libraries, so no additional installation is needed.

3. **Prepare your project files**

   Ensure all project files are in the same directory:
   
   ```
   TeamX_Connect_4_Agent.py
   search.py
   representation.py
   reasoning.py
   utils.py
   connect_4_main.pyc (provided by instructor)
   ```

4. **Rename the agent file (important)**

   Change `TeamX_Connect_4_Agent.py` to match your team name as specified by your instructor:
   
   For example, if you are Team 3, rename it to `Team3_Connect_4_Agent.py`

## Running the Agent

To have your agent play against another agent:

1. Make sure both agent files and the `connect_4_main.pyc` are in the same directory
2. Open a terminal and navigate to that directory
3. Run the following command:

   ```bash
   python connect_4_main.pyc Team3 Team4
   ```

   Replace `Team3` and `Team4` with the actual team names.

4. The game results will be written to `connect_4_result.txt` in the same directory

## Troubleshooting

If you encounter any issues:

1. **Agent not found error**:
   - Verify that your agent file is named correctly (e.g., `Team3_Connect_4_Agent.py`)
   - Make sure it's in the same directory as `connect_4_main.pyc`

2. **Import errors**:
   - Check that all module files (`search.py`, `representation.py`, etc.) are in the same directory

3. **Function errors**:
   - Verify that your agent implements the required functions:
     - `init_agent(player_symbol, board_num_rows, board_num_cols, board)`
     - `what_is_your_move(board, game_rows, game_cols, my_game_symbol)`

4. **Game crashes**:
   - Look for error messages in the terminal
   - Check `connect_4_result.txt` for any error information

## Performance Tuning

If your agent is running too slowly:

1. Open `TeamX_Connect_4_Agent.py`
2. Reduce the `MAX_DEPTH` value (e.g., from 5 to 4 or 3)
3. Save the file and run again

## Customization

You can customize the agent's strategy:

1. Modify the weights in `reasoning.py` to change the evaluation priorities
2. Adjust the search depth in `TeamX_Connect_4_Agent.py`
3. Add new pattern recognition features to `representation.py`

## Testing Your Agent

Before submitting, test your agent against the example agents:

```bash
python connect_4_main.pyc TeamX Team1
python connect_4_main.pyc TeamX Team2
```

Check the `connect_4_result.txt` file to see the outcome of each game.