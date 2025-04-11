# Agentic Learning System

This is a domain-agnostic learning system that uses LLM-powered reasoning to continuously improve its approach to solving problems from a dataset. The system employs an exploration/exploitation strategy that balances trying new approaches with refining successful ones.

## Features

- **LLM-Driven Learning**: Uses LLM reasoning for all key decisions instead of hardcoded rules
- **Dynamic Exploration/Exploitation**: Automatically adjusts the balance based on performance trends
- **Systematic Error Analysis**: Performs in-depth analysis of errors to guide improvements
- **Performance Tracking**: Maintains a comprehensive history of iterations and results
- **Domain-Agnostic Design**: Can adapt to different problem domains without domain-specific instructions

## Requirements

- Python 3.7+
- Google Gemini API key

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/agentic-learning-system.git
   cd agentic-learning-system
   ```

2. Set up your environment:
   ```
   # Set your Gemini API key
   export GEMINI_API_KEY=your_api_key_here

   # Install required packages
   pip install google-generativeai
   ```

## Usage

The system is designed to work with a dataset in JSON format. For the included example, the dataset should be structured as follows:

```json
{
  "calendar_scheduling_example_0": {
    "prompt_0shot": "...",
    "golden_plan": "..."
  },
  "calendar_scheduling_example_1": {
    "prompt_0shot": "...",
    "golden_plan": "..."
  },
  ...
}
```

### Running the System

```bash
python run_script.py --iterations 10 --dataset calendar_scheduling.json --prefix calendar_scheduling_example_
```

Arguments:
- `--iterations` or `-i`: Number of iterations to run (default: 5)
- `--dataset` or `-d`: Path to the dataset file (default: calendar_scheduling.json)
- `--prefix` or `-p`: Prefix for example keys in the dataset (default: calendar_scheduling_example_)

### Output

The system will create:
- An `archive` directory with detailed information about each iteration
- A `scripts` directory containing all generated scripts
- A summary of performance trends at the end of execution

## How It Works

1. **Initialization**: The system starts with an initial explore/exploit balance (default: 70/30).

2. **Iteration Process**:
   - The system samples examples from the dataset
   - Decides whether to explore (try a new approach) or exploit (refine existing approach)
   - Generates a Python script to solve the examples
   - Executes the script on each example
   - Evaluates the results against the correct answers
   - Analyzes errors to identify patterns and root causes
   - Adjusts the explore/exploit balance based on performance
   - Archives the iteration data and summary

3. **Learning and Adaptation**:
   - Over time, the system learns which approaches work best for the problem domain
   - It adjusts its strategy based on performance trends
   - It focuses on fixing the most critical issues identified in error analysis
   - It gradually improves its accuracy through systematic iteration

## Example Workflow

```
=== Starting Iteration 0 ===
Current explore/exploit balance: 70/30
Strategy for this iteration: Exploration
Generating script with LLM...
Approach summary: The script uses regex pattern matching to extract meeting parameters...
Executing script on samples...
  Processing sample 1/5...
    Result: Here is the proposed time: Monday, 15:00 - 16:00
  ...
Performance: 0.40 accuracy (2/5 correct)
Primary issue identified: Incorrect parsing of time constraints
Adjusting explore/exploit balance...
New explore/exploit balance: 75/25

=== Starting Iteration 1 ===
...
```

## Extending the System

The system is designed to be extensible:

1. **Different Datasets**: You can use it with any dataset that has the expected structure (questions and correct answers).

2. **Alternative LLMs**: Modify the `call_llm` function in `agent_system.py` to use a different LLM.

3. **Custom Evaluation Metrics**: Extend the evaluation logic in `evaluate_with_llm` to include additional metrics beyond exact matches.

4. **Enhanced Prompts**: Customize the prompts in `generate_script_with_llm` and other functions to better guide the LLM.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.