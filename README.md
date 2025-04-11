# Improved Agentic Learning System

This system uses LLM-driven reasoning to iteratively improve its approach to solving problems from a dataset. It employs a dynamic exploration/exploitation strategy and adapts its testing approach based on performance.

## Key Features

### 1. Example Rotation and Progressive Testing

- The system works through dataset examples sequentially, encountering new examples with each iteration
- It starts with small batch sizes (5 examples) and gradually increases as performance improves
- For promising scripts, it runs "progressive testing" on all previously seen examples to verify robustness

### 2. LLM-Driven Decision Making

- Batch size adjustment is determined by LLM reasoning about performance trends
- Explore/exploit balance is adapted based on LLM analysis of accuracy and coverage
- Best script selection considers both accuracy and testing coverage
- Answer evaluation uses semantic matching via LLM, not just exact string matching

### 3. Validation Interface

A separate validation script allows testing the best solution on any range of dataset examples:

```bash
python validate_script.py --start 900 --end 999
```

## How to Run

1. Set your Gemini API key:
   ```
   export GEMINI_API_KEY=your_api_key_here
   ```

2. Run the system for a specified number of iterations:
   ```
   python run_script.py --iterations 10
   ```

## System Workflow

1. **Initialization**: 
   - Starting explore rate: 70%, exploit rate: 30%
   - Starting batch size: 5 examples

2. **Each Iteration**:
   - Retrieves the next N examples from the dataset (where N is the current batch size)
   - Decides whether to explore (try a new approach) or exploit (refine existing approach)
   - Generates a script using LLM guidance
   - Tests the script on the batch
   - Evaluates results using LLM-based semantic matching
   - For promising scripts, runs progressive testing on all previously seen examples
   - Adjusts explore/exploit balance and batch size based on performance

3. **Performance Tracking**:
   - Tracks both batch performance and progressive testing performance
   - Maintains detailed logs of all iterations and performance metrics
   - Identifies the best script based on both accuracy and coverage

## Output and Results

The system provides:
- Detailed iteration logs in the archive directory
- Performance trend summaries
- Information about the best script
- A convenient interface for validating scripts on held-out examples

## Dataset Format

The system expects a JSON dataset with example keys following a pattern (e.g., "calendar_scheduling_example_0"):

```json
{
  "calendar_scheduling_example_0": {
    "prompt_0shot": "...",  // The input question
    "golden_plan": "..."    // The expected answer
  },
  "calendar_scheduling_example_1": {
    ...
  }
}
```