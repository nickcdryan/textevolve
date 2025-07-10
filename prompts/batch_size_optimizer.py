import json

def get_batch_size_optimization_prompt(current_batch_size, current_accuracy, total_examples_seen, performance_history):
    """
    Generate prompt and system instruction for batch size optimization.

    Args:
        current_batch_size: Current batch size being used
        current_accuracy: Current accuracy (float, e.g., 0.85)
        total_examples_seen: Total number of examples seen so far
        performance_history: List of performance data for recent iterations

    Returns:
        tuple: (prompt, system_instruction)
    """
    system_instruction = "You are a Batch Size Optimizer. Your task is to analyze performance trends and recommend the optimal batch size for testing, balancing between stability and throughput."

    prompt = f"""
As an AI optimization system, you need to determine the appropriate batch size for testing.

Current batch size: {current_batch_size}
Current accuracy: {current_accuracy:.2f}
Total examples seen so far: {total_examples_seen}

Recent performance history:
{json.dumps(performance_history, indent=2)}

Based on this information, determine if the batch size should be adjusted.
Consider:

1. Recent performance trend
2. Stability of results
3. Need for more diverse examples

Rules:
- Batch size should be between 3 and 10
- Increase batch size when performance is stable and good. For example, if every script tested on batches of 5 has been 100% accurate, then we cannot differentiate between them and tell how good they are relative to one another. In this case, you would increase the batch size to 10. 
- Decrease batch size if performance is consistently poor. For example, if every script tested on batches of 10 has been 0% accurate, then we are simply wasting compute and data examples, and it would be sufficient to just test on the minimum batch size. In this case you would decrease the batch size to 5.
- Keep batch size stable when exploring new approaches
- Remember: batch size should ONLY be increased when the current batch size is performing well and we want to test more diverse examples

Return only a JSON object with:
{{"new_batch_size": <integer>, "rationale": "<brief explanation>"}}
"""

    return prompt, system_instruction