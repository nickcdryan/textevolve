import json

def get_progressive_testing_prompt(current_iteration, current_accuracy, current_batch_size, 
   baseline_accuracy, performance_history):
   """Generate prompt for progressive testing decision"""
   
   system_instruction = """You are a Testing Efficiency Advisor. Your job is to decide whether to run progressive testing on the current iteration's script. Progressive testing takes extra time and resources, so only recommend it when the results look genuinely promising and warrant deeper evaluation."""
   
   baseline_text = f"{baseline_accuracy:.2f}" if baseline_accuracy is not None else "unknown"
   
   prompt = f"""
   Decide whether to run progressive testing on iteration {current_iteration}'s script.
   
   CURRENT ITERATION RESULTS:
   - Accuracy: {current_accuracy:.2f}
   - Batch size: {current_batch_size} examples
   - Baseline accuracy: {baseline_text}
   
   PERFORMANCE HISTORY:
   {json.dumps(performance_history, indent=2)}
   
   PROGRESSIVE TESTING INFO:
   Progressive testing runs the script on all previously seen examples (up to 10-20) to get a higher-resolution view of performance. This takes extra time and API calls, so we should be selective about when to use it.
   
   DECISION CRITERIA:
   
   1. BASELINE CALIBRATION:
   - Compare current accuracy to baseline, not absolute thresholds
   - If baseline was low, moderate improvements might be worth testing
   - If baseline was high, need substantial improvements to justify testing
   
   2. BATCH SIZE CONSIDERATIONS:
   - Small batches (≤3): Results are very noisy, need strong signals
   - Medium batches (4-7): Moderate confidence in results
   - Large batches (8+): More reliable, lower bar for testing
   
   3. PERFORMANCE CONTEXT:
   - Is this significantly better than recent iterations?
   - Is this the best result so far, or close to it?
   - Have we done progressive testing recently on similar results?
   
   4. CONSERVATIVE APPROACH:
   - Only recommend testing for genuinely promising results
   - Avoid testing marginal improvements or noisy small-batch results
   - Consider opportunity cost of API calls and time
   
   5. EARLY VS LATE ITERATIONS:
   - Early iterations (1-5): Higher bar since we're still exploring
   - Later iterations (6+): More selective, focus on clear improvements
   
   EXAMPLES OF WHEN TO TEST:
   - Current accuracy is substantially above baseline and recent performance
   - Results look stable across reasonable batch size
   - This represents a potential breakthrough approach
   - We haven't done progressive testing in several iterations
   
   EXAMPLES OF WHEN NOT TO TEST:
   - Marginal improvement over recent results
   - Very small batch size with noisy results
   - Similar performance to recent iterations that were already tested
   - Baseline suggests this performance level isn't particularly impressive
   
   Analyze the current results in context and decide whether progressive testing is justified.
   
   Your response must end with: "DECISION: [YES/NO]"
   Provide reasoning, then your final choice.
   """
   
   return prompt, system_instruction