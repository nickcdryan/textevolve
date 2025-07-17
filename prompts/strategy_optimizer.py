import json

def get_strategy_optimization_prompt(current_iteration, baseline_accuracy, performance_history):
    """Generate prompt for direct strategy selection"""

    system_instruction = """You are a Strategy Selector for an iterative learning system. Your job is to choose exactly ONE strategy for the next iteration: EXPLORE, EXPLOIT, or REFINE."""

    # Fix the baseline accuracy formatting
    baseline_text = f"{baseline_accuracy:.2f}" if baseline_accuracy is not None else "unknown"

    prompt = f"""
    Choose the strategy for iteration {current_iteration} of an AI learning system.

    BASELINE CONTEXT:
    The baseline script (iteration 0) that just calls an LLM without any other programming got {baseline_text} accuracy. Keep this in mind as you adjust your strategy. If the baseline is pretty low then it's a harder dataset, and if the baseline is pretty high then it's an easier dataset - you should use this to contextualize the scores you get on each iteration when deciding strategy.

    STRATEGY OPTIONS:
    1. EXPLORE: Generate completely novel approaches, test new hypotheses
    2. EXPLOIT: Combine elements from multiple successful approaches  
    3. REFINE: Target specific weaknesses in the single best script

    PERFORMANCE HISTORY:
    {json.dumps(performance_history, indent=2)}

    DECISION PRINCIPLES:

    1. EXPLORATION BIAS (Early Iterations):
       - Favor EXPLORE for first 5-8 iterations to build approach diversity
       - Only exploit/refine if you have genuinely exceptional results relative to baseline

    2. NOISE AWARENESS:
       - Small batch sizes (≤3) make results very noisy - don't over-interpret single results
       - Look for consistent patterns across multiple iterations
       - One bad/good result with small batches might just be luck

    3. BASELINE CALIBRATION:
       - Always compare performance to the baseline, not absolute thresholds
       - What counts as "good" depends entirely on how hard the baseline suggests this dataset is
       - If baseline was high, you need high performance to justify exploitation
       - If baseline was low, moderate improvements might justify exploitation

    4. DIVERSITY MAINTENANCE:
       - Even when doing well, occasionally explore (every 4-5 iterations)
       - Avoid getting stuck in local optima

    5. EXPLOITATION SIGNALS:
       - Multiple approaches performing well above baseline
       - Consistent patterns across iterations
       - At least 5+ iterations of exploration completed

    6. REFINEMENT SIGNALS:
       - One approach clearly superior to others
       - Specific weaknesses identified that could be fixed
       - Good baseline-relative performance that could be incremented

    CURRENT CONTEXT:
    - This is iteration {current_iteration}
    - Early exploration phase: {"Yes" if current_iteration < 8 else "No"}

    Analyze the performance history considering batch size noise and baseline calibration.

    Your response must end with: "STRATEGY: [EXPLORE/EXPLOIT/REFINE]"
    Provide reasoning, then your final choice.
    """

    return prompt, system_instruction