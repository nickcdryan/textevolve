import json

def get_batch_learnings_prompt(iteration_data, accuracy, sample_questions, script_code, error_examples, capability_insights):
    """
    Generate prompt and system instruction for extracting learnings from batch results.

    Args:
        iteration_data: Dictionary containing iteration information
        accuracy: Float representing current accuracy
        sample_questions: List of sample questions from the dataset
        script_code: String containing script code excerpt
        error_examples: List of error example dictionaries
        capability_insights: String containing capability insights

    Returns:
        tuple: (prompt, system_instruction)
    """
    system_instruction = (
        "You are a Knowledge Synthesizer. Your role is to extract concrete, "
        "dataset-specific insights from experiment results, focusing on patterns "
        "in the data, effective strategies for this specific task, and precise failure modes."
    )

    prompt = f"""
    Extract specific, concrete learnings from this iteration's results, focusing on dataset-specific insights:

    Iteration: {iteration_data.get("iteration")}
    Strategy: {iteration_data.get("strategy", "Unknown")}
    Accuracy: {accuracy:.2f}
    Approach summary: {iteration_data.get("approach_summary", "No summary available")}

    Sample questions from dataset:
    {json.dumps(sample_questions, indent=2)}

    Script approach (excerpt):
    ```python
    {script_code}
    ```

    Primary issue identified: {iteration_data.get("performance", {}).get("error_analysis", {}).get("primary_issue", "None identified")}

    Error patterns:
    {json.dumps(iteration_data.get("performance", {}).get("error_analysis", {}).get("error_patterns", []), indent=2)}

    Error examples (first {len(error_examples)} failures):
    {json.dumps(error_examples[:3], indent=2)}

    {capability_insights}

    Based on this information, provide specific learnings in the following format:

    1. DATASET PATTERNS: Identify 2-3 specific patterns or characteristics in this dataset. What format do questions take? What structures appear repeatedly? What's unique about this task?

    2. WORKING STRATEGIES: What specific techniques worked well for this particular dataset and why?

    3. FAILURE MODES: What specific aspects of the dataset or task caused failures? Describe exactly how and why the approach failed on specific examples.

    4. EXPERIMENT RESULTS: What did we learn from this specific experimental approach? What hypotheses were confirmed or rejected?

    5. NEXT STEPS: What specific adaptations should be made for this particular dataset and task?

    Focus on concrete, specific insights that are directly tied to the dataset and task at hand, not general principles of system design.
    Keep your summary focused on what we've learned about solving THIS specific dataset problem.
    """
    return prompt, system_instruction