import json

def get_strategy_optimization_prompt(current_iteration, baseline_accuracy, performance_history, 
                                   structured_learning_insights=None):
    """Generate prompt for direct strategy selection with structured learning insights"""

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

    STRUCTURED LEARNING INSIGHTS:
    {json.dumps(structured_learning_insights, indent=2) if structured_learning_insights else "No structured learning insights available yet."}

    ENHANCED DECISION PRINCIPLES:

    1. EVIDENCE-BASED STRATEGY SELECTION:
       - Use STRUCTURED LEARNING INSIGHTS to inform decisions
       - If insights show SUCCESSFUL_PATTERN with HIGH transferability → Consider EXPLOIT
       - If insights show COMPLEXITY_MISMATCH → Consider REFINE with different complexity
       - If insights show IMPLEMENTATION_BUG → Consider REFINE to fix execution
       - If insights show multiple failed approaches → Consider EXPLORE new direction

    2. PATTERN RECOGNITION SIGNALS:
       - EXPLOIT when: Multiple SUCCESSFUL_PATTERN learnings with similar conditions
       - REFINE when: PARTIAL_SUCCESS learnings show promising approaches needing fixes
       - EXPLORE when: No successful patterns found, or all recent attempts failed conceptually

    3. COMPLEXITY MATCH ANALYSIS:
       - If recent learnings show OVER_ENGINEERED approaches → REFINE to simplify
       - If recent learnings show UNDER_ENGINEERED approaches → EXPLORE more sophisticated methods
       - If complexity matches well → EXPLOIT or REFINE existing good approaches

    4. IMPLEMENTATION QUALITY FOCUS:
       - Scripts with GOOD/EXCELLENT quality but low accuracy → REFINE approach logic
       - Scripts with POOR/FAIR quality but good approach → REFINE implementation
       - Scripts with both poor quality and approach → EXPLORE new directions

    5. ADAPTIVE EXPLORATION:
       - Early iterations (≤5): Moderate exploration bias, but evidence can override
       - Mid iterations (6-15): Evidence-driven decisions, no hardcoded bias
       - Later iterations (15+): Focus on exploitation/refinement unless all approaches plateau

    6. NOISE AWARENESS & EVIDENCE STRENGTH:
       - Small batch sizes (≤3): Require STRONG evidence strength for major strategy changes
       - Weight decisions by evidence_strength (STRONG > MEDIUM > WEAK)
       - Don't over-interpret single results, look for patterns in learning insights

    7. TRANSFERABILITY CONSIDERATION:
       - HIGH transferability learnings are more reliable for exploitation
       - LOW transferability learnings suggest exploring different approaches
       - Consider dataset_type consistency when applying past learnings

    CURRENT CONTEXT:
    - This is iteration {current_iteration}
    - Early exploration phase: {"Yes" if current_iteration < 6 else "No"}

    ANALYSIS INSTRUCTIONS:
    1. First analyze the STRUCTURED LEARNING INSIGHTS for patterns and evidence
    2. Consider what the insights reveal about successful vs failed approaches
    3. Look for complexity mismatches, implementation issues, or successful patterns
    4. Weight evidence by strength (STRONG > MEDIUM > WEAK) and transferability
    5. Apply the enhanced decision principles based on learning patterns
    6. Consider performance history as secondary confirmation of learning insights
    7. Make evidence-based decision rather than following hardcoded iteration rules

    SPECIFIC DECISION LOGIC:
    - If learnings show SUCCESSFUL_PATTERN with HIGH transferability → Strong EXPLOIT signal
    - If learnings show COMPLEXITY_MISMATCH → REFINE with complexity adjustment
    - If learnings show IMPLEMENTATION_BUG + good concepts → REFINE implementation  
    - If learnings show consistent failures across different approaches → EXPLORE
    - If no strong learning patterns yet → Mild exploration bias (early iterations only)

    Your response must end with: "STRATEGY: [EXPLORE/EXPLOIT/REFINE]"
    
    Provide detailed reasoning based on structured learning insights, then your final choice.
    """

    return prompt, system_instruction