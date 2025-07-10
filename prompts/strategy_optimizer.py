import json

def get_strategy_optimization_prompt(performance_context, explore_rate, exploit_rate, refine_rate, 
                                   context, performance_history, capability_context=None):
    """
    Generate prompt and system instruction for strategy optimization.

    Args:
        performance_context: Dict containing baseline performance calibration data
        explore_rate: Current exploration rate percentage
        exploit_rate: Current exploitation rate percentage  
        refine_rate: Current refinement rate percentage
        context: Dict with current system status (balance, iterations, etc.)
        performance_history: List of recent performance data
        capability_context: Optional dict with capability insights

    Returns:
        tuple: (prompt, system_instruction)
    """

    # Role-specific system instruction for strategy optimizer
    system_instruction = "You are a Strategy Optimizer specializing in adaptive learning systems. Your role is to analyze performance patterns relative to baseline capabilities and determine the optimal balance between three distinct approaches: exploration, exploitation, and refinement."
    
    # Create prompt for LLM to reason about calibrated strategy adjustment
    prompt = f"""
    You're optimizing the strategy balance for an iterative learning system with three distinct modes.
    CRITICAL: All performance assessments must be made relative to the baseline performance, not absolute thresholds.
    
    PERFORMANCE CALIBRATION CONTEXT:
    {json.dumps(performance_context, indent=2)}
    
    KEY INSIGHTS FROM CALIBRATION:
    - Baseline LLM performance: {performance_context.get('baseline_accuracy', 'Unknown'):.2f}
    - Dataset difficulty: {performance_context.get('dataset_difficulty', 'Unknown')}
    - Current performance category: {performance_context.get('performance_category', 'Unknown')}
    - Relative improvement over baseline: {performance_context.get('relative_improvement', 0):.3f} ({performance_context.get('relative_percentage', 0):+.1f}%)
    - Should exploit based on threshold: {performance_context.get('should_exploit', False)}
    
    THREE STRATEGY MODES:
    1. EXPLORE ({getattr(self, 'explore_rate', 60)}%): Generate completely novel approaches and test new hypotheses
       - Use when performance is poor relative to baseline or when stuck at local optima
       - Use when dataset appears easy but current performance suggests untapped potential
    
    2. EXPLOIT ({getattr(self, 'exploit_rate', 20)}%): Combine elements from multiple successful approaches
       - Use when multiple approaches show promise above baseline
       - Use when performance is good but could benefit from combining strengths
    
    3. REFINE ({getattr(self, 'refine_rate', 20)}%): Target and fix specific weaknesses in the single best script
       - Use when one approach is clearly superior and specific improvements are identified
       - Use when performance is strong and incremental gains are the goal
    
    Current system status:
    - Current balance: {context["current_balance"]}
    - Iterations completed: {context["iterations_completed"]}
    - Best accuracy so far: {context["best_accuracy"]:.2f} (from iteration {context["best_iteration"]})
    - Total examples seen: {context["total_examples_seen"]}
    
    Performance history with baseline context (from newest to oldest):
    {json.dumps(performance_history[-5:] if len(performance_history) > 5 else performance_history, indent=2)}
    
    {"Capability insights:" if capability_context else ""}
    {json.dumps(capability_context, indent=2) if capability_context else ""}
    
    CALIBRATED DECISION GUIDELINES:
    
    1. FOR EASY DATASETS (baseline ≥ 80%):
       - Even 85-90% performance may indicate significant untapped potential
       - Favor exploration unless achieving 95%+ consistently
       - High standards for what constitutes "good enough" performance
    
    2. FOR MODERATE DATASETS (baseline 50-80%):
       - Performance 20+ points above baseline justifies some exploitation
       - Balance exploration with refinement of successful approaches
       - Look for opportunities to combine successful techniques
    
    3. FOR HARD DATASETS (baseline 20-50%):
       - Performance 10+ points above baseline is genuinely good
       - Focus on exploitation and refinement of working approaches
       - Exploration should target specific capability gaps
    
    4. FOR VERY HARD DATASETS (baseline < 20%):
       - Any consistent improvement over baseline is valuable
       - Prioritize refinement of anything that works
       - Exploration should be very targeted based on error analysis
    
    5. ANTI-PATTERNS TO AVOID:
       - Don't over-explore when you have genuinely good performance for the dataset difficulty
       - Don't over-exploit when performance suggests the dataset has much higher potential
       - Don't assume absolute performance levels without baseline context
    
    Determine the optimal balance considering baseline-relative performance, not absolute thresholds.
    
    Provide a JSON object with:
    {{"explore_rate": <explore_percentage>, "exploit_rate": <exploit_percentage>, "refine_rate": <refine_percentage>, "rationale": "<detailed_explanation_referencing_baseline_context>"}}
    """
    return prompt, system_instruction