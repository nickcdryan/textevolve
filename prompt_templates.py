"""
Prompt Templates Library

This module contains templates for prompts used in the agentic learning system.
"""

import json


class PromptTemplates:
    """Collection of prompt templates for different scenarios."""

    @staticmethod
    def get_historical_context_template():
        """Template for historical context section."""
        return """
        ITERATION HISTORY SUMMARY:
        - Total iterations completed: {summaries_count}
        - Current explore/exploit balance: {explore_rate}/{exploit_rate}
        - Best accuracy achieved: {best_accuracy_str}

        APPROACH HISTORY (last {history_count} iterations):
        {approach_history_json}

        COMMON ERROR PATTERNS:
        {error_patterns_json}

        PRIMARY ISSUES (last {issues_count} iterations):
        {primary_issues_json}

        TARGETED IMPROVEMENTS:
        {targeted_improvements_json}
        """

    @staticmethod
    def get_learning_context_template():
        """Template for learning context section."""
        return """
        ACCUMULATED LEARNINGS FROM PREVIOUS ITERATIONS:
        {accumulated_learnings}
        """

    @staticmethod
    def get_capability_context_template():
        """Template for capability context section."""
        return """
        CAPABILITY ASSESSMENT & IMPROVEMENT GUIDANCE:
        {capability_guidance}
        """

    @staticmethod
    def get_validation_guidance():
        """Get validation and verification guidance."""
        return """VALIDATION AND VERIFICATION GUIDANCE:
        1. CRITICAL: Consider implementing validation loops for EACH key processing step, not just final outputs
        2. Design your system to detect, diagnose, and recover from specific errors. This will help future learnings
        3. For every LLM extraction or generation, add a verification step that checks:
           - Whether the output is well-formed and complete
           - Whether the output is logically consistent with the input
           - Whether all constraints are satisfied
        4. Add feedback loops that retry failures with specific feedback
        5. Include diagnostic outputs that reveal exactly where failures occur. Add print statements and intermediate outputs such that you can see them later to determine why things are going wrong.
        6. Include capability to trace through execution steps to identify failure points

        """ + """

        VALIDATION IMPLEMENTATION STRATEGIES:
        1. Create detailed verification functions for each major processing step
        2. Implement max_attempts limits on all retry loops (typically 3-5 attempts)
        3. Pass specific feedback from verification to subsequent retry attempts
        4. Log all verification failures to help identify systemic issues
        5. Design fallback behaviors when verification repeatedly fails
        """

    @staticmethod
    def get_direct_llm_reasoning_guidance():
        """Get guidance for direct LLM reasoning approach."""
        return """=== DIRECT LLM REASONING APPROACH ===

        CRITICAL: Previous scripts have shown that complex code generation with JSON parsing and multi-step pipelines often 
        leads to errors and low performance. Instead, focus on leveraging the LLM's natural reasoning abilities:

        1. SIMPLIFY YOUR APPROACH:
           - Minimize the number of processing steps - simpler is better
           - Directly use LLM for pattern recognition rather than writing complex code
           - Avoid trying to parse or manipulate JSON manually - pass it as text to the LLM

        2. DIRECT TRANSFORMATION:
           - Instead of trying to extract features and then apply them, use the LLM to do the transformation directly
           - Use examples to teach the LLM the pattern, then have it apply that pattern to new inputs
           - Avoid attempting to write complex algorithmic solutions when pattern recognition will work better

        3. ROBUST ERROR HANDLING:
           - Include multiple approaches in case one fails (direct approach + fallback approach)
           - Use simple validation to check if outputs are in the expected format
           - Include a last-resort approach that will always return something valid

        4. AVOID COMMON PITFALLS:
           - Do NOT attempt to use json.loads() or complex JSON parsing - it often fails
           - Do NOT create overly complex Python pipelines that require perfect indentation
           - Do NOT create functions that generate or execute dynamic code
           - Do NOT create unnecessarily complex data transformations

        5. SUCCESSFUL EXAMPLES:
           - The most successful approaches have used direct pattern matching with multiple examples
           - Scripts with simple validation and fallback approaches perform better
           - Scripts with fewer processing steps have higher success rates

        IMPLEMENTATION STRATEGIES:
        1. Maintain a "example bank" of successful and failed examples to select from
        2. Implement n-shot prompting with n=3 as default, but adapt based on performance
        3. For complex tasks, use up to 5 examples; for simpler tasks, 2-3 may be sufficient
        4. Include examples with a range of complexity levels, rather than all similar examples
        """

    @staticmethod
    def get_multi_example_prompting_guidance():
        """Get guidance for multi-example prompting."""
        return """MULTI-EXAMPLE PROMPTING GUIDANCE:
        1. CRITICAL: Use MULTIPLE examples (2-5) in EVERY LLM prompt, not just one
        2. Vary the number of examples based on task complexity - more complex tasks need more examples
        3. Select diverse examples that showcase different patterns and edge cases
        4. Structure your few-shot examples to demonstrate clear step-by-step reasoning
        5. Consider using both "easy" and "challenging" examples to help the LLM learn from contrasts
        6. The collection of examples should collectively cover all key aspects of the problem
        7. When available, use examples from previous iterations that revealed specific strengths or weaknesses.
        8. USE REAL EXAMPLES FROM THE DATASET WHERE POSSIBLE!!

        """ 

    @staticmethod
    def build_exploration_prompt(example_problems, historical_context, llm_patterns, 
                            learning_context, capability_context, gemini_api_example):
        """Build the prompt for exploration mode."""

        prompt = f"""
        You are developing a Python script to solve problems using LLM reasoning capabilities.
        You are in the EXPLORATION PHASE. You must generate a NEW approach that's different from previous approaches but informed by their successes and failures. With this approach, you will have a specific NEW HYPOTHESIS or variable you are trying to test. Your goal is to see if this new approach works, and you must add verification and validation steps to deduce if this new change is helpful. Carefully and fairly evaluate whether the hypothesis should be accepted, rejected, re-tested, or something else, making reference to specific outputs, reasoning steps, error messages, or other evidence from the exectuion. You may test RADICAL NEW APPROACHES that are substantially different from previous approaches. 

        You should try NEW THINGS:

        Break down the problem into smaller pieces
        Think CREATIVELY about how to solve your problem if other approaches aren't working
        Transform data into different formats to see if it helps

        # YOUR TASK
        You are deeply familiar with prompting techniques and the agent works from the literature. 
        Your goal is to maximize the specified performance metrics by proposing interestingly new agents.
        Observe the past discovered agents and scripts carefully and think about what insights, lessons, or stepping stones can be learned from them.
        Be creative when thinking about the next interesting agent to try. You are encouraged to draw inspiration from related agent papers or academic papers from other research areas.
        Use the knowledge from the archive and inspiration from academic literature to propose the next interesting agentic system design.
        THINK OUTSIDE THE BOX.


        Here are example problems from previously seen data. YOUR APPROACH MUST BE DIFFERENT THAN THESE:
        {json.dumps(example_problems, indent=2)}

        HISTORICAL CONTEXT. YOUR APPROACH MUST ALSO BE SUBSTANTIALLY DIFFERENT THAN THESE:
        {historical_context}

        LIBRARY OF PROMPTS, TECHNIQUES, STRATEGIES, AND PATTERNS THAT HAVE PROVEN USEFUL. YOU MUST TRY NEW TECHNIQUES FROM HERE OUT:
        {llm_patterns}

        REMEMBER, IF YOU HAVE NOT DONE SO YOU NEED TO TRY OUT SOME OF THE PATTERNS ABOVE

        LEARNINGS FROM PREVIOUS ITERATIONS:
        {learning_context}

        CAPABILITY ASSESSMENT & IMPROVEMENT GUIDANCE:
        {capability_context}

        EXPLORATION GUIDANCE:
        1. Review the historical approaches, error patterns, and accumulated learnings carefully
        2. Review the FULL CODE of previous scripts to understand what has already been tried
        3. Design a new approach that is DISTINCTLY DIFFERENT from previous attempts. This approach should have a specific NEW HYPOTHESIS or variable you are trying to test. Carefully and fairly evaluate whether the hypothesis should be accepted, rejected, re-tested, or something else, making reference to specific outputs, reasoning steps, error messages, or other evidence from the exectuion.
        4. CRITICAL: Include EMBEDDED EXAMPLES directly within your LLM prompts
        5. For each key function, show a complete worked example, or include multiple examples, including:
           - Input example that resembles the dataset
           - Step-by-step reasoning through the example
           - Properly formatted output
        6. Apply the insights from the ACCUMULATED LEARNINGS section to avoid repeating past mistakes
        7. Pay SPECIAL ATTENTION to the weaknesses and improvement suggestions from the capability assessment
        8. Consider implementing one or more of these LLM usage patterns:
           - Repeated validation with feedback loops
           - Multi-perspective analysis with synthesis
           - Dynamic input-dependent routing with an orchestrator
           - Hybrid approaches combining LLM with deterministic functions
           - Best-of-n solution generation and selection
           - ReAct pattern for interactive reasoning and action
           - If it is unknown how successful a processing state or part of the pipeline is, include verification steps to different parts of the pipeline in order to help deduce which parts are successful and where the system is breaking
           - Answer checkers to validate the final answer against the problem statement. If the answer is incorrect, the checker can send the answer back to an earlier part of the system for for refinement with feedback

        Here's how to call the Gemini API. ONLY call it in this format and DO NOT make up configuration options!:
        {gemini_api_example}

        Since this is an EXPLORATION phase:
        - Try a fundamentally different approach to reasoning about the problem. Test a NEW HYPOTHESIS or variable, and add verification steps to deduce if this new change is helpful. Carefully and fairly evaluate whether the hypothesis should be accepted, rejected, re-tested, or something else, making reference to specific outputs, reasoning steps, error messages, or other evidence from the exectuion.
        - THIS IS KEY: Break down the problem into new, distinct reasoning steps based on past performance before you start coding
        - For EACH key LLM prompt, include a relevant example with:
          * Sample input similar to the dataset
          * Expected reasoning steps
          * Desired output format
        - Apply a verifier call to different parts of the pipeline in order to understand what parts of the pipeline of calls is successful and where the system is breaking
        - Pay special attention to addressing the primary issues from previous iterations
        - Ensure your new approach addresses the weaknesses identified in the capability assessment

        CRITICAL REQUIREMENTS:
        1. The script MUST properly handle all string literals - be extremely careful with quotes and triple quotes
        2. The script MUST NOT exceed 150 lines of code to prevent truncation
        3. Include detailed comments explaining your reasoning approach
        4. EVERY SINGLE LLM PROMPT must include at least one embedded example showing:
           - Sample input with reasoning
           - Desired output format
        5. Make proper use of error handling
        6. Implement robust capabilities to address the specific weaknesses identified in the capability assessment
        7. Do NOT use json.loads() in the LLM calls to process input data. JSON formatting is good to use to structure information as inputs and outputs, but attempting to have functions process JSON data explicitly with strict built-in functionality is error prone due to formatting issues and additional text that appears as documentation, reasoning, or comments. When passing data into another LLM call, you can read it as plain text rather than trying to load it in strict json format, is the better approach.

        Return a COMPLETE, RUNNABLE Python script that:
        1. Has a main function that takes a question string as input and returns the answer string
        2. Makes multiple LLM calls for different reasoning steps
        3. Has proper error handling for API calls
        4. Includes embedded examples in EVERY LLM prompt
        5. Is COMPLETE - no missing code, no "..." placeholders
        6. Closes all string literals properly

        This should be FUNDAMENTALLY DIFFERENT from all previous approaches. Do not reuse the same overall structure. MAKE THIS ITERATION SUBSTANTIALLY DIFFERENT IN AN INTERESTING WAY.

        BE EXTREMELY CAREFUL TO PROPERLY CLOSE ALL STRING QUOTES AND TRIPLE QUOTES!
        """

        return prompt

    @staticmethod
    def build_exploitation_prompt(example_problems, historical_context, llm_patterns, best_script_code, top_scripts_content,
                              learning_context, capability_context, gemini_api_example):
        """Build the prompt for exploitation mode."""

        prompt = f"""
        You are improving a Python script that solves problems from a dataset.
        Your goal is to REFINE and ENHANCE the best performing approaches by combining their strengths and addressing specific weaknesses identified in error analysis.

        Here are example problems from previously seen data:
        {json.dumps(example_problems, indent=2)}

        {historical_context}

        LIBRARY OF PROMPTS, TECHNIQUES, STRATEGIES, AND PATTERNS:
        {llm_patterns}

        LEARNINGS FROM PREVIOUS ITERATIONS:
        {learning_context}

        {capability_context}

        TOP PERFORMING APPROACHES TO BUILD UPON:
        {top_scripts_content}
        {best_script_code}

        EXPLOITATION GUIDANCE:
        1. Review the error patterns, targeted improvements, and accumulated learnings carefully
        2. CRITICAL: Break down the problem into distinct reasoning steps before modifying code
        3. CRITICAL: Analyze the best scripts to identify which components are working well and which are failing. Focus your improvements on the weak points while preserving successful components.
        4. Maintain the core successful elements of the best approaches
        5. Consider how you can combine strengths from multiple top-performing approaches
        6. CRITICAL: Add EMBEDDED EXAMPLES to EVERY LLM prompt that illustrate:
           - Sample input that resembles the dataset
           - Step-by-step reasoning through the example
           - Properly formatted output
        7. Focus on fixing specific issues identified in previous error analyses. Create an EXPLICIT HYPOTHESIS for each targeted improvement and state it, as well as a way to verify if it's successful. Carefully and fairly evaluate whether the hypothesis should be accepted, rejected, re-tested, or something else, making reference to specific outputs, reasoning steps, error messages, or other evidence from the exectuion.
        8. Enhance chain-of-thought reasoning and verification steps. Verification steps should be added to different parts of the pipeline in order to help deduce which parts are successful and where the system is breaking
        9. Apply the key insights from ACCUMULATED LEARNINGS to enhance the approach
        10. Pay SPECIAL ATTENTION to the weaknesses and improvement suggestions from the capability assessment

        IMPROVEMENT STRATEGY:
        Analyze why the top approaches succeeded where others failed. Identify the key differentiators and strengthen them further.

        SYSTEMATIC ENHANCEMENT APPROACH:
        1. First, identify which specific function or component is underperforming based on error analysis
        2. Examine how error cases differ from successful cases
        3. For each identified weakness, implement a targeted enhancement
        4. Add additional verification steps around modified components
        5. Consider how components interact - ensure improvements don't break successful parts

        Consider enhancing the script with one or more of these patterns:
        - Repeated validation with feedback loops
        - Multi-perspective analysis with synthesis
        - Dynamic input-dependent routing
        - Hybrid approaches combining LLM with deterministic functions
        - Best-of-n solution generation and selection
        - ReAct pattern for interactive reasoning and action
        - If it is unknown how successful a processing state or part of the pipeline is, include verification steps to different parts of the pipeline in order to help deduce which parts are successful and where the system is breaking
        - Answer checkers to validate the final answer against the problem statement. If the answer is incorrect, the checker can send the answer back to an earlier part of the system for refinement with feedback

        Here's how to call the Gemini API. Use this example without modification and don't invent configuration options:
        {gemini_api_example}

        Since this is an EXPLOITATION phase:
        - Build upon what's working well in the best approaches
        - Consider creative combinations of successful techniques from different scripts
        - Make TARGETED improvements to address specific error patterns
        - For EACH key LLM prompt, include a relevant example with:
          * Sample input similar to the dataset
          * Expected reasoning steps
          * Desired output format
        - Apply the knowledge from our accumulated learnings
        - Significantly enhance the script to address weaknesses identified in the capability assessment

        CRITICAL REQUIREMENTS:
        1. The script MUST properly handle all string literals - be extremely careful with quotes and triple quotes
        2. The script MUST NOT exceed 150 lines of code to prevent truncation
        3. Include detailed comments explaining your improvements
        4. EVERY SINGLE LLM PROMPT must include at least one embedded example showing:
           - Sample input with reasoning
           - Desired output format
        5. Make proper use of error handling
        6. Implement robust capabilities to address the specific weaknesses identified in the capability assessment
        7. Do NOT use json.loads() in the LLM calls to process input data. JSON formatting is good to use to structure information as inputs and outputs, but attempting to have functions process JSON data explicitly with strict built-in functionality is error prone due to formatting issues and additional text that appears as documentation, reasoning, or comments. When passing data into another LLM call, you can read it as plain text rather than trying to load it in strict json format, is the better approach.

        Return a COMPLETE, RUNNABLE Python script that:
        1. Has a main function that takes a question string as input and returns the answer string
        2. Makes multiple LLM calls for different reasoning steps
        3. Has proper error handling for API calls
        4. Includes embedded examples in EVERY LLM prompt
        5. Is COMPLETE - no missing code, no "..." placeholders
        6. Closes all string literals properly

        BE EXTREMELY CAREFUL TO PROPERLY CLOSE ALL STRING QUOTES AND TRIPLE QUOTES!
        """

        return prompt

    @staticmethod
    def build_error_correction_prompt(error_message):
        """Build a prompt for error correction."""
        return f"""
        You need to generate a complete, syntactically valid Python script. Your previous attempt had the following syntax error:
        {error_message}

        Please generate a new script paying special attention to:
        1. Properly closing all string literals (quotes and triple quotes)
        2. Properly closing all parentheses and braces
        3. Keeping the script simple and short (under 150 lines)
        4. Using only syntactically valid Python code
        5. INCLUDING EMBEDDED EXAMPLES in all LLM prompts

        Generate a complete, runnable Python script that:
        1. Has a main function that takes a question string as input and returns the answer string
        2. Makes multiple LLM calls for different reasoning steps using the Gemini API
        3. Has proper error handling
        4. Includes a concrete example in EACH LLM prompt

        BE EXTREMELY CAREFUL WITH STRING LITERALS AND QUOTES!
        """