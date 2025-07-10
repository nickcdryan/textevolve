import json

def get_explore_instructions(example_problems, historical_context, last_scripts_context, 
                           learning_context, capability_context, llm_api_example):
    """
    Generate exploration-specific instructions and context.

    Args:
        example_problems: List of example problems from dataset
        historical_context: Historical performance and approach data  
        last_scripts_context: Context about the last 5 scripts tried
        learning_context: Accumulated learnings from previous iterations
        capability_context: Capability assessment and improvement guidance
        gemini_api_example: Standard API usage example

    Returns:
        str: Complete exploration prompt
    """
    return f"""
You are developing a Python script to solve problems using LLM reasoning capabilities.
You are in the EXPLORATION PHASE. You must generate a NEW approach that's different from previous approaches but informed by their successes and failures. With this approach, you will have a specific NEW HYPOTHESIS or variable you are trying to test. Your goal is to see if this new approach works, and you must add verification and validation steps to deduce if this new change is helpful. You may also test RADICAL NEW APPROACHES that are substantially different from previous approaches. 

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


Here are example problems from previously seen data:
{json.dumps(example_problems, indent=2)}

HISTORICAL CONTEXT:
{historical_context}

PREVIOUSLY TRIED APPROACHES (LAST 5 SCRIPTS). YOUR APPROACH MUST BE SUBSTANTIVELY DIFFERENT THAN THESE:
{last_scripts_context}

LEARNINGS FROM PREVIOUS ITERATIONS:
{learning_context}

CAPABILITY ASSESSMENT & IMPROVEMENT GUIDANCE:
{capability_context}

EXPLORATION GUIDANCE:
1. Review the historical approaches, error patterns, and accumulated learnings carefully
2. Review the FULL CODE of previous scripts to understand what has already been tried
3. Design a new approach that is DISTINCTLY DIFFERENT from previous attempts. This approach should have a specific NEW HYPOTHESIS or variable you are trying to test. 
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

Here's how to call the Gemini API. Use this example without modification and don't invent configuration options:
{llm_api_example}

Since this is an EXPLORATION phase:
- Try a fundamentally different approach to reasoning about the problem. Test a NEW HYPOTHESIS or variable, and add verification steps to deduce if this new change is helpful.
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

This should be FUNDAMENTALLY DIFFERENT from all previous approaches. Do not reuse the same overall structure.

BE EXTREMELY CAREFUL TO PROPERLY CLOSE ALL STRING QUOTES AND TRIPLE QUOTES!
"""



def get_exploit_instructions(example_problems, historical_context, top_scripts_analysis, 
       learning_context, capability_context, llm_api_example):
   """
   Generate exploitation-specific instructions and context.
   
   Args:
   example_problems: List of example problems from dataset
   historical_context: Historical performance and approach data
   top_scripts_analysis: Analysis of top performing scripts to combine
   learning_context: Accumulated learnings from previous iterations
   capability_context: Capability assessment and improvement guidance
   gemini_api_example: Standard API usage example
   
   Returns:
   str: Complete exploitation prompt
   """
   return f"""
   You are creating a NEW Python script by SYNTHESIZING the best elements from multiple successful approaches.
   Your goal is to identify what makes each approach successful and combine these strengths into a superior hybrid solution.
   
   Here are example problems from previously seen data:
   {json.dumps(example_problems, indent=2)}
   
   {historical_context}
   
   {learning_context}
   
   {capability_context}
   
   MULTIPLE TOP PERFORMING APPROACHES TO SYNTHESIZE:
   {top_scripts_analysis}
   
   EXPLOITATION SYNTHESIS GUIDANCE:
   1. ANALYZE EACH TOP SCRIPT to identify:
      - What specific techniques make each approach successful?
      - What unique strengths does each approach have?
      - What weaknesses or limitations does each approach have?
      - Which components could be combined effectively?
   
   2. IDENTIFY SYNTHESIS OPPORTUNITIES:
      - Which successful techniques from different scripts could work together?
      - How can you combine the best reasoning patterns from multiple approaches?
      - What hybrid approach would leverage strengths while avoiding weaknesses?
      - Can you create a multi-stage pipeline using the best parts of each?
   
   3. CREATE A HYBRID APPROACH that:
      - Takes the most effective reasoning techniques from each top script
      - Combines different successful verification/validation strategies
      - Integrates the best error handling approaches
      - Merges effective prompt engineering techniques from multiple scripts
      - Creates a more robust solution than any individual approach
   
   4. SPECIFIC SYNTHESIS STRATEGIES:
      - If Script A excels at information extraction and Script B excels at reasoning, combine both
      - If Script A has great verification and Script B has great generation, merge the pipelines
      - If multiple scripts use different successful prompting styles, create a multi-perspective approach
      - If different scripts handle different types of errors well, create comprehensive error handling
   
   5. AVOID SIMPLE COPYING:
      - Don't just take one script and make minor changes
      - Don't just concatenate approaches without thoughtful integration
      - Create something that's genuinely better than the sum of its parts
      - Ensure the hybrid approach addresses weaknesses that individual scripts had
   
   CRITICAL REQUIREMENTS FOR SYNTHESIS:
   1. The script MUST be a true hybrid that combines elements from multiple top approaches
   2. Include a clear comment explaining which elements came from which approaches
   3. EVERY LLM PROMPT must include embedded examples showing:
      - Sample input similar to the dataset
      - Expected reasoning steps
      - Desired output format
   4. The hybrid should be more robust than any individual approach
   5. Address the weaknesses identified in the capability assessment through synthesis
   
   Here's how to call the Gemini API. Use this example without modification:
   {llm_api_example}
   
   SYNTHESIS IMPLEMENTATION:
   - Create a main function that orchestrates the combined approach
   - Integrate the best reasoning patterns from multiple scripts
   - Combine the most effective verification strategies
   - Merge successful prompt engineering techniques
   - Create comprehensive error handling that addresses issues from all approaches
   
   Return a COMPLETE, RUNNABLE Python script that represents a true synthesis of the top approaches:
   1. Has a main function that takes a question string as input and returns the answer string
   2. Combines reasoning techniques from multiple successful scripts
   3. Integrates the best verification and error handling from different approaches
   4. Includes embedded examples in EVERY LLM prompt
   5. Is COMPLETE - no missing code, no "..." placeholders
   6. Closes all string literals properly
   7. Includes comments explaining which techniques came from which top scripts
   
   BE EXTREMELY CAREFUL TO PROPERLY CLOSE ALL STRING QUOTES AND TRIPLE QUOTES!
   CREATE A TRUE HYBRID THAT'S BETTER THAN ANY INDIVIDUAL APPROACH!
   """


def get_refine_instructions(example_problems, historical_context, best_script_to_refine,
      best_script_successes, best_script_errors, learning_context, 
      capability_context, llm_api_example):
   """
   Generate refinement-specific instructions and context.
   
   Args:
   example_problems: List of example problems from dataset
   historical_context: Historical performance and approach data
   best_script_to_refine: Dict with info about the best script to refine
   success_samples: List of successful examples from the best script
   error_samples: List of failed examples from the best script
   learning_context: Accumulated learnings from previous iterations
   capability_context: Capability assessment and improvement guidance
   gemini_api_example: Standard API usage example
   
   Returns:
   str: Complete refinement prompt
   """
   return f"""
   You are performing SURGICAL REFINEMENT of the single best-performing script.
   Your goal is to identify specific weaknesses in this script and make targeted improvements while preserving its strengths.
   
   Here are example problems from previously seen data:
   {json.dumps(example_problems, indent=2)}
   
   {historical_context}
   
   {learning_context}
   
   {capability_context}
   
   BEST SCRIPT TO REFINE:
   Iteration: {best_script_to_refine.get('iteration', 'Unknown')}
   Accuracy: {best_script_to_refine.get('accuracy', 0):.2f}
   Approach Summary: {best_script_to_refine.get('approach_summary', 'No summary available')}
   
   CURRENT BEST SCRIPT CODE:
   ```python
   {best_script_to_refine.get('script', '')}
   ```
   
   SPECIFIC SUCCESS CASES (what the script does well):
   {json.dumps(best_script_successes[:3], indent=2)}
   
   SPECIFIC FAILURE CASES (what needs improvement):
   {json.dumps(best_script_errors[:3], indent=2)}
   
   REFINEMENT ANALYSIS GUIDANCE:
   1. IDENTIFY THE CORE STRENGTH:
      - What specific technique or approach makes this script successful?
      - Which components are working well and must be preserved?
      - What is the script's main competitive advantage?
   
   2. PINPOINT SPECIFIC WEAKNESSES:
      - Where exactly do the failures occur in the processing pipeline?
      - What specific patterns cause the script to fail?
      - Are failures due to information extraction, reasoning, formatting, or verification?
      - Can you identify the exact function or step where problems arise?
   
   3. FORM A SPECIFIC HYPOTHESIS:
      - What is the ONE most critical weakness to address?
      - What specific change would most likely improve performance?
      - How can you fix this weakness without breaking the existing strengths?
      - What verification can you add to test if your fix works?
   
   4. SURGICAL IMPROVEMENT STRATEGY:
      - Make the MINIMUM changes necessary to address the identified weakness
      - Preserve all successful components and logic
      - Add targeted verification for the specific area being improved
      - Enhance error handling for the identified failure mode
      - Add debugging output to verify the fix is working
   
   SPECIFIC REFINEMENT TECHNIQUES:
   - If failures are in information extraction: Improve prompts, add verification, better parsing
   - If failures are in reasoning: Add chain-of-thought, verification loops, multi-step reasoning
   - If failures are in formatting: Add output validation, format checking, retry logic
   - If failures are inconsistent: Add confidence scoring, multiple attempts, consensus approaches
   
   CRITICAL REFINEMENT REQUIREMENTS:
   1. Preserve the core successful approach - don't change what's working
   2. Make targeted, minimal changes focused on the specific weakness identified
   3. Add verification steps specifically for the area being improved
   4. Include debugging output to verify improvements are working
   5. EVERY LLM PROMPT must include embedded examples
   6. Test your hypothesis with additional verification
   
   Here's how to call the Gemini API. Use this example without modification:
   {llm_api_example}
   
   REFINEMENT IMPLEMENTATION:
   State Your Hypothesis: Clearly comment what specific weakness you're addressing and how
   Preserve Strengths: Keep all successful components intact
   Targeted Fix: Implement the minimal change needed to address the weakness
   Add Verification: Include checks to ensure your fix is working
   Debug Output: Add print statements to track the improvement
   
   Return a COMPLETE, RUNNABLE Python script that:
   1. Preserves the successful core approach of the original script
   2. Makes targeted improvements to address the specific identified weakness
   3. Includes a clear comment stating your improvement hypothesis
   4. Adds verification specifically for the improved component
   5. Includes embedded examples in EVERY LLM prompt
   6. Is COMPLETE - no missing code, no "..." placeholders
   7. Closes all string literals properly
   
   REFINEMENT HYPOTHESIS: [State your specific hypothesis about what to improve and why in a comment]
   
   BE EXTREMELY CAREFUL TO PROPERLY CLOSE ALL STRING QUOTES AND TRIPLE QUOTES!
   MAKE SURGICAL IMPROVEMENTS WHILE PRESERVING THE SCRIPT'S CORE STRENGTHS!
   """