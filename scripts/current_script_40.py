import os
import re
import math

def main(question):
    """Transforms a grid based on patterns in training examples using LLM-driven pattern recognition.
    This approach uses a "Hierarchical Rule Extraction and Application with Local Contextual Analysis" strategy.

    Hypothesis: By first extracting high-level rules and then refining them with local contextual analysis of cell neighborhoods, we can improve transformation accuracy. This combines global pattern recognition with localized adjustments. The key difference is splitting extraction into high-level, then refining with contextual analysis.

    Previous approaches failed in generalizing and accurately applying rules. This approach will attempt to address that through local contextualization and hierarchical refinement of rules.
    """
    return solve_grid_transformation(question)

def solve_grid_transformation(problem_text, max_attempts=3):
    """Solves the grid transformation problem by extracting high-level rules and refining with local context."""

    system_instruction = "You are an expert at identifying grid transformation patterns. First extract high-level rules, then analyze local context to refine them."
    
    # STEP 1: Extract High-Level Transformation Rule - with examples!
    high_level_rule_prompt = f"""
    Analyze the problem and extract the high-level transformation rule.

    Example 1:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[1, 0], [0, 1]]\n\nOutput Grid:\n[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n
    High-Level Rule: The input grid is expanded, with original '1's placed diagonally.

    Example 2:
    Problem: Grid Transformation Task\n\n=== TRAINING EXAMPLES ===\n\nExample 1:\nInput Grid:\n[[2, 8], [8, 2]]\n\nOutput Grid:\n[[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]\n
    High-Level Rule: Each element is expanded into a 2x2 block of the same value.

    Problem: {problem_text}
    High-Level Rule:
    """
    
    extracted_high_level_rule = call_llm(high_level_rule_prompt, system_instruction)
    print(f"Extracted High-Level Rule: {extracted_high_level_rule}") # Diagnostic

    # STEP 2: Analyze Local Context - with examples!
    local_context_prompt = f"""
    Analyze the local context (neighboring cells) to refine the high-level rule.

    High-Level Rule: {extracted_high_level_rule}
    Problem: {problem_text}

    Example 1:
    High-Level Rule: The input grid is expanded, with original '1's placed diagonally.
    Local Context Analysis: '1's are placed on the main diagonal, other cells are '0'.

    Example 2:
    High-Level Rule: Each element is expanded into a 2x2 block of the same value.
    Local Context Analysis: Each cell expands without consideration of neighboring values.

    Local Context Analysis:
    """
    
    local_context_analysis = call_llm(local_context_prompt, system_instruction)
    print(f"Local Context Analysis: {local_context_analysis}") # Diagnostic

    # STEP 3: Apply Refined Rule - with examples!
    apply_refined_rule_prompt = f"""
    Apply the refined transformation rule (high-level rule + local context analysis) to the input grid.

    High-Level Rule: {extracted_high_level_rule}
    Local Context Analysis: {local_context_analysis}
    Problem: {problem_text}

    Example 1:
    High-Level Rule: The input grid is expanded, with original '1's placed diagonally.
    Local Context Analysis: '1's are placed on the main diagonal, other cells are '0'.
    Input Grid: [[1, 0], [0, 1]]
    Transformed Grid: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    Example 2:
    High-Level Rule: Each element is expanded into a 2x2 block of the same value.
    Local Context Analysis: Each cell expands without consideration of neighboring values.
    Input Grid: [[2, 8], [8, 2]]
    Transformed Grid: [[2, 2, 8, 8], [2, 2, 8, 8], [8, 8, 2, 2], [8, 8, 2, 2]]

    Transformed Grid:
    """
    
    #Attempt to refine rule and apply
    for attempt in range(max_attempts):
        try:
            transformed_grid_text = call_llm(apply_refined_rule_prompt, system_instruction)
            print(f"Transformed Grid Text: {transformed_grid_text}") # Diagnostic

            # STEP 4: Basic validation: does the output look like a grid?
            if "[" not in transformed_grid_text or "]" not in transformed_grid_text:
                print(f"Attempt {attempt+1} failed: Output does not resemble a grid. Retrying...")
                continue

            return transformed_grid_text

        except Exception as e:
            print(f"Attempt {attempt+1} failed with error: {e}. Retrying...")

    # Fallback approach if all attempts fail
    return "[[0,0,0],[0,0,0],[0,0,0]]"


def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        if system_instruction:
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                ),
                contents=prompt
            )
        else:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Error: {str(e)}"