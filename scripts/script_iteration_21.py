import os
import re
import math

# EXPLORATION: Region-Based Transformation with Rule Selection via LLM and Rule Application with Explicit Mapping
# HYPOTHESIS: The LLM can be used to identify regions and rules on those regions, then it can more reliably transform them with this information.
# We're attempting to improve performance in cases with inconsistent operations that past systems struggled with.
# We will try dividing the grid into regions and applying the mapping transformation on each region by prompting LLM.

def solve_grid_transformation(question):
    """Solves grid transformation problems by region-based rule selection and application."""
    
    # Step 1: Identify Regions and Select Rules with Explanation
    region_rule_selection_result = identify_regions_and_rules(question)
    if not region_rule_selection_result["is_valid"]:
        return f"Error: Could not identify regions and rules. {region_rule_selection_result['error']}"
    
    regions_and_rules = region_rule_selection_result["regions_and_rules"]
    
    # Step 2: Apply Transformation with Explicit Mapping
    transformed_grid = apply_transformation(question, regions_and_rules)
    return transformed_grid

def identify_regions_and_rules(question):
    """Identifies regions in the grid and selects transformation rules for each region."""
    system_instruction = "You are an expert at identifying regions in grids and selecting appropriate transformation rules for each region."
    
    prompt = f"""
    Given the following grid transformation problem, analyze the training examples and identify distinct regions within the grid. For each region, select an appropriate transformation rule.
    
    Example 1:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[1, 1, 1],
     [0, 0, 0],
     [2, 2, 2]]
    Output Grid:
    [[3, 3, 3],
     [0, 0, 0],
     [4, 4, 4]]
    Regions and Rules:
    Region 1: Top row. Rule: Add 2 to each element.
    Region 2: Middle row. Rule: No transformation.
    Region 3: Bottom row. Rule: Add 2 to each element.
    
    Problem:
    {question}
    
    Regions and Rules:
    """
    
    regions_and_rules = call_llm(prompt, system_instruction)
    
    # Simple validation to ensure that *something* was output
    if regions_and_rules and regions_and_rules.strip():
        return {"is_valid": True, "regions_and_rules": regions_and_rules, "error": None}
    else:
        return {"is_valid": False, "regions_and_rules": None, "error": "Failed to identify regions and rules."}

def apply_transformation(question, regions_and_rules):
    """Applies the transformation rules to the test input grid, explicitly mapping values."""
    system_instruction = "You are an expert at applying transformation rules to grids, focusing on explicit value mapping."
    
    prompt = f"""
    Given the following grid transformation problem and the identified regions and rules, apply the rules to the test input grid. Provide ONLY the transformed grid.
    
    Example 1:
    Problem:
    Input Grid:
    [[1, 1, 1],
     [0, 0, 0],
     [2, 2, 2]]
    Regions and Rules:
    Region 1: Top row. Rule: Add 2 to each element.
    Region 2: Middle row. Rule: No transformation.
    Region 3: Bottom row. Rule: Add 2 to each element.
    Output Grid:
    [[3, 3, 3],
     [0, 0, 0],
     [4, 4, 4]]

    Problem:
    {question}
    Regions and Rules:
    {regions_and_rules}
    Output Grid:
    """
    
    transformed_grid = call_llm(prompt, system_instruction)
    return transformed_grid

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template or invent configuration options. This is how you call the LLM."""
    try:
        from google import genai
        from google.genai import types
    
        # Initialize the Gemini client
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
        # Call the API with system instruction if provided
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

def main(question):
    """Main function to solve the grid transformation task."""
    try:
        answer = solve_grid_transformation(question)
        return answer
    except Exception as e:
        return f"Error in main function: {str(e)}"