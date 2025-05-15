import os
import re
import math

# EXPLORATION: Iterative Transformation Discovery and Application with Structural Similarity
# HYPOTHESIS: By using structural similarity analysis between input and output grids, combined with an iterative refinement loop, we can improve pattern generalization and output grid correctness. The structural similarity should assist the LLM in understanding the visual transformations happening.
# The core idea is to have one agent discover potential transformation rules based on visual similarity, and another agent validate/refine those rules by attempting to transform the test grid iteratively. If transformation results in a STRUCTURALLY SIMILAR result with the training output, then we will say it is valid.

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems through iterative transformation and structural similarity analysis."""

    # 1. Discover Initial Transformation Rules
    discovery_result = discover_transformation_rules(question)
    if not discovery_result["is_valid"]:
        return f"Error: Could not discover initial transformation rules. {discovery_result['error']}"
    transformation_rules = discovery_result["transformation_rules"]

    # 2. Iteratively Refine and Apply Transformation
    refined_grid = iteratively_refine_and_apply(question, transformation_rules, max_attempts)
    return refined_grid

def discover_transformation_rules(question):
    """Discovers potential transformation rules by analyzing structural similarity between training input/output grids."""
    system_instruction = "You are an expert in discovering transformation rules in grid-based problems. Analyze the visual similarities and differences between the input and output grids to determine how the transformation works."

    prompt = f"""
    Given the following grid transformation problem, analyze the training examples and discover potential transformation rules by identifying structural similarities and differences between the input and output grids.

    Example:
    Problem:
    === TRAINING EXAMPLES ===
    Input Grid:
    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]]
    Output Grid:
    [[2, 2, 2],
     [1, 1, 1],
     [2, 2, 2]]
    Transformation Rules:
    Rows that contain only 0s in the input are replaced with rows of 2s in the output, while rows containing only 1s remain unchanged.

    Problem:
    {question}

    Transformation Rules:
    """

    transformation_rules = call_llm(prompt, system_instruction)

    # Basic validation to ensure *something* was output. More thorough validation happens in iteratively_refine_and_apply
    if transformation_rules and transformation_rules.strip():
        return {"is_valid": True, "transformation_rules": transformation_rules, "error": None}
    else:
        return {"is_valid": False, "transformation_rules": None, "error": "Failed to discover transformation rules."}

def iteratively_refine_and_apply(question, transformation_rules, max_attempts):
    """Iteratively refines and applies transformation rules by analyzing the structural similarity between the transformed grid and training grids."""
    system_instruction = "You are an expert in applying transformation rules to grids. You will iteratively refine your approach by comparing your results with structural similarities in the training grids. If they appear similar, then that means you are on the right track."

    prompt = f"""
    Given the following grid transformation problem and transformation rules, iteratively refine and apply the rules to the test input grid. Analyze the structural similarity between the transformed grid and the training grids to determine if the rules are being applied correctly.

    Problem:
    {question}
    Transformation Rules:
    {transformation_rules}

    Generate the transformed grid as a list of lists.
    """
    # This starts the process in a way which has the system iteratively refine and generate.
    transformed_grid = call_llm(prompt, system_instruction)

    for attempt in range(max_attempts): # Set max attemps
        # Add code to refine the transformation and check the validity of the output
        # This is where the magic is going to happen. The first pass of the LLM is probably going to be garbage, but
        # the fact that this iterates might be helpful.
        # 1. Compare the transformed_grid with the TRAINING output to see what the STRUCTURAL SIMILARITY is (e.g., if they are the same)
        # 2. If the TRANSFORMED_GRID has greater STRUCTURAL SIMILARITY, then the TRANSFORMED_GRID becomes the transformation_rules in the next iteration.
        # 3. With each iteration, the output converges as the LLM realizes and validates the approach it is using.

        validation_prompt = f"""
            Given the following grid transformation problem and transformation rules, iteratively refine and apply the rules to the test input grid.
            {question}
            The previously transformed grid:
            {transformed_grid}
            Now check for the validity and compare with training dataset, re-evaluate with a STRUCTURAL SIMILARITY approach.

            Does the generated grid transformation look STRUCTURALLY SIMILAR? Provide an explanation and update the results.
            """
        transformed_grid = call_llm(validation_prompt, system_instruction)

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