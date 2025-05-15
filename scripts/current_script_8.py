import os
import re
import math

# HYPOTHESIS: Instead of trying to extract rigid rules, use multiple LLM calls to perform analogical reasoning 
# by comparing the test input to each training example individually. This approach leverages the LLM's
# ability to identify similarities without requiring explicit rule extraction. This will be more robust
# to minor variations in grid size and patterns.

def solve_grid_transformation(question, max_attempts=3):
    """Solves grid transformation problems using analogical reasoning."""

    # 1. Extract training examples and test input from the question
    training_examples = extract_training_examples(question)
    test_input = extract_test_input(question)

    # 2. Apply analogical reasoning to each training example and generate a potential output
    potential_outputs = []
    for i, example in enumerate(training_examples):
        potential_output = analogical_reasoning(test_input, example["input_grid"], example["output_grid"])
        potential_outputs.append({"example_id": i, "output": potential_output})

    # 3. Select the best output based on similarity to the training examples
    best_output = select_best_output(test_input, potential_outputs, training_examples)

    return best_output

def extract_training_examples(question):
    """Extracts training examples from the question."""
    system_instruction = "You are an expert at extracting training examples."
    prompt = f"""
    Given the following grid transformation problem, extract the training examples into a list of dictionaries, 
    where each dictionary has the keys "input_grid" and "output_grid".
    
    Example:
    Question:
    === TRAINING EXAMPLES ===
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    === TEST INPUT ===
    [[5, 6], [7, 8]]
    
    Training Examples:
    [
      {{"input_grid": "[[1, 2], [3, 4]]", "output_grid": "[[2, 3], [4, 5]]"}}
    ]
    
    Question:
    {question}
    
    Training Examples:
    """
    
    examples_string = call_llm(prompt, system_instruction)
    
    # Basic string parsing due to avoidance of json.loads()
    examples = []
    example_splits = examples_string.strip("[]\n").split("}")
    for split in example_splits:
      if not split.strip():
        continue

      try:
          input_grid = split.split('input_grid": "[[')[1].split(']]"')[0]
          input_grid = "[[" + input_grid + "]]"

          output_grid = split.split('output_grid": "[[')[1].split(']]"')[0]
          output_grid = "[[" + output_grid + "]]"

          examples.append({"input_grid": input_grid, "output_grid": output_grid})
      except IndexError:
          continue

    return examples

def extract_test_input(question):
    """Extracts the test input grid from the question."""
    system_instruction = "You are an expert at extracting the test input grid."
    prompt = f"""
    Given the following grid transformation problem, extract the test input grid.
    
    Example:
    Question:
    === TRAINING EXAMPLES ===
    Input Grid: [[1, 2], [3, 4]]
    Output Grid: [[2, 3], [4, 5]]
    === TEST INPUT ===
    [[5, 6], [7, 8]]
    
    Test Input:
    [[5, 6], [7, 8]]
    
    Question:
    {question}
    
    Test Input:
    """
    test_input = call_llm(prompt, system_instruction)
    return test_input

def analogical_reasoning(test_input, training_input, training_output):
    """Applies analogical reasoning to generate a potential output."""
    system_instruction = "You are an expert at analogical reasoning for grid transformations."
    prompt = f"""
    Given the following test input grid, training input grid, and training output grid, use analogical reasoning to generate a potential output grid.
    
    Example:
    Test Input: [[5, 6], [7, 8]]
    Training Input: [[1, 2], [3, 4]]
    Training Output: [[2, 3], [4, 5]]
    Potential Output: [[6, 7], [8, 9]] (Each element in the test input is incremented by 1, similar to the training example.)
    
    Test Input: {test_input}
    Training Input: {training_input}
    Training Output: {training_output}
    Potential Output:
    """
    potential_output = call_llm(prompt, system_instruction)
    return potential_output

def select_best_output(test_input, potential_outputs, training_examples):
    """Selects the best output based on similarity to the training examples."""
    system_instruction = "You are an expert at selecting the best output from a list of potential outputs."
    prompt = f"""
    Given the following test input grid, a list of potential output grids, and the training examples, select the best output grid that most closely matches the transformation logic.
    
    Training examples: {training_examples}
    Test Input: {test_input}
    Potential Outputs: {potential_outputs}
    
    Reason step-by-step to figure out:
    1. Is there a transformation pattern related to all examples?
    2. Which potential ouput best matches what would be expected?
    Output:
    """
    best_output = call_llm(prompt, system_instruction)
    return best_output

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