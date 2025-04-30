import os
import re
import math

def main(question):
    """
    Solves grid transformation tasks by using a "Transformation Decomposition and Targeted Analogy" approach.

    Hypothesis: This approach will improve accuracy by decomposing transformations into component steps,
    identifying relevant analogies between training examples, and applying transformations based on these analogies.
    """
    try:
        # 1. Extract relevant grid data and perform validation.
        extracted_data = extract_data(question)
        if "Error" in extracted_data:
            return f"Data extraction error: {extracted_data}"

        # 2. Decompose the transformation using targeted analogy from training examples.
        transformation = decompose_and_transform(extracted_data)
        if "Error" in transformation:
            return f"Transformation decomposition error: {transformation}"

        return transformation

    except Exception as e:
        return f"Unexpected error: {str(e)}"

def extract_data(question):
    """Extracts training and test data from the problem question."""
    system_instruction = "You are an expert at extracting structured data, especially from grid transformation problems."
    prompt = f"""
    Extract the training examples and test input from the question. Ensure that both training examples and test input are present. Format the output as a dictionary-like string.

    Example:
    Question: Grid Transformation Task Training Examples: [{{'input': [[1, 2], [3, 4]], 'output': [[4, 3], [2, 1]]}}] Test Input: [[5, 6], [7, 8]]
    Extracted Data: {{'training_examples': '[{{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}}]', 'test_input': '[[5, 6], [7, 8]]'}}

    Question: {question}
    Extracted Data:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting data: {str(e)}"

def decompose_and_transform(extracted_data):
    """Decomposes the transformation into simpler rules, and applies these rules based on the training examples.

    This version aims to explicitly find an analogous element in the training data, and use that analogous element to help
    determine the transform instead of the entire training set.

    It checks the training and test sets for elements that have the same value in both sets, and assumes that that may be the element that gets transformed into the test element"""
    system_instruction = "You are an expert at transforming and decomposing data into simple transformation rules."
    prompt = f"""
    Decompose the transformation and apply it to the test data. Use the training examples to guide your application of the test data

    Example:
    Extracted Data: {{"training_examples": "[{{'input': [[1, 2], [3, 4]], 'output': [[4, 3], [2, 1]]}}]", "test_input": "[[5, 6], [7, 8]]"}}
    Transformation: Each number takes on the value of its reflection across both diagonals of its training data. Therefore the solution is [[8,7],[6,5]].

    Extracted Data: {extracted_data}
    Transformation:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting data: {str(e)}"

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