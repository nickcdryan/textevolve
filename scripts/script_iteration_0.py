import os
import re
import math

def main(question):
    """
    This script solves questions based on a given passage by:
    1. Analyzing the question type (numerical, logical, etc.).
    2. Extracting the relevant information from the passage.
    3. Generating and verifying the answer with LLM.

    This approach uses a chain-of-thought process with specific verification steps at each stage
    to ensure accuracy. The primary hypothesis is that focusing on the intent of the question
    before diving into extraction will lead to better results.
    """

    # Step 1: Determine the question type
    question_type = determine_question_type(question)
    if "Error" in question_type:
        return question_type  # Return error message

    # Step 2: Extract relevant information from the passage
    extracted_info = extract_relevant_info(question, question_type)
    if "Error" in extracted_info:
        return extracted_info

    # Step 3: Generate the answer
    generated_answer = generate_answer(extracted_info, question_type)
    if "Error" in generated_answer:
        return generated_answer

    # Step 4: Verify the generated answer
    verified_answer = verify_answer(generated_answer, question)
    if "Error" in verified_answer:
        return verified_answer

    return verified_answer

def determine_question_type(question):
    """Determine the type of the question (numerical, logical, etc.)."""
    system_instruction = "You are an expert at classifying question types."
    prompt = f"""
    Determine the type of question given the following examples. Return the type only.

    Example 1:
    Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
    Type: Numerical

    Example 2:
    Question: Who caught the final touchdown of the game?
    Type: Identification

    Question: {question}
    Type:
    """
    try:
        question_type = call_llm(prompt, system_instruction)
        if not question_type:
            return "Error: Could not determine question type"
        return question_type
    except Exception as e:
        return f"Error: {str(e)}"

def extract_relevant_info(question, question_type):
    """Extract relevant information from the passage."""
    system_instruction = "You are an expert at extracting relevant information."
    prompt = f"""
    Extract relevant information from the question based on the given question type.
    Return the extracted information as a plain text summary.

    Example 1:
    Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
    Type: Numerical
    Extracted Info: Chris Johnson's first touchdown yards, Jason Hanson's first field goal yards.

    Example 2:
    Question: Who caught the final touchdown of the game?
    Type: Identification
    Extracted Info: Player who caught the final touchdown.

    Question: {question}
    Type: {question_type}
    Extracted Info:
    """
    try:
        extracted_info = call_llm(prompt, system_instruction)
        if not extracted_info:
            return "Error: Could not extract information."
        return extracted_info
    except Exception as e:
        return f"Error: {str(e)}"

def generate_answer(extracted_info, question_type):
    """Generate the answer based on extracted information and question type."""
    system_instruction = "You are an expert at generating correct answers."
    prompt = f"""
    Generate an answer to the question based on the extracted information.

    Example 1:
    Extracted Info: Chris Johnson's first touchdown yards, Jason Hanson's first field goal yards.
    Question Type: Numerical
    Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
    Answer: Find yards for each event and add them.

    Example 2:
    Extracted Info: Player who caught the final touchdown.
    Question Type: Identification
    Question: Who caught the final touchdown of the game?
    Answer: Identify the player.

    Extracted Info: {extracted_info}
    Question Type: {question_type}
    Question: {question}
    Answer:
    """
    try:
        answer = call_llm(prompt, system_instruction)
        if not answer:
            return "Error: Could not generate answer."
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

def verify_answer(generated_answer, question):
    """Verify the generated answer to ensure correctness."""
    system_instruction = "You are an expert at verifying the correctness of answers."
    prompt = f"""
    Verify if the generated answer is correct and makes sense given the question.

    Example 1:
    Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
    Generated Answer: Find yards for each event and add them.
    Is Correct: Verify that the required information to "Find yards for each event and add them" is known and possible.

    Example 2:
    Question: Who caught the final touchdown of the game?
    Generated Answer: Identify the player.
    Is Correct: Verify that it makes sense to "Identify the player"

    Question: {question}
    Generated Answer: {generated_answer}
    Is Correct:
    """
    try:
        is_correct = call_llm(prompt, system_instruction)
        if not is_correct:
            return "Error: Could not verify answer."
        return is_correct
    except Exception as e:
        return f"Error: {str(e)}"

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