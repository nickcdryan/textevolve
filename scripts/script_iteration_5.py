import os
import re
import math

def main(question):
    """
    This script solves questions based on a given passage by:
    1. Determining the question type with examples.
    2. Extracting the relevant information with examples.
    3. Generating the answer with examples.
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
    generated_answer = generate_answer(extracted_info, question_type, question)
    if "Error" in generated_answer:
        return generated_answer

    return generated_answer # Directly return the generated answer

def determine_question_type(question):
    """Determine the type of the question (numerical, identification, etc.) with examples."""
    system_instruction = "You are an expert at classifying question types."
    prompt = f"""
    Determine the type of question given the following examples. Return the type only.

    Example 1:
    Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
    Type: Numerical

    Example 2:
    Question: Who caught the final touchdown of the game?
    Type: Identification

    Example 3:
    Question: Which star has a smaller mass, Nu Phoenicis or Gliese 915?
    Type: Comparative
    
    Example 4:
    Question: Who died first John of Ibelin or Isabella II?
    Type: Sequencing

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
    """Extract relevant information from the passage with examples, tailored to question type."""
    system_instruction = "You are an expert at extracting relevant information."
    prompt = f"""
    Extract relevant information from the passage based on the given question type.
    Return the extracted information as a plain text summary.

    Example 1:
    Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
    Type: Numerical
    Extracted Info: Chris Johnson's first touchdown yards, Jason Hanson's first field goal yards.

    Example 2:
    Question: Who caught the final touchdown of the game?
    Type: Identification
    Extracted Info: Player who caught the final touchdown.

    Example 3:
    Question: Which star has a smaller mass, Nu Phoenicis or Gliese 915?
    Type: Comparative
    Extracted Info: Mass of Nu Phoenicis, Mass of Gliese 915.
    
    Example 4:
    Question: Who died first John of Ibelin or Isabella II?
    Type: Sequencing
    Extracted Info: Date of death of John of Ibelin, Date of death of Isabella II.

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

def generate_answer(extracted_info, question_type, question):
    """Generate the answer based on extracted information and question type with examples."""
    system_instruction = "You are an expert at generating correct answers."
    prompt = f"""
    Generate an answer to the question based on the extracted information.

    Example 1:
    Extracted Info: Chris Johnson's first touchdown yards = 40, Jason Hanson's first field goal yards = 30.
    Question Type: Numerical
    Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
    Answer: 40 + 30 = 70 yards

    Example 2:
    Extracted Info: Player who caught the final touchdown = Mark Clayton
    Question Type: Identification
    Question: Who caught the final touchdown of the game?
    Answer: Mark Clayton

    Example 3:
    Extracted Info: Mass of Nu Phoenicis = 1.2 solar masses, Mass of Gliese 915 = 0.85 solar masses.
    Question Type: Comparative
    Question: Which star has a smaller mass, Nu Phoenicis or Gliese 915?
    Answer: Gliese 915
    
    Example 4:
    Extracted Info: Date of death of John of Ibelin = 1236, Date of death of Isabella II = 1228.
    Question Type: Sequencing
    Question: Who died first John of Ibelin or Isabella II?
    Answer: Isabella II

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