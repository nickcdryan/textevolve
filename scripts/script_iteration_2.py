import os
import json
import re

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
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

def extract_information_with_examples(problem):
    """Extract key information from the problem statement using embedded examples."""
    system_instruction = "You are an information extraction specialist. Identify all entities, relationships, and constraints."

    prompt = f"""
    Extract key information from this problem. Focus on entities, relationships, and constraints.

    Example:
    Question: You need to schedule a meeting for John and Jennifer for half an hour...
    Extracted Information:
    {{ "participants": ["John", "Jennifer"], "duration": "30 minutes", ... }}

    Now, extract information from this new problem:
    {problem}
    """

    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error during information extraction: {str(e)}"

def find_available_time_with_examples(extracted_info):
    """Find an available time slot using the extracted information."""
    system_instruction = "You are a scheduling assistant that finds available time slots given participants' schedules."
    prompt = f"""
    Given the extracted information, find an available time slot.

    Example:
    Extracted Information: {{ "participants": ["John", "Jennifer"], "duration": "30 minutes", ... }}
    Available Time: Monday, 14:30 - 15:00

    Now, find an available time slot given this extracted information:
    {extracted_info}
    """

    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error during time slot finding: {str(e)}"

def verify_solution_with_examples(problem, proposed_solution):
    """Verify if the proposed solution satisfies all constraints using embedded examples."""
    system_instruction = "You are a solution verifier. Verify solutions satisfy all constraints."

    prompt = f"""
    Verify if this proposed solution satisfies all constraints.

    Example:
    Problem: You need to schedule a meeting for John and Jennifer...
    Proposed Solution: Schedule the meeting on Wednesday from 13:00 to 13:30.
    Verification Result: INVALID - Conflicts with Jennifer's schedule.

    Problem:
    {problem}

    Proposed Solution:
    {proposed_solution}
    """

    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error during solution verification: {str(e)}"

def main(question):
    """Main function to schedule a meeting."""
    try:
        extracted_info = extract_information_with_examples(question)
        if "Error" in extracted_info:
            return extracted_info

        available_time = find_available_time_with_examples(extracted_info)
        if "Error" in available_time:
            return available_time

        verification_result = verify_solution_with_examples(question, available_time)
        if "Error" in verification_result:
            return verification_result

        return available_time  # Or verification_result, depending on desired output
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"