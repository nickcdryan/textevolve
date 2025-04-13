import json
import re
import os

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

def extract_info_with_examples(question):
    """Extracts information using LLM with examples."""
    system_instruction = "You are an expert meeting scheduler."
    prompt = f"""
    Extract the following information from the question, and return it as a JSON object: participants, duration, valid_hours, valid_days, availability.

    Example:
    Question: You need to schedule a meeting for John and Jennifer for half an hour between 9:00 to 17:00 on Monday. John is free all day. Jennifer is busy 10:00-12:00.
    JSON:
    {{
        "participants": ["John", "Jennifer"],
        "duration": "30 minutes",
        "valid_hours": "9:00-17:00",
        "valid_days": ["Monday"],
        "availability": {{
            "John": [],
            "Jennifer": ["10:00-12:00"]
        }}
    }}

    Now, for the question: {question}
    """
    return call_llm(prompt, system_instruction)

def find_available_time_with_examples(info_json):
    """Finds available time using LLM and extracted info."""
    system_instruction = "You are an expert meeting scheduler."
    prompt = f"""
    Given the following information about a meeting, find a valid time slot that works for everyone.

    Example:
    Info:
    {{
        "participants": ["John", "Jennifer"],
        "duration": "30 minutes",
        "valid_hours": "9:00-17:00",
        "valid_days": ["Monday"],
        "availability": {{
            "John": [],
            "Jennifer": ["10:00-12:00"]
        }}
    }}
    Reasoning:
    John is available all day. Jennifer is busy from 10:00 to 12:00. The meeting duration is 30 minutes, and it should be on Monday between 9:00 and 17:00. So, a possible time is 9:00-9:30.
    Solution: Monday, 9:00 - 9:30

    Now, given the following information: {info_json}
    """
    return call_llm(prompt, system_instruction)

def verify_solution_with_examples(question, solution):
    """Verifies the proposed solution against the original question."""
    system_instruction = "You are a solution verification expert."
    prompt = f"""
    Given the original question and a proposed solution, verify if the solution is correct.
    If there are errors, explain and say what a better solution would be.

    Example:
    Question: You need to schedule a meeting for John and Jennifer for half an hour between 9:00 to 17:00 on Monday. John is free all day. Jennifer is busy 10:00-12:00.
    Proposed solution: Monday, 10:30-11:00
    Reasoning:
    John is available. But Jennifer is busy from 10:00 to 12:00. So, this time slot is invalid.
    Solution: Invalid. A better solution is Monday, 9:00-9:30.

    Now, given the question: {question}, and solution: {solution}
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule meetings."""
    try:
        info_json = extract_info_with_examples(question)
        available_time = find_available_time_with_examples(info_json)
        verification_result = verify_solution_with_examples(question, available_time)

        if "Invalid" in verification_result:
            return "Error: " + verification_result
        else:
            return "Here is the proposed time: " + available_time
    except Exception as e:
        return f"Error: {str(e)}"