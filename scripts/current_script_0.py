import os
import json

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

def extract_info_with_examples(problem):
    """Extract structured information (JSON) with examples in the prompt."""
    system_instruction = "You are an expert at extracting structured information."
    prompt = f"""
    Extract information from the problem into a JSON format.

    Example:
    Problem: Schedule a meeting for John and Jane for 30 minutes on Monday between 9:00 and 17:00. John is busy from 10:00-11:00, and Jane is busy from 14:00-15:00.
    Output:
    {{
        "participants": ["John", "Jane"],
        "duration": "30 minutes",
        "day": "Monday",
        "start_time": "9:00",
        "end_time": "17:00",
        "John_busy": ["10:00-11:00"],
        "Jane_busy": ["14:00-15:00"]
    }}

    Now, extract the information from this problem:
    {problem}
    """
    return call_llm(prompt, system_instruction)

def find_available_time_with_examples(extracted_info):
    """Find an available meeting time using the extracted information."""
    system_instruction = "You are an expert at finding available meeting times."
    prompt = f"""
    Given the extracted information, find a suitable meeting time.

    Example:
    Extracted Info:
    {{
        "participants": ["John", "Jane"],
        "duration": "30 minutes",
        "day": "Monday",
        "start_time": "9:00",
        "end_time": "17:00",
        "John_busy": ["10:00-11:00"],
        "Jane_busy": ["14:00-15:00"]
    }}
    Output: Monday, 9:00-9:30

    Now, find a suitable meeting time for the following information:
    {extracted_info}
    """
    return call_llm(prompt, system_instruction)

def verify_solution_with_examples(problem, solution):
    """Verify the proposed solution."""
    system_instruction = "You are a solution verification expert."
    prompt = f"""
    Verify if the proposed solution is correct.

    Example:
    Problem: Schedule a meeting for John and Jane for 30 minutes on Monday between 9:00 and 17:00. John is busy from 10:00-11:00, and Jane is busy from 14:00-15:00.
    Solution: Monday, 9:00-9:30
    Output: Valid

    Problem: Schedule a meeting for John and Jane for 30 minutes on Monday between 9:00 and 17:00. John is busy from 10:00-11:00, and Jane is busy from 14:00-15:00.
    Solution: Monday, 10:30-11:00
    Output: Invalid, conflicts with John's schedule

    Now, verify if the following solution is correct:
    Problem: {problem}
    Solution: {solution}
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""
    try:
        extracted_info = extract_info_with_examples(question)
        available_time = find_available_time_with_examples(extracted_info)
        verification = verify_solution_with_examples(question, available_time)

        if "Invalid" in verification:
            return "No suitable time found."
        else:
            return f"Here is the proposed time: {available_time}"
    except Exception as e:
        return f"Error: {str(e)}"