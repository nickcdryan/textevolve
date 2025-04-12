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

def extract_information(problem):
    """Extract key information from the problem statement using chain-of-thought with examples."""
    system_instruction = "You are an information extraction expert."
    prompt = f"""
    Extract key information from the problem.

    Example:
    Problem: You need to schedule a meeting for John and Jennifer for half an hour between 9:00 and 17:00 on Monday. John is free all day. Jennifer is busy 10:00-11:00 and 14:00-15:00.
    Extracted Information:
    {{
      "participants": ["John", "Jennifer"],
      "duration": "30 minutes",
      "start_time": "9:00",
      "end_time": "17:00",
      "day": "Monday",
      "John": "Free all day",
      "Jennifer": ["10:00-11:00", "14:00-15:00"]
    }}

    Problem: {problem}
    Extracted Information:
    """
    return call_llm(prompt, system_instruction)

def find_available_time(extracted_info_json):
    """Find an available time slot using the extracted information with examples."""
    system_instruction = "You are a scheduling assistant."
    prompt = f"""
    Find a suitable meeting time given the following information.

    Example:
    Extracted Information:
    {{
      "participants": ["John", "Jennifer"],
      "duration": "30 minutes",
      "start_time": "9:00",
      "end_time": "17:00",
      "day": "Monday",
      "John": "Free all day",
      "Jennifer": ["10:00-11:00", "14:00-15:00"]
    }}
    Solution: Monday, 9:00 - 9:30

    Extracted Information:
    {extracted_info_json}
    Solution:
    """
    return call_llm(prompt, system_instruction)

def verify_solution(problem, proposed_solution):
    """Verify the proposed solution with the original problem with examples."""
    system_instruction = "You are a solution verification expert."
    prompt = f"""
    Verify if the proposed solution is valid based on the problem.

    Example:
    Problem: You need to schedule a meeting for John and Jennifer for half an hour between 9:00 and 17:00 on Monday. John is free all day. Jennifer is busy 10:00-11:00 and 14:00-15:00.
    Proposed Solution: Monday, 10:30 - 11:00
    Verification: Invalid, Jennifer is busy from 10:00-11:00

    Problem: {problem}
    Proposed Solution: {proposed_solution}
    Verification:
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""
    try:
        extracted_info = extract_information(question)
        proposed_solution = find_available_time(extracted_info)
        verification_result = verify_solution(question, proposed_solution)

        if "Invalid" in verification_result:
            return "No valid time found."
        else:
            return f"Here is the proposed time: {proposed_solution}"
    except Exception as e:
        return f"Error: {str(e)}"