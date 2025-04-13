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
    """Extract key information from the problem statement using an LLM with examples."""
    system_instruction = "You are an expert information extractor for scheduling problems."

    prompt = f"""
    Extract the key information about the scheduling problem and format it as a JSON object.
    Pay close attention to participants, duration, valid hours, valid days and individual availabilities.
    Differentiate preferences from hard constraints.

    Example Input:
    You need to schedule a meeting for John and Jennifer for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. 
    Here are the existing schedules for everyone during the days: 
    John has no meetings the whole week.
    Jennifer has meetings on Monday during 9:00 to 11:00, 11:30 to 13:00, 13:30 to 14:30, 15:00 to 17:00, Tuesday during 9:00 to 11:30, 12:00 to 17:00, Wednesday during 9:00 to 11:30, 12:00 to 12:30, 13:00 to 14:00, 14:30 to 16:00, 16:30 to 17:00.
    John would like to avoid more meetings on Monday after 14:30.

    Example Output:
    {{
      "participants": ["John", "Jennifer"],
      "duration": "30 minutes",
      "valid_hours": "9:00-17:00",
      "valid_days": ["Monday", "Tuesday", "Wednesday"],
      "availability": {{
        "John": {{
          "Monday": ["9:00-17:00"],
          "Tuesday": ["9:00-17:00"],
          "Wednesday": ["9:00-17:00"]
        }},
        "Jennifer": {{
          "Monday": ["11:00-11:30", "13:00-13:30", "14:30-15:00"],
          "Tuesday": ["11:30-12:00"],
          "Wednesday": ["11:30-12:00", "12:30-13:00", "14:00-14:30", "16:00-16:30"]
        }}
      }},
      "preferences": {{
        "John": "Avoid Monday after 14:30"
      }}
    }}

    Now, extract information from this new problem:
    {problem}
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error extracting information: {str(e)}"

def find_available_time(extracted_info_json):
    """Find an available time slot given the extracted information using LLM reasoning."""
    system_instruction = "You are an expert at finding available time slots for meetings."

    prompt = f"""
    Given the following extracted information about a scheduling problem, find a valid time slot for the meeting.
    Consider the duration, valid hours, valid days, and individual availabilities. Find the earliest available slot.
    Consider preferences last, but do not violate hard constraints.

    Extracted Information:
    {extracted_info_json}

    Reasoning:
    First, consider the valid days. Then, consider each participant's availability on those days within the valid hours.
    Look for a time slot that accommodates everyone, giving priority to earliest times.

    Example:
    If John is available all day Monday and Jennifer is available Monday 13:00-14:00, then a valid meeting time is Monday 13:00-13:30 for a 30-minute meeting.

    Now, find the available time for the information provided:
    """
    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error finding available time: {str(e)}"

def verify_solution(problem, proposed_solution):
    """Verify if the proposed solution satisfies all constraints using LLM with examples."""
    system_instruction = "You are a solution verifier, ensuring that the proposed solution is correct."

    prompt = f"""
    You are given a problem and a proposed solution. Determine if the solution is valid.
    List all the constraints present in the problem and verify them one by one.
    Return "Valid" if all constraints are satisfied, else return "Invalid" and list the violated constraints.

    Example Problem:
    You need to schedule a meeting for John and Jennifer for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. 
    Here are the existing schedules for everyone during the days: 
    John has no meetings the whole week.
    Jennifer has meetings on Monday during 9:00 to 11:00, 11:30 to 13:00, 13:30 to 14:30, 15:00 to 17:00, Tuesday during 9:00 to 11:30, 12:00 to 17:00, Wednesday during 9:00 to 11:30, 12:00 to 12:30, 13:00 to 14:00, 14:30 to 16:00, 16:30 to 17:00.
    John would like to avoid more meetings on Monday after 14:30.

    Proposed Solution:
    Schedule the meeting on Wednesday from 13:00 to 13:30.

    Verification:
    Jennifer is busy 13:00-14:00 on Wednesday, therefore the proposed solution is invalid.

    Problem:
    {problem}

    Proposed Solution:
    {proposed_solution}
    """

    try:
        return call_llm(prompt, system_instruction)
    except Exception as e:
        return f"Error verifying solution: {str(e)}"

def main(question):
    """Main function to schedule a meeting."""
    try:
        extracted_info_json = extract_information(question)
        if "Error" in extracted_info_json:
            return extracted_info_json

        available_time = find_available_time(extracted_info_json)
        if "Error" in available_time:
            return available_time
        
        verification_result = verify_solution(question, available_time)
        if "Error" in verification_result:
            return verification_result

        return f"Here is the proposed time: {available_time}"

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"