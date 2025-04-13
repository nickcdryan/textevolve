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

def extract_information_with_examples(problem):
    """Extract key information from the problem statement using embedded examples."""
    system_instruction = "You are an information extraction specialist focusing on identifying key entities and constraints."
    
    prompt = f"""
    Extract key information from this problem statement. Focus on identifying all entities, relationships, and constraints.
    
    Example usage:
    
    Question:
    You need to schedule a meeting for John and Jennifer for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. 
    Here are the existing schedules for everyone during the days: 
    John has no meetings the whole week.
    Jennifer has meetings on Monday during 9:00 to 11:00, 11:30 to 13:00, 13:30 to 14:30, 15:00 to 17:00, Tuesday during 9:00 to 11:30, 12:00 to 17:00, Wednesday during 9:00 to 11:30, 12:00 to 12:30, 13:00 to 14:00, 14:30 to 16:00, 16:30 to 17:00.
    John would like to avoid more meetings on Monday after 14:30. Find a time that works for everyone's schedule and constraints.
    
    Let's think step by step.
    
    The key entities are:
    - John (participant)
    - Jennifer (participant)
    
    The key constraints are:
    - Meeting duration: 30 minutes (half an hour)
    - Valid meeting hours: 9:00-17:00
    - Valid days: Monday, Tuesday, or Wednesday
    - John's availability: All week (no meetings)
    - Jennifer's availability:
      * Monday: Busy 9:00-11:00, 11:30-13:00, 13:30-14:30, 15:00-17:00
      * Tuesday: Busy 9:00-11:30, 12:00-17:00
      * Wednesday: Busy 9:00-11:30, 12:00-12:30, 13:00-14:00, 14:30-16:00, 16:30-17:00
    - Preferences: John prefers to avoid meetings on Monday after 14:30
    
    Extracted Information (JSON format):
    {{
      "participants": ["John", "Jennifer"],
      "duration": "30 minutes",
      "valid_hours": "9:00-17:00",
      "valid_days": ["Monday", "Tuesday", "Wednesday"],
      "availability": {{
        "John": "All times",
        "Jennifer": {{
          "Monday": ["9:00-11:00", "11:30-13:00", "13:30-14:30", "15:00-17:00"],
          "Tuesday": ["9:00-11:30", "12:00-17:00"],
          "Wednesday": ["9:00-11:30", "12:00-12:30", "13:00-14:00", "14:30-16:00", "16:30-17:00"]
        }}
      }},
      "preferences": {{
        "John": "Avoid Monday after 14:30"
      }}
    }}
    
    Now, extract information from this new problem:
    {problem}
    """
    
    return call_llm(prompt, system_instruction)

def find_available_times_with_examples(extracted_info):
    """Find available time slots based on the extracted information using examples."""
    system_instruction = "You are a scheduling assistant, finding available time slots based on participant availability."

    prompt = f"""
    Given the extracted information about participant availability, find a suitable time slot for the meeting.
    
    Example usage:
    
    Extracted Information (JSON format):
    {{
      "participants": ["John", "Jennifer"],
      "duration": "30 minutes",
      "valid_hours": "9:00-17:00",
      "valid_days": ["Monday", "Tuesday", "Wednesday"],
      "availability": {{
        "John": "All times",
        "Jennifer": {{
          "Monday": ["9:00-11:00", "11:30-13:00", "13:30-14:30", "15:00-17:00"],
          "Tuesday": ["9:00-11:30", "12:00-17:00"],
          "Wednesday": ["9:00-11:30", "12:00-12:30", "13:00-14:00", "14:30-16:00", "16:30-17:00"]
        }}
      }},
      "preferences": {{
        "John": "Avoid Monday after 14:30"
      }}
    }}
    
    Let's think step by step:
    
    1. Identify common available days: Monday, Tuesday, Wednesday
    2. Identify common available hours within the valid range (9:00-17:00) for at least 30 minutes.
       - John is available all times
       - Jennifer's availability:
          * Monday: 11:00-11:30, 13:00-13:30, 14:30-15:00
          * Tuesday: 11:30-12:00
          * Wednesday: 11:30-12:00, 12:30-13:00, 14:00-14:30, 16:00-16:30
    3. Consider John's preference: Avoid Monday after 14:30
    
    Possible meeting times:
    - Tuesday, 11:30-12:00
    - Wednesday, 11:30-12:00
    
    Proposed Time:
    Tuesday, 11:30 - 12:00
    
    Now, based on this new extracted information:
    {extracted_info}
    """
    
    return call_llm(prompt, system_instruction)

def verify_solution_with_examples(problem, proposed_solution):
    """Verify if the proposed solution satisfies all constraints using embedded examples."""
    system_instruction = "You are a critical evaluator who verifies if solutions satisfy all constraints."
    
    prompt = f"""
    Verify if this proposed solution satisfies all constraints in the problem.
    
    Example usage:
    
    Problem:
    You need to schedule a meeting for John and Jennifer for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. 
    Here are the existing schedules for everyone during the days: 
    John has no meetings the whole week.
    Jennifer has meetings on Monday during 9:00 to 11:00, 11:30 to 13:00, 13:30 to 14:30, 15:00 to 17:00, Tuesday during 9:00 to 11:30, 12:00 to 17:00, Wednesday during 9:00 to 11:30, 12:00 to 12:30, 13:00 to 14:00, 14:30 to 16:00, 16:30 to 17:00.
    John would like to avoid more meetings on Monday after 14:30.
    
    Proposed Solution:
    Schedule the meeting on Wednesday from 13:00 to 13:30.
    
    Verification:
    Let me check each constraint:
    1. Duration: The meeting is scheduled for 30 minutes (13:00-13:30) ✓
    2. Work hours: Meeting time 13:00-13:30 is within 9:00-17:00 ✓
    3. Valid day: Wednesday is one of the allowed days ✓
    4. John's availability: John has no meetings all week ✓
    5. Jennifer's availability on Wednesday:
       - Jennifer is busy 9:00-11:30, 12:00-12:30, 13:00-14:00, 14:30-16:00, 16:30-17:00
       - The proposed time 13:00-13:30 overlaps with Jennifer's busy time 13:00-14:00 ✗
    6. John's preference: Not applicable (not Monday after 14:30) ✓
    
    Result: INVALID - The solution conflicts with Jennifer's schedule on Wednesday from 13:00-14:00.
    
    Problem:
    {problem}
    
    Proposed Solution:
    {proposed_solution}
    
    Verification:
    """
    
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""
    try:
        # 1. Extract information
        extracted_info = extract_information_with_examples(question)
        if "Error" in extracted_info:
            return "Error during information extraction."
        
        # 2. Find available times
        available_times = find_available_times_with_examples(extracted_info)
        if "Error" in available_times:
            return "Error during time slot finding."

        # 3. Verify Solution
        verification = verify_solution_with_examples(question, available_times)
        if "INVALID" in verification or "Error" in verification:
           return "No valid meeting time found."

        return available_times
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"