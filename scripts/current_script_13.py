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

def extract_meeting_constraints(text):
    """Extract meeting constraints from the input text using LLM with examples."""
    system_instruction = "You are an expert meeting scheduler who extracts key constraints."
    prompt = f"""
    Extract the key constraints from the text below, including participants, duration, days, and schedules.
    
    Example:
    Text: You need to schedule a meeting for John and Jane for half an hour between the work hours of 9:00 to 17:00 on Monday. John is busy 9-10, Jane is busy 10-11.
    
    Reasoning: 
    1. Participants: John, Jane
    2. Duration: half an hour
    3. Days: Monday
    4. John's schedule: Busy 9:00-10:00
    5. Jane's schedule: Busy 10:00-11:00
    
    Output:
    {{
      "participants": ["John", "Jane"],
      "duration": "0:30",
      "days": ["Monday"],
      "schedules": {{
        "John": {{"Monday": ["9:00-10:00"]}},
        "Jane": {{"Monday": ["10:00-11:00"]}}
      }}
    }}
    
    Now extract constraints from this text:
    {text}
    """
    try:
        response = call_llm(prompt, system_instruction)
        constraints = json.loads(response)
        return constraints
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}, Response: {response}")
        return None
    except Exception as e:
        print(f"Error extracting meeting constraints: {str(e)}")
        return None

def verify_constraints(constraints, original_text):
    """Verify the extracted constraints for consistency and completeness using LLM."""
    system_instruction = "You are a meticulous verifier who checks extracted information."
    prompt = f"""
    Here are the extracted meeting constraints:
    {json.dumps(constraints, indent=2)}
    
    Original Text:
    {original_text}
    
    Example of verification:
    Constraints: {{"participants": ["A", "B"], "duration": "1:00", "days": ["Monday"], "schedules": {{"A": {{"Monday": ["9:00-10:00"]}}, "B": {{"Monday": ["10:00-11:00"]}}}}}}
    Reasoning:
    1. Check if all participants are present in schedules.
    2. Check if duration is valid.
    3. Check if days are valid.
    Result: Valid

    Are the constraints consistent with the original text and complete? Respond with "Valid" or "Invalid".
    """
    try:
        response = call_llm(prompt, system_instruction)
        if "Valid" in response:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error verifying constraints: {str(e)}")
        return False

def find_available_time(constraints):
    """Find an available time slot based on the extracted constraints using LLM."""
    system_instruction = "You are a meeting scheduler who finds available time slots."
    prompt = f"""
    Find an available time slot based on these constraints:
    {json.dumps(constraints, indent=2)}
    
    Example:
    Constraints: {{"participants": ["A", "B"], "duration": "0:30", "days": ["Monday"], "schedules": {{"A": {{"Monday": ["9:00-10:00"]}}, "B": {{"Monday": ["10:00-11:00"]}}}}}}
    Reasoning:
    1. Check availability on Monday between 9:00-17:00
    2. A is busy 9:00-10:00, B is busy 10:00-11:00
    3. A and B are both available after 11:00. Propose the earliest available time, 11:00-11:30.
    Solution: Monday, 11:00-11:30

    Provide the earliest available time slot in the format "Day, Start Time-End Time".
    """
    try:
        response = call_llm(prompt, system_instruction)
        return response
    except Exception as e:
        print(f"Error finding available time: {str(e)}")
        return None

def main(question):
    """Main function to schedule a meeting."""
    constraints = extract_meeting_constraints(question)
    if constraints is None:
        return "Error: Could not extract meeting details."

    if not verify_constraints(constraints, question):
        return "Error: Extracted constraints are invalid."

    available_time = find_available_time(constraints)
    if available_time is None:
        return "Error: No available time slots found."

    return f"Here is the proposed time: {available_time}"