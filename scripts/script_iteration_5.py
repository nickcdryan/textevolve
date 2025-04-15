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
    """Extracts meeting constraints using an LLM-driven approach with examples."""
    system_instruction = "You are an expert at extracting meeting constraints from text."
    prompt = f"""
    Extract the meeting constraints from the following text.

    Example:
    Input: You need to schedule a meeting for Ronald and Ann for one hour between the work hours of 9:00 to 17:00 on Monday. Ronald has meetings on Monday during 9:30 to 10:30; Ann is busy on Monday during 9:30 to 10:00.
    
    Let's think step by step:
    1. Identify the participants: Ronald and Ann
    2. Meeting duration: one hour
    3. Working hours: 9:00 to 17:00
    4. Day: Monday
    5. Ronald's busy times: 9:30 to 10:30
    6. Ann's busy times: 9:30 to 10:00
    
    Output:
    {{
      "participants": ["Ronald", "Ann"],
      "duration": "60",
      "work_hours": "9:00 to 17:00",
      "day": "Monday",
      "Ronald": ["9:30 to 10:30"],
      "Ann": ["9:30 to 10:00"]
    }}
    
    Now extract information from this new text:
    {text}
    """
    try:
        response = call_llm(prompt, system_instruction)
        return json.loads(response)
    except Exception as e:
        print(f"Error extracting constraints: {str(e)}")
        return {}
        
def verify_constraints(constraints):
    """Verify the extracted constraints for correctness."""
    system_instruction = "You are a constraint verifier that checks to ensure the details extracted from a prompt are correct."
    prompt = f"""
    Verify the meeting constraints for correctness. Ensure that the participant names, duration, working hours, days, and schedules are correct. Provide feedback for any errors, following this example format.
    
    Example:
    Input: 
    {{
      "participants": ["Ronald", "Ann"],
      "duration": "60",
      "work_hours": "9:00 to 17:00",
      "day": "Monday",
      "Ronald": ["9:30 to 10:30"],
      "Ann": ["9:30 to 10:00"]
    }}
    
    Let's think step by step:
    1. Check if all the participant names are correct
    2. Check if the meeting duration is correct
    3. Check if the working hours are correct
    4. Check if the day is correct
    5. Verify each participant's schedule is correctly captured
    
    Output:
    {{
      "verification_status": "VALID",
      "feedback": "No errors found."
    }}
    
    Now verify this new input:
    {constraints}
    """
    try:
        response = call_llm(prompt, system_instruction)
        return json.loads(response)
    except Exception as e:
        print(f"Error verifying constraints: {str(e)}")
        return {"verification_status": "ERROR", "feedback": str(e)}

def find_available_time_slot(constraints):
    """Finds an available time slot given the meeting constraints."""
    system_instruction = "You are an expert meeting scheduler that finds the best time slot for a meeting."
    prompt = f"""
    Find an available time slot that meets all the provided constraints, following the below format.
    
    Example:
    Input:
    {{
      "participants": ["Ronald", "Ann"],
      "duration": "60",
      "work_hours": "9:00 to 17:00",
      "day": "Monday",
      "Ronald": ["9:30 to 10:30"],
      "Ann": ["9:30 to 10:00"]
    }}
    
    Let's think step by step:
    1. Determine the working hours: 9:00 to 17:00 on Monday
    2. Consider the meeting duration: 60 minutes
    3. Identify Ronald's busy times: 9:30 to 10:30
    4. Identify Ann's busy times: 9:30 to 10:00
    5. Find a time slot that works for both Ronald and Ann
    
    Output:
    {{
      "proposed_time": "Monday, 10:30 - 11:30"
    }}
    
    Now provide a time slot for this new input:
    {constraints}
    """
    try:
        response = call_llm(prompt, system_instruction)
        return json.loads(response)
    except Exception as e:
        print(f"Error finding available time slot: {str(e)}")
        return {}

def main(question):
    """Main function to schedule a meeting given a question."""
    # Extract meeting constraints
    constraints = extract_meeting_constraints(question)
    
    # Verify extracted constraints
    verification = verify_constraints(constraints)
    
    if verification["verification_status"] != "VALID":
        return "Error: " + verification["feedback"]

    # Find an available time slot
    time_slot = find_available_time_slot(constraints)
    if "proposed_time" in time_slot:
        return "Here is the proposed time: " + time_slot["proposed_time"]
    else:
        return "No suitable time slot found."