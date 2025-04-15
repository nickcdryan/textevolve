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

def extract_meeting_constraints(text):
    """Extract meeting constraints using LLM with embedded examples."""
    system_instruction = "You are an expert at extracting meeting constraints."
    prompt = f"""
    Extract meeting constraints from the text, including participants, duration, days, and schedules.
    
    Example:
    Input: You need to schedule a meeting for Brenda and Bruce for half an hour between 9:00 to 17:00 on Monday, Tuesday or Wednesday. Brenda is busy on Monday 9:30-10:00, Tuesday 9:00-9:30. Bruce is busy on Monday 10:00-10:30, Tuesday 9:00-17:00.
    Let's think step by step.
    Participants: Brenda, Bruce
    Duration: Half an hour
    Days: Monday, Tuesday, Wednesday
    Brenda's schedule: Monday 9:30-10:00, Tuesday 9:00-9:30
    Bruce's schedule: Monday 10:00-10:30, Tuesday 9:00-17:00
    Output:
    {{
      "participants": ["Brenda", "Bruce"],
      "duration": "0:30",
      "days": ["Monday", "Tuesday", "Wednesday"],
      "schedules": {{
        "Brenda": {{"Monday": ["9:30-10:00"], "Tuesday": ["9:00-9:30"]}},
        "Bruce": {{"Monday": ["10:00-10:30"], "Tuesday": ["9:00-17:00"]}}
      }}
    }}
    
    Now extract from this text:
    {text}
    """
    return call_llm(prompt, system_instruction)

def find_available_time_slots(constraints_json):
    """Find available time slots using LLM with embedded examples."""
    system_instruction = "You are an expert at finding available time slots."
    prompt = f"""
    Given the meeting constraints, find the available time slots.
    
    Example:
    Input:
    {{
      "participants": ["Brenda", "Bruce"],
      "duration": "0:30",
      "days": ["Monday", "Tuesday"],
      "schedules": {{
        "Brenda": {{"Monday": ["9:30-10:00"], "Tuesday": ["9:00-9:30"]}},
        "Bruce": {{"Monday": ["10:00-10:30"], "Tuesday": ["9:00-17:00"]}}
      }}
    }}
    Let's think step by step.
    Available time slots:
    Monday: 9:00-9:30
    Output:
    {{
        "Monday": ["9:00-9:30"]
    }}
    
    Now find available time slots from these constraints:
    {constraints_json}
    """
    return call_llm(prompt, system_instruction)

def verify_solution(question, proposed_solution):
    """Verify if the proposed solution is valid using LLM with embedded examples."""
    system_instruction = "You are an expert at verifying meeting schedules."
    prompt = f"""
    Verify if the proposed solution is valid given the question.
    
    Example:
    Question: You need to schedule a meeting for Brenda and Bruce for half an hour on Monday or Tuesday. Brenda is busy on Monday 9:30-10:00. Bruce is busy on Monday 10:00-10:30, Tuesday 9:00-17:00.
    Proposed Solution: Monday, 9:00-9:30
    Let's think step by step.
    Brenda is available on Monday 9:00-9:30. Bruce is available on Monday 9:00-9:30. The duration is half an hour.
    Output: VALID
    
    Now verify if this solution is valid:
    Question: {question}
    Proposed Solution: {proposed_solution}
    """
    return call_llm(prompt, system_instruction)

def main(question):
    """Main function to schedule a meeting."""
    try:
        # 1. Extract meeting constraints
        constraints_json = extract_meeting_constraints(question)
        
        # 2. Find available time slots
        available_time_slots_json = find_available_time_slots(constraints_json)

        # 3. Return the first available time slot
        available_time_slots = json.loads(available_time_slots_json)
        for day, slots in available_time_slots.items():
            if slots:
                first_slot = slots[0]
                proposed_solution = f"{day}, {first_slot}"
                
                # 4. Verify the solution
                verification_result = verify_solution(question, proposed_solution)
                if "VALID" in verification_result:
                    return f"Here is the proposed time: {proposed_solution}"
                else:
                    return "No valid time found."
        
        return "No valid time found."
    except Exception as e:
        return f"Error: {str(e)}"