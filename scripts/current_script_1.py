import json
import os
import re
import math

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

def extract_meeting_info(question, max_attempts=3):
    """Extract meeting information from the question using LLM with robust JSON parsing and retry."""
    system_instruction = "You are an expert at extracting meeting scheduling information."
    prompt = f"""
    Extract the following information from the question: participants, duration, work hours, days, and existing schedules. Return a JSON object.

    Example:
    Question: You need to schedule a meeting for John and Jennifer for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. John has no meetings the whole week. Jennifer has meetings on Monday during 9:00 to 11:00.
    Extracted Info:
    {{
      "participants": ["John", "Jennifer"],
      "duration": "half an hour",
      "work_hours": "9:00 to 17:00",
      "days": ["Monday", "Tuesday", "Wednesday"],
      "John": "no meetings the whole week",
      "Jennifer": "Monday during 9:00 to 11:00"
    }}

    Question: {question}
    """

    for attempt in range(max_attempts):
        response = call_llm(prompt, system_instruction)
        try:
            # Attempt to parse the JSON response
            meeting_info = json.loads(response)
            return meeting_info  # Return if parsing is successful
        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1} failed: JSONDecodeError: {e}")
            if attempt == max_attempts - 1:
                return None  # Return None after max attempts

def filter_slots_by_constraints(time_slots, constraints, max_attempts=3):
    """Filter available time slots based on constraints using LLM."""
    system_instruction = "You are an expert at filtering time slots based on scheduling constraints."
    prompt = f"""
    Given the following time slots and constraints, filter out the slots that do not meet the constraints.

    Example:
    Time Slots: ["Monday, 13:00 - 13:30", "Tuesday, 10:00 - 10:30"]
    Constraints: "John is not available on Monday after 14:00"
    Filtered Slots: ["Tuesday, 10:00 - 10:30"]

    Time Slots: {time_slots}
    Constraints: {constraints}
    """
    for attempt in range(max_attempts):
        response = call_llm(prompt, system_instruction)
        try:
            # Split the time slots in the response
            filtered_slots = [slot.strip() for slot in response.split(",")]
            return filtered_slots
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {str(e)}")
            if attempt == max_attempts - 1:
                return time_slots

def main(question):
    """Main function to schedule a meeting."""
    # Step 1: Extract meeting information
    structured_info = extract_meeting_info(question)
    if not structured_info:
        return "Error: Could not extract meeting information."

    # Step 2: Generate dummy time slots (replace with actual logic)
    time_slots = ["Monday, 9:00 - 9:30", "Monday, 13:00 - 13:30", "Tuesday, 10:00 - 10:30"]

    # Step 3: Extract constraints (combine schedule and preferences)
    constraints = ""
    for key, value in structured_info.items():
        if key not in ["participants", "duration", "work_hours", "days"]:
            constraints += f"{key} is {value}. "

    # Step 4: Filter time slots by constraints
    filtered_slots = filter_slots_by_constraints(time_slots, constraints)

    # Step 5: Select the best time slot (replace with selection logic)
    if filtered_slots:
        return f"Here is the proposed time: {filtered_slots[0]}"
    else:
        return "No suitable time slots found."