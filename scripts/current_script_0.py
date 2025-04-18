import os
import json
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

def main(question):
    """Main function to schedule meetings with LLM reasoning."""
    try:
        # 1. Extract information about participants, schedules, and constraints
        extracted_info = extract_meeting_info(question)
        if "Error" in extracted_info:
            return "Error extracting meeting information."

        # 2. Generate possible meeting slots
        possible_slots = generate_meeting_slots(extracted_info)
        if "Error" in possible_slots:
            return "Error generating possible meeting slots."

        # 3. Filter slots based on constraints and preferences using LLM for reasoning
        filtered_slots = filter_meeting_slots(extracted_info, possible_slots)
        if "Error" in filtered_slots:
            return "Error filtering meeting slots."

        # 4. Select the best slot based on preferences
        best_slot = select_best_meeting_slot(extracted_info, filtered_slots)
        if "Error" in best_slot:
            return "Error selecting the best meeting slot."

        return best_slot
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def extract_meeting_info(question):
    """Extract meeting information using LLM with an embedded example."""
    system_instruction = "You are an expert at extracting information from meeting scheduling requests."
    prompt = f"""
    Extract information from the following meeting scheduling request, including participants, schedules, constraints, and preferences.

    Example:
    Input: You need to schedule a meeting for Joyce, Christine and Alexander for half an hour between the work hours of 9:00 to 17:00 on Monday. Joyce has meetings on Monday during 11:00 to 11:30, 13:30 to 14:00, 14:30 to 16:30; Christinehas no meetings the whole day. Alexander has meetings on Monday during 9:00 to 11:00, 12:00 to 12:30, 13:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00; Christine can not meet on Monday before 12:00.
    Output: {{"participants": ["Joyce", "Christine", "Alexander"], "duration": "half an hour", "day": "Monday", "start_time": "9:00", "end_time": "17:00", "Joyce": ["11:00-11:30", "13:30-14:00", "14:30-16:30"], "Christine": [], "Alexander": ["9:00-11:00", "12:00-12:30", "13:30-15:00", "15:30-16:00", "16:30-17:00"], "constraints": ["Christine can not meet before 12:00"]}}

    Input: {question}
    Output:
    """
    return call_llm(prompt, system_instruction)

def generate_meeting_slots(extracted_info):
    """Generate a list of possible meeting slots based on the extracted information."""
    try:
        info = json.loads(extracted_info)
        day = info.get("day", "Monday")
        start_time = 9
        end_time = 17
        duration = 30  # minutes

        slots = []
        current_time = start_time * 60  # Convert to minutes

        while current_time + duration <= end_time * 60:
            hour = current_time // 60
            minute = current_time % 60
            start_str = f"{hour:02}:{minute:02}"
            end_hour = (current_time + duration) // 60
            end_minute = (current_time + duration) % 60
            end_str = f"{end_hour:02}:{end_minute:02}"
            slots.append(f"{day}, {start_str} - {end_str}")
            current_time += 30  # Increment by 30 minutes

        return json.dumps(slots)
    except Exception as e:
        return f"Error generating slots: {str(e)}"

def filter_meeting_slots(extracted_info, possible_slots):
    """Filter the possible meeting slots based on constraints using LLM reasoning."""
    system_instruction = "You are an expert meeting scheduler who filters possible slots based on availability and constraints."
    prompt = f"""
    Given the extracted information and possible meeting slots, filter the slots to find times that work for everyone, considering their schedules and constraints.

    Example:
    Extracted Info: {{"participants": ["Joyce", "Christine", "Alexander"], "duration": "half an hour", "day": "Monday", "start_time": "9:00", "end_time": "17:00", "Joyce": ["11:00-11:30", "13:30-14:00", "14:30-16:30"], "Christine": [], "Alexander": ["9:00-11:00", "12:00-12:30", "13:30-15:00", "15:30-16:00", "16:30-17:00"], "constraints": ["Christine can not meet before 12:00"]}}
    Possible Slots: ["Monday, 09:00 - 09:30", "Monday, 09:30 - 10:00", "Monday, 10:00 - 10:30", "Monday, 10:30 - 11:00", "Monday, 11:00 - 11:30", "Monday, 11:30 - 12:00", "Monday, 12:00 - 12:30", "Monday, 12:30 - 13:00", "Monday, 13:00 - 13:30", "Monday, 13:30 - 14:00", "Monday, 14:00 - 14:30", "Monday, 14:30 - 15:00", "Monday, 15:00 - 15:30", "Monday, 15:30 - 16:00", "Monday, 16:00 - 16:30"]
    Filtered Slots: ["Monday, 12:30 - 13:00"]

    Extracted Info: {extracted_info}
    Possible Slots: {possible_slots}
    Filtered Slots:
    """
    return call_llm(prompt, system_instruction)

def select_best_meeting_slot(extracted_info, filtered_slots):
    """Select the best meeting slot based on preferences."""
    system_instruction = "You are an expert at selecting the best meeting slot from available options, considering preferences."
    prompt = f"""
    Given the extracted information and filtered meeting slots, select the best slot based on stated preferences. If no preferences are stated, return the first available slot.

    Example:
    Extracted Info: {{"participants": ["David", "Ethan", "Bradley", "Natalie"], "duration": "half an hour", "day": "Monday", "start_time": "9:00", "end_time": "17:00", "David": ["14:00-14:30", "16:30-17:00"], "Ethan": ["13:00-13:30", "14:30-15:00"], "Bradley": ["9:30-10:30", "11:00-12:00", "13:30-14:00", "15:30-17:00"], "Natalie": ["9:30-10:00", "10:30-12:00", "12:30-15:30", "16:00-17:00"], "constraints": ["Natalie would rather not meet after 10:30"]}}
    Filtered Slots: ["Monday, 09:00 - 09:30"]
    Best Slot: Here is the proposed time: Monday, 9:00 - 9:30

    Extracted Info: {extracted_info}
    Filtered Slots: {filtered_slots}
    Best Slot:
    """
    return call_llm(prompt, system_instruction)