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
    """Main function to schedule meetings given constraints."""

    # 1. Extract structured information from the question
    structured_info = extract_meeting_info(question)

    # 2. Find available time slots given extracted info
    available_slots = find_available_time_slots(structured_info)

    # 3. Filter available slots based on constraints
    filtered_slots = filter_slots_by_constraints(available_slots, structured_info)

    # 4. Select best slot if multiple exist, otherwise, return the only slot
    best_slot = select_best_time_slot(filtered_slots, structured_info)

    return best_slot

def extract_meeting_info(question):
    """Extract structured information from the question using LLM."""
    system_instruction = "You are an expert meeting scheduler who extracts key pieces of information."
    prompt = f"""
    Extract the meeting participants, duration, working hours, days, existing schedules, and preferences from the following text.
    Provide the output as JSON.

    Example:
    Input: You need to schedule a meeting for Joyce, Christine and Alexander for half an hour between the work hours of 9:00 to 17:00 on Monday. Joyce has meetings on Monday during 11:00 to 11:30, 13:30 to 14:00, 14:30 to 16:30; Christine has no meetings the whole day. Alexander has meetings on Monday during 9:00 to 11:00, 12:00 to 12:30, 13:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00; Christine can not meet on Monday before 12:00.
    Output:
    {{
        "participants": ["Joyce", "Christine", "Alexander"],
        "duration": 30,
        "working_hours": [900, 1700],
        "days": ["Monday"],
        "schedules": {{
            "Joyce": [["Monday", 1100, 1130], ["Monday", 1330, 1400], ["Monday", 1430, 1630]],
            "Christine": [],
            "Alexander": [["Monday", 900, 1100], ["Monday", 1200, 1230], ["Monday", 1330, 1500], ["Monday", 1530, 1600], ["Monday", 1630, 1700]]
        }},
        "constraints": {{
            "Christine": [["Monday", "before", 1200]]
        }}
    }}

    Now extract the information from:
    {question}
    """
    try:
        structured_info_str = call_llm(prompt, system_instruction)
        structured_info = json.loads(structured_info_str)
        return structured_info
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"Error extracting information: {e}")
        return None

def find_available_time_slots(structured_info):
    """Find available time slots given the structured information"""
    # This is a placeholder. Implement this function's logic.
    return "Placeholder - Implement find_available_time_slots()"

def filter_slots_by_constraints(available_slots, structured_info):
    """Filter the available time slots based on the constraints using LLM."""
    system_instruction = "You are an expert at filtering available time slots based on constraints."
    prompt = f"""
    Given available time slots and constraints, filter out the invalid time slots.

    Example:
    Available Slots: [["Monday", 1230, 1300], ["Monday", 1300, 1330]]
    Constraints: Christine can not meet on Monday before 1300.
    Filtered Slots: [["Monday", 1300, 1330]]

    Available Slots: {available_slots}
    Constraints: {structured_info.get("constraints", "")}
    Filtered Slots:
    """
    try:
        filtered_slots_str = call_llm(prompt, system_instruction)
        # In a real implementation, you would parse filtered_slots_str into a list
        return filtered_slots_str # Returning the string for this example.
    except Exception as e:
        print(f"Error filtering slots: {e}")
        return None

def select_best_time_slot(filtered_slots, structured_info):
    """Select the best time slot among the filtered slots."""
    # This is a placeholder. Implement this function's logic.
    return "Placeholder - Implement select_best_time_slot()"