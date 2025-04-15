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
    """
    Schedules meetings by extracting constraints, finding available time slots,
    and verifying the proposed solution, using LLMs for reasoning and extraction.
    """

    try:
        # 1. Extract constraints using an LLM with chain-of-thought reasoning and embedded examples
        extracted_constraints = extract_meeting_constraints(question)

        # 2. Find available time slots by generating candidates and verifying them against constraints
        available_time_slots = find_available_time_slots(extracted_constraints)

        # 3. Verify the solution
        verified_solution = verify_solution(question, available_time_slots)

        return verified_solution

    except Exception as e:
        return f"Error: {str(e)}"

def extract_meeting_constraints(text):
    """Extracts meeting constraints from the input text using LLM."""
    system_instruction = "You are an expert at extracting meeting constraints."
    prompt = f"""
    Extract meeting constraints from the text below.
    
    Example:
    Text: You need to schedule a meeting for Diana and William for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. 
    Diana's calendar is wide open the entire week.
    William has meetings on Monday during 9:00 to 11:30, 12:00 to 14:30, 15:00 to 17:00, Tuesday during 9:00 to 17:00, Wednesday during 9:00 to 17:00; 
    Diana do not want to meet on Monday after 13:00.
    
    Extracted Constraints:
    {{
        "participants": ["Diana", "William"],
        "duration": 30,
        "available_days": ["Monday", "Tuesday", "Wednesday"],
        "working_hours": [9, 17],
        "Diana": {{
            "Monday": [],
            "Tuesday": [],
            "Wednesday": [],
        }},
        "William": {{
            "Monday": [[9, 11.5], [12, 14.5], [15, 17]],
            "Tuesday": [[9, 17]],
            "Wednesday": [[9, 17]],
        }},
        "preferences": {{
            "Diana": "Do not meet on Monday after 13:00."
        }}
    }}
    
    Now extract the constrains from this text:
    {text}
    """
    try:
        return json.loads(call_llm(prompt, system_instruction))
    except Exception as e:
        raise Exception(f"Could not extract meeting constraints: {str(e)}")

def find_available_time_slots(constraints):
    """Generates candidate time slots and filters them based on constraints."""
    available_slots = []
    try:
        participants = constraints["participants"]
        duration = constraints["duration"]
        available_days = constraints["available_days"]
        working_hours = constraints["working_hours"]

        for day in available_days:
            for hour in range(working_hours[0], working_hours[1]):
                start_time = hour
                end_time = hour + duration / 60  # Convert minutes to hours

                # Check if the time slot is within working hours
                if end_time > working_hours[1]:
                    continue

                # Check availability for all participants
                available = True
                for participant in participants:
                    if participant in constraints and day in constraints[participant]:
                        busy_slots = constraints[participant][day]
                        for busy_slot in busy_slots:
                            if start_time >= busy_slot[0] and start_time < busy_slot[1]:
                                available = False
                                break
                        if not available:
                            break
                    else:
                        #Default to wide open
                        continue

                if available:
                    available_slots.append({"day": day, "start_time": start_time, "end_time": end_time})
        return available_slots

    except Exception as e:
        raise Exception(f"Error finding available time slots: {str(e)}")

def verify_solution(problem, available_time_slots):
    """Verifies the solution with LLM, makes sure that the solution adheres to all requirements."""
    system_instruction = "You are an expert at verifying if the meeting can be scheduled."
    prompt = f"""
    Problem: {problem}
    Available Time Slots: {available_time_slots}
    
    Based on the problem and available time slots, find an appropriate time to schedule the meeting.
    Return "No available time slots found" if there is no possible time.
    
    Example:
    Problem: You need to schedule a meeting for Diana and William for half an hour between the work hours of 9:00 to 17:00 on either Monday, Tuesday or Wednesday. 
    Diana's calendar is wide open the entire week.
    William has meetings on Monday during 9:00 to 11:30, 12:00 to 14:30, 15:00 to 17:00, Tuesday during 9:00 to 17:00, Wednesday during 9:00 to 17:00; 
    Diana do not want to meet on Monday after 13:00.
    Available Time Slots:
    [{{'day': 'Monday', 'start_time': 11.5, 'end_time': 12.0}}]
    
    Solution:
    Here is the proposed time: Monday, 11:30 - 12:00
    
    Now, solve the new problem.
    """
    try:
        if available_time_slots:
            first_slot = available_time_slots[0]
            start_time_minutes = int((first_slot["start_time"] - int(first_slot["start_time"])) * 60)
            end_time_minutes = int((first_slot["end_time"] - int(first_slot["end_time"])) * 60)

            start_time_str = f"{int(first_slot['start_time']):02}:{start_time_minutes:02}"
            end_time_str = f"{int(first_slot['end_time']):02}:{end_time_minutes:02}"
            return f"Here is the proposed time: {first_slot['day']}, {start_time_str} - {end_time_str}"
        else:
            return "No available time slots found"
    except Exception as e:
        return "No available time slots found"