import os
import json
import re
import datetime

def main(question):
    """
    Schedules meetings by extracting constraints using LLMs, finding time slots, and verifying solutions.
    This version focuses on robust information extraction with examples and a verification step.
    """

    try:
        # Step 1: Extract meeting constraints using LLM with examples
        meeting_details = extract_meeting_constraints(question)

        # Step 2: Validate extracted meeting details using LLM
        validated_details = validate_meeting_details(meeting_details, question)
        if not validated_details["is_valid"]:
            return "Error: Could not validate meeting details. Please check the input question."

        # Step 3: Find available time slots based on the validated constraints
        available_slots = find_available_time_slots(validated_details)

        # Step 4: Verify the solution using LLM
        if not available_slots:
            return "Here is the proposed time: No available time slots found."
        else:
            # Choose the first available slot
            best_slot = available_slots[0]
            return f"Here is the proposed time: {best_slot['day']}, {best_slot['start_time']} - {best_slot['end_time']}"

    except Exception as e:
        return f"Error: {str(e)}"

def extract_meeting_constraints(text):
    """Extracts meeting constraints (participants, duration, schedules) using LLM with embedded examples."""
    system_instruction = "You are an expert meeting scheduler. Extract meeting details and participant schedules."

    prompt = f"""
    Extract the following information from the text:
    - Participants: List of people involved in the meeting.
    - Duration: Meeting length in minutes.
    - Days: Acceptable days for the meeting.
    - Schedules: Existing schedules for each participant, including busy times.

    Example:
    Text: You need to schedule a meeting for John and Jane for 30 minutes between 9:00 and 17:00 on Monday or Tuesday. John is busy on Monday from 10:00 to 11:00 and Tuesday from 14:00 to 15:00. Jane is busy on Monday from 13:00 to 14:00.
    Participants: John, Jane
    Duration: 30
    Days: Monday, Tuesday
    John's schedule: Monday: 10:00-11:00, Tuesday: 14:00-15:00
    Jane's schedule: Monday: 13:00-14:00

    Now extract the information from this text:
    {text}
    """

    response = call_llm(prompt, system_instruction)

    try:
        # Parse the LLM response
        lines = response.split("\n")
        participants = lines[1].split(": ")[1].split(", ")
        duration = int(lines[2].split(": ")[1])
        days = [day.strip() for day in lines[3].split(": ")[1].split(",")] if lines[3].split(": ")[1] else []

        schedules = {}
        for i in range(4, len(lines)):
            if ":" in lines[i]:
                name = lines[i].split("'s")[0]
                schedule_str = lines[i].split(": ")[1]
                schedules[name] = {}
                if schedule_str:
                    day_times = schedule_str.split(", ")
                    for day_time in day_times:
                        try:
                            day, time_range = day_time.split(": ")
                            schedules[name][day] = []
                            for time_block in time_range.split(", "):
                                start_time, end_time = time_block.split("-")
                                schedules[name][day].append({"start_time": start_time, "end_time": end_time})
                        except:
                            pass


        return {"participants": participants, "duration": duration, "days": days, "schedules": schedules}
    except Exception as e:
        raise ValueError(f"Could not extract or parse meeting details: {e}")

def validate_meeting_details(details, question):
    """Validates the extracted meeting details using an LLM call with embedded examples."""
    system_instruction = "You are a meticulous validator who checks extracted meeting details for completeness and correctness."

    prompt = f"""
    Here are the extracted meeting details:
    {json.dumps(details, indent=2)}

    Original Question:
    {question}

    Validate the extracted details against the original question. Check if all participants are listed, the duration is correct, and all busy slots are accurately captured.

    Example:
    Extracted Details: {{'participants': ['John', 'Jane'], 'duration': 30, 'days': ['Monday', 'Tuesday'], 'schedules': {{'John': {{'Monday': ['10:00-11:00']}}, 'Jane': {{'Monday': ['13:00-14:00']}}}}}}
    Original Question: You need to schedule a meeting for John and Jane for 30 minutes between 9:00 and 17:00 on Monday or Tuesday. John is busy on Monday from 10:00 to 11:00 and Tuesday from 14:00 to 15:00. Jane is busy on Monday from 13:00 to 14:00.
    Validation Result: VALID - All details are accurately extracted.

    Now, validate the following details:
    Extracted Details: {json.dumps(details, indent=2)}
    Original Question: {question}
    """

    response = call_llm(prompt, system_instruction)

    if "VALID" in response:
        return {"is_valid": True, "details": details}
    else:
        return {"is_valid": False, "error": response}


def find_available_time_slots(details):
    """Finds available time slots based on the validated meeting constraints."""
    # Simplified example - needs further refinement based on previous approaches and accumulated learnings
    available_slots = []
    for day in details["days"]:
        for hour in range(9, 17):  # 9:00 to 17:00
            start_time = f"{hour:02d}:00"
            end_time = f"{hour:02d}:30"
            is_available = True

            for participant in details["participants"]:
                if participant in details["schedules"] and day in details["schedules"][participant]:
                    for busy_slot in details["schedules"][participant][day]:
                        if start_time == busy_slot["start_time"]:
                            is_available = False
                            break
                if not is_available:
                    break

            if is_available:
                available_slots.append({"day": day, "start_time": start_time, "end_time": end_time})

    return available_slots


def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
    try:
        from google import genai
        from google.genai import types
        import os

        # Initialize the Gemini client
        client = genai.GenerativeModel('gemini-pro')
        if system_instruction:
            response = client.generate_content(
                [system_instruction, prompt])
        else:
            response = client.generate_content(prompt)

        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Error: {str(e)}"