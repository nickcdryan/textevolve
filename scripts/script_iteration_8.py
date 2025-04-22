import os
import re
import math

def main(question):
    """Schedules meetings using a hybrid approach: LLM for understanding, deterministic code for availability checking.

    HYPOTHESIS: Combining LLM understanding with precise deterministic time calculations will improve accuracy and reliability.
    This approach uses a ReAct-like loop, with the LLM extracting, Python calculating, and the LLM validating/adjusting.
    """
    try:
        # 1. Extract meeting information using LLM (Extraction Agent)
        extracted_info = extract_meeting_info(question)
        if "Error" in extracted_info:
            return extracted_info

        # 2. Calculate available time slots using deterministic Python code
        available_slots = calculate_available_slots(extracted_info)

        # 3. Propose a meeting time using LLM, considering available slots
        proposed_time = propose_meeting_time(available_slots, extracted_info, question)

        # 4. Deterministic verification of the proposed time, adjust if needed
        verified_time = verify_proposed_time(proposed_time, extracted_info)

        return verified_time

    except Exception as e:
        return f"Error processing the request: {str(e)}"

def extract_meeting_info(question):
    """Extracts meeting details (participants, duration, days, schedules) using LLM with examples."""
    system_instruction = "You are an expert at extracting precise meeting details."
    prompt = f"""
    Extract meeting details. Return a Python dictionary.

    Example 1:
    Question: Schedule John and Mary for 30 minutes on Monday. John is busy 9-10, Mary 11-12.
    Extraction: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00"], "Mary": ["11:00-12:00"]}}}}

    Example 2:
    Question: Alice, Bob, and Charlie for 1 hour on Tuesday/Wednesday. Alice busy Tue 14-15, Bob Wed 10-11.
    Extraction: {{"participants": ["Alice", "Bob", "Charlie"], "duration": 60, "days": ["Tuesday", "Wednesday"], "schedules": {{"Alice": ["14:00-15:00 (Tuesday)"], "Bob": ["10:00-11:00 (Wednesday)"], "Charlie": []}}}}

    Question: {question}
    Extraction:
    """
    return call_llm(prompt, system_instruction)

def calculate_available_slots(extracted_info):
    """Calculates available time slots using deterministic Python code."""
    try:
        info = eval(extracted_info)
        participants = info["participants"]
        duration = info["duration"]
        days = info["days"]
        schedules = info["schedules"]

        # Implement deterministic logic to calculate available slots based on the extracted info
        # This is a simplified example. A full implementation would be more complex
        available_slots = []
        for day in days:
            for hour in range(9, 17):  # 9:00 to 16:00
                start_time = f"{hour:02d}:00"
                end_time = f"{(hour + math.ceil(duration/60)):02d}:00"
                is_available = True
                for person in participants:
                    if person in schedules and any(start_time in busy_slot and day in busy_slot for busy_slot in schedules[person]):
                        is_available = False
                        break
                if is_available and hour + math.ceil(duration/60) <= 17: # meeting end time must be <= 17:00
                    available_slots.append(f"{day}, {start_time}-{end_time}")
        return available_slots
    except:
        return "Error: could not parse available slots. Please check schedule format is correct."

def propose_meeting_time(available_slots, extracted_info, question):
    """Proposes a meeting time using LLM, considering available slots."""
    system_instruction = "You are an expert at scheduling meetings, suggesting the BEST time to have the meeting."
    prompt = f"""
    Given the available slots, propose a suitable meeting time in the format 'Here is the proposed time: [Day], [Start Time]-[End Time]'.

    Example:
    Available Slots: ['Monday, 10:00-10:30', 'Monday, 14:00-14:30']
    Meeting Details: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"]}}
    Proposed Time: Here is the proposed time: Monday, 10:00-10:30

    Available Slots: {available_slots}
    Meeting Details: {extracted_info}
    Proposed Time:
    """
    return call_llm(prompt, system_instruction)

def verify_proposed_time(proposed_time, extracted_info):
    """Deterministically verifies if the proposed time works, adjusts if needed."""
    try:
        if "Error" in proposed_time:
            return proposed_time
        time_match = re.search(r"Here is the proposed time: (\w+), (\d{1,2}:\d{2})-(\d{1,2}:\d{2})", proposed_time)
        if not time_match:
            return "Error: Could not parse the meeting time."

        day = time_match.group(1)
        start_time = time_match.group(2)
        end_time = time_match.group(3)
        extracted_info = eval(extracted_info)
        participants = extracted_info["participants"]
        schedules = extracted_info["schedules"]

        # Perform deterministic conflict checking
        for person in participants:
            if person in schedules:
                for busy_slot in schedules[person]:
                    if day in busy_slot:
                        busy_start, busy_end = re.search(r"(\d{1,2}:\d{2})-(\d{1,2}:\d{2})", busy_slot).groups()
                        if start_time < busy_end and end_time > busy_start:
                            return "Error: Proposed time conflicts with existing schedule."
        return f"Here is the proposed time: {day}, {start_time}-{end_time}"

    except Exception as e:
        return f"Error during verification: {str(e)}"

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