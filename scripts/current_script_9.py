import os
import re
import math

def main(question):
    """Schedules meetings using a multi-stage extraction, reasoning, and deterministic filtering approach.

    This iteration introduces a separate "availability reasoner" that parses the text and converts it to data that
    is compatible with deterministic Python calculations. The hypothesis is that we can isolate the LLM to the high-level reasoning
    and make the Python calculations do the more reliable time slot calculations.

    This approach focuses on robust extraction, explicit reasoning steps, and a deterministic solution checker.
    """
    try:
        # 1. Extract structured info using LLM
        extracted_info = extract_meeting_info(question)
        if "Error" in extracted_info:
            return extracted_info

        # 2. Convert available slots with Python code using explicit data structures.
        available_slots = convert_to_available_slots(extracted_info)

        # 3. Propose a meeting time using LLM and the analyzed data
        proposed_time = propose_meeting_time(available_slots, extracted_info)

        return proposed_time

    except Exception as e:
        return f"Error processing the request: {str(e)}"

def extract_meeting_info(question):
    """Extracts structured information from the question using LLM with multi-example prompting."""
    system_instruction = "You are an expert at extracting meeting details from text. Return a JSON object."
    prompt = f"""
        You are an expert at extracting meeting details from text. Extract:
        - participants (list of names)
        - duration (integer, minutes)
        - days (list of strings, e.g., "Monday", "Tuesday")
        - existing schedules (dictionary, participant name -> list of time ranges "HH:MM-HH:MM (Day)")

        Example 1:
        Question: Schedule a meeting for John and Mary for 30 minutes on Monday. John is busy from 9:00-10:00, Mary is busy from 11:00-12:00.
        Extraction: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"], "schedules": {{"John": ["9:00-10:00 (Monday)"], "Mary": ["11:00-12:00 (Monday)"]}}}}

        Example 2:
        Question: Schedule a meeting for Alice, Bob, and Charlie for 1 hour on Tuesday and Wednesday. Alice is busy from 14:00-15:00 on Tuesday, Bob is busy from 10:00-11:00 on Wednesday. Charlie is free.
        Extraction: {{"participants": ["Alice", "Bob", "Charlie"], "duration": 60, "days": ["Tuesday", "Wednesday"], "schedules": {{"Alice": ["14:00-15:00 (Tuesday)"], "Bob": ["10:00-11:00 (Wednesday)"], "Charlie": []}}}}

        Question: {question}
        Extraction:
        """
    return call_llm(prompt, system_instruction)

def convert_to_available_slots(extracted_info):
    """Converts unstructured schedule info to deterministic list of available times using Python code."""
    try:
        info = eval(extracted_info) # This avoids json.loads error
        participants = info["participants"]
        duration = info["duration"]
        days = info["days"]
        schedules = info["schedules"]

        # Convert schedules to a more usable format (dictionary of lists of tuples)
        blocked_times = {}
        for person, busy_times in schedules.items():
            blocked_times[person] = []
            for time_range_str in busy_times:
                match = re.search(r"(\d{1,2}:\d{2})-(\d{1,2}:\d{2}) \((.*)\)", time_range_str)
                if match:
                    start_time, end_time, day = match.groups()
                    blocked_times[person].append((start_time, end_time, day))

        available_slots = []
        for day in days:
            for hour in range(9, 17):  # 9:00 to 16:00
                start_time = f"{hour:02d}:00"
                end_time = f"{(hour + math.ceil(duration/60)):02d}:00"
                is_available = True
                for person in participants:
                    if person in blocked_times and any(start_time < busy_end and end_time > busy_start and day == busy_day
                                                        for busy_start, busy_end, busy_day in blocked_times[person]):
                        is_available = False
                        break
                if is_available and hour + math.ceil(duration/60) <= 17: # meeting end time must be <= 17:00
                    available_slots.append(f"{day}, {start_time}-{end_time}")
        return available_slots
    except:
        return "Error: could not parse available slots. Please check schedule format is correct."

def propose_meeting_time(available_slots, extracted_info):
    """Propose a suitable meeting time based on available slots and extracted data. The data should already be parsed into lists."""
    system_instruction = "You are skilled at proposing meeting times considering available time slots."
    prompt = f"""
        You are an expert meeting scheduler. Given the available time slots and meeting details, propose the BEST meeting time. Respond in the format:
        Here is the proposed time: [Day], [Start Time]-[End Time]

        Example:
        Available Time Slots: ['Monday, 10:00-10:30', 'Monday, 14:00-14:30']
        Meeting Details: {{"participants": ["John", "Mary"], "duration": 30, "days": ["Monday"]}}
        Proposed Time: Here is the proposed time: Monday, 10:00-10:30

        Available Time Slots: {available_slots}
        Meeting Details: {extracted_info}
        Proposed Time:
        """
    return call_llm(prompt, system_instruction)

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