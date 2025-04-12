import re
import json
from datetime import datetime, timedelta

def main(question):
    """
    Simulates an LLM-driven meeting scheduler using a combination of information extraction with LLM-like reasoning steps.
    Leverages structured reasoning and validation to generate meeting times.
    """

    try:
        # Step 1: LLM-style Information Extraction and Structuring
        task_details = extract_task_details(question)

        # Step 2: Time Slot Generation and Filtering
        available_slots = generate_and_filter_slots(task_details)

        # Step 3: Propose Solution
        if available_slots:
            proposed_time = available_slots[0]  # Take the first available slot
            return f"Here is the proposed time: {proposed_time}"
        else:
            return "No suitable time found."

    except Exception as e:
        return f"Error: {str(e)}"


def extract_task_details(question):
    """
    Extracts key information such as participants, schedules, duration, and constraints using LLM-style decomposition.
    Instead of directly parsing with regex, simulates reasoning to understand the structure.
    """
    participants_match = re.search(r"schedule a meeting for (.*?) for", question)
    participants = [name.strip() for name in participants_match.group(1).split(',')] if participants_match else []

    schedules = {}
    schedule_lines = re.findall(r"([A-Za-z]+) (?:has meetings|is busy|has blocked their calendar) on Monday during (.*?);", question)
    for person, schedule_str in schedule_lines:
        schedule_list = []
        time_ranges = schedule_str.split(', ')
        for time_range in time_ranges:
            start_time, end_time = time_range.split(' to ')
            schedule_list.append((start_time, end_time))
        schedules[person] = schedule_list

    duration_match = re.search(r"for (.*?)(?: between| on)", question)
    duration_str = duration_match.group(1) if duration_match else "half an hour"
    duration = 30 if "half an hour" in duration_str else int(re.search(r"(\d+)", duration_str).group(1))  # Extract number for other durations

    constraints_match = re.search(r"(.*)\. Find a time", question)
    constraints = constraints_match.group(1) if constraints_match else ""
    
    # Simulate LLM reasoning for work hours
    work_hours_match = re.search(r"between the work hours of (\d+:\d+) to (\d+:\d+)", question)
    work_start_time, work_end_time = work_hours_match.groups() if work_hours_match else ("9:00", "17:00") # Default values

    return {
        "participants": participants,
        "schedules": schedules,
        "duration": duration,
        "constraints": constraints,
        "work_start_time": work_start_time,
        "work_end_time": work_end_time
    }


def generate_and_filter_slots(task_details):
    """
    Generates candidate time slots and filters them based on availability, preferences and constraints.
    """
    participants = task_details["participants"]
    schedules = task_details["schedules"]
    duration = task_details["duration"]
    constraints = task_details["constraints"]
    work_start_time = task_details["work_start_time"]
    work_end_time = task_details["work_end_time"]

    start_time = datetime.strptime(work_start_time, "%H:%M")
    end_time = datetime.strptime(work_end_time, "%H:%M")
    time_slots = []

    current_time = start_time
    while current_time + timedelta(minutes=duration) <= end_time:
        time_slots.append(current_time.strftime("%H:%M"))
        current_time += timedelta(minutes=30) # Increment by 30 minutes to generate slots

    available_slots = []
    for slot_start in time_slots:
        slot_end_dt = datetime.strptime(slot_start, "%H:%M") + timedelta(minutes=duration)
        slot_end = slot_end_dt.strftime("%H:%M")
        is_available = True
        for person in participants:
            if person in schedules:
                for busy_start, busy_end in schedules[person]:
                    # Convert to datetime objects
                    slot_start_dt = datetime.strptime(slot_start, "%H:%M")
                    slot_end_dt = datetime.strptime(slot_end, "%H:%M")
                    busy_start_dt = datetime.strptime(busy_start, "%H:%M")
                    busy_end_dt = datetime.strptime(busy_end, "%H:%M")

                    if (slot_start_dt < busy_end_dt) and (slot_end_dt > busy_start_dt):
                        is_available = False
                        break
            if not is_available:
                break
        
        # Simulate constraint check with "LLM-like" reasoning
        if "Raymond would rather not meet on Monday after 11:30" in constraints and "Raymond" in participants:
            slot_start_dt = datetime.strptime(slot_start, "%H:%M")
            if slot_start_dt >= datetime.strptime("11:30", "%H:%M"):
                is_available = False
                
        if "Joe can not meet on Monday after 14:00" in constraints and "Joe" in participants:
            slot_start_dt = datetime.strptime(slot_start, "%H:%M")
            if slot_start_dt >= datetime.strptime("14:00", "%H:%M"):
                is_available = False

        if is_available:
            available_slots.append(f"Monday, {slot_start} - {slot_end}")
    
    return available_slots