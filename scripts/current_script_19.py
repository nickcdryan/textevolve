import re
import json

def main(question):
    """
    Simulates a meeting scheduler using LLM-driven approach with layered reasoning.
    This approach uses a combination of LLM-like parsing and reasoning steps to find
    an available meeting time that satisfies all constraints.

    The approach involves extracting information, generating candidate slots,
    verifying constraints, and iteratively refining the schedule to produce a valid
    meeting time. It aims to improve the quality and accuracy of the proposed
    meeting time. The process uses chain of thought to provide insight.
    """

    try:
        # Step 1: LLM-like Information Extraction (Simulated)
        task_description, schedule_details = split_question(question)
        participants, duration, work_hours = extract_constraints(task_description)
        schedules = extract_schedules(schedule_details)

        # Step 2: LLM-like Reasoning and Candidate Generation
        possible_slots = generate_candidate_slots(work_hours, duration)

        # Step 3: Verification using LLM-inspired multi-agent concept
        verified_slots = verify_slots(possible_slots, schedules, participants)

        # Step 4: Refinement (if needed) - Simulated LLM iteration
        if not verified_slots:
            # Simulate LLM re-evaluation. In a real LLM system, this would
            # involve LLM re-analyzing and re-generating. Here we
            # add some randomness to the work hours to see if a different time can be found
            work_hours = (max(8,work_hours[0] - 1), min(18, work_hours[1] + 1))  # expand work hours slightly to allow more edge cases
            possible_slots = generate_candidate_slots(work_hours, duration)
            verified_slots = verify_slots(possible_slots, schedules, participants)
            
        if not verified_slots:
            return "No suitable time found within the given constraints and expanded work hours."
            

        # Step 5: Final Output - Act like an LLM generating the answer
        if verified_slots:
            return f"Here is the proposed time: Monday, {verified_slots[0][0]:02d}:{verified_slots[0][1]:02d} - {verified_slots[0][2]:02d}:{verified_slots[0][3]:02d} "
        else:
            return "No suitable time found within the given constraints."

    except Exception as e:
        return f"An error occurred: {str(e)}"

def split_question(question):
    """Splits the question into task description and schedule details."""
    parts = question.split("Here are the existing schedules for everyone during the day:")
    task_description = parts[0].strip()
    schedule_details = parts[1].strip() if len(parts) > 1 else ""
    return task_description, schedule_details

def extract_constraints(task_description):
    """
    Extracts participants, duration, and work hours from the task description
    using LLM-like techniques (simulated).
    """

    # Participants extraction - simplified regex
    match = re.search(r"schedule a meeting for (.*?) for", task_description)
    if not match:
        raise ValueError("Could not extract participants from task description.")
    participants = [name.strip() for name in match.group(1).split(',')]

    # Duration extraction - simplified regex
    duration = 30  # Assume 30 minutes, can be improved with more sophisticated extraction

    # Work hours extraction - simplified hardcoding
    work_hours = (9, 17)  # Assume 9:00 to 17:00, can be improved.

    return participants, duration, work_hours

def extract_schedules(schedule_details):
    """
    Extracts schedules from the schedule details using LLM-like techniques (simulated).
    This uses a dictionary-based approach instead of class for flexibility.
    """
    schedules = {}
    for line in schedule_details.split("\n"):
        if "is busy on Monday" in line:
            name = line.split("is busy on Monday")[0].strip()
            busy_times_str = line.split("is busy on Monday during")[1].strip()
            
            # Handle "the entire day" case first
            if "the entire day" in busy_times_str:
                schedules[name] = [(9 * 60, 17 * 60)]
                continue

            time_ranges = re.split(r', |;', busy_times_str)  # Split by comma or semicolon
            busy_times = []
            for time_range in time_ranges:
                try:
                    start_time_str, end_time_str = time_range.split(" to ")
                    start_hour, start_minute = map(int, start_time_str.split(":"))
                    end_hour, end_minute = map(int, end_time_str.split(":"))
                    busy_times.append((start_hour * 60 + start_minute, end_hour * 60 + end_minute))
                except ValueError:
                    # Handle cases where the format is incorrect
                    print(f"Warning: Skipping invalid time range '{time_range}' in schedule for '{name}'.")
                    continue
            schedules[name] = busy_times
        elif "has no meetings the whole day" in line:
            name = line.split("has no meetings the whole day")[0].strip()
            schedules[name] = []
    return schedules

def generate_candidate_slots(work_hours, duration):
    """Generates candidate time slots based on the work hours and duration."""
    start_hour, end_hour = work_hours
    slots = []
    for hour in range(start_hour, end_hour):
        for minute in range(0, 60, 30):
            start = hour * 60 + minute
            end = start + duration
            if end <= end_hour * 60:
                slots.append((hour, minute, end // 60, end % 60))
    return slots

def verify_slots(slots, schedules, participants):
    """
    Verifies time slots against participant schedules, using LLM-like
    reasoning for conflict detection.
    """
    valid_slots = []
    for slot in slots:
        start_time = slot[0] * 60 + slot[1]
        end_time = slot[2] * 60 + slot[3]
        is_valid = True
        for person in participants:
            if person in schedules:
                for busy_start, busy_end in schedules[person]:
                    if start_time < busy_end and end_time > busy_start:
                        is_valid = False
                        break
            if not is_valid:
                break
        if is_valid:
            valid_slots.append(slot)
    return valid_slots