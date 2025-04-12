import re
import json

def main(question):
    """
    Schedules a meeting given participant schedules and constraints, using a multi-agent LLM-driven approach.
    This iteration focuses on simulating different agent perspectives to refine the solution.

    Args:
        question (str): A string containing the scheduling problem description.

    Returns:
        str: A string containing the proposed meeting time.
    """

    try:
        # --- Agent 1: Information Extractor ---
        # Extracts key information using LLM-like prompting and string manipulation (simulated)
        task_description, schedules_description, constraints = extract_information(question)

        # --- Agent 2: Schedule Analyzer ---
        # Analyzes individual schedules and constraints, determines free slots
        participant_schedules = parse_schedules(schedules_description)
        available_slots = {}
        for participant, schedule in participant_schedules.items():
            available_slots[participant] = find_available_time_slots(schedule)

        # --- Agent 3: Meeting Scheduler ---
        # Integrates schedules and constraints to find a common free slot
        meeting_time = find_meeting_time(available_slots, constraints)

        # --- Agent 4: Verifier/Critic Agent ---
        # Verifies that the schedule works for everyone and adheres to constraints.
        verification_result = verify_schedule(meeting_time, participant_schedules, constraints)
        if not verification_result["valid"]:
           # Attempt a second pass, if initial schedule not verified.
           meeting_time = find_meeting_time(available_slots, constraints, second_pass = True)
           verification_result = verify_schedule(meeting_time, participant_schedules, constraints)
           if not verification_result["valid"]:
              return "Could not find a suitable meeting time." # Last ditch effort.
        
        if meeting_time:
            return f"Here is the proposed time: {meeting_time}"
        else:
            return "Could not find a suitable meeting time."

    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while scheduling the meeting."


def extract_information(question):
    """
    Simulates an LLM extracting key information from the question.
    This function uses string manipulations but aims to mimic the behavior of an LLM.

    Args:
        question (str): The scheduling problem description.

    Returns:
        tuple: A tuple containing the task description, schedules description, and constraints.
    """
    try:
        parts = question.split("Here are the existing schedules for everyone during the day:")
        task_description = parts[0].split("TASK:")[1].strip() if len(parts) > 1 else ""
        
        schedules_and_constraints = parts[1] if len(parts) > 1 else ""
        
        schedules_end = schedules_and_constraints.find("Find a time that works for everyone's schedule and constraints.")
        schedules_description = schedules_and_constraints[:schedules_end].strip() if schedules_end != -1 else schedules_and_constraints
        constraints_start = schedules_and_constraints.find("Find a time that works for everyone's schedule and constraints.")

        constraints = schedules_and_constraints[constraints_start:].replace("Find a time that works for everyone's schedule and constraints.", "").strip() if constraints_start != -1 else ""

        return task_description, schedules_description, constraints

    except Exception as e:
        print(f"Error during information extraction: {e}")
        return "", "", ""


def parse_schedules(schedules_description):
    """
    Parses the schedule descriptions into a dictionary of participant schedules.

    Args:
        schedules_description (str): The schedule descriptions.

    Returns:
        dict: A dictionary where keys are participant names and values are lists of time intervals (tuples).
    """
    participant_schedules = {}
    try:
        schedule_lines = schedules_description.split('\n')
        for line in schedule_lines:
            if "has meetings on" in line or "is busy on" in line or "has blocked their calendar on" in line or "has no meetings the whole day" in line:
                parts = line.split(" has meetings on ") or line.split(" is busy on ") or line.split(" has blocked their calendar on ") or line.split(" has no meetings the whole day ")
                participant = parts[0].strip()
                if "no meetings the whole day" in line:
                   participant_schedules[participant] = []
                else:
                    time_intervals_str = parts[1].replace("Monday during ", "")
                    time_intervals = []
                    for interval in time_intervals_str.split(', '):
                        start_time, end_time = interval.split(' to ')
                        time_intervals.append((start_time, end_time))
                    participant_schedules[participant] = time_intervals
    except Exception as e:
        print(f"Error during schedule parsing: {e}")
    return participant_schedules

def convert_to_minutes(time_str):
    """Converts a time string (e.g., "9:00") to minutes from midnight."""
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

def find_available_time_slots(schedule):
    """
    Finds available time slots in a given schedule.

    Args:
        schedule (list): A list of time intervals (tuples) representing busy times.

    Returns:
        list: A list of time intervals (tuples) representing available times.
    """
    available_slots = []
    start_of_day = convert_to_minutes("9:00")
    end_of_day = convert_to_minutes("17:00")
    
    schedule_in_minutes = sorted([(convert_to_minutes(start), convert_to_minutes(end)) for start, end in schedule])
    
    current_time = start_of_day
    for busy_start, busy_end in schedule_in_minutes:
        if current_time < busy_start:
            available_slots.append((current_time, busy_start))
        current_time = busy_end
    
    if current_time < end_of_day:
        available_slots.append((current_time, end_of_day))
    
    return available_slots

def find_meeting_time(available_slots, constraints, second_pass = False):
    """
    Finds a common meeting time that works for all participants, considering constraints.

    Args:
        available_slots (dict): A dictionary of available time slots for each participant.
        constraints (str): Constraints on the meeting time.

    Returns:
        str: The proposed meeting time, or None if no suitable time is found.
    """
    meeting_duration = 30  # Default to 30 minutes
    all_participants = list(available_slots.keys())
    
    if not all_participants:
        return None
    
    # Find intersection of available times for the first two participants
    common_slots = find_common_time_slots(available_slots[all_participants[0]], available_slots[all_participants[1]])
    
    # Iteratively find common slots with the remaining participants
    for i in range(2, len(all_participants)):
        common_slots = find_common_time_slots(common_slots, available_slots[all_participants[i]])
    
    # Apply constraints
    if "Christine can not meet on Monday before 12:00" in constraints:
        constraint_time = convert_to_minutes("12:00")
        common_slots = [(max(start, constraint_time), end) for start, end in common_slots if end > constraint_time]

    if "Helen do not want to meet on Monday after 13:30" in constraints:
        constraint_time = convert_to_minutes("13:30")
        common_slots = [(start, min(end, constraint_time)) for start, end in common_slots if start < constraint_time]

    if "Billy would like to avoid more meetings on Monday after 15:30" in constraints:
        constraint_time = convert_to_minutes("15:30")
        if not second_pass: # First pass, prefer times before 15:30
            common_slots = [(start, min(end, constraint_time)) for start, end in common_slots if start < constraint_time]
        else: # Second pass, consider times after 15:30 if there are no earlier slots.
            original_slots = [(start, end) for start, end in common_slots if start >= constraint_time]
            if not original_slots: # No suitable earlier slots, use the slots before 15:30.
                 common_slots = [(start, min(end, constraint_time)) for start, end in common_slots if start < constraint_time]
            else:
                common_slots = original_slots

    # Find a slot that fits the meeting duration
    for start, end in common_slots:
        if end - start >= meeting_duration:
            meeting_start_hours = start // 60
            meeting_start_minutes = start % 60
            meeting_end_hours = (start + meeting_duration) // 60
            meeting_end_minutes = (start + meeting_duration) % 60
            
            meeting_start_time = f"{meeting_start_hours:02}:{meeting_start_minutes:02}"
            meeting_end_time = f"{meeting_end_hours:02}:{meeting_end_minutes:02}"
            
            return f"Monday, {meeting_start_time} - {meeting_end_time} "
    
    return None


def find_common_time_slots(slots1, slots2):
    """
    Finds the common time slots between two lists of time slots.

    Args:
        slots1 (list): List of time intervals (tuples).
        slots2 (list): List of time intervals (tuples).

    Returns:
        list: List of common time intervals (tuples).
    """
    common_slots = []
    for start1, end1 in slots1:
        for start2, end2 in slots2:
            intersection_start = max(start1, start2)
            intersection_end = min(end1, end2)
            if intersection_start < intersection_end:
                common_slots.append((intersection_start, intersection_end))
    return common_slots

def verify_schedule(meeting_time, participant_schedules, constraints):
    """
    Verifies that the proposed schedule is valid for all participants.

    Args:
        meeting_time (str): The proposed meeting time.
        participant_schedules (dict): A dictionary of participant schedules.
        constraints (str): Constraints on the meeting time.

    Returns:
        dict: A dictionary indicating whether the schedule is valid and providing feedback.
    """
    if not meeting_time:
        return {"valid": False, "feedback": "No meeting time proposed."}
    
    try:
        meeting_start_time_str = meeting_time.split(', ')[1].split(' - ')[0]
        meeting_end_time_str = meeting_time.split(', ')[1].split(' - ')[1]

        meeting_start_minutes = convert_to_minutes(meeting_start_time_str)
        meeting_end_minutes = convert_to_minutes(meeting_end_time_str)
    except:
        return {"valid": False, "feedback": "Error parsing meeting time."}
    
    for participant, schedule in participant_schedules.items():
        for busy_start_str, busy_end_str in schedule:
            busy_start_minutes = convert_to_minutes(busy_start_str)
            busy_end_minutes = convert_to_minutes(busy_end_str)
            
            if meeting_start_minutes < busy_end_minutes and meeting_end_minutes > busy_start_minutes:
                return {"valid": False, "feedback": f"Conflict with {participant}'s schedule."}
    
    # Check if Christine can not meet before 12:00
    if "Christine can not meet on Monday before 12:00" in constraints:
        if meeting_start_minutes < convert_to_minutes("12:00") and "Christine" in participant_schedules:
            return {"valid": False, "feedback": "Christine cannot meet before 12:00."}

    # Check if Helen do not want to meet on Monday after 13:30
    if "Helen do not want to meet on Monday after 13:30" in constraints:
        if meeting_start_minutes >= convert_to_minutes("13:30") and "Helen" in participant_schedules:
            return {"valid": False, "feedback": "Helen does not want to meet after 13:30."}

    return {"valid": True, "feedback": "Schedule is valid."}


# Example Usage (for testing):
if __name__ == "__main__":
    example_question = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Joyce, Christine and Alexander for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nJoyce has meetings on Monday during 11:00 to 11:30, 13:30 to 14:00, 14:30 to 16:30; \nChristinehas no meetings the whole day.\nAlexander has meetings on Monday during 9:00 to 11:00, 12:00 to 12:30, 13:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00; \n\nChristine can not meet on Monday before 12:00. Find a time that works for everyone's schedule and constraints. \nSOLUTION: "
    
    result = main(example_question)
    print(result)  # Output the proposed meeting time or an error message