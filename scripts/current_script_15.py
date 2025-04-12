import re

def main(question):
    """
    Schedules a meeting for a group of people, considering their schedules and constraints,
    by leveraging LLM-style reasoning through a series of simulated LLM calls. This approach
    aims to mimic how a large language model would process and reason about the scheduling problem.

    The "LLM calls" are simulated using string parsing and logical deductions within the script.

    Args:
        question (str): A string containing the task description, participants, schedules,
                         and any preferences or constraints.

    Returns:
        str: A string containing the proposed meeting time, formatted as "Here is the proposed time: ..."
             or "No suitable time found." if a meeting cannot be scheduled.
    """

    try:
        # Simulate LLM call to extract participants and their schedules.
        participants, schedules = extract_participants_and_schedules(question)

        # Simulate LLM call to extract constraints (meeting duration and time preferences).
        duration, work_hours, preferences = extract_constraints(question)

        # Simulate LLM call to generate potential time slots.
        time_slots = generate_time_slots(work_hours, duration)

        # Simulate LLM call to check availability for each time slot.
        available_slots = check_availability(time_slots, schedules, preferences)

        if available_slots:
            # Simulate LLM call to select the best time slot based on preferences (if any).
            best_time = select_best_time(available_slots, preferences)

            start_time, end_time = best_time
            return f"Here is the proposed time: Monday, {start_time} - {end_time} "
        else:
            return "No suitable time found."

    except Exception as e:
        return f"Error processing the request: {str(e)}"


def extract_participants_and_schedules(question):
    """
    Simulates an LLM call to extract participants and their schedules from the question text.

    Args:
        question (str): The input question string.

    Returns:
        tuple: A tuple containing a list of participants and a dictionary of their schedules.
    """
    participants_match = re.search(r"schedule a meeting for (.*?) for", question)
    participants = [p.strip() for p in participants_match.group(1).split(",")] if participants_match else []

    schedules = {}
    for participant in participants:
        schedule_match = re.search(rf"{participant} (?:is busy on|has meetings on) Monday during (.*?)(?:;|\.)", question)
        if schedule_match:
            schedule_str = schedule_match.group(1)
            schedule = []
            time_ranges = schedule_str.split(", ")
            for time_range in time_ranges:
                start, end = time_range.split(" to ")
                schedule.append((start, end))
            schedules[participant] = schedule
        else:
            schedules[participant] = []  # Assume no meetings if not explicitly mentioned.

    return participants, schedules


def extract_constraints(question):
    """
    Simulates an LLM call to extract meeting duration, work hours, and preferences from the question text.

    Args:
        question (str): The input question string.

    Returns:
        tuple: A tuple containing the meeting duration (minutes), work hours (start and end), and preferences.
    """
    duration_match = re.search(r"for (.*?) hour", question)
    duration_str = duration_match.group(1) if duration_match else "half"
    duration = 30 if "half" in duration_str else int(duration_str) * 60

    work_hours_match = re.search(r"between the work hours of (.*?) to (.*?) on Monday", question)
    work_hours = (work_hours_match.group(1), work_hours_match.group(2)) if work_hours_match else ("9:00", "17:00")

    preferences = {}
    if "would rather not meet on Monday before" in question:
        preference_match = re.search(r"would rather not meet on Monday before (.*?)[\.]", question)
        if preference_match:
            preferences["avoid_before"] = preference_match.group(1)
    if "would like to avoid more meetings on Monday after" in question:
        preference_match = re.search(r"would like to avoid more meetings on Monday after (.*?)[\.]", question)
        if preference_match:
            preferences["avoid_after"] = preference_match.group(1)

    if "earlist availability" in question:
        preferences["earliest"] = True

    return duration, work_hours, preferences


def generate_time_slots(work_hours, duration):
    """
    Simulates an LLM call to generate potential time slots within the given work hours.

    Args:
        work_hours (tuple): A tuple containing the start and end work hours (strings).
        duration (int): The meeting duration in minutes.

    Returns:
        list: A list of tuples, where each tuple represents a time slot (start and end times as strings).
    """
    start_hour, end_hour = work_hours
    start_minute = int(start_hour.split(":")[1])
    end_minute = int(end_hour.split(":")[1])

    start_hour = int(start_hour.split(":")[0])
    end_hour = int(end_hour.split(":")[0])

    time_slots = []
    current_hour = start_hour
    current_minute = start_minute

    while current_hour < end_hour or (current_hour == end_hour and current_minute <= end_minute - (duration if duration <= 60 else 60)):
        start_time = f"{current_hour:02}:{current_minute:02}"
        end_hour_slot = current_hour
        end_minute_slot = current_minute + duration

        if end_minute_slot >= 60:
             end_hour_slot += end_minute_slot // 60
             end_minute_slot %= 60

        end_time = f"{end_hour_slot:02}:{end_minute_slot:02}"

        if end_hour_slot > end_hour or (end_hour_slot == end_hour and end_minute_slot > end_minute):
            break

        time_slots.append((start_time, end_time))
        current_minute += 30  # Increment by 30 minutes for each time slot

        if current_minute >= 60:
            current_hour += current_minute // 60
            current_minute %= 60

    return time_slots


def check_availability(time_slots, schedules, preferences):
    """
    Simulates an LLM call to check the availability of each time slot against the participants' schedules.

    Args:
        time_slots (list): A list of potential time slots (tuples of start and end times).
        schedules (dict): A dictionary of participants and their schedules.
        preferences (dict): A dictionary of meeting preferences.

    Returns:
        list: A list of available time slots (tuples of start and end times).
    """
    available_slots = []

    for start_time, end_time in time_slots:
        available = True
        for participant, schedule in schedules.items():
            for busy_start, busy_end in schedule:
                if not (end_time <= busy_start or start_time >= busy_end):
                    available = False
                    break  # Participant is busy during this slot.

            if not available:
                break  # Time slot is not available.

        if available:

            # Check the "avoid_before" constraint.
            if "avoid_before" in preferences:
                 if start_time < preferences["avoid_before"]:
                     continue

            # Check the "avoid_after" constraint.
            if "avoid_after" in preferences:
                if start_time >= preferences["avoid_after"]:
                     continue
            available_slots.append((start_time, end_time))

    return available_slots

def select_best_time(available_slots, preferences):
    """
    Simulates an LLM call to select the best time slot from the available options based on preferences.

    Args:
        available_slots (list): A list of available time slots (tuples of start and end times).
        preferences (dict): A dictionary of meeting preferences.

    Returns:
        tuple: The selected time slot (start and end times as strings).
    """

    if "earliest" in preferences:
        return available_slots[0] if available_slots else None

    return available_slots[-1] if available_slots else None  # Return the last available time