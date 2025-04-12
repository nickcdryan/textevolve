import re
import datetime

def main(question):
    """
    This script uses a simulated annealing approach to find the best meeting time,
    considering participant availability and preferences. It attempts to mimic LLM
    reasoning by extracting relevant information and iteratively refining a proposed solution.
    """
    try:
        # Extract meeting details using LLM-style parsing (simulated with regex)
        participants, duration, schedules, preferences = extract_meeting_details(question)

        # Generate candidate time slots
        candidate_slots = generate_candidate_slots(duration)

        # Evaluate time slots using simulated annealing
        best_slot = simulated_annealing(candidate_slots, schedules, preferences)

        if best_slot:
            return f"Here is the proposed time: Monday, {best_slot}"
        else:
            return "Could not find a suitable time."

    except Exception as e:
        return f"Error: {e}"

def extract_meeting_details(question):
    """
    Extracts meeting participants, duration, schedules, and preferences from the question text
    using regular expressions. This simulates LLM-like information extraction.
    """
    # Extract participants
    match = re.search(r"schedule a meeting for (.*?) for", question)
    if not match:
        raise ValueError("Could not extract participants.")
    participants = [name.strip() for name in match.group(1).split(',')]

    # Extract duration (assuming it's always half an hour for simplicity)
    duration = 30  # minutes

    # Extract schedules
    schedules = {}
    for participant in participants:
        schedule_match = re.search(rf"{participant} has meetings on Monday during (.*?);", question)
        if schedule_match:
            schedules[participant] = parse_schedule(schedule_match.group(1))
        else:
            schedules[participant] = []  # Assume no meetings if not specified

    # Extract preferences (limited to "don't meet after" for now)
    preferences = {}
    for participant in participants:
        preference_match = re.search(rf"{participant} would rather not meet on Monday after (\d+:\d+)", question)
        if preference_match:
            preferences[participant] = preference_match.group(1)

    return participants, duration, schedules, preferences

def parse_schedule(schedule_str):
    """
    Parses a schedule string (e.g., "9:30 to 10:00, 12:00 to 12:30") into a list of time intervals.
    """
    intervals = []
    for interval_str in schedule_str.split(', '):
        start_str, end_str = interval_str.split(' to ')
        intervals.append((start_str, end_str))
    return intervals

def generate_candidate_slots(duration):
    """
    Generates a list of candidate time slots (start and end times) for the meeting,
    considering the work hours of 9:00 to 17:00.
    """
    start_time = datetime.datetime.strptime("09:00", "%H:%M").time()
    end_time = datetime.datetime.strptime("17:00", "%H:%M").time()
    current_time = start_time
    slots = []
    while current_time < end_time:
        next_time = (datetime.datetime.combine(datetime.date.today(), current_time) +
                     datetime.timedelta(minutes=duration)).time()
        if next_time <= end_time:
            slots.append((current_time.strftime("%H:%M"), next_time.strftime("%H:%M")))
        else:
            break  # No more slots possible

        current_time = (datetime.datetime.combine(datetime.date.today(), current_time) +
                        datetime.timedelta(minutes=15)).time() # increment by 15 mins for broader coverage

    return slots

def simulated_annealing(candidate_slots, schedules, preferences, temperature=100, cooling_rate=0.95, iterations=1000):
    """
    Performs simulated annealing to find the best meeting time, considering participant availability and preferences.
    """
    import random
    current_slot = random.choice(candidate_slots)
    best_slot = current_slot
    best_energy = calculate_energy(current_slot, schedules, preferences)

    for _ in range(iterations):
        new_slot = random.choice(candidate_slots)
        new_energy = calculate_energy(new_slot, schedules, preferences)

        # Decide whether to accept the new solution
        if new_energy < best_energy:
            best_slot = new_slot
            best_energy = new_energy

        # Metropolis criterion for accepting worse solutions
        if new_energy < calculate_energy(current_slot, schedules, preferences) or \
           random.random() < math.exp((calculate_energy(current_slot, schedules, preferences) - new_energy) / temperature):
            current_slot = new_slot

        temperature *= cooling_rate

    return best_slot

def calculate_energy(slot, schedules, preferences):
    """
    Calculates the "energy" of a given time slot based on participant availability and preferences.
    Lower energy indicates a better slot.
    """
    energy = 0

    # Check for schedule conflicts
    for participant, busy_intervals in schedules.items():
        for start, end in busy_intervals:
            if is_time_overlap(slot, (start, end)):
                energy += 10  # High penalty for conflicts

    # Check for preference violations
    for participant, preferred_time in preferences.items():
        if slot[0] > preferred_time:  # Start time is after preferred time
            energy += 5  # Medium penalty for preference violation

    return energy

def is_time_overlap(slot1, slot2):
    """
    Checks if two time intervals overlap.
    """
    start1, end1 = slot1
    start2, end2 = slot2
    return not (end1 <= start2 or end2 <= start1)

import math # Needed for simulated annealing
if __name__ == "__main__":
    question1 = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Gary, Douglas, Elizabeth and Daniel for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nGary has meetings on Monday during 9:30 to 10:00, 12:00 to 12:30; \nDouglas has meetings on Monday during 10:30 to 11:00, 11:30 to 12:00, 14:00 to 14:30, 16:30 to 17:00; \nElizabeth has meetings on Monday during 11:30 to 13:30, 14:00 to 15:00, 16:00 to 17:00; \nDaniel has blocked their calendar on Monday during 10:30 to 12:30, 14:00 to 17:00; \n\nGary would rather not meet on Monday after 10:00. Find a time that works for everyone's schedule and constraints. \nSOLUTION: "
    print(main(question1))

    question2 = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Kevin, Carolyn, Stephanie and Isabella for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nKevin has blocked their calendar on Monday during 10:30 to 11:00, 14:00 to 14:30; \nCarolyn is busy on Monday during 11:30 to 12:00, 16:00 to 16:30; \nStephanie is busy on Monday during 9:00 to 10:30, 11:00 to 12:30, 13:30 to 14:30, 15:00 to 17:00; \nIsabella has meetings on Monday during 9:00 to 9:30, 10:30 to 11:00, 11:30 to 14:30, 15:00 to 17:00; \n\nFind a time that works for everyone's schedule and constraints. \nSOLUTION: "
    print(main(question2))