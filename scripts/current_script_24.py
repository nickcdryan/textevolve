import re
import datetime

def main(question):
    """
    Schedules a meeting using a constraint satisfaction problem (CSP) approach, leveraging LLM-style reasoning
    through flexible parsing and iterative refinement of candidate solutions.

    The CSP is defined as follows:
        - Variables: Possible meeting timeslots
        - Domains: Each timeslot has a domain of "available" or "unavailable"
        - Constraints: Each participant's schedule restricts the domain of certain timeslots

    This version specifically incorporates a "preference scoring" system to handle soft constraints,
    simulating LLM's ability to weigh preferences.

    Args:
        question (str): A string describing the scheduling task and participant schedules.

    Returns:
        str: A string indicating the proposed meeting time, or a message if no suitable time is found.
    """

    try:
        # 1. Information Extraction using flexible regex and LLM-like reasoning (handling variations)
        participants = extract_participants(question)
        duration = extract_duration(question)
        schedules = extract_schedules(question, participants)
        preferences = extract_preferences(question, participants) # NEW: Extract preferences

        # 2. Generate Candidate Time Slots (CSP Variables)
        start_time = datetime.datetime(2024, 1, 1, 9, 0, 0)  # Arbitrary date, only time matters
        end_time = datetime.datetime(2024, 1, 1, 17, 0, 0)
        time_slots = generate_time_slots(start_time, end_time, duration)

        # 3. Constraint Evaluation (CSP Constraint Propagation) and Preference Scoring
        available_slots = []
        for slot_start, slot_end in time_slots:
            is_available = True
            score = 0 # Initialize preference score

            for participant, schedule in schedules.items():
                if is_time_blocked(slot_start, slot_end, schedule):
                    is_available = False
                    break  # Constraint violated

                # Apply preference scoring: higher score means better
                if participant in preferences:
                    preference_penalty = calculate_preference_penalty(slot_start, preferences[participant])
                    score -= preference_penalty # Deduct penalty for violating preference

            if is_available:
                available_slots.append(((slot_start, slot_end), score)) # Store score with the slot

        # 4. Solution Selection - Choose slot with best score (simulating LLM choosing preferred option)
        if available_slots:
            # Sort by score in descending order (higher score is better)
            best_slot = max(available_slots, key=lambda item: item[1])[0]
            start_time_str = best_slot[0].strftime("%H:%M")
            end_time_str = best_slot[1].strftime("%H:%M")
            return f"Here is the proposed time: Monday, {start_time_str} - {end_time_str} "
        else:
            return "No suitable meeting time found."

    except Exception as e:
        return f"Error scheduling meeting: {str(e)}"

def extract_participants(question):
    """
    Extracts participant names from the question using a more flexible regex.
    Handles variations in phrasing (simulating LLM robustness).
    """
    match = re.search(r"schedule a meeting for (.*?) for", question)
    if match:
        return [name.strip() for name in match.group(1).split(',')]
    else:
        raise ValueError("Could not extract participants from the question.")


def extract_duration(question):
    """
    Extracts the meeting duration (in minutes) from the question.
    """
    match = re.search(r"for (.*?)(?: between|on)", question)  # Modified regex for flexibility
    if match:
        duration_str = match.group(1).strip()
        if "half an hour" in duration_str:
            return 30
        else:
            try:
                return int(re.search(r"(\d+)", duration_str).group(1)) # Extract digits and convert to int
            except:
                raise ValueError("Could not extract duration from the question. Check formatting.")
    else:
        raise ValueError("Could not extract duration from the question.")

def extract_schedules(question, participants):
    """
    Extracts the schedules of each participant from the question using more robust parsing.
    """
    schedules = {}
    for participant in participants:
        schedule_match = re.search(rf"{participant}(?:'s| has) (.*?)[\n;]", question, re.IGNORECASE) # Added ignorecase
        if schedule_match:
            schedules[participant] = schedule_match.group(1).strip()
        else:
            schedules[participant] = "no meetings the whole day." # Default when no info provided
    return schedules

def extract_preferences(question, participants):
    """
    Extracts meeting time preferences for each participant from the question.
    Returns a dictionary of participant -> "not before HH:MM" or None if no preference.
    """
    preferences = {}
    for participant in participants:
        preference_match = re.search(rf"{participant} would rather not meet on Monday before (\d+:\d+)", question)
        if preference_match:
            preferences[participant] = preference_match.group(1)
        else:
            preferences[participant] = None  # No specific preference

    return preferences

def generate_time_slots(start_time, end_time, duration):
    """
    Generates a list of possible time slots between the start and end times, with the given duration.
    """
    time_slots = []
    current_time = start_time
    while current_time + datetime.timedelta(minutes=duration) <= end_time:
        time_slots.append((current_time, current_time + datetime.timedelta(minutes=duration)))
        current_time += datetime.timedelta(minutes=15)  # 15-minute increments
    return time_slots

def is_time_blocked(slot_start, slot_end, schedule_string):
    """
    Checks if a given time slot is blocked according to the schedule string.
    Uses more flexible parsing with LLM-like interpretation of schedule strings.
    """
    if "no meetings the whole day" in schedule_string.lower() or "free the entire day" in schedule_string.lower() : # Handle "free" days
        return False

    blocked_times = re.findall(r"(\d+:\d+)\s*to\s*(\d+:\d+)", schedule_string) # Extract blocked times

    for blocked_start_str, blocked_end_str in blocked_times:
        blocked_start = datetime.datetime.strptime(blocked_start_str, "%H:%M").time()
        blocked_end = datetime.datetime.strptime(blocked_end_str, "%H:%M").time()

        slot_start_time = slot_start.time()
        slot_end_time = slot_end.time()

        blocked_start_dt = datetime.datetime.combine(slot_start.date(), blocked_start) # Combine date for comparison
        blocked_end_dt = datetime.datetime.combine(slot_start.date(), blocked_end)
        slot_start_dt = datetime.datetime.combine(slot_start.date(), slot_start_time)
        slot_end_dt = datetime.datetime.combine(slot_start.date(), slot_end_time)

        if blocked_start_dt <= slot_start_dt < blocked_end_dt or blocked_start_dt < slot_end_dt <= blocked_end_dt or (slot_start_dt <= blocked_start_dt and slot_end_dt >= blocked_end_dt): #Time overlap
            return True

    return False

def calculate_preference_penalty(slot_start, preferred_time_str):
    """
    Calculates a penalty score based on how much the meeting time violates the preference.
    Higher penalty means a greater violation.
    """
    if preferred_time_str is None:
        return 0 # No penalty if no preference

    preferred_time = datetime.datetime.strptime(preferred_time_str, "%H:%M").time()
    slot_start_time = slot_start.time()

    preferred_datetime = datetime.datetime.combine(slot_start.date(), preferred_time)
    slot_start_datetime = datetime.datetime.combine(slot_start.date(), slot_start_time)

    if slot_start_datetime < preferred_datetime:
        time_difference = preferred_datetime - slot_start_datetime #Calculate difference
        return time_difference.total_seconds() / 3600 # Penalty proportional to time difference (in hours)
    else:
        return 0 #No penalty as meeting starts after preferred time.

# Example usage (for testing):
# question = "You need to schedule a meeting for Aaron, Sarah, Martha and Heather for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nAaron has blocked their calendar on Monday during 9:00 to 9:30, 11:30 to 12:00, 12:30 to 14:00, 15:30 to 16:00; \nSarah is busy on Monday during 10:30 to 11:30, 12:30 to 13:00, 13:30 to 14:30, 16:00 to 16:30; \nMartha is busy on Monday during 9:00 to 9:30, 10:30 to 12:00, 12:30 to 13:30, 14:00 to 14:30, 15:30 to 17:00; \nHeather has meetings on Monday during 9:00 to 10:00, 11:30 to 12:00, 13:00 to 14:30, 15:00 to 15:30, 16:00 to 16:30; \n\nSarah would rather not meet on Monday before 13:30. Find a time that works for everyone's schedule and constraints."
# answer = main(question)
# print(answer)