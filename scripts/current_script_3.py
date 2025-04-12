import re
import json

def main(question):
    """
    Schedules a meeting for given participants based on their availability and constraints,
    simulating an LLM-driven approach with specialized agent roles and knowledge bases.
    This approach leverages LLMs for complex reasoning and information extraction
    and Python code for deterministic calculations and verification.
    """

    try:
        # 1. Problem Decomposition Agent: Decompose the problem into subtasks.
        decomposition = decompose_problem(question)

        # 2. Information Extraction Agent: Extract relevant information.
        info = extract_information(decomposition)

        # 3. Schedule Generation Agent: Generate candidate schedules.
        candidate_schedules = generate_schedules(info)

        # 4. Schedule Verification Agent: Verify the proposed schedules.
        verified_schedules = verify_schedules(candidate_schedules, info)

        # 5. Preference Optimization Agent: Optimize schedules based on preferences.
        optimized_schedule = optimize_schedule(verified_schedules, info)

        # 6. Response Generation Agent: Format the response.
        answer = format_response(optimized_schedule)

        return answer

    except Exception as e:
        return f"Error: {str(e)}"

def decompose_problem(question):
    """
    Simulates an LLM decomposing the problem into smaller parts.
    In a real LLM-driven approach, this would involve calling an LLM
    to identify key elements.
    """
    # For now, simulate LLM decomposition
    return {
        "task": extract_task(question),
        "participants": extract_participants(question),
        "duration": extract_duration(question),
        "availability": extract_availability(question),
        "preferences": extract_preferences(question)
    }

def extract_task(question):
    """Extracts the task description from the input question using a simplified regex."""
    match = re.search(r"TASK:\s*(.*?)\n\n", question)
    return match.group(1) if match else ""


def extract_participants(question):
    """Extracts participant names from the input question using a simplified regex."""
    match = re.search(r"schedule a meeting for\s*(.*?)\s*for", question)
    if match:
        return [p.strip() for p in match.group(1).split(',')]
    return []


def extract_duration(question):
    """Extracts meeting duration from the input question using a simplified regex."""
    match = re.search(r"for\s*(.*?)\s*between", question)
    if match:
        duration_str = match.group(1).strip()
        if "hour" in duration_str:
            duration = int(duration_str.split(" ")[0]) * 60
        elif "half an hour" in duration_str:
            duration = 30
        else:
            duration = int(duration_str.split(" ")[0])
        return duration
    return 30  # Default duration

def extract_availability(question):
    """Extracts availability information for each participant using a simplified regex."""
    availability = {}
    participants = extract_participants(question)
    for participant in participants:
        match = re.search(r"{}\s*(.*?)(?:\n|\Z)".format(re.escape(participant)), question, re.DOTALL) #use \Z instead of $
        if match:
            availability[participant] = match.group(1).strip()
        else:
            availability[participant] = "free the entire day" #default
    return availability


def extract_preferences(question):
    """Extracts preferences from the question using a simplified regex."""
    match = re.search(r"want to meet on Monday\s*(.*?)Find", question)
    if match:
        return match.group(1).strip()
    else:
        return "" #no preference

def extract_information(decomposition):
    """
    Extracts and consolidates information needed for scheduling,
    simulating the function of an LLM.
    """
    participants = decomposition["participants"]
    availability = decomposition["availability"]
    preferences = decomposition["preferences"]
    duration = decomposition["duration"]
    
    blocked_times = {}
    for participant, schedule_str in availability.items():
        blocked_times[participant] = parse_schedule(schedule_str)

    return {
        "participants": participants,
        "duration": duration,
        "blocked_times": blocked_times,
        "preferences": preferences
    }


def parse_schedule(schedule_str):
    """Parses schedule strings to extract blocked time slots using regex.  Simulates an LLM."""
    blocked_slots = []
    time_pattern = r"(\d{1,2}:\d{2})\s*to\s*(\d{1,2}:\d{2})"
    matches = re.findall(time_pattern, schedule_str)
    for start_time, end_time in matches:
        blocked_slots.append((start_time, end_time))
    return blocked_slots

def generate_schedules(info):
    """Generates candidate schedules based on available information."""
    start_time = 9 * 60  # Start at 9:00
    end_time = 17 * 60   # End at 17:00
    duration = info["duration"]
    blocked_times = info["blocked_times"]
    
    available_times = []
    
    # Convert blocked times to minutes
    blocked_times_minutes = {}
    for person, blocked in blocked_times.items():
        blocked_times_minutes[person] = []
        for start, end in blocked:
            blocked_times_minutes[person].append((time_to_minutes(start), time_to_minutes(end)))
    
    # Iterate through possible start times
    current_time = start_time
    while current_time + duration <= end_time:
        is_available = True
        for person, blocked in blocked_times_minutes.items():
            for block_start, block_end in blocked:
                if current_time < block_end and current_time + duration > block_start:
                    is_available = False
                    break
            if not is_available:
                break

        if is_available:
            available_times.append(current_time)
        current_time += 30  # Check every 30 minutes
    
    schedules = []
    for time in available_times:
        schedules.append({"start": time, "end": time + duration})

    return schedules

def verify_schedules(schedules, info):
    """Verifies if the generated schedules work for all participants and constraints."""
    verified_schedules = []
    blocked_times_minutes = {}
    for person, blocked in info["blocked_times"].items():
        blocked_times_minutes[person] = []
        for start, end in blocked:
            blocked_times_minutes[person].append((time_to_minutes(start), time_to_minutes(end)))

    for schedule in schedules:
        is_valid = True
        for person, blocked in blocked_times_minutes.items():
            for block_start, block_end in blocked:
                if schedule["start"] < block_end and schedule["end"] > block_start:
                    is_valid = False
                    break
            if not is_valid:
                break
        if is_valid:
            verified_schedules.append(schedule)
    return verified_schedules


def optimize_schedule(verified_schedules, info):
    """Optimizes the schedule based on given preferences.  Currently it selects first available."""
    preferences = info["preferences"]
    if len(verified_schedules) == 0:
        return None

    #Simulate preference for no meetings before certain time
    if preferences:
        preferred_start_time_match = re.search(r"before (\d{1,2}:\d{2})", preferences)
        if preferred_start_time_match:
            preferred_start_time = time_to_minutes(preferred_start_time_match.group(1))
            
            for schedule in verified_schedules:
                if schedule["start"] >= preferred_start_time:
                    return schedule
            #If no suitable schedule is found after preferred time, return the first available schedule
            return verified_schedules[0]
    # If no preferences, return the first valid schedule
    return verified_schedules[0]


def format_response(schedule):
    """Formats the final response into a human-readable string."""
    if schedule is None:
        return "No suitable time found."
    
    start_time = minutes_to_time(schedule["start"])
    end_time = minutes_to_time(schedule["end"])
    return f"Here is the proposed time: Monday, {start_time} - {end_time} "

def time_to_minutes(time_str):
    """Converts a time string (HH:MM) to minutes since midnight."""
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

def minutes_to_time(minutes):
    """Converts minutes since midnight to a time string (HH:MM)."""
    hours = minutes // 60
    minutes = minutes % 60
    return "{:02d}:{:02d}".format(hours, minutes)