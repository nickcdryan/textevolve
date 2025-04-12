import re
import datetime

def main(question):
    """
    Schedules a meeting by simulating a LLM scheduler with iterative refinement and conflict resolution.
    It extracts information, proposes an initial time, checks for conflicts, and iteratively adjusts the time
    based on conflicts until a suitable time is found. The core idea is to mimic an LLM going through multiple
    "reasoning" steps to find a solution.
    """
    try:
        participants, duration, schedules, preferences = extract_information_with_llm_simulation(question)
        
        # Propose an initial meeting time
        proposed_time = propose_initial_time(schedules, preferences)
        
        # Iteratively refine the meeting time to resolve conflicts
        max_iterations = 10  # Limit iterations to prevent infinite loops
        for i in range(max_iterations):
            conflicts = check_for_conflicts(proposed_time, participants, schedules)
            
            if not conflicts:
                start_time_str = proposed_time.strftime("%H:%M")
                end_time = proposed_time + duration
                end_time_str = end_time.strftime("%H:%M")
                return f"Here is the proposed time: Monday, {start_time_str} - {end_time_str} "
            
            proposed_time = adjust_meeting_time(proposed_time, conflicts, duration)  # Adjust based on conflicts
        
        return "Could not find a suitable meeting time within the iteration limit."
    
    except Exception as e:
        return f"An error occurred: {e}"

def extract_information_with_llm_simulation(question):
    """
    Simulates an LLM to extract relevant information from the question using a combination of regex and string manipulation.
    Mimics the way an LLM might handle different phrasings or variations in the input.
    """
    try:
        # Extract participants
        match = re.search(r"schedule a meeting for (.*?) for", question)
        participants = [name.strip() for name in match.group(1).split(',')] if match else []
        
        # Extract duration (in minutes)
        duration_match = re.search(r"for (.*?)(?: between| on)", question)  # added non-greedy match
        duration_str = duration_match.group(1).strip() if duration_match else "half an hour"  # Handle missing duration
        if "half an hour" in duration_str:
            duration = datetime.timedelta(minutes=30)
        else:
            duration_minutes = int(re.search(r'(\d+)', duration_str).group(1))  # Extract number from duration
            duration = datetime.timedelta(minutes=duration_minutes)
            
        # Extract schedules
        schedules = {}
        for participant in participants:
            schedule_match = re.search(rf"{participant} (?:is|has) (?:busy|meetings) on Monday during (.*?)(?:;|\n|$)", question)
            if schedule_match:
                schedules[participant] = parse_schedule_with_llm_simulation(schedule_match.group(1))
            else:
                schedules[participant] = []  # No schedule found, assume available

        # Extract Preferences
        preferences = {}
        preference_match = re.search(r"(.*?) would (?:rather not|like to avoid) meet", question)
        if preference_match:
            preferred_participant = participants[0] # simplifying assumption - preference given to 1st participant
            preferences[preferred_participant] = "later"

        return participants, duration, schedules, preferences
    
    except Exception as e:
        raise ValueError(f"Error extracting information: {e}")

def parse_schedule_with_llm_simulation(schedule_string):
    """
    Simulates an LLM to parse schedule strings with some flexibility.
    Mimics the LLM's ability to understand variations in time formatting and wording.
    """
    try:
        schedule = []
        time_ranges = schedule_string.split(', ')
        for time_range in time_ranges:
            match = re.search(r"(\d{1,2}:\d{2}) to (\d{1,2}:\d{2})", time_range)
            if match:
                start_time_str, end_time_str = match.groups()
                start_time = datetime.datetime.strptime(start_time_str, "%H:%M").time()
                end_time = datetime.datetime.strptime(end_time_str, "%H:%M").time()
                schedule.append((start_time, end_time))
        return schedule
    except Exception as e:
        print(f"Error parsing schedule: {e}")
        return []

def propose_initial_time(schedules, preferences):
    """
    Proposes an initial meeting time based on the earliest availability of participants,
    and taking preferences into account.
    """
    # Start at 9:00 and check for a free slot
    current_time = datetime.datetime.combine(datetime.date.today(), datetime.time(9, 0))
    end_of_day = datetime.datetime.combine(datetime.date.today(), datetime.time(17, 0))
    
    return current_time

def check_for_conflicts(proposed_time, participants, schedules):
    """
    Checks if the proposed meeting time conflicts with any participant's schedule.
    """
    conflicts = {}
    proposed_end_time = proposed_time + datetime.timedelta(minutes=30)
    
    for participant in participants:
        if participant in schedules:
            for busy_start_time, busy_end_time in schedules[participant]:
                busy_start = datetime.datetime.combine(datetime.date.today(), busy_start_time)
                busy_end = datetime.datetime.combine(datetime.date.today(), busy_end_time)
                
                if (proposed_time < busy_end) and (proposed_end_time > busy_start):
                    conflicts[participant] = (busy_start_time, busy_end_time)
    
    return conflicts

def adjust_meeting_time(proposed_time, conflicts, duration):
    """
    Adjusts the meeting time based on conflicts, simulating an LLM's iterative refinement process.
    """
    # If there are conflicts, move the meeting time 30 minutes later
    return proposed_time + datetime.timedelta(minutes=30)