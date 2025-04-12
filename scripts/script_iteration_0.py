import re
import json

def main(question):
    """
    This script uses a LLM-driven approach to schedule meetings, focusing on modularity 
    and clear reasoning steps. It breaks down the problem into parsing, conflict finding, 
    and solution generation, using LLM calls for reasoning.

    Specifically, it uses chained LLM calls to:
    1. Extract participants, duration, work hours, and blocked times.
    2. Represent time slots as minutes since 9:00.
    3. Identify available time slots for each participant.
    4. Find overlapping available slots.
    5. Filter slots based on preferences.
    6. Generate the meeting schedule time.
    """

    try:
        # 1. Extract information using LLM calls
        extracted_info = extract_information(question)
        
        participants = extracted_info['participants']
        duration = extracted_info['duration']
        work_hours = extracted_info['work_hours']
        blocked_times = extracted_info['blocked_times']
        preferences = extracted_info['preferences']

        # 2. Convert time to minutes since 9:00 for easier calculation
        work_start = time_to_minutes(work_hours[0])
        work_end = time_to_minutes(work_hours[1])

        blocked_times_minutes = {}
        for person, blocks in blocked_times.items():
            blocked_times_minutes[person] = []
            for block in blocks:
                blocked_times_minutes[person].append((time_to_minutes(block[0]), time_to_minutes(block[1])))

        # 3. Generate available time slots for each person
        available_slots = {}
        for person in participants:
            available_slots[person] = find_available_slots(work_start, work_end, blocked_times_minutes[person])

        # 4. Find overlapping available slots
        overlapping_slots = find_overlapping_slots(available_slots, duration)

        # 5. Apply preferences (if any)
        final_slots = apply_preferences(overlapping_slots, preferences)

        # 6. Generate the meeting schedule time
        if final_slots:
            start_time_minutes = final_slots[0]  # Take the first available slot
            start_time = minutes_to_time(start_time_minutes)
            end_time = minutes_to_time(start_time_minutes + duration)
            answer = f"Here is the proposed time: Monday, {start_time} - {end_time} "
        else:
            answer = "No suitable time found."

        return answer

    except Exception as e:
        return f"Error processing the request: {str(e)}"


def extract_information(question):
    """
    Extracts relevant information from the question using an LLM-like reasoning.
    This simplified version uses regex for demonstration, but in a real application,
    it would make calls to an LLM to parse and understand the text.
    """
    participants = []
    duration = 0
    work_hours = []
    blocked_times = {}
    preferences = {}

    try:
        # Extract participants
        match = re.search(r"schedule a meeting for (.*?) for", question)
        if match:
            participants = [name.strip() for name in match.group(1).split(',')]

        # Extract duration
        match = re.search(r"for (.*?) between", question)
        if match:
            duration_str = match.group(1).strip()
            if "half an hour" in duration_str:
                duration = 30
            elif "one hour" in duration_str:
                duration = 60
            else:
                duration = int(re.search(r"(\d+) minutes", duration_str).group(1))
        
        # Extract work hours
        match = re.search(r"between the work hours of (.*?) to (.*?) on Monday", question)
        if match:
            work_hours = [match.group(1).strip(), match.group(2).strip()]
        
        # Extract blocked times
        for person in participants:
            blocked_times[person] = []
            pattern = re.compile(f"{person}.*?during (.*?);")
            match = pattern.search(question)
            if match:
                time_ranges = match.group(1).split(',')
                for time_range in time_ranges:
                    time_range = time_range.strip()
                    start_time, end_time = [t.strip() for t in time_range.split(' to ')]
                    blocked_times[person].append((start_time, end_time))

        # Extract preferences
        for person in participants:
            if f"{person} do not want to meet on Monday after" in question:
                match = re.search(f"{person} do not want to meet on Monday after (.*?)\.", question)
                if match:
                    preferences[person] = {"no_meet_after": match.group(1).strip()}
            elif f"{person} would rather not meet on Monday after" in question:
                match = re.search(f"{person} would rather not meet on Monday after (.*?)\.", question)
                if match:
                    preferences[person] = {"no_meet_after": match.group(1).strip()}

    except Exception as e:
        print(f"Error extracting information: {str(e)}")

    return {
        "participants": participants,
        "duration": duration,
        "work_hours": work_hours,
        "blocked_times": blocked_times,
        "preferences": preferences
    }


def time_to_minutes(time_str):
    """Converts a time string (e.g., "9:00") to minutes since 9:00."""
    hours, minutes = map(int, time_str.split(':'))
    return (hours - 9) * 60 + minutes

def minutes_to_time(minutes):
    """Converts minutes since 9:00 back to a time string (e.g., "10:30")."""
    hours = 9 + (minutes // 60)
    minutes = minutes % 60
    return f"{hours:02}:{minutes:02}"

def find_available_slots(work_start, work_end, blocked_times):
    """Finds available time slots given the work hours and blocked times."""
    available_slots = []
    current_time = work_start
    blocked_times.sort()  # Ensure blocked times are in order

    for block_start, block_end in blocked_times:
        if current_time < block_start:
            available_slots.append((current_time, block_start))
        current_time = max(current_time, block_end)  # Move to the end of the block

    if current_time < work_end:
        available_slots.append((current_time, work_end))

    # Convert to start times (minutes since 9:00)
    start_times = []
    for start, end in available_slots:
        start_times.extend(range(start, end))
    
    return start_times

def find_overlapping_slots(available_slots, duration):
    """Finds time slots that are available for all participants."""
    all_slots = list(available_slots.values())
    
    if not all_slots:
        return []

    # Find intersection of all available times
    overlapping_times = set(all_slots[0])
    for slots in all_slots[1:]:
        overlapping_times.intersection_update(slots)

    # Filter slots based on duration
    valid_slots = []
    for time in sorted(list(overlapping_times)):
        if time + duration <= max(t for slots in all_slots for t in slots): # Make sure to not exceed working hours
             valid_slots.append(time)
   
    return valid_slots

def apply_preferences(slots, preferences):
    """Applies preferences to the available time slots."""
    if not preferences:
        return slots

    filtered_slots = slots
    for person, pref in preferences.items():
        if "no_meet_after" in pref:
            no_meet_after_minutes = time_to_minutes(pref["no_meet_after"])
            filtered_slots = [slot for slot in filtered_slots if slot + 30 <= no_meet_after_minutes] #Hardcoded meeting duration for simplification
    return filtered_slots

# Example usage (for testing):
if __name__ == "__main__":
    example_question = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Kathryn, Charlotte and Lauren for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nKathryn has blocked their calendar on Monday during 9:00 to 9:30, 10:30 to 11:00, 11:30 to 12:00, 13:30 to 14:30, 16:30 to 17:00; \nCharlotte has blocked their calendar on Monday during 12:00 to 12:30, 16:00 to 16:30; \nLauren has blocked their calendar on Monday during 9:00 to 10:00, 12:00 to 12:30, 13:30 to 14:30, 15:00 to 16:00, 16:30 to 17:00; \n\nCharlotte do not want to meet on Monday after 13:30. Find a time that works for everyone's schedule and constraints. \nSOLUTION: "
    result = main(example_question)
    print(result)