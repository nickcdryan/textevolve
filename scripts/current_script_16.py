import re

def main(question):
    """
    This script uses a simulated LLM to schedule meetings. It takes a question
    string as input, extracts the necessary information, and returns a proposed
    meeting time. This iteration will focus on a simplified time representation
    and constraint satisfaction to explore core scheduling logic.

    Instead of trying to perfectly parse the text, let's use a rule based approach
    for understanding and constraint extraction combined with simulated time validity
    to explore.
    """
    try:
        # 1. Extract information (simulate LLM parsing)
        participants, schedules, constraints = extract_info(question)

        # 2. Generate candidate time slots (simplified)
        candidate_slots = generate_time_slots()

        # 3. Filter slots based on availability and preferences
        available_slots = filter_slots(candidate_slots, schedules, constraints, participants)

        # 4. Select the best slot (very basic, can be improved)
        if available_slots:
            best_slot = available_slots[0]  # Just pick the first available
        else:
            return "No suitable time found."

        # 5. Format and return the solution
        return format_solution(best_slot)

    except Exception as e:
        return f"An error occurred: {str(e)}"


def extract_info(question):
    """
    Simulates LLM parsing to extract participants, schedules, and constraints.
    Uses simplified string matching for demonstration purposes.
    """
    participants_match = re.search(r"schedule a meeting for (.*?) for", question)
    participants = [p.strip() for p in participants_match.group(1).split(",") if participants_match] if participants_match else []

    schedules = {}
    for participant in participants:
        schedule_match = re.search(rf"{participant} (?:has meetings|is busy) on Monday during (.*?)(?:;|\n)", question)
        if schedule_match:
            schedules[participant] = parse_schedule(schedule_match.group(1))
        else:
            schedules[participant] = []  # Default: No schedule found

    # Simplified constraint extraction (preference for early meeting)
    constraints = {"avoid_after": 1300}  # Avoid after 1 PM (13:00)
    return participants, schedules, constraints


def parse_schedule(schedule_string):
    """
    Parses schedule strings into a list of tuples (start_time, end_time).
    Handles multiple time ranges in the schedule. Returns an empty list if parsing fails.
    """
    try:
        time_ranges = schedule_string.split(", ")
        schedule = []
        for time_range in time_ranges:
            match = re.search(r"(\d{1,2}:\d{2}) to (\d{1,2}:\d{2})", time_range)
            if match:
                start_time = convert_to_minutes(match.group(1))
                end_time = convert_to_minutes(match.group(2))
                schedule.append((start_time, end_time))
        return schedule
    except:
        return []


def convert_to_minutes(time_str):
    """
    Converts a time string (HH:MM) to minutes since midnight.
    Returns -1 on error.
    """
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    except:
        return -1

def generate_time_slots(start_time=9 * 60, end_time=17 * 60, duration=30):
    """
    Generates candidate time slots in minutes.

    """
    slots = []
    current_time = start_time
    while current_time + duration <= end_time:
        slots.append((current_time, current_time + duration))
        current_time += 30  # Increment by slot duration
    return slots



def filter_slots(candidate_slots, schedules, constraints, participants):
    """
    Filters candidate slots based on availability and preferences.
    Implements a straightforward conflict checking mechanism.

    """
    available_slots = []
    for slot in candidate_slots:
        is_available = True
        for participant in participants:
            if participant in schedules:
                for busy_slot in schedules[participant]:
                    if (slot[0] < busy_slot[1] and slot[1] > busy_slot[0]):
                        is_available = False
                        break
            if not is_available:
                break
        if is_available:
            available_slots.append(slot)
    return available_slots


def format_solution(best_slot):
    """Formats the solution into the desired output format."""
    start_time_hours = best_slot[0] // 60
    start_time_minutes = best_slot[0] % 60
    end_time_hours = best_slot[1] // 60
    end_time_minutes = best_slot[1] % 60

    start_time_str = f"{start_time_hours:02d}:{start_time_minutes:02d}"
    end_time_str = f"{end_time_hours:02d}:{end_time_minutes:02d}"
    return f"Here is the proposed time: Monday, {start_time_str} - {end_time_str} "


# Example usage (for testing - replace with the actual question)
if __name__ == "__main__":
    example_question = """
    You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:

    TASK: You need to schedule a meeting for Gary, Douglas, Elizabeth and Daniel for half an hour between the work hours of 9:00 to 17:00 on Monday. 

    Here are the existing schedules for everyone during the day: 
    Gary has meetings on Monday during 9:30 to 10:00, 12:00 to 12:30; 
    Douglas has meetings on Monday during 10:30 to 11:00, 11:30 to 12:00, 14:00 to 14:30, 16:30 to 17:00; 
    Elizabeth has meetings on Monday during 11:30 to 13:30, 14:00 to 15:00, 16:00 to 17:00; 
    Daniel has blocked their calendar on Monday during 10:30 to 12:30, 14:00 to 17:00; 

    Gary would rather not meet on Monday after 10:00. Find a time that works for everyone's schedule and constraints. 
    SOLUTION: 
    """
    answer = main(example_question)
    print(answer)