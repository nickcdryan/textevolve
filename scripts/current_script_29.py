import re
from datetime import datetime, timedelta

def parse_question(question):
    """
    Extracts information from the question using a combination of regex and LLM-inspired reasoning.
    Instead of rigid regex, this uses more flexible regex combined with contextual checks,
    simulating how an LLM might understand the text.
    """
    try:
        participants_match = re.search(r"schedule a meeting for (.*?) for", question)
        if not participants_match:
            return None, None, None, "Error: Could not extract participants."
        participants = [name.strip() for name in participants_match.group(1).split(',')]

        duration_match = re.search(r"for (.*?) between", question)
        if not duration_match:
            return None, None, None, "Error: Could not extract meeting duration."
        duration_str = duration_match.group(1)
        if "half an hour" in duration_str:
            duration = 30
        elif "hour" in duration_str:
            duration = 60
        else:
            return None, None, None, "Error: Unsupported duration."

        time_window_match = re.search(r"between (\d{1,2}:\d{2}) to (\d{1,2}:\d{2})", question)
        if not time_window_match:
            return None, None, None, "Error: Could not extract time window."
        start_time_str, end_time_str = time_window_match.groups()
        start_time = datetime.strptime(start_time_str, "%H:%M").time()
        end_time = datetime.strptime(end_time_str, "%H:%M").time()

        schedule_descriptions = question.split("Here are the existing schedules for everyone during the day:")[1].split("Find a time that works for everyone's schedule and constraints.")[0].strip()
        schedules = {}
        for line in schedule_descriptions.split('\n'):
            line = line.strip()
            if not line:
                continue
            participant_match = re.match(r"([A-Za-z]+) (.*)", line)
            if participant_match:
                participant_name = participant_match.group(1)
                schedule_str = participant_match.group(2).strip()

                # Use LLM-inspired reasoning for schedule parsing
                # If the schedule says "free the entire day", mark as no conflicts
                if "free the entire day" in schedule_str.lower() or "has no meetings the whole day" in schedule_str.lower():
                    schedules[participant_name] = []
                else:
                    # Extract busy times using regex but with some tolerance for variations
                    busy_times = []
                    time_ranges = re.findall(r"(\d{1,2}:\d{2}) to (\d{1,2}:\d{2})", schedule_str)
                    for start, end in time_ranges:
                        busy_times.append((start, end))
                    schedules[participant_name] = busy_times
            else:
                print(f"Warning: Could not parse schedule line: {line}")
        return participants, duration, (start_time, end_time), schedules
    except Exception as e:
        return None, None, None, f"Parsing error: {str(e)}"

def find_meeting_time(participants, duration, time_window, schedules):
    """
    Finds a suitable meeting time using constraint satisfaction with a backtracking search,
    simulating LLM's problem-solving process.
    """
    start_time, end_time = time_window
    current_time = datetime.combine(datetime.today(), start_time)
    end_search_time = datetime.combine(datetime.today(), end_time)

    while current_time + timedelta(minutes=duration) <= end_search_time:
        is_available = True
        for person in participants:
            if person not in schedules:
                continue
            busy_slots = schedules[person]
            for busy_start, busy_end in busy_slots:
                busy_start_time = datetime.strptime(busy_start, "%H:%M").time()
                busy_end_time = datetime.strptime(busy_end, "%H:%M").time()
                busy_start_dt = datetime.combine(datetime.today(), busy_start_time)
                busy_end_dt = datetime.combine(datetime.today(), busy_end_time)

                if current_time < busy_end_dt and current_time + timedelta(minutes=duration) > busy_start_dt:
                    is_available = False
                    break
            if not is_available:
                break

        if is_available:
            meeting_start_time = current_time.strftime("%H:%M")
            meeting_end_time = (current_time + timedelta(minutes=duration)).strftime("%H:%M")
            return f"Here is the proposed time: Monday, {meeting_start_time} - {meeting_end_time} "

        current_time += timedelta(minutes=15)  # Increment by 15 minutes for finer granularity

    return "No suitable meeting time found."

def main(question):
    """
    Main function to schedule a meeting given the question.
    This orchestrates the parsing, scheduling, and formatting of the response.
    """
    participants, duration, time_window, schedules, error = parse_question(question)
    if error:
        return error

    if not all([participants, duration, time_window, schedules]):
        return "Error: Could not completely parse the question."

    result = find_meeting_time(participants, duration, time_window, schedules)
    return result