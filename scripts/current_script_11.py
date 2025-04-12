import re
import datetime

def main(question):
    """
    This script uses a simulated LLM approach with a "negotiation" strategy
    to find a suitable meeting time. It extracts information about participants
    and their availability and iteratively attempts to find a slot that works
    for everyone.  This approach simulates a negotiation by progressively
    relaxing constraints until a solution is found.

    The approach prioritizes LLM reasoning for decision-making while using
    Python for time calculations.
    """
    try:
        # 1. Extract information using LLM-simulated reasoning
        participants, schedules, duration, constraints = extract_info_llm(question)

        # 2. Initial proposed time based on a heuristic (e.g., earliest possible)
        proposed_time = find_initial_time(schedules, duration)

        # 3. Iterative negotiation (constraint relaxation)
        solution = negotiate_meeting_time(participants, schedules, duration, constraints, proposed_time)

        if solution:
            return f"Here is the proposed time: {solution}"
        else:
            return "No suitable meeting time found."

    except Exception as e:
        return f"Error: {str(e)}"


def extract_info_llm(question):
    """
    Simulates LLM information extraction.  Extracts participants, schedules,
    duration, and any additional constraints. LLM reasoning is simulated
    by prioritizing certain information and applying heuristics.

    Here, we'll extract information using regex and string manipulation,
    but imagine this function is implemented via calls to an actual LLM.
    """
    try:
        # Extract participants
        match = re.search(r"schedule a meeting for (.*?) for", question)
        if not match:
            raise ValueError("Could not extract participants.")
        participants = [name.strip() for name in match.group(1).split(',')]

        # Extract duration (assuming half an hour if not specified)
        duration = 30  # Default to 30 minutes
        if "hour" in question:
            duration_match = re.search(r"for (\w+\s?)hour", question)

            if duration_match:
                duration_str = duration_match.group(1).strip()
                if duration_str == "half":
                     duration = 30
                elif duration_str == "one":
                     duration = 60
                else:
                     raise ValueError("Could not parse meeting duration")
            else:
                duration_match = re.search(r"for (\d+) minutes", question)
                if duration_match:
                    duration = int(duration_match.group(1))
        # Extract schedules (simulating LLM understanding of schedules)
        schedules = {}
        schedule_lines = re.findall(r"(\w+) has meetings on Monday during (.*?);", question)
        for name, schedule_str in schedule_lines:
            schedule_list = []
            time_slots = schedule_str.split(", ")
            for slot in time_slots:
                start_time_str, end_time_str = slot.split(" to ")
                start_time = datetime.datetime.strptime(start_time_str, "%H:%M").time()
                end_time = datetime.datetime.strptime(end_time_str, "%H:%M").time()
                schedule_list.append((start_time, end_time))
            schedules[name] = schedule_list

        # Extract constraints (simulating LLM constraint extraction)
        constraints = {}
        if "avoid more meetings on Monday after" in question:
          pref_match = re.search(r"avoid more meetings on Monday after (\d+:\d+)", question)
          if pref_match:
            constraints["avoid_after"] = datetime.datetime.strptime(pref_match.group(1), "%H:%M").time()
        if "do not want to meet on Monday before" in question:
            pref_match = re.search(r"do not want to meet on Monday before (\d+:\d+)", question)
            if pref_match:
                constraints["avoid_before"] = datetime.datetime.strptime(pref_match.group(1), "%H:%M").time()

        return participants, schedules, duration, constraints

    except Exception as e:
        raise ValueError(f"Error extracting information: {str(e)}")


def find_initial_time(schedules, duration):
    """
    Simulates LLM suggesting an initial meeting time.  This is a heuristic-based
    suggestion based on available time slots.
    """
    start_time = datetime.time(9, 0) #Earliest possible time
    return start_time


def negotiate_meeting_time(participants, schedules, duration, constraints, proposed_start_time):
    """
    Simulates an LLM negotiating a meeting time by iteratively checking
    availability and relaxing constraints if needed.

    The negotiation involves checking for conflicts and adjusting the proposed
    meeting time.
    """
    current_time = proposed_start_time
    end_time = datetime.time(17, 0)

    while current_time <= end_time:
        # Calculate meeting end time
        current_datetime = datetime.datetime.combine(datetime.date.today(), current_time)
        meeting_end_datetime = current_datetime + datetime.timedelta(minutes=duration)
        meeting_end_time = meeting_end_datetime.time()

        # Check for time constraints
        if "avoid_before" in constraints and current_time < constraints["avoid_before"]:
            current_datetime = datetime.datetime.combine(datetime.date.today(), current_time)
            current_datetime += datetime.timedelta(minutes=30)
            current_time = current_datetime.time()
            continue

        # Check if the time works for all participants
        available = True
        for person in participants:
            if person in schedules:
                for busy_start, busy_end in schedules[person]:
                    if (current_time < busy_end and meeting_end_time > busy_start):
                        available = False
                        break
            if not available:
                break

        if available:
            # Check if the time adheres to the constraints
            if "avoid_after" in constraints and current_time > constraints["avoid_after"]:
                current_datetime = datetime.datetime.combine(datetime.date.today(), current_time)
                current_datetime += datetime.timedelta(minutes=30)
                current_time = current_datetime.time()
                continue

            meeting_start_str = current_time.strftime("%H:%M")
            meeting_end_str = meeting_end_time.strftime("%H:%M")
            return f"Monday, {meeting_start_str} - {meeting_end_str} "

        # If not available, increment the time and try again (simulating negotiation)
        current_datetime = datetime.datetime.combine(datetime.date.today(), current_time)
        current_datetime += datetime.timedelta(minutes=30)
        current_time = current_datetime.time()

    return None