import re
import datetime

def main(question):
    """
    This script employs a simulated LLM approach to meeting scheduling,
    focusing on information extraction via string parsing and prioritizing
    later meeting times while avoiding conflicts.

    It extracts participants, schedules, and constraints from the question.
    Then it finds a suitable time, starting from the latest possible time slot
    and iterating backward. This approach is specifically designed to explore
    the possibility of favoring later meeting times when multiple slots are available.
    """

    try:
        # Extract meeting details from the question
        participants = extract_participants(question)
        schedules = extract_schedules(question)
        constraints = extract_constraints(question)

        # Set default values and extract duration
        work_hours_start = 9
        work_hours_end = 17
        meeting_duration = 0.5  # default to 30 minutes

        duration_match = re.search(r"for (\w+(?:\s\w+)?(?:-minute|-hour)?)", question)
        if duration_match:
            duration_str = duration_match.group(1)
            if "hour" in duration_str:
                try:
                    if "half" in duration_str:
                        meeting_duration = 0.5
                    elif "quarter" in duration_str:
                        meeting_duration = 0.25
                    else:
                        num_hours = re.search(r"(\d+)", duration_str)
                        if num_hours:
                            meeting_duration = float(num_hours.group(1))
                        else:
                            meeting_duration = 1.0  # Default to 1 hour if no number is found
                except:
                    meeting_duration = 0.5
            elif "minute" in duration_str:
                try:
                    num_minutes = re.search(r"(\d+)", duration_str)
                    if num_minutes:
                        meeting_duration = float(num_minutes.group(1)) / 60.0
                    else:
                        meeting_duration = 0.5
                except:
                    meeting_duration = 0.5

        # Find a suitable meeting time, prioritizing later times
        proposed_time = find_suitable_time_prioritize_late(
            participants, schedules, work_hours_start, work_hours_end, meeting_duration, constraints
        )

        if proposed_time:
            return f"Here is the proposed time: Monday, {proposed_time}"
        else:
            return "No suitable time found."

    except Exception as e:
        return f"Error: {str(e)}"


def extract_participants(question):
    """Extracts the names of the participants from the question."""
    try:
        match = re.search(r"schedule a meeting for (.*?) for", question)
        if match:
            participants = [name.strip() for name in match.group(1).split(",")]
            return participants
        else:
            return []
    except:
        return []


def extract_schedules(question):
    """Extracts the schedules of each participant from the question."""
    schedules = {}
    try:
        lines = question.split("\n")
        for line in lines:
            if "has blocked their calendar on Monday during" in line or "is busy on Monday during" in line:
                name = line.split(" ")[0]
                schedule_str = line.split("during ")[1].strip()
                schedule_str = schedule_str.replace(';', '')
                time_slots = schedule_str.split(", ")
                schedules[name] = []
                for slot in time_slots:
                    start, end = slot.split(" to ")
                    schedules[name].append((start, end))
    except:
        pass
    return schedules


def extract_constraints(question):
    """Extracts any additional constraints mentioned in the question."""
    constraints = []
    try:
        if "would rather not meet on Monday before" in question:
            match = re.search(r"would rather not meet on Monday before (\d+:\d+)", question)
            if match:
                constraints.append(("time_preference", match.group(1)))
    except:
        pass
    return constraints


def find_suitable_time_prioritize_late(participants, schedules, work_hours_start, work_hours_end, meeting_duration, constraints):
    """
    Finds a suitable meeting time by checking availability, prioritizing later times.
    """
    available_time_slots = []
    for hour in range(work_hours_start, work_hours_end):
        for minute in [0, 30]:
            start_time = datetime.time(hour, minute)
            end_hour = hour
            end_minute = minute + int(meeting_duration * 60)
            if end_minute >= 60:
                end_hour += 1
                end_minute -= 60
            if end_hour > work_hours_end:
                continue

            end_time = datetime.time(end_hour, end_minute)

            is_available = True
            for participant in participants:
                if participant in schedules:
                    for busy_slot_start, busy_slot_end in schedules[participant]:
                        try:
                            busy_start_hour, busy_start_minute = map(int, busy_slot_start.split(':'))
                            busy_end_hour, busy_end_minute = map(int, busy_slot_end.split(':'))

                            busy_start_time = datetime.time(busy_start_hour, busy_start_minute)
                            busy_end_time = datetime.time(busy_end_hour, busy_end_minute)

                            if start_time < busy_end_time and end_time > busy_start_time:
                                is_available = False
                                break
                        except ValueError:
                            is_available = False
                            break
                if not is_available:
                    break

            if is_available:
                available_time_slots.append((start_time, end_time))

    # Prioritize later meeting times
    available_time_slots.sort(reverse=True)

    # Apply preferences if any
    preferred_time = None
    for start_time, end_time in available_time_slots:
        valid_time = True
        for constraint_type, constraint_value in constraints:
            if constraint_type == "time_preference":
                pref_hour, pref_minute = map(int, constraint_value.split(':'))
                pref_time = datetime.time(pref_hour, pref_minute)
                if start_time < pref_time:
                    valid_time = False
                    break
        if valid_time:
            preferred_time = start_time.strftime("%H:%M") + " - " + end_time.strftime("%H:%M")
            break

    return preferred_time


# Example usage (for testing)
if __name__ == "__main__":
    question = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Aaron, Sarah, Martha and Heather for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nAaron has blocked their calendar on Monday during 9:00 to 9:30, 11:30 to 12:00, 12:30 to 14:00, 15:30 to 16:00; \nSarah is busy on Monday during 10:30 to 11:30, 12:30 to 13:00, 13:30 to 14:30, 16:00 to 16:30; \nMartha is busy on Monday during 9:00 to 9:30, 10:30 to 12:00, 12:30 to 13:30, 14:00 to 14:30, 15:30 to 17:00; \nHeather has meetings on Monday during 9:00 to 10:00, 11:30 to 12:00, 13:00 to 14:30, 15:00 to 15:30, 16:00 to 16:30; \n\nSarah would rather not meet on Monday before 13:30. Find a time that works for everyone's schedule and constraints. \nSOLUTION: "
    answer = main(question)
    print(answer)