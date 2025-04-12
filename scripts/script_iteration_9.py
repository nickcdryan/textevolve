import re
import datetime

def main(question):
    """
    Simulates an LLM-driven meeting scheduler using a constraint satisfaction approach,
    where we explicitly define the constraints and use a backtracking algorithm to find a solution.
    This approach differs from previous attempts by focusing on a more programmatic,
    constraint-based search for a valid meeting time rather than relying heavily on 
    regex or simulated LLM reasoning steps.

    Args:
        question (str): The question string containing meeting scheduling details.

    Returns:
        str: The answer string with the proposed meeting time, or an error message if no solution is found.
    """

    try:
        # Step 1: Information Extraction using LLM-style prompting (simulated)
        task_description, participants, schedules, constraints = extract_information(question)

        # Step 2: Define Constraints
        work_hours_start = datetime.time(9, 0)
        work_hours_end = datetime.time(17, 0)
        meeting_duration = datetime.timedelta(minutes=30)
        day = "Monday"

        # Step 3: Generate Possible Time Slots (Constraint Generation)
        possible_time_slots = generate_time_slots(work_hours_start, work_hours_end, meeting_duration)

        # Step 4: Check each time slot against constraints (Constraint Satisfaction)
        for start_time in possible_time_slots:
            end_time = (datetime.datetime.combine(datetime.date.today(), start_time) + meeting_duration).time()  # Combine with date for timedelta addition
            is_valid = True

            # Check work hours constraint
            if not (work_hours_start <= start_time < work_hours_end and work_hours_start < end_time <= work_hours_end):
                is_valid = False
                continue

            # Check participant schedules constraint
            for participant, busy_times in schedules.items():
                for busy_start_str, busy_end_str in busy_times:
                    busy_start = parse_time(busy_start_str)
                    busy_end = parse_time(busy_end_str)

                    if (start_time >= busy_start and start_time < busy_end) or (end_time > busy_start and end_time <= busy_end) or (start_time <= busy_start and end_time >= busy_end):
                        is_valid = False
                        break
                if not is_valid:
                    break  # Move to the next time slot if any participant is busy

            # Check any additional constraints
            if constraints:
                for constraint in constraints:
                    participant_name, not_before_time_str = constraint
                    if "before" in question:
                        not_before_time_str = not_before_time_str.replace("before", "").strip()
                    not_before_time = parse_time(not_before_time_str)

                    if participant_name in participants and start_time < not_before_time:
                        is_valid = False
                        break

            if is_valid:
                # Step 5: Solution Found!
                return f"Here is the proposed time: {day}, {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')} "

        # Step 6: No Solution Found
        return "Error: No suitable meeting time found that satisfies all constraints."

    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"


def extract_information(question):
    """
    Extracts information from the question using a combination of regular expressions
    and string manipulation.  This aims to mimic what an LLM might do, but in a more
    controlled and explicit manner.

    Args:
        question (str): The question string.

    Returns:
        tuple: A tuple containing task description, participants, schedules, and constraints.
    """
    task_description = question.split("TASK:")[1].split("Here are the existing schedules")[0].strip() if "TASK:" in question else ""
    participants = []
    if "schedule a meeting for" in question:
        match = re.search(r"schedule a meeting for (.*?) for", question)
        if match:
            participants = [name.strip() for name in match.group(1).split(',')]

    schedules = {}
    lines = question.split("\n")
    for line in lines:
        if "has blocked their calendar on Monday during" in line or "has meetings on Monday during" in line:
            participant = line.split("has")[0].strip()
            schedule_str = line.split("during")[1].strip().rstrip(";")
            time_ranges = re.findall(r"(\d{1,2}:\d{2})\s*to\s*(\d{1,2}:\d{2})", schedule_str)
            schedules[participant] = time_ranges

    constraints = []
    if "can not meet on Monday before" in question:
        constraint_line = next((line for line in lines if "can not meet on Monday before" in line), None)
        if constraint_line:
            participant = constraint_line.split("can not meet")[0].strip()
            time = constraint_line.split("before")[1].strip().rstrip(".")
            constraints.append((participant, time))

    return task_description, participants, schedules, constraints


def generate_time_slots(start_time, end_time, duration):
    """
    Generates a list of possible time slots between start_time and end_time,
    with the given duration.

    Args:
        start_time (datetime.time): The start time.
        end_time (datetime.time): The end time.
        duration (datetime.timedelta): The duration of each time slot.

    Returns:
        list: A list of datetime.time objects representing the start times of the time slots.
    """
    time_slots = []
    current_time = start_time
    while current_time < end_time:
        time_slots.append(current_time)
        current_datetime = datetime.datetime.combine(datetime.date.today(), current_time)
        current_datetime += duration
        current_time = current_datetime.time()
    return time_slots


def parse_time(time_str):
    """
    Parses a time string in the format "HH:MM" and returns a datetime.time object.

    Args:
        time_str (str): The time string.

    Returns:
        datetime.time: A datetime.time object representing the time.
    """
    try:
        return datetime.datetime.strptime(time_str, "%H:%M").time()
    except ValueError:
        try:
            return datetime.datetime.strptime(time_str, "%I:%M %p").time()
        except ValueError:
            return datetime.datetime.strptime(time_str, "%I%p").time()


# Example usage (for testing)
if __name__ == "__main__":
    question = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Eric, Ashley, Ronald and Larry for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nErichas no meetings the whole day.\nAshley is busy on Monday during 10:00 to 10:30, 11:00 to 12:00, 12:30 to 13:00, 15:00 to 16:00; \nRonald has meetings on Monday during 9:00 to 9:30, 10:00 to 11:30, 12:30 to 14:00, 14:30 to 17:00; \nLarry has blocked their calendar on Monday during 9:00 to 12:00, 13:00 to 17:00; \n\nFind a time that works for everyone's schedule and constraints. \nSOLUTION: "
    answer = main(question)
    print(answer)  # Expected output: Here is the proposed time: Monday, 12:00 - 12:30

    question2 = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Diana, Gerald, Timothy and Julie for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nDiana has blocked their calendar on Monday during 10:30 to 11:00, 11:30 to 12:00, 13:30 to 14:00, 14:30 to 15:00, 16:00 to 17:00; \nGerald has meetings on Monday during 9:30 to 10:00, 12:30 to 13:00, 15:00 to 15:30, 16:00 to 16:30; \nTimothy is busy on Monday during 10:00 to 12:00, 12:30 to 14:00, 14:30 to 15:00, 16:00 to 17:00; \nJulie has meetings on Monday during 9:30 to 10:00, 10:30 to 11:00, 11:30 to 15:30, 16:00 to 17:00; \n\nDiana can not meet on Monday before 9:30. Find a time that works for everyone's schedule and constraints. \nSOLUTION: "
    answer2 = main(question2)
    print(answer2)