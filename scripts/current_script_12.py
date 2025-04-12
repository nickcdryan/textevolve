import re
import datetime

def main(question):
    """
    Schedules a meeting by simulating a multi-agent negotiation process,
    where each participant is an agent with their own constraints. This
    approach differs from previous attempts by using a more sophisticated
    time representation and negotiation loop.

    Args:
        question (str): A string containing the scheduling problem.

    Returns:
        str: A string containing the proposed meeting time.
    """

    try:
        # 1. Extract information using LLM-like reasoning with regex (for simplicity)
        participants = extract_participants(question)
        schedules = extract_schedules(question, participants)
        duration = extract_duration(question)  # Duration in minutes
        work_hours_start, work_hours_end = extract_work_hours(question)

        # 2. Represent time as datetime objects for easier manipulation
        start_time = datetime.datetime.strptime(work_hours_start, "%H:%M").time()
        end_time = datetime.datetime.strptime(work_hours_end, "%H:%M").time()
        current_time = datetime.datetime.combine(datetime.date.today(), start_time)
        end_of_day = datetime.datetime.combine(datetime.date.today(), end_time)

        # 3. Negotiation Loop: Iterate through time slots until a consensus is reached
        while current_time + datetime.timedelta(minutes=duration) <= end_of_day:
            is_available = True
            for participant in participants:
                if has_conflict(current_time, duration, schedules[participant]):
                    is_available = False
                    break

            if is_available:
                meeting_start_str = current_time.strftime("%H:%M")
                meeting_end_str = (current_time + datetime.timedelta(minutes=duration)).strftime("%H:%M")
                return f"Here is the proposed time: Monday, {meeting_start_str} - {meeting_end_str} "

            current_time += datetime.timedelta(minutes=30)  # Increment by 30 minutes

        return "No suitable time found within the given constraints."

    except Exception as e:
        return f"An error occurred: {str(e)}"


def extract_participants(question):
    """
    Extracts the list of participants from the question string.

    Args:
        question (str): The scheduling problem description.

    Returns:
        list: A list of participant names.
    """
    match = re.search(r"schedule a meeting for (.*?) for", question)
    if match:
        return [name.strip() for name in match.group(1).split(',')]
    return []

def extract_schedules(question, participants):
    """
    Extracts the schedules for each participant from the question string.

    Args:
        question (str): The scheduling problem description.
        participants (list): A list of participant names.

    Returns:
        dict: A dictionary where keys are participant names and values are lists of
              time intervals representing their busy times.
    """
    schedules = {}
    for participant in participants:
        pattern = rf"{participant}.*?(Monday during (.*?);|calendar is wide open the entire day.)"
        match = re.search(pattern, question)
        if match:
            schedule_str = match.group(2)
            if schedule_str is None:  # calendar is wide open
                schedules[participant] = []
            else:
                time_intervals = []
                interval_strings = schedule_str.split(', ')
                for interval_str in interval_strings:
                    time_match = re.search(r"(\d{1,2}:\d{2}) to (\d{1,2}:\d{2})", interval_str)
                    if time_match:
                        start_time = time_match.group(1)
                        end_time = time_match.group(2)
                        time_intervals.append((start_time, end_time))
                schedules[participant] = time_intervals
        else:
            schedules[participant] = []  # Default to empty schedule if not found
    return schedules

def extract_duration(question):
    """
    Extracts the meeting duration from the question string.

    Args:
        question (str): The scheduling problem description.

    Returns:
        int: The meeting duration in minutes.
    """
    match = re.search(r"for (.*?) between", question)
    if match:
        duration_str = match.group(1).strip()
        if "half an hour" in duration_str:
            return 30
        elif "one hour" in duration_str:
            return 60
    return 30  # Default duration

def extract_work_hours(question):
    """
    Extracts the work hours from the question string.

    Args:
        question (str): The scheduling problem description.

    Returns:
        tuple: A tuple containing the start and end work hours in HH:MM format.
    """
    match = re.search(r"between the work hours of (.*?) to (.*?) on Monday", question)
    if match:
        return match.group(1), match.group(2)
    return "09:00", "17:00"  # Default work hours

def has_conflict(current_time, duration, schedule):
    """
    Checks if the proposed meeting time conflicts with a participant's schedule.

    Args:
        current_time (datetime): The proposed meeting start time.
        duration (int): The meeting duration in minutes.
        schedule (list): A list of time intervals representing the participant's busy times.

    Returns:
        bool: True if there is a conflict, False otherwise.
    """
    meeting_start = current_time.time()
    meeting_end = (current_time + datetime.timedelta(minutes=duration)).time()

    for start_str, end_str in schedule:
        schedule_start = datetime.datetime.strptime(start_str, "%H:%M").time()
        schedule_end = datetime.datetime.strptime(end_str, "%H:%M").time()

        # Convert everything to datetime objects for comparison
        current_datetime = datetime.datetime.combine(datetime.date.today(), current_time.time())
        schedule_start_datetime = datetime.datetime.combine(datetime.date.today(), schedule_start)
        schedule_end_datetime = datetime.datetime.combine(datetime.date.today(), schedule_end)

        meeting_start_datetime = datetime.datetime.combine(datetime.date.today(), meeting_start)
        meeting_end_datetime = datetime.datetime.combine(datetime.date.today(), meeting_end)
    
        if meeting_start_datetime < schedule_end_datetime and meeting_end_datetime > schedule_start_datetime:
            return True
    return False


# Example Usage (for testing - not needed in the final submission)
if __name__ == "__main__":
    question1 = """You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:

TASK: You need to schedule a meeting for John, Ralph, Daniel and Keith for half an hour between the work hours of 9:00 to 17:00 on Monday. 

Here are the existing schedules for everyone during the day: 
John has meetings on Monday during 9:00 to 9:30, 10:00 to 10:30, 11:00 to 11:30, 13:00 to 14:30; 
Ralph has meetings on Monday during 13:30 to 14:00, 14:30 to 15:00, 15:30 to 16:30; 
Daniel has blocked their calendar on Monday during 9:00 to 9:30, 10:30 to 11:00, 12:00 to 13:00, 13:30 to 16:30; 
Keith has blocked their calendar on Monday during 10:30 to 14:30, 15:00 to 17:00; 

Find a time that works for everyone's schedule and constraints. 
SOLUTION: """
    answer1 = main(question1)
    print(answer1)

    question2 = """You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:

TASK: You need to schedule a meeting for Joe, Diana, Harold and Philip for half an hour between the work hours of 9:00 to 17:00 on Monday. 

Here are the existing schedules for everyone during the day: 
Joe's calendar is wide open the entire day.
Diana has blocked their calendar on Monday during 10:30 to 11:00, 12:30 to 13:00, 14:30 to 15:00, 15:30 to 16:00; 
Harold is busy on Monday during 10:00 to 11:00, 11:30 to 16:30; 
Philip has meetings on Monday during 9:00 to 9:30, 10:30 to 12:00, 12:30 to 13:30, 14:00 to 17:00; 

Find a time that works for everyone's schedule and constraints. 
SOLUTION: """
    answer2 = main(question2)
    print(answer2)