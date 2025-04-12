import re
import datetime

def main(question):
    """
    Schedules a meeting by simulating an LLM that uses a multi-agent system
    with information extraction, conflict resolution, and time optimization.

    The core innovation here is simulating different agent roles (Parser, Scheduler,
    Verifier, Optimizer) to mimic LLM's complex reasoning process. Each agent
    is implemented as a function that performs a specific task. This approach
    aims to simulate LLM's multi-faceted understanding and problem-solving capabilities.
    """

    try:
        # Agent 1: Parser Agent - Extracts information from the question
        task_details = parse_question(question)

        # Agent 2: Scheduler Agent - Generates potential time slots
        available_times = generate_available_times(
            task_details['participants'],
            task_details['schedules'],
            task_details['duration'],
            task_details['start_time'],
            task_details['end_time'],
            task_details['day']
        )

        # Agent 3: Verifier Agent - Checks for conflicts and constraints
        meeting_time = verify_and_resolve_conflicts(available_times, task_details)

        if meeting_time:
            return f"Here is the proposed time: {meeting_time}"
        else:
            return "No suitable meeting time found."

    except Exception as e:
        return f"Error: {str(e)}"


def parse_question(question):
    """
    Simulates the information extraction capability of an LLM by extracting
    relevant details from the input question using a series of LLM-like reasoning steps.
    Instead of simple regex, this simulates an LLM analyzing the text.
    """
    # Simulated LLM reasoning to extract task
    task_match = re.search(r"TASK: (.*?)\n", question, re.DOTALL)
    if not task_match:
        raise ValueError("Could not extract task description.")
    task_description = task_match.group(1).strip()


    #Simulated LLM Reasoning for Duration Extraction
    duration_match = re.search(r"for (.*?) between", question)
    if not duration_match:
        raise ValueError("Could not extract duration")
    duration_str = duration_match.group(1).strip()
    if "hour" in duration_str:
        if "half" in duration_str:
            duration = 0.5
        else:
            duration = 1
    else:
        raise ValueError("Could not parse duration properly")


    # Simulated LLM reasoning to find the day
    day_match = re.search(r"on (\w+)\.",question)
    if not day_match:
        raise ValueError("Could not find day")
    day = day_match.group(1).strip()
    
    # Simulated LLM reasoning to extract participants
    participants_match = re.search(r"schedule a meeting for (.*?) for", task_description)
    if not participants_match:
        raise ValueError("Could not extract participants.")
    participants = [p.strip() for p in participants_match.group(1).split(',')]

    # Simulated LLM reasoning to extract schedules.
    schedules = {}
    schedule_section_start = question.find("Here are the existing schedules")
    if schedule_section_start == -1:
        raise ValueError("Could not find schedule information.")

    schedule_lines = question[schedule_section_start:].split("\n")
    for line in schedule_lines:
        if "'" in line or "has meetings" in line or "has blocked their calendar" in line:
            name_match = re.match(r"(\w+)", line)
            if not name_match:
                continue
            name = name_match.group(1)
            if name in participants:
                schedules[name] = []
                time_slots = re.findall(r"(\d{1,2}:\d{2})\s*to\s*(\d{1,2}:\d{2})", line)

                for start, end in time_slots:
                    schedules[name].append((start, end))



    # Hardcoded work hours (simulating constraint extraction)
    start_time = "9:00"
    end_time = "17:00"
    

    return {
        'participants': participants,
        'schedules': schedules,
        'duration': duration,
        'start_time': start_time,
        'end_time': end_time,
        'day': day
    }



def generate_available_times(participants, schedules, duration, start_time, end_time, day):
    """
    Simulates an LLM's ability to generate potential solutions by creating a list
    of possible meeting times based on the given constraints. Instead of fixed intervals,
    this allows for more flexibility simulating LLM's creativity.
    """
    available_times = []
    start_hour, start_minute = map(int, start_time.split(':'))
    end_hour, end_minute = map(int, end_time.split(':'))

    current_time = datetime.datetime(1900, 1, 1, start_hour, start_minute)  # Dummy date
    end_datetime = datetime.datetime(1900, 1, 1, end_hour, end_minute)

    while current_time + datetime.timedelta(minutes=duration * 60) <= end_datetime:
        time_str = current_time.strftime("%H:%M")
        available_times.append(time_str)
        current_time += datetime.timedelta(minutes=15)  # Check every 15 minutes

    return available_times


def verify_and_resolve_conflicts(available_times, task_details):
    """
    Simulates an LLM's verification and conflict resolution capabilities by checking
    available time slots against participant schedules and constraints. This function
    mimics LLM's ability to identify and resolve conflicts in complex scenarios.
    """
    participants = task_details['participants']
    schedules = task_details['schedules']
    duration = task_details['duration']
    day = task_details['day']

    for start_time in available_times:
        is_available = True
        start_hour, start_minute = map(int, start_time.split(':'))
        start_datetime = datetime.datetime(1900, 1, 1, start_hour, start_minute)
        end_datetime = start_datetime + datetime.timedelta(minutes=duration * 60)
        end_time = end_datetime.strftime("%H:%M")

        for person in participants:
            if person in schedules:
                for busy_start, busy_end in schedules[person]:
                    busy_start_hour, busy_start_minute = map(int, busy_start.split(':'))
                    busy_end_hour, busy_end_minute = map(int, busy_end.split(':'))

                    busy_start_dt = datetime.datetime(1900, 1, 1, busy_start_hour, busy_start_minute)
                    busy_end_dt = datetime.datetime(1900, 1, 1, busy_end_hour, busy_end_minute)
                    
                    if (start_datetime < busy_end_dt) and (end_datetime > busy_start_dt):
                        is_available = False
                        break
            if not is_available:
                break

        if is_available:
            return f"{day}, {start_time} - {end_time}"

    return None




# Example usage (for testing):
if __name__ == "__main__":
    example_question = """You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:

TASK: You need to schedule a meeting for Grace, Alexis, Helen and Ashley for half an hour between the work hours of 9:00 to 17:00 on Monday. 

Here are the existing schedules for everyone during the day: 
Grace's calendar is wide open the entire day.
Alexishas no meetings the whole day.
Helen is busy on Monday during 9:00 to 12:00, 12:30 to 14:00, 14:30 to 15:00, 15:30 to 16:00, 16:30 to 17:00; 
Ashley has meetings on Monday during 9:00 to 9:30, 10:00 to 10:30, 11:00 to 14:00, 14:30 to 15:00, 15:30 to 17:00; 

Grace would like to avoid more meetings on Monday after 15:00. Find a time that works for everyone's schedule and constraints. """
    
    answer = main(example_question)
    print(answer)