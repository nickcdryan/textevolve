import re
import datetime

def main(question):
    """
    Schedules a meeting for a group of people given their availability.

    This approach will use a simulated LLM that leverages chain-of-thought reasoning
    to determine the best meeting time. Instead of directly extracting information
    with regex (which has been done in previous iterations), this approach will
    simulate LLM calls for each step, breaking down the problem into smaller, manageable parts.

    Args:
        question (str): A string containing the meeting scheduling task and participant schedules.

    Returns:
        str: A string containing the proposed meeting time, or an error message if no suitable time is found.
    """

    try:
        # --- Simulated LLM Reasoning Steps ---

        # 1. Decomposition & Information Extraction
        task_details = llm_decompose_task(question)

        # 2. Schedule Parsing
        schedules = llm_parse_schedules(task_details["schedules"])

        # 3. Constraint Identification
        constraints = llm_identify_constraints(task_details)

        # 4. Time Slot Generation & Conflict Resolution
        proposed_time = llm_find_available_slot(schedules, constraints)

        # 5. Solution Formatting
        answer = format_answer(proposed_time)

        return answer

    except Exception as e:
        return f"Error: Could not schedule meeting. {str(e)}"


# --- Simulated LLM Functions ---

def llm_decompose_task(question):
    """
    Simulates an LLM decomposing the task into sub-components
    and extracting high-level information.

    This function identifies participants, meeting duration, work hours, and schedules.
    """

    participants_match = re.search(r"schedule a meeting for (.*?) for", question)
    participants = [p.strip() for p in participants_match.group(1).split(",") if participants_match] if participants_match else []

    duration_match = re.search(r"for (.*?) between", question)
    duration = duration_match.group(1) if duration_match else "one hour"  # Default duration

    schedules_start = question.find("Here are the existing schedules")
    schedules = question[schedules_start:] if schedules_start != -1 else ""

    return {
        "participants": participants,
        "duration": duration,
        "schedules": schedules
    }


def llm_parse_schedules(schedule_text):
    """
    Simulates an LLM parsing the schedules of each participant
    using chain-of-thought reasoning. Instead of strict regex, this uses a more
    flexible matching to simulate the robustness of an LLM.
    """

    schedules = {}
    schedule_lines = schedule_text.split("\n")

    for line in schedule_lines:
        if "calendar is wide open" in line:
            name = line.split("'s")[0].split(" ")[-1]  # Extract name
            schedules[name] = []  # Empty list indicates fully available
        elif "has meetings on Monday" in line or "is busy on Monday" in line or "has blocked their calendar on Monday" in line:
            name = line.split(" ")[0]
            schedule_str = line.split("during")[1].strip()
            times = re.findall(r"(\d{1,2}:\d{2})\s*to\s*(\d{1,2}:\d{2})", schedule_str)
            schedules[name] = [(start, end) for start, end in times]
        elif len(line.strip()) > 0 and "Here are the existing schedules" not in line:
          #Handles other sentences in the prompt that could be confused as schedules
          continue
    return schedules


def llm_identify_constraints(task_details):
    """
    Simulates an LLM identifying constraints, including work hours and preferences.

    """
    work_hours_start = 9
    work_hours_end = 17
    preferences = []
    for p in task_details["participants"]:
        if "do not want to meet" in task_details["schedules"]:
            start_time_match = re.search(r"before (\d{1,2}:\d{2})", task_details["schedules"])
            if start_time_match:
                pref_time = start_time_match.group(1)
                preferences.append((p,pref_time))
    return {
        "work_hours_start": work_hours_start,
        "work_hours_end": work_hours_end,
        "meeting_duration": task_details["duration"],
        "preferences": preferences
    }

def llm_find_available_slot(schedules, constraints):
    """
    Simulates an LLM finding a valid time slot by reasoning through
    the constraints and available times.
    """

    start_time = constraints["work_hours_start"]
    end_time = constraints["work_hours_end"]
    duration = constraints["meeting_duration"]

    if "half an hour" in duration:
        meeting_duration_minutes = 30
    else:
        meeting_duration_minutes = 60

    for hour in range(start_time, end_time):
        for minute in range(0, 60, 30 if meeting_duration_minutes == 30 else 60):  # check every 30 min for half hour meetings
            meeting_start_time = datetime.datetime(1900, 1, 1, hour, minute)
            meeting_end_time = meeting_start_time + datetime.timedelta(minutes=meeting_duration_minutes)

            # Check if within work hours
            if meeting_end_time.hour > end_time or (meeting_end_time.hour == end_time and meeting_end_time.minute > 0):
                continue

            is_available = True
            for person, busy_times in schedules.items():
                for busy_start, busy_end in busy_times:
                    busy_start_time = datetime.datetime.strptime(busy_start, "%H:%M").time()
                    busy_end_time = datetime.datetime.strptime(busy_end, "%H:%M").time()
                    
                    busy_start_dt = datetime.datetime.combine(datetime.date.today(), busy_start_time)
                    busy_end_dt = datetime.datetime.combine(datetime.date.today(), busy_end_time)
                    meeting_start_dt = datetime.datetime.combine(datetime.date.today(), meeting_start_time.time())
                    meeting_end_dt = datetime.datetime.combine(datetime.date.today(), meeting_end_time.time())
                    
                    if meeting_start_dt < busy_end_dt and meeting_end_dt > busy_start_dt:
                        is_available = False
                        break
                if not is_available:
                    break

            if is_available:
                # Check preferences
                preference_met = True
                for person, pref_time in constraints["preferences"]:
                    pref_time_dt = datetime.datetime.strptime(pref_time, "%H:%M").time()
                    pref_time_dt = datetime.datetime.combine(datetime.date.today(), pref_time_dt)
                    if person in schedules.keys() and meeting_start_time.time() < pref_time_dt.time():
                        preference_met = False
                        break

                if preference_met:
                    return meeting_start_time.strftime("%H:%M") + " - " + meeting_end_time.strftime("%H:%M")

    return None


# --- Helper Functions ---

def format_answer(time_slot):
    """
    Formats the answer string.
    """
    if time_slot:
        return f"Here is the proposed time: Monday, {time_slot} "
    else:
        return "No suitable time found."


# --- Execution ---
if __name__ == "__main__":
    example_question = "You are an expert at scheduling meetings. You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. Note there exists a solution that works with existing schedule of every participant. Here are a few example tasks and solutions:\n\nTASK: You need to schedule a meeting for Jeffrey, Jack and Judith for half an hour between the work hours of 9:00 to 17:00 on Monday. \n\nHere are the existing schedules for everyone during the day: \nJeffrey's calendar is wide open the entire day.\nJack has meetings on Monday during 10:30 to 11:30, 13:00 to 13:30, 14:00 to 14:30, 16:00 to 17:00; \nJudith is busy on Monday during 9:00 to 10:00, 10:30 to 11:00, 11:30 to 14:00, 14:30 to 15:00, 15:30 to 17:00; \n\nJeffrey do not want to meet on Monday before 14:00. Find a time that works for everyone's schedule and constraints. \nSOLUTION: "
    answer = main(example_question)
    print(answer)