import re
import datetime

def main(question):
    """
    Schedules a meeting based on participant availability and constraints, using an LLM-inspired, iterative refinement approach.
    This approach moves away from rigid regex parsing to leverage LLM-style iterative refinement.
    """
    try:
        # Initial extraction of information (can be made more robust with LLM calls in future iterations)
        task_description, participant_schedules_str = question.split("Here are the existing schedules for everyone during the day: \n")
        
        participants = extract_participants(task_description)
        schedules = parse_schedules(participant_schedules_str)
        duration = extract_duration(task_description)
        work_hours = (9, 17)  # Assuming 9:00 to 17:00 always

        # Initial candidate time slots (coarse-grained)
        candidate_slots = generate_candidate_slots(work_hours, duration)

        # Iterative refinement of candidate slots based on constraints
        refined_slots = refine_candidate_slots(candidate_slots, schedules, participants)

        if refined_slots:
            # Select the first available slot as the proposed meeting time
            start_time = refined_slots[0]
            end_time = (datetime.datetime.combine(datetime.date.today(), start_time) + datetime.timedelta(minutes=duration)).time()
            answer = f"Here is the proposed time: Monday, {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')} "
        else:
            answer = "No suitable meeting time found."

        return answer

    except Exception as e:
        return f"Error: Could not schedule meeting. {str(e)}"


def extract_participants(task_description):
    """Extracts participant names from the task description."""
    match = re.search(r"schedule a meeting for (.*?) for", task_description)
    if match:
        return [name.strip() for name in match.group(1).split(',')]
    else:
        return []

def parse_schedules(schedule_string):
    """Parses the schedules of each participant, handling missing data."""
    schedules = {}
    lines = schedule_string.strip().split('\n')
    for line in lines:
        if "calendar is wide open" in line:
            name = line.split("'s calendar")[0]
            schedules[name] = []  # No blocked times
        elif "has no meetings" in line:
            name = line.split("has no meetings")[0]
            schedules[name] = []
        else:
            match = re.match(r"(\w+) has meetings on Monday during (.*);", line) # handles "meetings" format
            if match:
                 name = match.group(1)
                 times = match.group(2).split(", ")
                 blocked_times = []
                 for time_range in times:
                      try:
                           start, end = time_range.split(" to ")
                           blocked_times.append((start, end))
                      except: # skip badly formatted string
                           pass
                 schedules[name] = blocked_times
            else:
                 match = re.match(r"(\w+) has blocked their calendar on Monday during (.*);", line) # handles "blocked" format
                 if match:
                      name = match.group(1)
                      times = match.group(2).split(", ")
                      blocked_times = []
                      for time_range in times:
                           try:
                                start, end = time_range.split(" to ")
                                blocked_times.append((start, end))
                           except: # skip badly formatted string
                                pass
                      schedules[name] = blocked_times

    return schedules

def extract_duration(task_description):
    """Extracts the meeting duration in minutes from the task description."""
    match = re.search(r"for (.*?) between", task_description)
    if match:
        duration_str = match.group(1).strip()
        if "half an hour" in duration_str:
            return 30
        elif "one hour" in duration_str:
            return 60
    return 30  # Default to 30 minutes


def generate_candidate_slots(work_hours, duration):
    """Generates candidate time slots within the work hours."""
    start_hour, end_hour = work_hours
    slots = []
    current_time = datetime.time(start_hour, 0)
    end_time = datetime.time(end_hour, 0)

    while current_time < end_time:
        slots.append(current_time)
        current_time = (datetime.datetime.combine(datetime.date.today(), current_time) + datetime.timedelta(minutes=30)).time()  # Increment by 30 mins
    return slots


def refine_candidate_slots(candidate_slots, schedules, participants):
    """Refines candidate time slots by checking for conflicts with participant schedules."""
    refined_slots = []
    for slot in candidate_slots:
        is_available = True
        for participant in participants:
            if participant in schedules:
                blocked_times = schedules[participant]
                for blocked_start, blocked_end in blocked_times:
                    blocked_start_time = datetime.datetime.strptime(blocked_start, "%H:%M").time()
                    blocked_end_time = datetime.datetime.strptime(blocked_end, "%H:%M").time()

                    if not (slot >= blocked_end_time or (datetime.datetime.combine(datetime.date.today(), slot) + datetime.timedelta(minutes=extract_duration("for half an hour between"))).time() <= blocked_start_time):
                        is_available = False
                        break
            else:
                #If person doesn't have a schedule, assume they are busy
                is_available = False
                break # No schedule found for person, just move on

        if is_available:
            refined_slots.append(slot)

    return refined_slots