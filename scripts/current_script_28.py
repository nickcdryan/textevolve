import re
import datetime

def main(question):
    """
    Schedules a meeting given constraints on participants' schedules,
    meeting duration, and time preferences, using an LLM-inspired approach.

    This approach simulates LLM reasoning by using a series of functions
    that mimic information extraction, constraint satisfaction, and
    iterative refinement, without directly calling an LLM.
    """
    try:
        # 1. Extract information using regex, simulating LLM parsing
        meeting_details = extract_meeting_details(question)
        schedules = extract_schedules(question)
        preferences = extract_preferences(question)

        # 2. Generate candidate meeting times
        candidate_times = generate_candidate_times(meeting_details['start_time'],
                                                     meeting_details['end_time'],
                                                     meeting_details['duration'])

        # 3. Filter and refine candidate times based on constraints and preferences
        available_times = filter_available_times(candidate_times, schedules)
        refined_times = refine_times_with_preferences(available_times, preferences)

        # 4. Select the best meeting time
        if refined_times:
            best_time = refined_times[0]
            return f"Here is the proposed time: Monday, {best_time}"
        else:
            return "No suitable meeting time found."

    except Exception as e:
        return f"Error: Could not schedule meeting. {str(e)}"


def extract_meeting_details(question):
    """
    Extracts meeting duration, start and end times, and participants from the question.
    """
    try:
        participants_match = re.search(r"schedule a meeting for (.*?) for", question)
        if not participants_match:
            raise ValueError("Could not extract participants.")
        participants = [name.strip() for name in participants_match.group(1).split(',')]

        duration_match = re.search(r"for (.*?) between", question)
        if not duration_match:
            raise ValueError("Could not extract meeting duration.")
        duration_str = duration_match.group(1)
        if "half an hour" in duration_str:
            duration = 30
        elif "hour" in duration_str:
            duration_num = int(duration_str.split(" ")[0])
            duration = duration_num * 60
        else:
            raise ValueError("Unsupported duration format.")

        time_range_match = re.search(r"between the work hours of (.*?) to (.*?) on Monday", question)
        if not time_range_match:
            raise ValueError("Could not extract time range.")
        start_time_str = time_range_match.group(1)
        end_time_str = time_range_match.group(2)

        return {
            "participants": participants,
            "duration": duration,
            "start_time": start_time_str,
            "end_time": end_time_str
        }
    except Exception as e:
        raise ValueError(f"Error extracting meeting details: {str(e)}")


def extract_schedules(question):
    """
    Extracts existing schedules for each participant from the question.
    """
    schedules = {}
    lines = question.split("\n")
    for line in lines:
        if "has meetings on Monday during" in line or "has blocked their calendar on Monday during" in line or "'s calendar is wide open the entire day." in line or "is free the entire day." in line:
            participant_match = re.search(r"^(.*?) (?:has meetings on Monday during|has blocked their calendar on Monday during|'s calendar is wide open the entire day.|is free the entire day.)", line)

            if participant_match:
                participant = participant_match.group(1).strip()
                if "'s calendar is wide open the entire day." in line or "is free the entire day." in line:
                  schedules[participant] = []
                  continue
                schedule_match = re.search(r"during (.*?);", line)
                if schedule_match:
                    schedule_str = schedule_match.group(1)
                    schedule_entries = schedule_str.split(", ")
                    schedules[participant] = []
                    for entry in schedule_entries:
                        time_range = entry.split(" to ")
                        if len(time_range) == 2:
                            schedules[participant].append(time_range)
    return schedules


def extract_preferences(question):
    """
    Extracts meeting time preferences from the question.
    """
    preferences = {}
    if "would rather not meet on Monday after" in question:
        preference_match = re.search(r"would rather not meet on Monday after (.*?)\.", question)
        if preference_match:
            preferences['avoid_after'] = preference_match.group(1)

    if "can not meet on Monday after" in question:
      preference_match = re.search(r"can not meet on Monday after (.*?)\.", question)
      if preference_match:
          preferences['avoid_after'] = preference_match.group(1)
    return preferences


def generate_candidate_times(start_time_str, end_time_str, duration):
    """
    Generates candidate meeting times in 30-minute increments.
    """
    start_time = datetime.datetime.strptime(start_time_str, "%H:%M").time()
    end_time = datetime.datetime.strptime(end_time_str, "%H:%M").time()
    current_time = datetime.datetime.combine(datetime.date.today(), start_time)
    end_datetime = datetime.datetime.combine(datetime.date.today(), end_time)

    candidate_times = []
    while current_time + datetime.timedelta(minutes=duration) <= end_datetime:
        candidate_times.append(current_time.strftime("%H:%M - ") +
                               (current_time + datetime.timedelta(minutes=duration)).strftime("%H:%M"))
        current_time += datetime.timedelta(minutes=30)

    return candidate_times


def filter_available_times(candidate_times, schedules):
    """
    Filters out candidate meeting times that conflict with existing schedules.
    """
    available_times = []
    for time_slot in candidate_times:
        start_time_str, end_time_str = time_slot.split(" - ")
        start_time = datetime.datetime.strptime(start_time_str, "%H:%M").time()
        end_time = datetime.datetime.strptime(end_time_str, "%H:%M").time()

        is_available = True
        for participant, busy_slots in schedules.items():
            for busy_slot in busy_slots:
                busy_start_str, busy_end_str = busy_slot
                busy_start = datetime.datetime.strptime(busy_start_str, "%H:%M").time()
                busy_end = datetime.datetime.strptime(busy_end_str, "%H:%M").time()

                if start_time < busy_end and end_time > busy_start:
                    is_available = False
                    break
            if not is_available:
                break

        if is_available:
            available_times.append(time_slot)

    return available_times


def refine_times_with_preferences(available_times, preferences):
    """
    Refines available meeting times based on preferences.
    """
    refined_times = available_times[:]  # Create a copy

    if 'avoid_after' in preferences:
        avoid_after_time = datetime.datetime.strptime(preferences['avoid_after'], "%H:%M").time()
        refined_times = [time for time in refined_times
                         if datetime.datetime.strptime(time.split(" - ")[0], "%H:%M").time() < avoid_after_time]

    return refined_times